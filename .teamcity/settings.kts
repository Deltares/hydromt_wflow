import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.PullRequests
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildFeatures.emailNotifier
import jetbrains.buildServer.configs.kotlin.buildFeatures.notifications
import jetbrains.buildServer.configs.kotlin.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.schedule
import jetbrains.buildServer.configs.kotlin.triggers.vcs
import jetbrains.buildServer.configs.kotlin.vcs.GitVcsRoot

/*
This file is meant to live in hydromt_wflow's own .teamcity/ directory
(versioned settings), which is why hydromt_wflow itself is referenced via
DslContext.settingsRoot rather than a second, hand-declared VCS root.

Five build configurations, one template stack:

  SystemTest              - manual/debug run, no trigger
  SystemTestPrCheck        - hydromt_wflow PR -> GitHub check
  SystemTestDev             - Wflow.jl master commit -> email on failure
  SystemTestLatestRelease   - Wflow.jl tag push -> email on failure
  SystemTestOldestSupported - nightly schedule -> email on failure

PR-triggered and Wflow.jl-triggered builds are deliberately split into
different build configurations. A single build config can't publish a
GitHub check on one trigger cause and an email on another, so honouring
"no commit status for wflow.jl, email instead" (see team recap) means the
two trigger causes need separate build types, even though they share the
exact same build/run steps template.
*/

version = "2026.1"

project {

    vcsRoot(WflowJl)

    buildType(SystemTest)
    buildType(SystemTestPrCheck)
    buildType(SystemTestDev)
    buildType(SystemTestLatestRelease)
    buildType(SystemTestOldestSupported)

    template(WflowSystemTestTemplate)
    template(GitHubPrTemplate)
    template(WflowJlEmailTemplate)
    template(WflowWindowsAgentTemplate)

    params {
        // Explicit version pins agreed with the Wflow.jl team - bump these by
        // hand when a new release should become "latest"/"oldest supported".
        param("wflow.dev.branch", "master")
        param("wflow.latest.release", "v1.0.3")
        param("wflow.oldest.supported.release", "v1.0.0")

        // Who gets paged when a Wflow.jl-triggered or nightly run breaks.
        // There's no hydromt_wflow PR to attach a GitHub check to in that
        // case, so we email instead (see team recap).
        param("notify.email", "wflow-ci@deltares.nl")
    }
}

// ---------------------------------------------------------------------------
// Build types
// ---------------------------------------------------------------------------

object SystemTest : BuildType({
    templates(WflowSystemTestTemplate, WflowWindowsAgentTemplate)
    name = "System test (manual)"
    description = "Build and run an SBM from scratch (artifact_data), then convert it to a sediment model and run that. No trigger - use 'Run...' and override wflow.cli.branch.filter to point at a specific wflow_cli build."

    params {
        param("wflow.cli.branch.filter", "+:refs/tags/%wflow.latest.release%")
    }
})

object SystemTestPrCheck : BuildType({
    templates(WflowSystemTestTemplate, GitHubPrTemplate, WflowWindowsAgentTemplate)
    name = "System test (PR check)"
    description = "Runs on every hydromt_wflow PR against the latest supported Wflow.jl release and publishes a GitHub check."

    params {
        param("wflow.cli.branch.filter", "+:refs/tags/%wflow.latest.release%")
        text("status.check.name", "System test (PR)", allowEmpty = false)
    }
})

object SystemTestDev : BuildType({
    templates(WflowSystemTestTemplate, WflowJlEmailTemplate, WflowWindowsAgentTemplate)
    name = "System test (Wflow-dev)"
    description = "Runs system test using the latest build of Wflow.jl %wflow.dev.branch%. Triggered by Wflow.jl, not a hydromt_wflow PR - failures are emailed, not posted as a GitHub check."

    params {
        param("wflow.cli.branch.filter", "+:refs/heads/%wflow.dev.branch%")
    }

    triggers {
        vcs {
            id = "TRIGGER_862"
            triggerRules = "+:root=${WflowJl.id}:**"
            branchFilter = "+:refs/heads/%wflow.dev.branch%"
        }
    }
})

object SystemTestLatestRelease : BuildType({
    templates(WflowSystemTestTemplate, WflowJlEmailTemplate, WflowWindowsAgentTemplate)
    name = "System test (Wflow latest release)"
    description = "Runs system test using the latest tagged Wflow.jl release (%wflow.latest.release%). Triggered by new Wflow.jl tags; failures are emailed."

    params {
        param("wflow.cli.branch.filter", "+:refs/tags/%wflow.latest.release%")
    }

    triggers {
        vcs {
            id = "TRIGGER_863"
            triggerRules = "+:root=${WflowJl.id}:**"
            branchFilter = "+:refs/tags/*"
        }
    }
})

object SystemTestOldestSupported : BuildType({
    templates(WflowSystemTestTemplate, WflowJlEmailTemplate, WflowWindowsAgentTemplate)
    name = "System test (Wflow oldest supported)"
    description = "Nightly canary against the oldest release we still claim to support (%wflow.oldest.supported.release%). Also doubles as the 'catch a silent upstream dependency regression' check, since nothing else re-runs this pipeline without a hydromt_wflow or Wflow.jl commit."

    params {
        param("wflow.cli.branch.filter", "+:refs/tags/%wflow.oldest.supported.release%")
    }

    triggers {
        schedule {
            id = "TRIGGER_NIGHTLY"
            schedulingPolicy = daily {
                hour = 2
            }
            triggerBuild = always()
            withPendingChangesOnly = false
        }
    }
})

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------

object WflowSystemTestTemplate : Template({
    name = "System Test Template"
    description = "VcsRoots, Dependencies & Build steps"

    params {
        text(
            "wflow.cli.branch.filter", "",
            description = "Newline-delimited set of rules in the form of +|-:logical branch name (with an optional * placeholder) picking which wflow_cli build to fetch. Every build type using this template must set this.",
            allowEmpty = false
        )
    }

    vcs {
        root(DslContext.settingsRoot, "+:. => ./hydromt_wflow")
    }

    steps {
        script {
            name = "Build and run sbm"
            id = "Build_sbm"
            workingDir = "hydromt_wflow"
            scriptContent = """
                @if not exist "%teamcity.agent.work.dir%\wflow_cli\bin\wflow_cli.exe" (
                    echo ERROR: wflow_cli.exe not found at "%teamcity.agent.work.dir%\wflow_cli\bin\wflow_cli.exe"
                    exit /b 1
                )

                REM Build
                pixi run build-system-test-sbm "%teamcity.agent.work.dir%\system-test\wflow_sbm"

                REM Run
                "%teamcity.agent.work.dir%\wflow_cli\bin\wflow_cli.exe" "%teamcity.agent.work.dir%\system-test\wflow_sbm\wflow_sbm.toml"
            """.trimIndent()
        }
        script {
            name = "Build and run sediment"
            id = "build_and_run_sediment"
            workingDir = "hydromt_wflow"
            scriptContent = """
                REM Build
                pixi run build-system-test-sediment "%teamcity.agent.work.dir%\system-test\wflow_sbm"

                REM Run
                "%teamcity.agent.work.dir%\wflow_cli\bin\wflow_cli.exe" "%teamcity.agent.work.dir%\system-test\wflow_sbm\wflow_sediment.toml"
            """.trimIndent()
        }
    }

    dependencies {
        artifacts(AbsoluteId("wflow_BuildWflowCliWindows")) {
            id = "ARTIFACT_DEPENDENCY_7064"
            buildRule = lastSuccessful("%wflow.cli.branch.filter%")
            cleanDestination = true
            artifactRules = """+:wflow_cli.zip!/wflow_cli/** => %teamcity.agent.work.dir%\wflow_cli"""
        }
    }
})

object GitHubPrTemplate : Template({
    name = "GitHub PR Template"
    description = "PR trigger + commit status publisher for hydromt_wflow-PR-triggered builds only. Do not combine with WflowJlEmailTemplate."

    params {
        text("status.check.name", "", allowEmpty = false)
    }

    triggers {
        vcs {
            id = "TRIGGER_858"
            triggerRules = "+:root=${DslContext.settingsRoot.id}:**"
            branchFilter = """
                -:*
                +pr:*
                +:<default>
            """.trimIndent()
        }
    }

    features {
        commitStatusPublisher {
            id = "BUILD_EXT_521"
            vcsRootExtId = "${DslContext.settingsRoot.id}"
            publisher = github {
                statusCheckName = "%status.check.name%"
                githubUrl = "https://api.github.com"
                authType = vcsRoot()
            }
        }
        pullRequests {
            id = "BUILD_EXT_522"
            vcsRootExtId = "${DslContext.settingsRoot.id}"
            provider = github {
                authType = vcsRoot()
                filterTargetBranch = """
                    +:refs/heads/main
                    +:refs/heads/release/*
                """.trimIndent()
                filterAuthorRole = PullRequests.GitHubRoleFilter.MEMBER
                ignoreDrafts = true
            }
        }
    }
})

object WflowJlEmailTemplate : Template({
    name = "Wflow.jl VCS root + email on failure"
    description = "Attaches Wflow.jl (for triggering/version pinning) and emails on failure instead of publishing a GitHub check, since these builds aren't tied to a hydromt_wflow commit or PR. Do not combine with GitHubPrTemplate."

    vcs {
        root(WflowJl)
    }

    features {
        notifications {
            id = "BUILD_EXT_EMAIL"
            notifierSettings = emailNotifier {
                email = "%notify.email%"
            }
            buildFailed = true
        }
    }
})

object WflowWindowsAgentTemplate : Template({
    name = "Windows Agent Template"
    description = "Requires Windows os"

    requirements {
        contains("teamcity.agent.jvm.os.name", "Windows", "RQ_858")
    }
})

object WflowJl : GitVcsRoot({
    name = "Wflow.jl"
    url = "https://github.com/Deltares/Wflow.jl.git"
    branch = "master"
    branchSpec = """
        +:refs/heads/master
        +:refs/tags/(v*)
    """.trimIndent()
})
