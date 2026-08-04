import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.PullRequests
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
// import jetbrains.buildServer.configs.kotlin.buildFeatures.emailNotifier  // unresolved on this TeamCity instance - email notifier feature disabled; see WflowJlEmailTemplate below
// import jetbrains.buildServer.configs.kotlin.buildFeatures.notifications  // only used by the disabled WflowJlEmailTemplate - uncomment alongside it
import jetbrains.buildServer.configs.kotlin.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.buildSteps.powerShell
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.schedule
import jetbrains.buildServer.configs.kotlin.triggers.vcs
import jetbrains.buildServer.configs.kotlin.vcs.GitVcsRoot

/*
This file is meant to live in hydromt_wflow's own .teamcity/ directory
(versioned settings), which is why hydromt_wflow itself is referenced via
DslContext.settingsRoot rather than a second, hand-declared VCS root.

Five build configurations, one template stack:

  SystemTestPrCheckStable   - hydromt_wflow PR -> GitHub check, uses latest wflow.jl release
  SystemTestPrCheckDev      - hydromt_wflow PR -> GitHub check, uses wflow.jl@main
  SystemTestDev             - nightly schedule, uses wflow.jl@main, full profile
  SystemTestLatestRelease   - nightly schedule, uses latest wflow.jl release, full profile
  SystemTestOldestSupported - nightly schedule, uses oldest supported wflow.jl release, full profile
*/

version = "2026.1"

project {

    vcsRoot(WflowJl)

    buildType(SystemTestPrCheckStable)
    buildType(SystemTestPrCheckDev)
    buildType(SystemTestDev)
    buildType(SystemTestLatestRelease)
    buildType(SystemTestOldestSupported)

    template(WflowSystemTestTemplate)
    template(GitHubPrTemplate)
    // template(WflowJlEmailTemplate)
    template(WflowWindowsAgentTemplate)

    params {
        // Explicit version pins agreed with the Wflow.jl team.
        // - wflow.latest.release tracks a release *branch* on the wflow_cli
        //   build config, so "latest" always resolves to the newest build on
        //   that branch without needing a manual bump per patch release.
        // - wflow.oldest.supported.release is pinned to an exact *tag* - bump
        //   this by hand when the oldest release we still support changes.
        param("wflow.dev.branch", "main")
        param("wflow.latest.release", "release/v1.0")
        param("wflow.oldest.supported.release", "v1.0.0")

        // Who gets paged when a nightly run breaks.
        // There's no hydromt_wflow PR to attach a GitHub check to in that
        // case, so we email instead (see team recap).
        // param("notify.email", "wflow-ci@deltares.nl")

        // Earth Data Hub credentials for regression testing (secure parameters)
        password("env.EARTHDATAHUB_APIKEY", "credentialsJSON:6c0dcda0-ac7a-4304-915f-566fb8b37b54", description = "API key generated 28/7 via luuk.blom@deltares.nl at https://earthdatahub.destine.eu/getting-started#configuring-netrc", display = ParameterDisplay.HIDDEN, readOnly = true)
    }
}

// ---------------------------------------------------------------------------
// Build types
// ---------------------------------------------------------------------------

object SystemTestPrCheckStable : BuildType({
    templates(WflowSystemTestTemplate, GitHubPrTemplate, WflowWindowsAgentTemplate)
    name = "System test (PR check, latest release)"
    description = "Runs on every hydromt_wflow PR against the latest supported Wflow.jl release and publishes a GitHub check."

    params {
        param("wflow.cli.branch.filter", "+:%wflow.latest.release%")
        param("regression.profile", "pr")
        text("status.check.name", "Regression test (Wflow.jl @ %wflow.latest.release%)", allowEmpty = false)
    }
})

object SystemTestPrCheckDev : BuildType({
    templates(WflowSystemTestTemplate, GitHubPrTemplate, WflowWindowsAgentTemplate)
    name = "System test (PR check, wflow main)"
    description = "Runs on every hydromt_wflow PR against the latest build from Wflow.jl main and publishes a GitHub check."

    params {
        param("wflow.cli.branch.filter", "+:%wflow.dev.branch%")
        param("regression.profile", "pr")
        text("status.check.name", "Regression test (Wflow.jl @ %wflow.dev.branch%)", allowEmpty = false)
    }
})

object SystemTestDev : BuildType({
    templates(WflowSystemTestTemplate, WflowWindowsAgentTemplate) // WflowJlEmailTemplate,
    name = "System test (Nightly, Wflow main)"
    description = "Nightly run using the latest build of Wflow.jl %wflow.dev.branch% and the full regression profile."

    params {
        param("wflow.cli.branch.filter", "+:%wflow.dev.branch%")
        param("regression.profile", "all")
    }

    triggers {
        schedule {
            id = "TRIGGER_NIGHTLY_DEV"
            schedulingPolicy = daily {
                hour = 2
            }
            triggerBuild = always()
            withPendingChangesOnly = false
        }
    }
})

object SystemTestLatestRelease : BuildType({
    templates(WflowSystemTestTemplate, WflowWindowsAgentTemplate) // WflowJlEmailTemplate,
    name = "System test (Nightly, Wflow latest release)"
    description = "Nightly run using the latest build of the Wflow.jl %wflow.latest.release% release branch and the full regression profile."

    params {
        param("wflow.cli.branch.filter", "+:%wflow.latest.release%")
        param("regression.profile", "all")
    }

    triggers {
        schedule {
            id = "TRIGGER_NIGHTLY_LATEST_RELEASE"
            schedulingPolicy = daily {
                hour = 3
            }
            triggerBuild = always()
            withPendingChangesOnly = false
        }
    }
})

object SystemTestOldestSupported : BuildType({
    templates(WflowSystemTestTemplate, WflowWindowsAgentTemplate) // WflowJlEmailTemplate,
    name = "System test (Nightly, Wflow oldest supported)"
    description = "Nightly canary against the oldest release we still claim to support (%wflow.oldest.supported.release%). Also doubles as the 'catch a silent upstream dependency regression' check, since nothing else re-runs this pipeline without a hydromt_wflow or Wflow.jl commit."

    params {
        param("wflow.cli.branch.filter", "+:%wflow.oldest.supported.release%")
        param("regression.profile", "all")
    }

    triggers {
        schedule {
            id = "TRIGGER_NIGHTLY_OLDEST_SUPPORTED"
            schedulingPolicy = daily {
                hour = 4
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
        text(
            "regression.profile", "",
            description = "Basin profile for regression tasks (pr|all).",
            allowEmpty = false
        )
    }

    vcs {
        root(DslContext.settingsRoot, "+:. => ./hydromt_wflow")
    }

    steps {
        powerShell {
            name = "Connect to P drive"
            id = "Map_P_drive"
            scriptMode = script {
                content = """
                    if (-not (Test-Path 'P:\')) {
                        net use P: \\directory.intra\PROJECT /persistent:no
                    } else {
                        Write-Host 'P: drive already available, skipping net use.'
                    }
                """.trimIndent()
            }
        }
        powerShell {
            name = "Setup earthdatahub credentials"
            id = "Setup_earthdatahub_netrc"
            scriptMode = script {
                content = """
                    ${'$'}netrcFile = '%teamcity.build.checkoutDir%\_netrc'
                    Write-Host "##teamcity[setParameter name='env.NETRC' value='${'$'}netrcFile']"

                    if (Test-Path ${'$'}netrcFile) {
                        ${'$'}existing = Get-Content ${'$'}netrcFile -Raw -ErrorAction SilentlyContinue
                        if (${'$'}existing -match 'api\.earthdatahub\.destine\.eu') {
                            Write-Host 'earthdatahub entry already exists in _netrc, skipping'
                            exit 0
                        }
                        Add-Content -Path ${'$'}netrcFile -Value ''
                    }

                    @(
                        'machine api.earthdatahub.destine.eu'
                        'login apikey'
                        'password %env.EARTHDATAHUB_APIKEY%'
                    ) | Add-Content -Path ${'$'}netrcFile
                    Write-Host 'earthdatahub credentials added to _netrc'
                """.trimIndent()
            }
        }
        powerShell {
            name = "Build and run regression pipeline"
            id = "Build_run_regression_pipeline"
            workingDir = "hydromt_wflow"
            scriptMode = script {
                content = """
                    ${'$'}wflowCli = '%teamcity.build.checkoutDir%\wflow_cli\bin\wflow_cli.exe'
                    if (-not (Test-Path ${'$'}wflowCli)) {
                        Write-Error "wflow_cli.exe not found at ${'$'}wflowCli"
                        exit 1
                    }
                    ${'$'}env:WFLOW_CLI = ${'$'}wflowCli
                    pixi run regression-pipeline %regression.profile%
                    if (${'$'}LASTEXITCODE -ne 0) { exit ${'$'}LASTEXITCODE }
                """.trimIndent()
            }
        }
        powerShell {
            name = "Assert regression metrics"
            id = "assert_regression_metrics"
            workingDir = "hydromt_wflow"
            scriptMode = script {
                content = """
                    pixi run regression-assert '%regression.profile%'
                    if (${'$'}LASTEXITCODE -ne 0) { exit ${'$'}LASTEXITCODE }
                """.trimIndent()
            }
        }
    }

    dependencies {
        artifacts(AbsoluteId("wflow_BuildWflowCliWindows")) {
            id = "ARTIFACT_DEPENDENCY_7064"
            buildRule = lastSuccessful("%wflow.cli.branch.filter%")
            cleanDestination = true
            artifactRules = """+:wflow_cli.zip!/wflow_cli/** => %teamcity.build.checkoutDir%\wflow_cli"""
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

/* uncomment when email is enabled / fixed on teamcity
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
*/

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
    branch = "main"
    branchSpec = """
        +:refs/heads/main
        +:refs/tags/(v*)
    """.trimIndent()
})
