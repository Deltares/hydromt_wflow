import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.PullRequests
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.vcs
import jetbrains.buildServer.configs.kotlin.vcs.GitVcsRoot

/*
The settings script is an entry point for defining a TeamCity
project hierarchy. The script should contain a single call to the
project() function with a Project instance or an init function as
an argument.

VcsRoots, BuildTypes, Templates, and subprojects can be
registered inside the project using the vcsRoot(), buildType(),
template(), and subProject() methods respectively.

To debug settings scripts in command-line, run the

    mvnDebug org.jetbrains.teamcity:teamcity-configs-maven-plugin:generate

command and attach your debugger to the port 8000.

To debug in IntelliJ Idea, open the 'Maven Projects' tool window (View
-> Tool Windows -> Maven Projects), find the generate task node
(Plugins -> teamcity-configs -> teamcity-configs:generate), the
'Debug' option is available in the context menu for the task.
*/

version = "2026.1"

project {

    vcsRoot(WflowJl)

    buildType(SystemTest)
    buildType(SystemTestDev)

    template(GitHubPrTemplate)
    template(WflowWindowsAgentTemplate)
    template(WflowSystemTestTemplate)

    params {
        param("wflow.oldest.supported.release", "v1.0.0")
        param("wflow.dev.branch", "master")
        param("wflow.latest.release", "v1.0.3")
    }
}

object SystemTest : BuildType({
    templates(WflowSystemTestTemplate)
    name = "System test"
    description = "Build and run an SBM from scratch (artifact_data), then convert it to a sediment model and run that."
})

object SystemTestDev : BuildType({
    templates(WflowSystemTestTemplate, GitHubPrTemplate, WflowWindowsAgentTemplate)
    name = "System test (dev)"
    description = "Runs system test using the latest build of wflow.jl"

    params {
        text("status.check.name", "System test (Wflow-dev)", allowEmpty = false)
    }

    vcs {
        root(WflowJl)
    }

    triggers {
        vcs {
            id = "TRIGGER_862"
            branchFilter = """
                +:refs/heads/master
                +:refs/tags/*
            """.trimIndent()
        }
    }

    dependencies {
        dependency(AbsoluteId("wflow_BuildWflowCliWindows")) {
            snapshot {
                onDependencyFailure = FailureAction.FAIL_TO_START
            }

            artifacts {
                id = "ARTIFACT_DEPENDENCY_7064"
                buildRule = sameChain()
                cleanDestination = true
                artifactRules = """+:wflow_cli.zip!/wflow_cli/** => %teamcity.agent.work.dir%\wflow_cli"""
            }
        }
    }
})

object GitHubPrTemplate : Template({
    name = "GitHub PR Template"
    description = "vcs triggers, PR build feature, commit status publisher"

    params {
        text("status.check.name", "", allowEmpty = false)
    }

    vcs {
        root(DslContext.settingsRoot)
    }

    triggers {
        vcs {
            id = "TRIGGER_858"
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

object WflowSystemTestTemplate : Template({
    name = "System Test Template"
    description = "VcsRoots, Dependencies & Build steps"

    params {
        text("wflow.cli.version.latest", "v1.1.0",
              regex = """^v\d+\.\d+\.\d+${'$'}""", validationMessage = "Must be a valid 'vx.y.z' version (for example, v1.1.0)")
        text("wflow.cli.branch.filter", "", description = "Newline-delimited set of rules in the form of +|-:logical branch name (with an optional * placeholder)", allowEmpty = false)
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
