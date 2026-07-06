import jetbrains.buildServer.configs.kotlin.*
import jetbrains.buildServer.configs.kotlin.buildFeatures.commitStatusPublisher
import jetbrains.buildServer.configs.kotlin.buildFeatures.pullRequests
import jetbrains.buildServer.configs.kotlin.buildSteps.script
import jetbrains.buildServer.configs.kotlin.triggers.schedule
import jetbrains.buildServer.configs.kotlin.triggers.vcs
import jetbrains.buildServer.configs.kotlin.vcs.GitVcsRoot

version = "2026.1"

project {
    description = "HydroMT Wflow system tests"

    params {
        param("wflow.dev.branch", "master")
        param("wflow.latest.release", "v1.0.3")
        param("wflow.oldest.supported.release", "v1.0.0")
    }

    vcsRoot(HydromtWflow)

    template(SystemTestTemplate)
    template(GitHubPrTemplate)
    template(NightlyTriggerTemplate)

    buildType(SystemTestLatest)
    buildType(SystemTestDev)
    buildType(SystemTestOldest)
}

// =============================================================================
// VCS Roots
// =============================================================================

object HydromtWflow : GitVcsRoot({
    name = "hydromt_wflow"
    url = "https://github.com/Deltares/hydromt_wflow"
    branch = "main"
    branchSpec = """
        +:refs/heads/main
        +:refs/heads/release/*
    """.trimIndent()
    authMethod = password {
        userName = "deltares-service-account"
        password = "%github_deltares-service-account_access_token%"
    }
    agentCleanPolicy = AgentCleanPolicy.ON_BRANCH_CHANGE
    agentCleanFilesPolicy = AgentCleanFilesPolicy.ALL_UNTRACKED
    param("submoduleCheckout", "CHECKOUT")
    param("useAlternates", "AUTO")
})

// =============================================================================
// Templates
// =============================================================================

object SystemTestTemplate : Template({
    name = "System Test Template"
    description = "Parameterized system test: builds and runs a model against a specific Wflow CLI version"

    params {
        param("wflow.cli.branch.filter", "")
        select("system.test.model", "piave", label = "Test model",
            description = "Test model to build and run",
            options = listOf("piave" to "Piave", "moselle" to "Moselle"))
        param("system.test.root", """%teamcity.agent.work.dir%\system-test\%system.test.model%""")
    }

    vcs {
        root(HydromtWflow, "+:. => ./hydromt_wflow")
    }

    steps {
        script {
            id = "build_and_run_sbm"
            name = "Build and run SBM"
            workingDir = "hydromt_wflow"
            scriptContent = """
                @if not exist "%teamcity.agent.work.dir%\wflow_cli\bin\wflow_cli.exe" (
                    echo ERROR: wflow_cli.exe not found at "%teamcity.agent.work.dir%\wflow_cli\bin\wflow_cli.exe"
                    exit /b 1
                )

                REM Build
                pixi run build-system-test-sbm "%system.test.root%\wflow_sbm"

                REM Run
                "%teamcity.agent.work.dir%\wflow_cli\bin\wflow_cli.exe" "%system.test.root%\wflow_sbm\wflow_sbm.toml"
            """.trimIndent()
        }
        script {
            id = "build_and_run_sediment"
            name = "Build and run sediment"
            workingDir = "hydromt_wflow"
            scriptContent = """
                REM Build
                pixi run build-system-test-sediment "%system.test.root%\wflow_sbm"

                REM Run
                "%teamcity.agent.work.dir%\wflow_cli\bin\wflow_cli.exe" "%system.test.root%\wflow_sbm\wflow_sediment.toml"
            """.trimIndent()
        }
        script {
            id = "assert_regression"
            name = "Assert regression"
            workingDir = "hydromt_wflow"
            scriptContent = """pixi run test-regression "%system.test.root%\wflow_sbm""""
        }
    }

    requirements {
        contains("teamcity.agent.jvm.os.name", "Windows")
    }

    dependencies {
        artifacts(AbsoluteId("wflow_BuildWflowCliWindows")) {
            buildRule = lastSuccessful("%wflow.cli.branch.filter%")
            cleanDestination = true
            artifactRules = """+:wflow_cli.zip!/wflow_cli/** => %teamcity.agent.work.dir%\wflow_cli"""
        }
    }
})

object GitHubPrTemplate : Template({
    name = "GitHub PR Template"
    description = "VCS triggers, PR build feature, commit status publisher"

    params {
        param("status.check.name", "")
    }

    vcs {
        root(HydromtWflow)
    }

    triggers {
        vcs {
            id = "TRIGGER_PR"
            branchFilter = """
                -:*
                +pr:*
                +:<default>
            """.trimIndent()
            enableQueueOptimization = true
            param("quietPeriodMode", "DO_NOT_USE")
        }
    }

    features {
        commitStatusPublisher {
            id = "BUILD_EXT_COMMIT_STATUS"
            vcsRootExtId = "${HydromtWflow.id}"
            publisher = github {
                githubUrl = "https://api.github.com"
                authType = vcsRoot()
            }
            param("build_custom_name", "%status.check.name%")
        }
        pullRequests {
            id = "BUILD_EXT_PR"
            vcsRootExtId = "${HydromtWflow.id}"
            provider = github {
                authType = vcsRoot()
                filterTargetBranch = """
                    +:refs/heads/main
                    +:refs/heads/release/*
                """.trimIndent()
                ignoreDrafts = true
            }
            param("filterAuthorRole", "MEMBER")
        }
    }
})

object NightlyTriggerTemplate : Template({
    name = "Nightly Trigger Template"
    description = "Scheduled nightly trigger for regression testing"

    triggers {
        schedule {
            id = "TRIGGER_NIGHTLY"
            schedulingPolicy = cron {
                hours = "2"
                minutes = "0"
                timezone = "Europe/Amsterdam"
            }
            enableQueueOptimization = true
            param("triggerBuildWithPendingChangesOnly", "true")
        }
    }
})

// =============================================================================
// Build Configurations (matrix)
// =============================================================================

object SystemTestLatest : BuildType({
    name = "System test (latest release)"
    description = "Piave SBM + Sediment against Wflow latest release (%wflow.latest.release%)"

    templates(SystemTestTemplate, GitHubPrTemplate)

    params {
        param("wflow.cli.branch.filter", "+:%wflow.latest.release%")
        param("system.test.model", "piave")
        param("status.check.name", "System test (Wflow %wflow.latest.release%)")
    }
})

object SystemTestDev : BuildType({
    name = "System test (dev)"
    description = "Piave SBM + Sediment against the latest Wflow.jl master build"

    templates(SystemTestTemplate, GitHubPrTemplate)

    params {
        param("wflow.cli.branch.filter", "+:%wflow.dev.branch%")
        param("system.test.model", "piave")
        param("status.check.name", "System test (Wflow-dev)")
    }
})

object SystemTestOldest : BuildType({
    name = "System test (oldest supported)"
    description = "Piave SBM + Sediment against Wflow oldest supported release (%wflow.oldest.supported.release%)"

    templates(SystemTestTemplate, NightlyTriggerTemplate)

    params {
        param("wflow.cli.branch.filter", "+:%wflow.oldest.supported.release%")
        param("system.test.model", "piave")
    }
})
