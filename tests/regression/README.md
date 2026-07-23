# Regression test suite

End-to-end regression tests that build a wflow model with HydroMT, run it with Wflow.jl,
and assert that key output metrics stay within tolerance of a stored baseline.

## Design

The suite is split into two independent stages that can run on different machines or
at different times.

**Stage 1 – pipeline** (`run_pipeline.py`)
Build the HydroMT model (SBM and sediment) and execute the Wflow.jl simulation.
Outputs are written to a directory of your choice (`ROOT`).

**Stage 2 – assertion** (`tests/test_system_regression.py`)
Load the model outputs produced by stage 1, compute scalar metrics, compare them
against the stored baseline, and fail if any deviation exceeds the configured tolerance.
TeamCity statistics are emitted as a side effect (see below).

The two stages are wired together via the `ROOT` directory path.

### Key files

```text
tests/regression/
  manifest.json            # maps profile names to basin lists
  regression_utils.py      # shared helpers (build, run, metrics, compare, report)
  run_pipeline.py          # stage 1 entry point
  generate_metrics.py      # regenerate baseline metrics from existing model outputs
  <basin>/
    config.json            # per-basin configuration (paths, metrics, tolerances)
    build_sbm.yml          # HydroMT build recipe for the SBM model
    build_sediment.yml     # HydroMT build recipe for the sediment model
    metrics/
      baseline_metrics.json  # committed baseline — what the output *should* look like
   test_regression.py   # pytest entry point for stage 2
```

### Profiles

`manifest.json` maps profile names to basin lists:

| Profile | Basins          | Used for               |
|---------|-----------------|------------------------|
| `pr`    | piave           | PR checks (fast)       |
| `all`   | piave, moselle  | Nightly full runs      |

### Metric specification

Each entry in `config.json → <model>.metrics` describes one group:

```jsonc
{
  "name": "outlet_discharge",   // identifier used in baseline and failure messages
  "variable": "river_q",        // variable name in the output NetCDF
  "selector": { "wflow_id": -1 }, // optional: isel coordinates to subset
  "aggregation": "sum",         // how to reduce spatial dims: sum | mean | max
  "metrics": ["mean", "peak", "total"], // statistics computed over the time axis
  "rel_tol": 0.01,              // relative tolerance (1%)
  "abs_tol": 1e-6               // absolute tolerance (used when expected ≈ 0)
}
```

The baseline file mirrors this structure, keyed by model and then metric name:

```json
{
  "sbm":      { "outlet_discharge": { "mean": 1.23, "peak": 4.56, "total": 789.0 } },
  "sediment": { "soil_loss":        { "mean": 0.01, "peak": 0.05, "total": 3.14  } }
}
```

---

## Running locally

All commands below assume you are in the repository root and use the `pixi` task runner.
Replace `<ROOT>` with a local directory for model outputs (e.g. `/tmp/regression-runs`).

### Full pipeline (build + run + assert)

```bash
# 1. Build and run models (requires wflow_cli on PATH or explicit path)
pixi run regression-pipeline pr <ROOT> wflow_cli

# 2. Assert metrics against baseline
pixi run regression-assert pr <ROOT>
```

Use profile `all` to include all basins instead of just piave.

### Partial runs (build only, no wflow execution)

```bash
# Build SBM model for a single basin
pixi run build-sbm piave

# Build sediment model for a single basin
pixi run build-sediment piave
```

The default `ROOT` for these tasks is `tests/regression/.runs` inside the repo.

### Regenerate baseline metrics

Run this after confirming that a deviation is intentional (see "Responding to failures" below):

```bash
pixi run regression-generate-metrics all <ROOT>
```

This overwrites the `baseline_metrics.json` files in `tests/regression/<basin>/metrics/`.
Commit the updated files together with the code change that caused the deviation.

### Assert against a pre-existing run directory

Stage 2 can be run independently if you already have model outputs:

```bash
pixi run regression-assert all /path/to/existing/run/root
```

---

## Responding to failures

When the assertion step fails, the test output explains what exceeded tolerance and
lists three follow-up options:

1. **Fix the regression** — identify the code change that shifted the output, correct it,
   re-run the pipeline, and verify the metrics pass.

2. **Accept the change and regenerate** — if the deviation is expected (e.g. a deliberate
   model improvement), regenerate the baseline metrics and commit the updated JSON files:
   ```bash
   pixi run regression-generate-metrics all <ROOT>
   ```
   Re-run the assertion to confirm the new baseline passes.

3. **Widen tolerances** — if the deviation is within acceptable physical bounds but
   exceeds the current threshold, update `rel_tol` / `abs_tol` in
   `tests/regression/<basin>/config.json`.

> **Warning — silent drift:** do not auto-accept or auto-regenerate metrics after every
> successful run. If the baseline is updated each time, each run only needs to stay within
> tolerance of the *previous* run rather than the original validated state. A small drift
> that passes every individual check can compound run-over-run until the model is
> silently producing physically wrong results. The baseline should represent a deliberate
> decision that "this output is physically correct". Verify any baseline update with the
> Wflow.jl team before committing.

---

## Adding a new basin

1. Create `tests/regression/<basin>/` with the following files:
   - `build_sbm.yml` — HydroMT build config for the SBM model.
   - `build_sediment.yml` — HydroMT build/update config for the sediment model.
   - `config.json` — points to the build configs, data catalogs, output paths, and
     metric specs. Use an existing basin as a template.

2. Create the baseline metrics directory and placeholder:
   ```bash
   mkdir tests/regression/<basin>/metrics/
   ```

3. Register the basin in `manifest.json` under the appropriate profile(s).

4. Run the full pipeline to generate model outputs, then generate the baseline:
   ```bash
   pixi run regression-pipeline all <ROOT> wflow_cli
   pixi run regression-generate-metrics all <ROOT>
   ```

5. Commit everything: build configs, `config.json`, `manifest.json`, and
   `baseline_metrics.json`.

---

## Adding a new metric to an existing basin

1. Add a metric spec to `tests/regression/<basin>/config.json` under the relevant model
   (`sbm` or `sediment`). Follow the metric specification schema above.

2. Generate updated baseline metrics from an existing run:
   ```bash
   pixi run regression-generate-metrics all <ROOT>
   ```

3. Commit the updated `config.json` and `baseline_metrics.json`.

---

## TeamCity integration

The TeamCity configuration lives in `.teamcity/settings.kts` (Kotlin DSL versioned settings).

### Build configurations

| Name | Trigger | Profile | Wflow.jl version |
|------|---------|---------|-----------------|
| System test (PR check, latest release) | Every PR to `main` / `release/*` | `pr` | latest release branch |
| System test (PR check, wflow master) | Every PR to `main` / `release/*` | `pr` | `master` |
| System test (Nightly, Wflow master) | Daily 02:00 | `all` | `master` |
| System test (Nightly, Wflow latest release) | Daily 03:00 | `all` | latest release branch |
| System test (Nightly, Wflow oldest supported) | Daily 04:00 | `all` | pinned oldest supported tag |

PR builds publish a GitHub commit status check. Nightly builds have email notification
configured (currently disabled pending TeamCity setup).

### Expected PR check behaviour

Both PR checks run against the same hydromt_wflow commit but use different Wflow.jl
versions (latest release vs master). It is **expected that one of them can fail** and
this is not necessarily a problem with hydromt_wflow.

Wflow.jl does not guarantee backwards-compatible config changes between versions. A
renamed parameter or changed default in the kernel can break the model run entirely.
hydromt_wflow handles config upgrades for users, but these regression tests deliberately
use the raw build output — they test the model output, not the upgrade path.

This means:

| Scenario | Meaning |
|----------|---------|
| Both checks pass | No breaking changes in either Wflow.jl version. |
| One check fails | A breaking change was introduced in that Wflow.jl version. Monitor it; coordinate with the Wflow.jl team when merging. |
| Both checks fail | A problem in hydromt_wflow itself. Must be fixed before merging. |

### Build steps per configuration

Each configuration runs two sequential steps:

1. **Build and run regression pipeline** — calls `pixi run regression-pipeline` with the
   profile and a work directory on the agent (`%teamcity.agent.work.dir%\system-test`).
2. **Assert regression metrics** — calls `pixi run regression-assert`, which runs pytest
   and fails the build if any metric is out of tolerance.

### wflow_cli artifact dependency

The Wflow.jl CLI binary is fetched as a TeamCity artifact from the `wflow_BuildWflowCliWindows`
build configuration. The branch filter (e.g. `+:release/v1.0`) selects which Wflow.jl
version to use. All builds run on a Windows agent.

### TeamCity statistics

`regression_utils.emit_teamcity_stats` prints a `##teamcity[buildStatisticValue ...]`
line for every metric value, using the key pattern:

```
regression_<basin>_<model>_<metric_name>_<metric_key>
```

These values appear on the build's Statistics tab and can be plotted as trend charts
across builds, making it easy to spot gradual drift even when values stay within tolerance.

---

## Updating the Kotlin DSL settings

The TeamCity configuration is defined in `.teamcity/settings.kts` as versioned Kotlin DSL.
Changes take effect when the file is committed and TeamCity picks up the update.

### Project-level version parameters

At the top of `settings.kts` the project block defines three parameters:

```kotlin
param("wflow.dev.branch",               "master")
param("wflow.latest.release",           "release/v1.0")
param("wflow.oldest.supported.release", "v1.0.0")
```

These are the only values that need to change when Wflow.jl ships a new release.

| Parameter | What it controls | How to update |
|-----------|-----------------|---------------|
| `wflow.dev.branch` | Branch on Wflow.jl used by the dev nightly and dev PR check. | Change only if the development branch is renamed (currently always `master`). |
| `wflow.latest.release` | Release *branch* tracked by the stable PR check and the latest-release nightly. Points to a branch (e.g. `release/v1.0`), so "latest build on that branch" is always resolved automatically — no bump needed for patch releases. | Update to the new branch name when a new minor/major release branch is cut on Wflow.jl (e.g. `release/v1.0` → `release/v2.0`). |
| `wflow.oldest.supported.release` | Exact *tag* used by the oldest-supported nightly. Pinned intentionally — it will not move until you change this. | Bump to the new oldest tag when the support window advances (e.g. drop v1.0.0, make v1.1.0 the new floor). Agree the new floor with the Wflow.jl team first. |

### When Wflow.jl cuts a new release

1. A new `release/vX.Y` branch is created on Wflow.jl.
2. Update `wflow.latest.release` to `"release/vX.Y"` in `settings.kts`.
3. If the previous latest release now becomes the oldest supported, update
   `wflow.oldest.supported.release` to the appropriate tag (e.g. `"v1.0.0"`).
   If the previous oldest is now dropped from support, bump it to the new floor.
4. Commit the change. TeamCity will apply it on the next sync.
5. Re-run the regression suite against the new release branch and update baseline
   metrics if the output has changed (see "Responding to failures" above).

### When hydromt_wflow cuts a new release

No changes to `settings.kts` are needed. The suite always runs against the current
commit via `DslContext.settingsRoot`, so PR checks and nightlies automatically cover
the new release branch once it is pushed.
