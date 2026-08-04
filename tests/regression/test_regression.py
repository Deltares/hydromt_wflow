"""System regression tests for basin SBM and sediment model metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.regression.regression_utils import (
    Metric,
    MetricsComparison,
    RegressionCheck,
    RuntimesComparison,
    RuntimeSpec,
    compare_metrics,
    compute_metrics,
    default_run_root,
    emit_teamcity_stats,
    get_basins_for_profile,
    load_basin_config,
    repo_root,
    report_failures,
    resolve_path,
)

pytestmark = pytest.mark.regression


def _resolved_run_root(config: pytest.Config) -> Path:
    regression_root = config.getoption("--regression-root")
    if regression_root:
        return Path(regression_root)

    return default_run_root()


def _resolved_basins(config: pytest.Config, project_root: Path) -> list[str]:
    profile = config.getoption("--regression-profile")
    return get_basins_for_profile(project_root, profile)


def pytest_generate_tests(metafunc):
    if "basin" not in metafunc.fixturenames:
        return
    project_root = repo_root()
    basins = _resolved_basins(metafunc.config, project_root)
    metafunc.parametrize("basin", basins)


def test_basin_regression_metrics(basin, request):
    project_root = repo_root()
    run_root = _resolved_run_root(request.config)
    basin_config = load_basin_config(project_root, basin)

    baseline_path = resolve_path(project_root, basin_config["baseline_metrics"])
    if not baseline_path.exists():
        raise AssertionError(
            f"Baseline metrics not found: {baseline_path}. "
            "Generate with: pixi run regression-generate-metrics <PROFILE> <ROOT>"
        )

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    # Load actual runtimes from the pipeline run
    runtimes_path = run_root / "runtimes.json"
    if not runtimes_path.exists():
        raise AssertionError(
            f"Runtimes file not found: {runtimes_path}. "
            "Run the pipeline with: pixi run regression-pipeline"
        )
    all_runtimes = json.loads(runtimes_path.read_text(encoding="utf-8"))
    runtimes_actual = all_runtimes.get(basin, {})
    if not runtimes_actual:
        raise AssertionError(
            f"No runtimes recorded for basin '{basin}' in {runtimes_path}. "
            "Run the pipeline with: pixi run regression-pipeline"
        )

    sbm_output = run_root / "wflow_sbm" / basin / basin_config["sbm"]["output_nc"]
    sediment_output = (
        run_root / "wflow_sediment" / basin / basin_config["sediment"]["output_nc"]
    )

    sbm_specs = [Metric.from_dict(m) for m in basin_config["sbm"]["metrics"]]
    sediment_specs = [Metric.from_dict(m) for m in basin_config["sediment"]["metrics"]]
    sbm_actual = compute_metrics(sbm_output, sbm_specs)
    sediment_actual = compute_metrics(sediment_output, sediment_specs)

    emit_teamcity_stats(basin, "sbm", sbm_actual)
    emit_teamcity_stats(basin, "sediment", sediment_actual)

    sbm_runtime_specs = {
        k: RuntimeSpec.from_dict(v)
        for k, v in basin_config["sbm"].get("runtime_specs", {}).items()
    }
    sediment_runtime_specs = {
        k: RuntimeSpec.from_dict(v)
        for k, v in basin_config["sediment"].get("runtime_specs", {}).items()
    }
    failures = []
    failures.extend(
        compare_metrics(
            RegressionCheck(
                basin=basin,
                model_type="sbm",
                metrics=MetricsComparison(
                    actual=sbm_actual,
                    baseline=baseline.get("sbm", {}),
                    specs=sbm_specs,
                ),
                runtimes=RuntimesComparison(
                    actual=runtimes_actual.get("sbm", {}),
                    baseline=baseline.get("sbm_runtimes", {}),
                    specs=sbm_runtime_specs,
                ),
            )
        )
    )
    failures.extend(
        compare_metrics(
            RegressionCheck(
                basin=basin,
                model_type="sediment",
                metrics=MetricsComparison(
                    actual=sediment_actual,
                    baseline=baseline.get("sediment", {}),
                    specs=sediment_specs,
                ),
                runtimes=RuntimesComparison(
                    actual=runtimes_actual.get("sediment", {}),
                    baseline=baseline.get("sediment_runtimes", {}),
                    specs=sediment_runtime_specs,
                ),
            )
        )
    )

    report = report_failures(failures)
    assert not failures, report
