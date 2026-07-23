"""System regression tests for basin SBM and sediment model metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.regression.regression_utils import (
    compare_metrics,
    compute_metrics,
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

    legacy_model_root = config.getoption("--model-root")
    if legacy_model_root:
        return Path(legacy_model_root).parent.parent

    pytest.skip("Pass --regression-root to run regression assertions.")


def _resolved_basins(config: pytest.Config, project_root: Path) -> list[str]:
    basins_arg = config.getoption("--regression-basins")
    if basins_arg:
        return [b.strip() for b in basins_arg.split(",") if b.strip()]

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
            "Generate with: pixi run regression-generate-metrics <ROOT>"
        )

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    sbm_output = run_root / "wflow_sbm" / basin / basin_config["sbm"]["output_nc"]
    sediment_output = (
        run_root / "wflow_sediment" / basin / basin_config["sediment"]["output_nc"]
    )

    sbm_actual = compute_metrics(sbm_output, basin_config["sbm"]["metrics"])
    sediment_actual = compute_metrics(
        sediment_output, basin_config["sediment"]["metrics"]
    )

    emit_teamcity_stats(basin, "sbm", sbm_actual)
    emit_teamcity_stats(basin, "sediment", sediment_actual)

    failures = []
    failures.extend(
        compare_metrics(
            actual=sbm_actual,
            baseline=baseline.get("sbm", {}),
            specs=basin_config["sbm"]["metrics"],
            model_name=f"{basin}.sbm",
        )
    )
    failures.extend(
        compare_metrics(
            actual=sediment_actual,
            baseline=baseline.get("sediment", {}),
            specs=basin_config["sediment"]["metrics"],
            model_name=f"{basin}.sediment",
        )
    )

    report = report_failures(failures)
    assert not failures, report
