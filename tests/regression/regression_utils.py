from __future__ import annotations

import json
import math
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_DEFAULT_REL_TOL_METRIC = 1e-3  # 0.1%
_DEFAULT_ABS_TOL_METRIC = 1e-6  # 1e-6 (for near-zero values)
_DEFAULT_REL_TOL_RUNTIME = 0.20  # 20%
_DEFAULT_ABS_TOL_RUNTIME = 120.0  # 120 seconds


@dataclass
class Metric:
    name: str
    variable: str
    selector: dict[str, float] | None = None
    selector_method: Literal["sel", "isel"] = "sel"
    aggregation: str = "mean"
    metrics: list[str] = field(default_factory=lambda: ["mean", "peak", "total"])
    rel_tol: float = _DEFAULT_REL_TOL_METRIC
    abs_tol: float = _DEFAULT_ABS_TOL_METRIC

    @classmethod
    def from_dict(cls, d: dict) -> "Metric":
        return cls(
            name=d["name"],
            variable=d["variable"],
            selector=d.get("selector"),
            selector_method=d.get("selector_method", "sel"),
            aggregation=d.get("aggregation", "mean"),
            metrics=d.get("metrics", ["mean", "peak", "total"]),
            rel_tol=float(d.get("rel_tol", _DEFAULT_REL_TOL_METRIC)),
            abs_tol=float(d.get("abs_tol", _DEFAULT_ABS_TOL_METRIC)),
        )


@dataclass
class RuntimeSpec:
    rel_tol: float = _DEFAULT_REL_TOL_RUNTIME
    abs_tol: float = _DEFAULT_ABS_TOL_RUNTIME

    @classmethod
    def from_dict(cls, d: dict) -> "RuntimeSpec":
        return cls(
            rel_tol=float(d.get("rel_tol", _DEFAULT_REL_TOL_RUNTIME)),
            abs_tol=float(d.get("abs_tol", _DEFAULT_ABS_TOL_RUNTIME)),
        )


@dataclass
class MetricsComparison:
    """Actual vs. baseline metrics for a single model."""

    actual: dict[str, dict[str, float]]
    baseline: dict[str, dict[str, float]]
    specs: list[Metric]


@dataclass
class RuntimesComparison:
    """Actual vs. baseline runtimes."""

    actual: dict[str, float]
    baseline: dict[str, float]
    specs: dict[str, RuntimeSpec] = field(default_factory=dict)


@dataclass
class RegressionCheck:
    """Complete regression check for one model (sbm or sediment)."""

    basin: str
    model_type: Literal["sbm", "sediment"]
    metrics: MetricsComparison
    runtimes: RuntimesComparison


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_run_root() -> Path:
    return repo_root() / "tests" / "regression" / ".runs"


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def load_manifest(project_root: Path) -> dict:
    manifest_path = project_root / "tests" / "regression" / "manifest.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_basin_config(project_root: Path, basin: str) -> dict:
    path = project_root / "tests" / "regression" / basin / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown basin '{basin}'. Expected config at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def get_basins_for_profile(project_root: Path, profile: str) -> list[str]:
    manifest = load_manifest(project_root)
    profiles = manifest.get("profiles", {})
    # If the value isn't a named profile, treat it as a literal basin name.
    return profiles.get(profile, [profile])


def list_profile_choices(project_root: Path) -> list[str]:
    """List valid values for the --profile/--regression-profile CLI options.

    Includes both named profiles (e.g. "pr", "all") from manifest.json and every
    individual basin directory that has a config.json, so a basin can be run on
    its own without also being listed as a profile. Computed dynamically so that
    registering a new basin/profile in manifest.json is the only step needed to
    make it selectable, matching the "Adding a new basin" instructions in
    tests/regression/README.md.
    """
    manifest = load_manifest(project_root)
    profiles = set(manifest.get("profiles", {}).keys())
    regression_dir = project_root / "tests" / "regression"
    basins = {
        p.parent.name for p in regression_dir.glob("*/config.json") if p.is_file()
    }
    return sorted(profiles | basins)


def _resolve_data_catalog_arg(project_root: Path, entry: str) -> str:
    if "/" in entry or "\\" in entry or entry.startswith("."):
        return str(resolve_path(project_root, entry))
    return entry


def _run(cmd: list[str]) -> float:
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )
    return elapsed


def _apply_common_build_args(
    cmd: list[str],
    project_root: Path,
    data_catalogs: list[str],
    force_overwrite: bool,
    verbosity: str,
) -> list[str]:
    for entry in data_catalogs:
        cmd.extend(["-d", _resolve_data_catalog_arg(project_root, entry)])
    if force_overwrite:
        cmd.append("--fo")
    if verbosity:
        cmd.append(verbosity)
    return cmd


def build_sbm(
    project_root: Path,
    basin_config: dict,
    basin: str,
    root: Path,
    force_overwrite: bool,
    verbosity: str,
) -> tuple[Path, float]:
    sbm_root = root / "wflow_sbm" / basin
    sbm_cfg = basin_config["sbm"]
    cmd = [
        "hydromt",
        "build",
        "wflow_sbm",
        str(sbm_root),
        "-i",
        str(resolve_path(project_root, sbm_cfg["build_config"])),
    ]
    cmd = _apply_common_build_args(
        cmd=cmd,
        project_root=project_root,
        data_catalogs=sbm_cfg["data_catalogs"],
        force_overwrite=force_overwrite,
        verbosity=verbosity,
    )
    elapsed = _run(cmd)
    return sbm_root, elapsed


def _validate_config(cfg_path: Path, model_dir: Path) -> None:
    """Validate that the base config file exists when resolved relative to the model output directory.

    HydroMT resolves config.read filenames relative to the model output directory.
    Raises FileNotFoundError with a clear message if the resolved path does not exist,
    so the caller knows to fix the relative path in the YAML recipe.
    """
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    for step in cfg.get("steps", []):
        if isinstance(step, dict) and "config.read" in step:
            cr = step["config.read"]
            if isinstance(cr, dict):
                fn = cr.get("filename")
                if fn and not Path(fn).is_absolute():
                    resolved = (model_dir / fn).resolve()
                    if not resolved.exists():
                        raise FileNotFoundError(
                            f"config.read filename '{fn}' in {cfg_path} does not exist "
                            f"when resolved relative to model dir {model_dir} "
                            f"(tried: {resolved}). Fix the relative path in the YAML."
                        )


def build_sediment(
    project_root: Path,
    basin_config: dict,
    basin: str,
    root: Path,
    force_overwrite: bool,
    verbosity: str,
) -> tuple[Path, float]:
    sbm_root = root / "wflow_sbm" / basin
    sediment_root = root / "wflow_sediment" / basin
    sediment_cfg = basin_config["sediment"]
    cfg_path = resolve_path(project_root, sediment_cfg["build_config"])
    _validate_config(cfg_path, sediment_root)

    cmd = [
        "hydromt",
        "update",
        "wflow_sediment",
        str(sbm_root),
        "-o",
        str(sediment_root),
        "-i",
        str(cfg_path),
    ]
    cmd = _apply_common_build_args(
        cmd=cmd,
        project_root=project_root,
        data_catalogs=sediment_cfg["data_catalogs"],
        force_overwrite=force_overwrite,
        verbosity=verbosity,
    )
    elapsed = _run(cmd)

    sbm_output = sbm_root / "run_default" / "output.nc"
    sediment_forcing = sediment_root / "run_default" / "output.nc"
    if not sbm_output.exists():
        raise FileNotFoundError(
            f"SBM output not found: {sbm_output}. It is likely that the SBM run failed or has not been executed yet."
        )

    sediment_forcing.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sbm_output, sediment_forcing)
    return sediment_root, elapsed


def run_wflow(wflow_cli: Path, config_toml: Path) -> float:
    start = time.time()
    _run([str(wflow_cli), str(config_toml)])
    elapsed = time.time() - start
    return elapsed


def _select_and_reduce(
    data_array: xr.DataArray,
    selector: dict[str, float] | None,
    selector_method: str,
    aggregation: str,
) -> xr.DataArray:
    """Apply the selector and reduce non-time dims, keeping the time series intact."""
    data = data_array
    if selector:
        if selector_method == "sel":
            data = data.sel(selector, method="nearest")
        else:
            data = data.isel(selector)
    if "time" in data.dims:
        reduce_dims = [dim for dim in data.dims if dim != "time"]
    else:
        reduce_dims = list(data.dims)
    if reduce_dims:
        if aggregation == "sum":
            data = data.sum(dim=reduce_dims, skipna=True)
        elif aggregation == "mean":
            data = data.mean(dim=reduce_dims, skipna=True)
        elif aggregation == "max":
            data = data.max(dim=reduce_dims, skipna=True)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
    return data


def _to_series(
    data_array: xr.DataArray,
    selector: dict[str, float] | None,
    selector_method: str,
    aggregation: str,
) -> np.ndarray:
    data = _select_and_reduce(data_array, selector, selector_method, aggregation)
    values = np.asarray(data.values, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    return values


def compute_metrics(
    output_nc: Path, specs: list[Metric]
) -> dict[str, dict[str, float]]:
    if not output_nc.exists():
        raise FileNotFoundError(
            f"Model output not found: {output_nc}. It is likely that the model run failed or has not been executed yet."
        )
    results: dict[str, dict[str, float]] = {}
    with xr.open_dataset(output_nc) as dataset:
        for spec in specs:
            if spec.variable not in dataset:
                raise KeyError(f"Variable '{spec.variable}' not found in {output_nc}")
            series = _to_series(
                data_array=dataset[spec.variable],
                selector=spec.selector,
                selector_method=spec.selector_method,
                aggregation=spec.aggregation,
            )
            metric_values: dict[str, float] = {}
            for metric in spec.metrics:
                if metric == "mean":
                    metric_values[metric] = float(np.nanmean(series))
                elif metric == "peak":
                    metric_values[metric] = float(np.nanmax(series))
                elif metric == "total":
                    metric_values[metric] = float(np.nansum(series))
                else:
                    raise ValueError(
                        f"Unsupported metric: {metric}, valid options are: 'mean', 'peak', 'total'"
                    )
            results[spec.name] = metric_values
    return results


def compare_metrics(check: RegressionCheck) -> list[str]:
    failures: list[str] = []

    # Compare metrics
    for spec in check.metrics.specs:
        if spec.name not in check.metrics.baseline:
            failures.append(
                f"[{check.basin}.{check.model_type}] Missing baseline metric group '{spec.name}'"
            )
            continue
        for metric_key in spec.metrics:
            expected = check.metrics.baseline[spec.name].get(metric_key)
            if expected is None:
                failures.append(
                    f"[{check.basin}.{check.model_type}] Missing baseline value for {spec.name}.{metric_key}"
                )
                continue
            observed = check.metrics.actual[spec.name][metric_key]
            expected = float(expected)
            if not math.isfinite(observed) or not math.isfinite(expected):
                failures.append(
                    f"[{check.basin}.{check.model_type}] {spec.name}.{metric_key} "
                    f"is not finite: observed={observed}, expected={expected}"
                )
                continue
            abs_err = abs(observed - expected)
            if abs(expected) <= spec.abs_tol:
                if abs_err > spec.abs_tol:
                    failures.append(
                        f"[{check.basin}.{check.model_type}] {spec.name}.{metric_key} abs_err={abs_err:.6e} > abs_tol={spec.abs_tol:.6e}"
                    )
                continue
            rel_err = abs_err / abs(expected)
            if rel_err > spec.rel_tol:
                failures.append(
                    f"[{check.basin}.{check.model_type}] {spec.name}.{metric_key} rel_err={rel_err:.6%} > rel_tol={spec.rel_tol:.6%}"
                )

    # Compare runtimes if provided
    for runtime_key, runtime_spec in check.runtimes.specs.items():
        if runtime_key not in check.runtimes.baseline:
            failures.append(
                f"[{check.basin}.{check.model_type}] Missing baseline runtime value for '{runtime_key}'"
            )
            continue
        expected = check.runtimes.baseline[runtime_key]
        observed = check.runtimes.actual.get(runtime_key)
        if observed is None:
            failures.append(
                f"[{check.basin}.{check.model_type}] Missing observed runtime value for '{runtime_key}'"
            )
            continue
        if not math.isfinite(observed) or not math.isfinite(expected):
            failures.append(
                f"[{check.basin}.{check.model_type}] runtime '{runtime_key}' "
                f"is not finite: observed={observed}, expected={expected}"
            )
            continue
        abs_err = abs(observed - expected)
        rel_err = abs_err / abs(expected) if expected != 0 else 0.0
        if rel_err > runtime_spec.rel_tol and abs_err > runtime_spec.abs_tol:
            failures.append(
                f"[{check.basin}.{check.model_type}] {runtime_key} rel_err={rel_err:.6%} > rel_tol={runtime_spec.rel_tol:.6%} AND abs_err={abs_err:.2f}s > abs_tol={runtime_spec.abs_tol:.2f}s"
            )
    return failures


def report_failures(failures: list[str]) -> str:
    """Build a human-readable failure report.

    To visualize a failing run (actual vs. baseline bar charts plus the
    underlying output time series) without re-building or re-running any
    model, use `plot_regression_report`, e.g.:
        from tests.regression.regression_utils import plot_regression_report
        plot_regression_report("<basin>")
    See also `build_regression_check`, `plot_metric_comparison` and
    `plot_metric_timeseries` for lower-level building blocks.
    """
    n = len(failures)
    lines = [
        f"Regression check failed: {n} metric(s) failed.",
        "",
        "Failures:",
    ]
    for f in failures:
        lines.append(f"  {f}")
    lines += [
        "",
        "Possible follow-up actions:",
        "  1. Fix the regression: Investigate recent code changes and correct the model",
        "     behaviour. Re-run the pipeline and verify metrics pass before merging.",
        "",
        "  2. Accept the change (if change in output is expected): If the deviation is intentional (e.g. a",
        "     model improvement or deliberate output change), regenerate the baseline",
        "     metrics and commit the updated files:",
        "       pixi run regression-generate-metrics <PROFILE> <ROOT>",
        "     Then re-run the tests to confirm they pass.",
        "",
        "  3. Widen tolerances: If the deviation is within acceptable bounds but exceeds",
        "     the current thresholds, update rel_tol / abs_tol in the relevant",
        "     tests/regression/<basin>/config.json and regenerate metrics.",
        "",
        "  4. Visualize: Plot actual vs. baseline metrics and the underlying output",
        "     time series (reads existing outputs, does not re-run anything):",
        "       from tests.regression.regression_utils import plot_regression_report",
        "       plot_regression_report('<basin>')",
        "",
    ]
    return "\n".join(lines)


def emit_teamcity_stats(
    basin: str, model_name: str, metrics: dict[str, dict[str, float]]
) -> None:
    for metric_name, metric_values in metrics.items():
        for metric_key, value in metric_values.items():
            key = f"regression_{basin}_{model_name}_{metric_name}_{metric_key}"
            safe_key = key.replace(" ", "_")
            print(
                f"##teamcity[buildStatisticValue key='{safe_key}' value='{value:.6e}']"
            )


def build_regression_check(
    project_root: Path,
    run_root: Path,
    basin: str,
    model_type: Literal["sbm", "sediment"],
) -> RegressionCheck:
    """Build a RegressionCheck for an already-completed run.

    Reads the existing model output netcdf, baseline metrics and recorded
    runtimes from disk without building or running any model.
    """
    basin_config = load_basin_config(project_root, basin)
    model_cfg = basin_config[model_type]
    baseline_path = resolve_path(project_root, basin_config["baseline_metrics"])
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    output_nc = run_root / f"wflow_{model_type}" / basin / model_cfg["output_nc"]
    specs = [Metric.from_dict(m) for m in model_cfg["metrics"]]
    actual = compute_metrics(output_nc, specs)

    runtimes_actual: dict[str, float] = {}
    runtimes_path = run_root / "runtimes.json"
    if runtimes_path.exists():
        all_runtimes = json.loads(runtimes_path.read_text(encoding="utf-8"))
        runtimes_actual = all_runtimes.get(basin, {}).get(model_type, {})

    runtime_specs = {
        k: RuntimeSpec.from_dict(v)
        for k, v in model_cfg.get("runtime_specs", {}).items()
    }

    return RegressionCheck(
        basin=basin,
        model_type=model_type,
        metrics=MetricsComparison(
            actual=actual,
            baseline=baseline.get(model_type, {}),
            specs=specs,
        ),
        runtimes=RuntimesComparison(
            actual=runtimes_actual,
            baseline=baseline.get(f"{model_type}_runtimes", {}),
            specs=runtime_specs,
        ),
    )


def plot_metric_comparison(check: RegressionCheck, ax: Axes | None = None) -> Axes:
    """Bar plot comparing actual vs. baseline metric values for a RegressionCheck.

    Actual bars are colored red when the observed value is outside the
    metric's rel_tol/abs_tol relative to baseline, blue otherwise. Each pair
    of bars is annotated with the relative error. A log y-scale is used since
    mean/peak/total values for a metric can differ by orders of magnitude.
    """
    labels: list[str] = []
    actual_vals: list[float] = []
    baseline_vals: list[float] = []
    rel_errs: list[float] = []
    failed: list[bool] = []

    for spec in check.metrics.specs:
        baseline_group = check.metrics.baseline.get(spec.name, {})
        actual_group = check.metrics.actual.get(spec.name, {})
        for metric_key in spec.metrics:
            expected = baseline_group.get(metric_key)
            observed = actual_group.get(metric_key)
            if expected is None or observed is None:
                continue
            expected = float(expected)
            observed = float(observed)
            abs_err = abs(observed - expected)
            rel_err = abs_err / abs(expected) if expected != 0 else float("nan")
            if abs(expected) <= spec.abs_tol:
                is_failed = abs_err > spec.abs_tol
            else:
                is_failed = rel_err > spec.rel_tol
            labels.append(f"{spec.name}\n{metric_key}")
            actual_vals.append(observed)
            baseline_vals.append(expected)
            rel_errs.append(rel_err)
            failed.append(is_failed)

    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 4))

    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, baseline_vals, width, label="baseline", color="tab:grey")
    bars = ax.bar(
        x + width / 2,
        actual_vals,
        width,
        label="actual",
        color=["tab:red" if f else "tab:blue" for f in failed],
    )
    if all(v > 0 for v in (*baseline_vals, *actual_vals)):
        ax.set_yscale("log")
    for bar, rel_err in zip(bars, rel_errs):
        if math.isnan(rel_err):
            continue
        ax.annotate(
            f"{rel_err:+.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(f"{check.basin} - {check.model_type} metrics (red = out of tolerance)")
    ax.legend()
    return ax


def plot_metric_timeseries(
    output_nc: Path,
    specs: list[Metric],
    axes: list[Axes] | None = None,
) -> list[Axes]:
    """Plot, per metric spec, the (spatially-reduced) time series it was computed from."""
    if axes is None:
        _, axes_arr = plt.subplots(
            len(specs), 1, figsize=(10, 3 * len(specs)), squeeze=False
        )
        axes = list(axes_arr[:, 0])

    with xr.open_dataset(output_nc) as dataset:
        for ax, spec in zip(axes, specs):
            if spec.variable not in dataset:
                raise KeyError(f"Variable '{spec.variable}' not found in {output_nc}")
            series = _select_and_reduce(
                data_array=dataset[spec.variable],
                selector=spec.selector,
                selector_method=spec.selector_method,
                aggregation=spec.aggregation,
            ).load()
            if "time" in series.dims:
                series.plot.line(ax=ax, x="time")
            else:
                ax.plot(series.values)
            ax.set_title(f"{spec.name} ({spec.variable})")
            ax.set_ylabel(spec.variable)

    return axes


def plot_regression_report(
    basin: str,
    run_root: Path | None = None,
    project_root: Path | None = None,
    save_path: Path | None = None,
) -> Figure:
    """Visualize regression metrics and their underlying time series for one basin.

    Reads the existing model outputs (sbm and sediment) and baseline metrics
    from disk; does not build or run any model. For each model type, plots a
    bar chart comparing actual vs. baseline metric values plus line plots of
    the time series each metric was computed from.
    """
    project_root = project_root or repo_root()
    run_root = run_root or default_run_root()
    basin_config = load_basin_config(project_root, basin)

    sbm_check = build_regression_check(project_root, run_root, basin, "sbm")
    sediment_check = build_regression_check(project_root, run_root, basin, "sediment")

    sbm_output = run_root / "wflow_sbm" / basin / basin_config["sbm"]["output_nc"]
    sediment_output = (
        run_root / "wflow_sediment" / basin / basin_config["sediment"]["output_nc"]
    )

    n_ts_rows = max(len(sbm_check.metrics.specs), len(sediment_check.metrics.specs))
    fig, axes = plt.subplots(1 + n_ts_rows, 2, figsize=(14, 3.5 * (1 + n_ts_rows)))

    plot_metric_comparison(sbm_check, ax=axes[0, 0])
    plot_metric_comparison(sediment_check, ax=axes[0, 1])

    plot_metric_timeseries(sbm_output, sbm_check.metrics.specs, axes=list(axes[1:, 0]))
    plot_metric_timeseries(
        sediment_output, sediment_check.metrics.specs, axes=list(axes[1:, 1])
    )

    # Hide unused time series rows for the model type with fewer metrics.
    for row in range(1, 1 + n_ts_rows):
        if row > len(sbm_check.metrics.specs):
            axes[row, 0].set_visible(False)
        if row > len(sediment_check.metrics.specs):
            axes[row, 1].set_visible(False)

    fig.suptitle(f"Regression report: {basin}")
    fig.tight_layout()

    if save_path is None:
        save_path = run_root / "reports" / f"{basin}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150)
    print(f"Regression report saved to {save_path}")

    return fig
