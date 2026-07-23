from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

_DEFAULT_REL_TOL = 1e-3
_DEFAULT_ABS_TOL = 1e-6


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    if profile not in profiles:
        raise ValueError(
            f"Unknown profile '{profile}'. Available: {', '.join(sorted(profiles))}"
        )
    return profiles[profile]


def _resolve_data_catalog_arg(project_root: Path, entry: str) -> str:
    if "/" in entry or "\\" in entry or entry.startswith("."):
        return str(resolve_path(project_root, entry))
    return entry


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


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
) -> Path:
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
    _run(cmd)
    return sbm_root


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
) -> Path:
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
    _run(cmd)

    sbm_output = sbm_root / "run_default" / "output.nc"
    sediment_forcing = sediment_root / "run_default" / "output.nc"
    if not sbm_output.exists():
        raise FileNotFoundError(
            f"SBM output not found: {sbm_output}. It is likely that the SBM run failed or has not been executed yet."
        )

    sediment_forcing.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sbm_output, sediment_forcing)
    return sediment_root


def run_wflow(wflow_cli: Path, config_toml: Path) -> None:
    _run([str(wflow_cli), str(config_toml)])


def _to_series(
    data_array: xr.DataArray,
    selector: dict[str, int] | None,
    aggregation: str,
) -> np.ndarray:
    data = data_array
    if selector:
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
    values = np.asarray(data.values, dtype=float)
    if values.ndim == 0:
        values = values.reshape(1)
    return values


def compute_metrics(output_nc: Path, specs: list[dict]) -> dict[str, dict[str, float]]:
    if not output_nc.exists():
        raise FileNotFoundError(
            f"Model output not found: {output_nc}. It is likely that the model run failed or has not been executed yet."
        )
    results: dict[str, dict[str, float]] = {}
    with xr.open_dataset(output_nc) as dataset:
        for spec in specs:
            name = spec["name"]
            variable = spec["variable"]
            if variable not in dataset:
                raise KeyError(f"Variable '{variable}' not found in {output_nc}")
            series = _to_series(
                data_array=dataset[variable],
                selector=spec.get("selector"),
                aggregation=spec.get("aggregation", "sum"),
            )
            metric_values: dict[str, float] = {}
            for metric in spec["metrics"]:
                if metric == "mean":
                    metric_values[metric] = float(np.nanmean(series))
                elif metric == "peak":
                    metric_values[metric] = float(np.nanmax(series))
                elif metric == "total":
                    metric_values[metric] = float(np.nansum(series))
                else:
                    raise ValueError(f"Unsupported metric: {metric}")
            results[name] = metric_values
    return results


def compare_metrics(
    actual: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    specs: list[dict],
    model_name: str,
) -> list[str]:
    failures: list[str] = []
    for spec in specs:
        metric_name = spec["name"]
        rel_tol = float(spec.get("rel_tol", _DEFAULT_REL_TOL))
        abs_tol = float(spec.get("abs_tol", _DEFAULT_ABS_TOL))
        if metric_name not in baseline:
            failures.append(
                f"[{model_name}] Missing baseline metric group '{metric_name}'"
            )
            continue
        for metric_key in spec["metrics"]:
            expected = baseline[metric_name].get(metric_key)
            if expected is None:
                failures.append(
                    f"[{model_name}] Missing baseline value for {metric_name}.{metric_key}"
                )
                continue
            observed = actual[metric_name][metric_key]
            expected = float(expected)
            abs_err = abs(observed - expected)
            if abs(expected) <= abs_tol:
                if abs_err > abs_tol:
                    failures.append(
                        f"[{model_name}] {metric_name}.{metric_key} abs_err={abs_err:.6e} > abs_tol={abs_tol:.6e}"
                    )
                continue
            rel_err = abs_err / abs(expected)
            if rel_err > rel_tol:
                failures.append(
                    f"[{model_name}] {metric_name}.{metric_key} rel_err={rel_err:.6%} > rel_tol={rel_tol:.6%}"
                )
    # if failures: how to regenerate metrics only if you know its correct... DANGER ZONE!
    return failures


def report_failures(failures: list[str]) -> str:
    n = len(failures)
    lines = [f"Regression check failed: {n} metric(s) failed.", "", "Failures:"]
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
        "       pixi run regression-generate-metrics --root <ROOT>",
        "     Then re-run the tests to confirm they pass.",
        "",
        "  3. Widen tolerances: If the deviation is within acceptable bounds but exceeds",
        "     the current thresholds, update rel_tol / abs_tol in the relevant",
        "     tests/regression/<basin>/config.json and regenerate metrics.",
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
