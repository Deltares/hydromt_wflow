from __future__ import annotations

import argparse
import json
from pathlib import Path

from regression_utils import (
    Metric,
    compute_metrics,
    default_run_root,
    get_basins_for_profile,
    load_basin_config,
    repo_root,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline regression metrics from model outputs."
    )
    parser.add_argument(
        "--root",
        default=default_run_root(),
        help="Root directory for model runs.",
    )
    parser.add_argument(
        "--profile",
        default="pr",
        choices=["all", "pr", "piave", "moselle"],
        help="Basin profile or individual basin name to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = repo_root()
    run_root = Path(args.root)

    runtimes_path = run_root / "runtimes.json"
    if not runtimes_path.exists():
        raise FileNotFoundError(
            f"Runtimes file not found: {runtimes_path}. "
            "Run the pipeline first with: pixi run regression-run-pipeline"
        )
    runtimes = json.loads(runtimes_path.read_text(encoding="utf-8"))

    for basin in get_basins_for_profile(project_root, args.profile):
        basin_config = load_basin_config(project_root, basin)

        sbm_output = run_root / "wflow_sbm" / basin / basin_config["sbm"]["output_nc"]
        sediment_output = (
            run_root / "wflow_sediment" / basin / basin_config["sediment"]["output_nc"]
        )

        sbm_specs = [Metric.from_dict(m) for m in basin_config["sbm"]["metrics"]]
        sediment_specs = [
            Metric.from_dict(m) for m in basin_config["sediment"]["metrics"]
        ]

        basin_runtimes = runtimes.get(
            basin, runtimes
        )  # support per-basin or flat layout
        payload = {
            "sbm": compute_metrics(sbm_output, sbm_specs),
            "sbm_runtimes": basin_runtimes.get("sbm", {}),
            "sediment": compute_metrics(sediment_output, sediment_specs),
            "sediment_runtimes": basin_runtimes.get("sediment", {}),
        }

        baseline_path = resolve_path(project_root, basin_config["baseline_metrics"])
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Generated baseline metrics for {basin}: {baseline_path}")


if __name__ == "__main__":
    main()
