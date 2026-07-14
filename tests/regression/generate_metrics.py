from __future__ import annotations

import argparse
import json
from pathlib import Path

from regression_utils import (
    compute_metrics,
    get_basins_for_profile,
    load_basin_config,
    repo_root,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline regression metrics from model outputs."
    )
    parser.add_argument("--root", required=True, help="Root directory for model runs.")
    parser.add_argument(
        "--profile",
        default="all",
        choices=["pr", "all"],
        help="Basin profile to process.",
    )
    parser.add_argument(
        "--basins",
        default="",
        help="Comma-separated basin list. Overrides --profile when provided.",
    )
    return parser.parse_args()


def _basins(args: argparse.Namespace, project_root: Path) -> list[str]:
    if args.basins.strip():
        return [b.strip() for b in args.basins.split(",") if b.strip()]
    return get_basins_for_profile(project_root, args.profile)


def main() -> None:
    args = parse_args()
    project_root = repo_root()
    run_root = Path(args.root)

    for basin in _basins(args, project_root):
        basin_config = load_basin_config(project_root, basin)

        sbm_output = run_root / "wflow_sbm" / basin / basin_config["sbm"]["output_nc"]
        sediment_output = (
            run_root / "wflow_sediment" / basin / basin_config["sediment"]["output_nc"]
        )

        payload = {
            "sbm": compute_metrics(sbm_output, basin_config["sbm"]["metrics"]),
            "sediment": compute_metrics(
                sediment_output, basin_config["sediment"]["metrics"]
            ),
        }

        baseline_path = resolve_path(project_root, basin_config["baseline_metrics"])
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
