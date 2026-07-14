from __future__ import annotations

import argparse
from pathlib import Path

from regression_utils import (
    build_sbm,
    build_sediment,
    get_basins_for_profile,
    load_basin_config,
    repo_root,
    run_wflow,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run regression model pipeline.")
    parser.add_argument("--root", required=True, help="Root directory for model runs.")
    parser.add_argument(
        "--profile",
        default="all",
        choices=["pr", "all"],
        help="Basin profile to run.",
    )
    parser.add_argument(
        "--basins",
        default="",
        help="Comma-separated basin list. Overrides --profile when provided.",
    )
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "sbm", "sediment"],
        help="Pipeline mode.",
    )
    parser.add_argument(
        "--wflow-cli",
        default="",
        help="Path to wflow_cli executable (required for --mode full).",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        default=False,
        help="Pass --fo to hydromt commands.",
    )
    parser.add_argument(
        "--verbosity",
        default="-vv",
        choices=["-v", "-vv", "-vvv"],
        help="HydroMT verbosity flag.",
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
    basins = _basins(args, project_root)
    wflow_cli = Path(args.wflow_cli) if args.wflow_cli else None

    if args.mode == "full" and wflow_cli is None:
        raise ValueError("--wflow-cli is required when --mode=full")

    for basin in basins:
        basin_config = load_basin_config(project_root, basin)

        if args.mode in {"full", "sbm"}:
            sbm_root = build_sbm(
                project_root=project_root,
                basin_config=basin_config,
                basin=basin,
                root=run_root,
                force_overwrite=args.force_overwrite,
                verbosity=args.verbosity,
            )
            if args.mode == "full":
                sbm_toml = sbm_root / basin_config["sbm"]["config_toml"]
                run_wflow(wflow_cli=wflow_cli, config_toml=sbm_toml)

        if args.mode in {"full", "sediment"}:
            sediment_root = build_sediment(
                project_root=project_root,
                basin_config=basin_config,
                basin=basin,
                root=run_root,
                force_overwrite=args.force_overwrite,
                verbosity=args.verbosity,
            )
            if args.mode == "full":
                sediment_toml = sediment_root / basin_config["sediment"]["config_toml"]
                run_wflow(wflow_cli=wflow_cli, config_toml=sediment_toml)


if __name__ == "__main__":
    main()
