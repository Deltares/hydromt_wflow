from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from regression_utils import (
    build_sbm,
    build_sediment,
    default_run_root,
    get_basins_for_profile,
    load_basin_config,
    repo_root,
    run_wflow,
)


def parse_args() -> argparse.Namespace:
    # Load environment variables from .env file to support WFLOW_CLI
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Run regression model pipeline (build + run)."
    )
    parser.add_argument(
        "--root",
        default=default_run_root(),
        required=True,
        help="Root directory for model runs.",
    )
    parser.add_argument(
        "--profile",
        default="pr",
        choices=["all", "pr", "piave", "moselle"],
        help="Basin profile or individual basin name to run.",
    )
    parser.add_argument(
        "--wflow-cli",
        type=Path,
        default=os.getenv("WFLOW_CLI"),
        help="Path to wflow_cli executable. Defaults to WFLOW_CLI env var.",
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
    args = parser.parse_args()
    if not args.wflow_cli:
        parser.error("--wflow-cli is required or set WFLOW_CLI environment variable")
    if not Path(args.wflow_cli).exists():
        parser.error(f"wflow_cli not found: {args.wflow_cli}")
    return args


def main() -> None:
    args = parse_args()
    project_root = repo_root()
    run_root = Path(args.root)
    basins = get_basins_for_profile(project_root, args.profile)

    for basin in basins:
        basin_config = load_basin_config(project_root, basin)
        runtimes: dict[str, dict[str, float]] = {"sbm": {}, "sediment": {}}

        # Build + run SBM
        sbm_root, sbm_build_time = build_sbm(
            project_root=project_root,
            basin_config=basin_config,
            basin=basin,
            root=run_root,
            force_overwrite=args.force_overwrite,
            verbosity=args.verbosity,
        )
        runtimes["sbm"]["build_time"] = sbm_build_time
        sbm_toml = sbm_root / basin_config["sbm"]["config_toml"]
        sbm_run_time = run_wflow(wflow_cli=args.wflow_cli, config_toml=sbm_toml)
        runtimes["sbm"]["kernel_runtime"] = sbm_run_time

        # Build + run sediment
        sediment_root, sediment_build_time = build_sediment(
            project_root=project_root,
            basin_config=basin_config,
            basin=basin,
            root=run_root,
            force_overwrite=args.force_overwrite,
            verbosity=args.verbosity,
        )
        runtimes["sediment"]["build_time"] = sediment_build_time
        sediment_toml = sediment_root / basin_config["sediment"]["config_toml"]
        sediment_run_time = run_wflow(
            wflow_cli=args.wflow_cli, config_toml=sediment_toml
        )
        runtimes["sediment"]["kernel_runtime"] = sediment_run_time

        # Write runtimes to JSON file for test consumption
        runtimes_path = run_root / "runtimes.json"
        runtimes_path.parent.mkdir(parents=True, exist_ok=True)
        runtimes_path.write_text(json.dumps(runtimes, indent=2), encoding="utf-8")

        print(f"\n=== {basin.upper()} RUNTIMES ===")
        print(
            f"SBM: build_time={runtimes['sbm'].get('build_time', 0):.2f}s, kernel_runtime={runtimes['sbm'].get('kernel_runtime', 0):.2f}s"
        )
        print(
            f"Sediment: build_time={runtimes['sediment'].get('build_time', 0):.2f}s, kernel_runtime={runtimes['sediment'].get('kernel_runtime', 0):.2f}s"
        )


if __name__ == "__main__":
    main()
