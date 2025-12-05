"""Module containing system tests for hydromt_wflow integration.

It assumes that julia and hydromt are installed and available in the system PATH.
"""

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from hydromt import initialize_logging

logger = logging.getLogger(__name__)

EXAMPLE_DIR = Path(__file__).parents[2] / "examples"
INTEGRATION_TESTS_DIR = Path(__file__).parent


def main():
    """Run the integration tests using hydromt_wflow and Julia.

    This function generates a water demand model, then runs it.
    Next, it generates a sediment model based on the water demand model
    and runs that as well.
    """
    initialize_logging()

    logger.info("Building sbm model with water demand data...")
    sbm_config = generate_water_demand_model()
    logger.info("Running sbm model...")
    run_wflow_julia(sbm_config)

    logger.info("Building sediment model from sbm model...")
    sediment_config = generate_sediment_model(sbm_config.parent)
    logger.info("Running sediment model...")
    run_wflow_julia(sediment_config)


def generate_water_demand_model() -> Path:
    """Generate a wflow_sbm model with water demand data."""
    root = Path(tempfile.gettempdir(), "wflow_sbm").resolve()
    build_workflow_yml: Path = INTEGRATION_TESTS_DIR / "wflow_build_sbm.yml"
    demand_data_catalog: Path = EXAMPLE_DIR / "data" / "demand" / "data_catalog.yml"

    command = [
        "hydromt",
        "build",
        "wflow_sbm",
        root.as_posix(),
        "-i",
        build_workflow_yml.as_posix(),
        "-d",
        demand_data_catalog.as_posix(),
        "-d",
        "artifact_data",
        "-v",
    ]

    shutil.rmtree(root, ignore_errors=True)
    run_command(command)
    config_file = root / "wflow_sbm.toml"
    assert config_file.exists(), f"Config file {config_file.as_posix()} does not exist."
    return config_file


def generate_sediment_model(sbm_root: Path) -> Path:
    """Generate a wflow_sediment model using forcings from the water demand model."""
    # backup sbm for debugging purposes
    backup_root = sbm_root.with_name(sbm_root.name + "_backup")
    shutil.rmtree(backup_root, ignore_errors=True)
    shutil.copytree(sbm_root, backup_root)

    update_workflow_yml: Path = (
        INTEGRATION_TESTS_DIR / "wflow_build_sediment_from_sbm.yml"
    )
    command = [
        "hydromt",
        "update",
        "wflow_sediment",
        sbm_root.as_posix(),
        "-i",
        update_workflow_yml.as_posix(),
        "-d",
        "artifact_data",
        "--fo",
        "-v",
    ]
    run_command(command)

    logger.info(f"Generated sediment model at {sbm_root.as_posix()}")
    config_file = sbm_root / "wflow_sediment.toml"
    assert config_file.exists(), f"Config file {config_file.as_posix()} does not exist."
    return config_file


def run_command(command: list[str]) -> bool:
    """Run a command as a subprocess and handle logging and errors."""
    logger.info(f"Running command: {' '.join(command)}")
    process = subprocess.run(command, text=True, capture_output=True)
    success = process.returncode == 0

    if process.stdout:
        logger.info(process.stdout)

    if not success:
        msg = (
            f"Command {' '.join(command)} failed with return code {process.returncode}"
        )
        logger.error(msg)
        logger.error(process.stderr)
        raise RuntimeError(msg)

    return success


def run_wflow_julia(toml_path: Path):
    """Run a wflow model using Julia and a TOML configuration file."""
    command = [
        "julia",
        "-e",
        f"""
        toml_path = raw\"{toml_path.as_posix()}\";
        using Wflow;
        Wflow.run(toml_path)
        """,
    ]
    run_command(command)


if __name__ == "__main__":
    main()
