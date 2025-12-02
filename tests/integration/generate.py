import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import xarray as xr

from hydromt_wflow.wflow_sbm import WflowSbmModel
from hydromt_wflow.wflow_sediment import WflowSedimentModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXAMPLE_DIR = Path(__file__).parents[2] / "examples"


def generate_water_demand_model() -> Path:
    """Generate a wflow_sbm model with water demand data.

    Taken from update_model_water_demand.ipynb example:
        ! hydromt build wflow_sbm "./wflow_piave_basin" -i wflow_build_basin.yml -d artifact_data -v
        ! hydromt update wflow_sbm "./wflow_piave_basin" -o "./wflow_piave_water_demand" -i wflow_update_water_demand.yml -d artifact_data -d ./data/demand/data_catalog.yml -v

    Returns
    -------
    Path
        Path to the built model.

    """  # noqa
    tmp_dir = Path(tempfile.gettempdir())
    root: Path = tmp_dir / "wflow_piave_basin"
    updated_root: Path = tmp_dir / "wflow_piave_water_demand"
    build_workflow_yml: Path = EXAMPLE_DIR / "wflow_build_basin.yml"
    update_workflow_yml: Path = EXAMPLE_DIR / "wflow_update_water_demand.yml"
    demand_data_catalog: Path = EXAMPLE_DIR / "data" / "demand" / "data_catalog.yml"

    build_command = [
        "hydromt",
        "build",
        "wflow_sbm",
        root.as_posix(),
        "-i",
        build_workflow_yml.as_posix(),
        "-d",
        "artifact_data",
        "-v",
    ]

    update_command = [
        "hydromt",
        "update",
        "wflow_sbm",
        root.as_posix(),
        "-o",
        updated_root.as_posix(),
        "-i",
        update_workflow_yml.as_posix(),
        "-d",
        "artifact_data",
        "-d",
        demand_data_catalog.as_posix(),
        "-v",
    ]
    if REBUILD:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(updated_root, ignore_errors=True)

    if not root.exists():
        _subprocess_run(build_command)
    if not updated_root.exists():
        _subprocess_run(update_command)

    config_file = updated_root / "wflow_sbm.toml"
    assert config_file.exists()

    logger.info(f"Generated water demand model at {updated_root.as_posix()}")
    return config_file


def generate_sediment_model(water_demand_toml: Path) -> Path:
    """Generate a wflow_sediment model using forcings from the water demand model.

    Hydromt build wflow_sediment "./wflow_test_sediment" -i wflow_sediment_build.yml -d artifact_data --fo -v
    hydromt update wflow_sediment "wflow_piave_subbasin" -o "./wflow_test_extend_sediment" -i wflow_extend_sediment.yml -d artifact_data --fo -v
    """  # noqa
    tmp_dir = Path(tempfile.gettempdir())
    root: Path = tmp_dir / "wflow_test_sediment"
    updated_root: Path = tmp_dir / "wflow_test_extend_sediment"
    build_workflow_yml: Path = EXAMPLE_DIR / "wflow_sediment_build.yml"
    update_workflow_yml: Path = EXAMPLE_DIR / "wflow_extend_sediment.yml"
    sediment_toml = updated_root / "wflow_sediment.toml"

    build_command = [
        "hydromt",
        "build",
        "wflow_sediment",
        root.as_posix(),
        "-i",
        build_workflow_yml.as_posix(),
        "-d",
        "artifact_data",
        "--fo",
        "-v",
    ]
    update_command = [
        "hydromt",
        "update",
        "wflow_sediment",
        root.as_posix(),
        "-o",
        updated_root.as_posix(),
        "-i",
        update_workflow_yml.as_posix(),
        "-d",
        "artifact_data",
        "--fo",
        "-v",
    ]
    if REBUILD:
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(updated_root, ignore_errors=True)

    if not updated_root.exists():
        _subprocess_run(build_command)
        _subprocess_run(update_command)

    # copy forcings from water demand model
    water_demand_model = WflowSbmModel(
        root=water_demand_toml.parent, mode="r", data_libs=["artifact_data"]
    )
    output_dir = water_demand_model.config.get_value("dir_output", abs_path=True)
    sbm_forcing_path = output_dir / water_demand_model.config.get_value(
        "output.netcdf_grid.path"
    )
    sediment_model = WflowSedimentModel(
        root=updated_root, mode="r", data_libs=["artifact_data"]
    )
    sediment_forcing_path: Path = sediment_model.config.get_value(
        "input.path_forcing", abs_path=True
    )
    sediment_forcing_path.unlink(missing_ok=True)
    shutil.copyfile(sbm_forcing_path, sediment_forcing_path)
    sediment_model.read()  # re-read to pick up new forcing file

    # Add river_q to forcing
    orig_ds = water_demand_model.forcing.data
    river_q = xr.load_dataset(sediment_forcing_path)
    river_q.raster.set_crs(water_demand_model.staticmaps.crs)

    # reproject river_q to sediment model grid
    river_q_reprojected = river_q["river_q"].raster.reproject_like(
        sediment_model.staticmaps.data
    )
    orig_ds["river_q"] = river_q_reprojected
    sediment_model.forcing.set(orig_ds)

    start_time = pd.to_datetime(orig_ds["time"].values.min())
    end_time = pd.to_datetime(orig_ds["time"].values.max())

    # fix config
    sediment_model.config.set("input.netcdf_scalar.path", sediment_forcing_path.name)
    sediment_model.config.set("input.netcdf_scalar.variable", "river_q")
    sediment_model.config.set("output.netcdf_scalar", None)
    sediment_model.config.set("time.starttime", start_time)
    sediment_model.config.set("time.endtime", end_time)
    sediment_model.config.set("output.netcdf_grid.path", "wflow_sediment_forcing.nc")

    sediment_model.write()

    return sediment_toml


def _subprocess_run(command: list[str]) -> bool:
    logger.info(f"Running command: {' '.join(command)}")
    process = subprocess.run(command, text=True, capture_output=True)
    success = process.returncode == 0

    if process.stdout:
        logger.info(process.stdout)

    if not success:
        logger.error(
            f"Command {' '.join(command)} failed with return code {process.returncode}"
        )
        logger.error(process.stderr)
        raise RuntimeError("Subprocess command failed.")

    return success


def _julia_run(toml_path: Path):
    run_command = [
        "julia",
        "-e",
        f"""
        toml_path = raw\"{toml_path.as_posix()}\";
        using Wflow;
        Wflow.run(toml_path)
        """,
    ]
    _subprocess_run(run_command)


REBUILD = True

if __name__ == "__main__":
    water_demand_toml = generate_water_demand_model()
    _julia_run(water_demand_toml)

    sediment_toml = generate_sediment_model(water_demand_toml)
    _julia_run(sediment_toml)
