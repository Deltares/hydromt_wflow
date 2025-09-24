"""Test plugin model class against hydromt.models.model_api."""

from os.path import abspath, dirname, join
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydromt_wflow.wflow_base import WflowBaseModel
from hydromt_wflow.wflow_sbm import WflowSbmModel
from hydromt_wflow.wflow_sediment import WflowSedimentModel

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")

_supported_models: dict[str, type[WflowBaseModel]] = {
    "wflow": WflowSbmModel,
    "wflow_sediment": WflowSedimentModel,
    "wflow_simple": WflowSbmModel,
}


def _compare_wflow_models(mod0: WflowBaseModel, mod1: WflowBaseModel):
    # check maps
    if mod0.staticmaps._data:
        eq, errors = mod0.staticmaps.test_equal(mod1.staticmaps)
        assert eq, f"staticmaps not equal: {errors}"

    # check geoms
    if mod0.geoms._data:
        eq, errors = mod0.geoms.test_equal(mod1.geoms)
        assert eq, f"geoms not equal: {errors}"

    # check config
    if mod0.config._data:
        # flatten
        eq, errors = mod0.config.test_equal(mod1.config)
        assert eq, f"config not equal: {errors}"


@pytest.mark.timeout(300)  # max 5 min
@pytest.mark.parametrize("model", list(_supported_models.keys()))
def test_model_build(tmpdir, model, example_models, example_inis):
    # get model type
    model_type = _supported_models[model]
    # create folder to store new model
    root = str(tmpdir.join(model))
    param_path = (
        Path(__file__).parent.parent / "hydromt_wflow" / "data" / "parameters_data.yml"
    )
    mod1 = model_type(
        root=root, mode="w", data_libs=["artifact_data", param_path.as_posix()]
    )

    # get ini file
    steps = example_inis[model]

    # Build model
    mod1.build(steps=steps)

    # Compare with model from examples folder
    # (need to read it again for proper geoms check)
    mod1 = model_type(root=root, mode="r")
    mod1.read()
    # get reference model
    mod0 = example_models[model]
    if mod0 is not None:
        mod0.read()
        # compare models
        _compare_wflow_models(mod0, mod1)


def test_base_model_init_should_raise():
    with pytest.raises(TypeError) as exc_info:
        WflowBaseModel()

    assert str(exc_info.value) == (
        "``WflowBaseModel`` is an abstract class and cannot be instantiated "
        "directly. Please use one of its subclasses defined as hydromt-entry "
        "points: [``WflowSbmModel``, ``WflowSedimentModel``]"
    )


@pytest.mark.timeout(60)  # max 1 min
def test_model_clip(
    tmpdir: Path,
    example_wflow_model: WflowSbmModel,
    clipped_wflow_model: WflowSbmModel,
):
    model = "wflow"

    # Clip method options
    destination = str(tmpdir.join(model))
    region = {
        "subbasin": [12.3006, 46.4324],
        "meta_streamorder": 4,
    }

    # Clip workflow, based on example model
    example_wflow_model.read()
    example_wflow_model.set_root(destination, mode="w")
    example_wflow_model.clip(region)
    example_wflow_model.write()

    # Compare with model from examples folder
    # (need to read it again for proper geoms check)
    mod1 = WflowSbmModel(root=destination, mode="r")
    mod1.read()
    # Read reference clipped model
    clipped_wflow_model.read()
    # compare models
    _compare_wflow_models(clipped_wflow_model, mod1)
    # check states
    eq, errors = clipped_wflow_model.states.test_equal(mod1.states)
    assert eq, f"states not equal: {errors}"


def test_model_clip_reservoir(
    tmpdir: Path,
    example_wflow_model: WflowSbmModel,
    reservoir_rating: dict[str, pd.DataFrame],
):
    model = "wflow"

    # Clip method options
    destination = str(tmpdir.join(model))
    region = {
        "subbasin": [12.3162, 46.1676],
        "meta_streamorder": 3,
    }

    # Read and add reservoir rating tables before clipping
    clip_model = WflowSbmModel(root=example_wflow_model.root.path, mode="r+")
    clip_model.read()

    for name, tbl in reservoir_rating.items():
        clip_model.set_tables(tbl, name=name)
    assert len(clip_model.tables.data) == 4
    assert "reservoir_sh_169986" in clip_model.tables.data
    assert "reservoir_hq_3367" in clip_model.tables.data

    # Clip
    clip_model.set_root(destination, mode="w")
    clip_model.clip(region)
    assert len(clip_model.tables.data) == 2
    assert "reservoir_sh_169986" in clip_model.tables.data
    assert "reservoir_hq_3367" not in clip_model.tables.data


def test_sediment_model_clip(
    tmpdir: Path,
    example_sediment_model: WflowSedimentModel,
):
    model = "wflow"

    # Clip method options
    destination = str(tmpdir.join(model))
    region = {
        "subbasin": [12.3162, 46.1676],
        "meta_streamorder": 3,
    }

    # Clip workflow, based on example model
    example_sediment_model.read()
    example_sediment_model.set_root(destination, mode="w")
    example_sediment_model.clip(region)

    # Check extent of the clipped model
    n_pixels = example_sediment_model.staticmaps.data["subcatchment"].sum()
    assert n_pixels == 97
    # This model should still have one reservoir after clipping
    assert "reservoirs" in example_sediment_model.geoms.data
    assert len(example_sediment_model.geoms.data["reservoirs"]) == 1
    assert "reservoir_area_id" in example_sediment_model.staticmaps.data
    assert example_sediment_model.get_config("model.reservoir__flag") is True


def test_model_inverse_clip(example_wflow_model: WflowSbmModel):
    # Clip method options
    region = {
        "subbasin": [12.3006, 46.4324],
        "meta_streamorder": 4,
    }

    inverse_clip_model = WflowSbmModel(root=example_wflow_model.root.path, mode="r+")
    # Clip workflow, based on example model
    inverse_clip_model.read()
    # Get number of active pixels from full model
    n_pixels_full = inverse_clip_model.staticmaps.data["subcatchment"].sum()
    inverse_clip_model.clip(region.copy(), inverse_clip=True)
    # Get number of active pixels from inversely clipped model
    n_pixels_inverse_clipped = inverse_clip_model.staticmaps.data["subcatchment"].sum()

    # Do clipping again, but normally
    clip_model = WflowSbmModel(root=example_wflow_model.root.path, mode="r+")
    clip_model.read()
    clip_model.clip(region.copy(), inverse_clip=False)
    # Get number of active pixels from clipped model
    n_pixels_clipped = clip_model.staticmaps.data["subcatchment"].sum()

    assert n_pixels_inverse_clipped < n_pixels_full
    assert n_pixels_full == n_pixels_inverse_clipped + n_pixels_clipped


def test_model_outputs(example_wflow_outputs):
    # Tests on outputs
    wflow = example_wflow_outputs

    # Check the gridded output
    assert isinstance(wflow.output_grid.data, xr.Dataset)
    assert "q_river" in wflow.output_grid.data
    assert wflow.output_grid.data.raster.x_dim == wflow.staticmaps.data.raster.x_dim

    # Check the netcdf scalar output
    assert isinstance(wflow.output_scalar.data, xr.Dataset)
    assert "Q" in wflow.output_scalar.data

    # Check the csv output
    assert len(wflow.output_csv.data) == len(wflow.get_config("output.csv.column"))

    # Checks for the csv columns
    # Q for gauges_grdc
    assert len(wflow.output_csv.data["river_q_gauges_grdc"].index) == 3
    assert np.isin(6349410, wflow.output_csv.data["river_q_gauges_grdc"].index)

    # Coordinates and values for coordinate.x and index.x for temp
    assert np.isclose(
        wflow.output_csv.data["temp_bycoord"]["x"].values,
        wflow.output_csv.data["temp_byindex"]["x"].values,
    )
    assert np.allclose(
        wflow.output_csv.data["temp_bycoord"].values,
        wflow.output_csv.data["temp_byindex"].values,
    )

    # Coordinates of the reservoir
    assert np.isclose(wflow.output_csv.data["reservoir_volume"]["y"], 46.16656)
