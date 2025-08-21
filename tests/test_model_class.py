"""Test plugin model class against hydromt.models.model_api."""

import warnings
from os.path import abspath, dirname, join
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hydromt_wflow.wflow import WflowModel
from hydromt_wflow.wflow_sediment import WflowSedimentModel

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")

_supported_models: dict[str, type[WflowModel]] = {
    "wflow": WflowModel,
    "wflow_sediment": WflowSedimentModel,
    "wflow_simple": WflowModel,
}


def _compare_wflow_models(mod0: WflowModel, mod1: WflowModel):
    # check maps
    invalid_maps = {}
    # invalid_maps_dtype = {}
    if mod0.staticmaps._data:
        maps = mod0.staticmaps.data.raster.vars
        assert np.all(mod0.crs == mod1.crs), "map crs grid"
        # check on dims names and values
        for dim in mod1.staticmaps.data.dims:
            try:
                xr.testing.assert_identical(
                    mod1.staticmaps.data[dim], mod0.staticmaps.data[dim]
                )
            except AssertionError:
                raise AssertionError(f"dim {dim} in map not identical")

        for name in maps:
            map0 = mod0.staticmaps.data[name].fillna(0)
            if name not in mod1.staticmaps.data:
                invalid_maps[name] = "KeyError"
                continue
            map1 = mod1.staticmaps.data[name].fillna(0)
            if (
                not np.allclose(map0, map1, atol=1e-3, rtol=1e-3)
                or map0.dtype != map1.dtype
            ):
                if len(map0.dims) > 2:  # 3 dim map
                    map0 = map0[0, :, :]
                    map1 = map1[0, :, :]
                # Check on dtypes
                err = (
                    ""
                    if map0.dtype == map1.dtype
                    else f"{map1.dtype} instead of {map0.dtype}"
                )
                # Check on nodata
                # hilariously np.nan == np.nan returns False, hence the additional check
                err = (
                    err
                    if map0.raster.nodata == map1.raster.nodata
                    or (np.isnan(map0.raster.nodata) and np.isnan(map1.raster.nodata))
                    else f"nodata {map1.raster.nodata} instead of \
{map0.raster.nodata}; {err}"
                )
                notclose = ~np.equal(map0, map1)
                ncells = int(np.sum(notclose))
                if ncells > 0:
                    # xy = map0.raster.idx_to_xy(np.where(notclose.ravel())[0])
                    # yxs = ", ".join([f"({y:.6f}, {x:.6f})" for x, y in zip(*xy)])
                    diff = (map0.values - map1.values)[notclose].mean()
                    err = f"mean diff ({ncells:d} cells): {diff:.4f}; {err}"
                invalid_maps[name] = err
    # invalid_map_str = ", ".join(invalid_maps)
    # assert (
    #    len(invalid_maps_dtype) == 0
    # ), f"{len(invalid_maps_dtype)} invalid dtype for maps: {invalid_maps_dtype}"
    assert len(invalid_maps) == 0, f"{len(invalid_maps)} invalid maps: {invalid_maps}"
    # check geoms
    if mod0.geoms._data:
        for name in mod0.geoms.data:
            geom0 = mod0.geoms.get(name)
            geom1 = mod1.geoms.get(name)
            assert geom0.index.size == geom1.index.size
            assert np.all(set(geom0.index) == set(geom1.index)), f"geom index {name}"
            assert geom0.columns.size == geom1.columns.size
            assert np.all(set(geom0.columns) == set(geom1.columns)), (
                f"geom columns {name}"
            )
            assert geom0.crs == geom1.crs, f"geom crs {name}"
            if not np.all(geom0.geometry == geom1.geometry):
                warnings.warn(f"New geom {name} different than the example one.")
    # check config
    if mod0.config._data:
        # flatten
        assert mod0.config.data == mod1.config.data, "config mismatch"


@pytest.mark.skip(reason="Issue with buffer in get_rasterdataset hydromt#1226")
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
    opt = example_inis[model]

    # Build model
    mod1.build(steps=opt["steps"])

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


@pytest.mark.skip(
    reason="Allow clipping again - current conflict for reading forcing and states"
)
@pytest.mark.timeout(60)  # max 1 min
def test_model_clip(
    tmpdir: Path, example_wflow_model: WflowModel, clipped_wflow_model: WflowModel
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
    example_wflow_model.clip_grid(region)
    example_wflow_model.clip_forcing()
    example_wflow_model.clip_states()
    example_wflow_model.write()

    # Compare with model from examples folder
    # (need to read it again for proper geoms check)
    mod1 = WflowModel(root=destination, mode="r")
    mod1.read()
    # Read reference clipped model
    clipped_wflow_model.read()
    # compare models
    _compare_wflow_models(clipped_wflow_model, mod1)


@pytest.mark.skip(
    reason="Allow clipping again - current conflict for reading forcing and states"
)
def test_model_inverse_clip(example_wflow_model: WflowModel):
    # Clip method options
    region = {
        "subbasin": [12.3006, 46.4324],
        "meta_streamorder": 4,
    }

    # Clip workflow, based on example model
    example_wflow_model.read()
    # Get number of active pixels from full model
    n_pixels_full = example_wflow_model.staticmaps.data["subcatchment"].sum()
    example_wflow_model.clip_grid(region.copy(), inverse_clip=True)
    # Get number of active pixels from inversely clipped model
    n_pixels_inverse_clipped = example_wflow_model.staticmaps.data["subcatchment"].sum()

    # Do clipping again, but normally
    example_wflow_model.read()
    example_wflow_model.clip_grid(region.copy(), inverse_clip=False)
    # Get number of active pixels from clipped model
    n_pixels_clipped = example_wflow_model.staticmaps.data["subcatchment"].sum()

    assert n_pixels_inverse_clipped < n_pixels_full
    assert n_pixels_full == n_pixels_inverse_clipped + n_pixels_clipped


def test_model_results(example_wflow_results):
    # Tests on results
    # Number of dict keys = 1 for netcdf_grid + 1 for netcdf_scalar + nb of csv.column
    assert len(example_wflow_results.results) == (
        2 + len(example_wflow_results.get_config("output.csv.column"))
    )

    # Check that the output and netcdf xr.Dataset are present
    assert "netcdf_grid" in example_wflow_results.results
    assert isinstance(example_wflow_results.results["netcdf_scalar"], xr.Dataset)

    # Checks for the csv columns
    # Q for gauges_grdc
    assert len(example_wflow_results.results["river_q_gauges_grdc"].index) == 3
    assert np.isin(6349410, example_wflow_results.results["river_q_gauges_grdc"].index)

    # Coordinates and values for coordinate.x and index.x for temp
    assert np.isclose(
        example_wflow_results.results["temp_bycoord"]["x"].values,
        example_wflow_results.results["temp_byindex"]["x"].values,
    )
    assert np.allclose(
        example_wflow_results.results["temp_bycoord"].values,
        example_wflow_results.results["temp_byindex"].values,
    )

    # Coordinates of the reservoir
    assert np.isclose(example_wflow_results.results["reservoir_volume"]["y"], 46.16656)
