"""Test plugin model class against hydromt.models.model_api."""

import warnings
from os.path import abspath, dirname, join

import numpy as np
import pytest
import xarray as xr

from hydromt_wflow.wflow import WflowModel
from hydromt_wflow.wflow_sediment import WflowSedimentModel

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")

_supported_models = {
    "wflow": WflowModel,
    "wflow_sediment": WflowSedimentModel,
    "wflow_simple": WflowModel,
}


def _compare_wflow_models(mod0, mod1):
    # check maps
    invalid_maps = {}
    # invalid_maps_dtype = {}
    if mod0._grid is not None:
        maps = mod0.grid.raster.vars
        assert np.all(mod0.crs == mod1.crs), "map crs grid"
        # check on dims names and values
        for dim in mod1._grid.dims:
            try:
                xr.testing.assert_identical(mod1._grid[dim], mod0._grid[dim])
            except AssertionError:
                raise AssertionError(f"dim {dim} in map not identical")

        for name in maps:
            map0 = mod0.grid[name].fillna(0)
            if name not in mod1.grid:
                invalid_maps[name] = "KeyError"
                continue
            map1 = mod1.grid[name].fillna(0)
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
                err = (
                    err
                    if map0.raster.nodata == map1.raster.nodata
                    else f"nodata {map1.raster.nodata} instead of \
{map0.raster.nodata}; {err}"
                )
                notclose = ~np.equal(map0, map1)
                ncells = int(np.sum(notclose))
                if ncells > 0:
                    # xy = map0.raster.idx_to_xy(np.where(notclose.ravel())[0])
                    # yxs = ", ".join([f"({y:.6f}, {x:.6f})" for x, y in zip(*xy)])
                    diff = (map0.values - map1.values)[notclose].mean()
                    err = f"diff ({ncells:d} cells): {diff:.4f}; {err}"
                invalid_maps[name] = err
    # invalid_map_str = ", ".join(invalid_maps)
    # assert (
    #    len(invalid_maps_dtype) == 0
    # ), f"{len(invalid_maps_dtype)} invalid dtype for maps: {invalid_maps_dtype}"
    assert len(invalid_maps) == 0, f"{len(invalid_maps)} invalid maps: {invalid_maps}"
    # check geoms
    if mod0._geoms:
        for name in mod0.geoms:
            geom0 = mod0.geoms[name]
            geom1 = mod1.geoms[name]
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
    if mod0._config:
        # flatten
        assert mod0._config == mod1._config, "config mismatch"


@pytest.mark.parametrize("model", list(_supported_models.keys()))
def test_model_class(model, example_models):
    mod = example_models[model]
    if mod is not None:
        mod.read()
        # run test_model_api() method
        non_compliant_list = mod._test_model_api()
        assert len(non_compliant_list) == 0


# @pytest.mark.timeout(300)  # max 5 min
# @pytest.mark.parametrize("model", list(_supported_models.keys()))
# def test_model_build(tmpdir, model, example_models, example_inis):
#     # get model type
#     model_type = _supported_models[model]
#     # create folder to store new model
#     root = str(tmpdir.join(model))
#     mod1 = model_type(root=root, mode="w", data_libs="artifact_data")
#     # Build method options
#     region = {
#         "subbasin": [12.2051, 45.8331],
#         "strord": 4,
#         "bounds": [11.70, 45.35, 12.95, 46.70],
#     }
#     # get ini file
#     opt = example_inis[model]
#     # Build model
#     mod1.build(region=region, opt=opt)
#     # Check if model is api compliant
#     non_compliant_list = mod1._test_model_api()
#     assert len(non_compliant_list) == 0

#     # Compare with model from examples folder
#     # (need to read it again for proper geoms check)
#     mod1 = model_type(root=root, mode="r")
#     mod1.read()
#     # get reference model
#     mod0 = example_models[model]
#     if mod0 is not None:
#         mod0.read()
#         # compare models
#         _compare_wflow_models(mod0, mod1)


def test_model_clip(tmpdir, example_wflow_model, clipped_wflow_model):
    model = "wflow"

    # Clip method options
    destination = str(tmpdir.join(model))
    region = {
        "subbasin": [12.3006, 46.4324],
        "wflow_streamorder": 4,
    }

    # Clip workflow, based on example model
    example_wflow_model.read()
    example_wflow_model.set_root(destination, mode="w")
    example_wflow_model.clip_grid(region)
    example_wflow_model.clip_forcing()
    example_wflow_model.write()
    # Check if model is api compliant
    non_compliant_list = example_wflow_model._test_model_api()
    assert len(non_compliant_list) == 0

    # Compare with model from examples folder
    # (need to read it again for proper geoms check)
    mod1 = WflowModel(root=destination, mode="r")
    mod1.read()
    # Read reference clipped model
    clipped_wflow_model.read()
    # compare models
    _compare_wflow_models(clipped_wflow_model, mod1)


def test_model_inverse_clip(tmpdir, example_wflow_model):
    # Clip method options
    region = {
        "subbasin": [12.3006, 46.4324],
        "wflow_streamorder": 4,
    }

    # Clip workflow, based on example model
    example_wflow_model.read()
    # Get number of active pixels from full model
    n_pixels_full = example_wflow_model.grid["wflow_subcatch"].sum()
    example_wflow_model.clip_grid(region, inverse_clip=True)
    # Get number of active pixels from inversely clipped model
    n_pixels_inverse_clipped = example_wflow_model.grid["wflow_subcatch"].sum()

    # Do clipping again, but normally
    example_wflow_model.read()
    example_wflow_model.clip_grid(region, inverse_clip=False)
    # Get number of active pixels from clipped model
    n_pixels_clipped = example_wflow_model.grid["wflow_subcatch"].sum()

    assert n_pixels_inverse_clipped < n_pixels_full
    assert n_pixels_full == n_pixels_inverse_clipped + n_pixels_clipped


def test_model_results(example_wflow_results):
    # Tests on results
    # Number of dict keys = 1 for output + 1 for netcdf + nb of csv.column
    assert len(example_wflow_results.results) == (
        2 + len(example_wflow_results.get_config("csv.column"))
    )

    # Check that the output and netcdf xr.Dataset are present
    assert "output" in example_wflow_results.results
    assert isinstance(example_wflow_results.results["netcdf"], xr.Dataset)

    # Checks for the csv columns
    # Q for gauges_grdc
    assert len(example_wflow_results.results["Q_gauges_grdc"].index) == 3
    assert np.isin(6349410, example_wflow_results.results["Q_gauges_grdc"].index)

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
    assert np.isclose(example_wflow_results.results["res-volume"]["y"], 46.16656)
