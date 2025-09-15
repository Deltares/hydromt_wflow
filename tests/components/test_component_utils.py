from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hydromt_wflow.components.utils import (
    _mount,
    _relpath,
    get_mask_layer,
    make_config_paths_relative,
)
from hydromt_wflow.components.utils import test_equal_grid_data as equal_grid_data


def test__mount():
    # Call the function on unix path
    m = _mount("/d/tmp/foo")
    # Assert the mount
    assert m == "/d/"

    # Call the function on windows path
    m = _mount("d:/tmp/foo")
    # Assert the mount
    assert m == "d:/"

    # Call the function on a relative path
    m = _mount("tmp/foo")
    # Assert that it's None
    assert m is None


def test__relpath_abs(tmp_path: Path):
    # Call the function
    p = _relpath(Path(tmp_path, "tmp/tmp.txt"), tmp_path)

    # Assert the output
    assert isinstance(p, str)
    assert p == "tmp/tmp.txt"

    # Path one above the current, also pass as a string
    in_p = Path(tmp_path.parent, "tmp.txt").as_posix()
    p = _relpath(in_p, tmp_path)

    # Assert the output
    assert p == "../tmp.txt"


def test__relpath_rel(tmp_path: Path):
    # Call the function on a path that is already relative
    p = _relpath("tmp/tmp.txt", tmp_path)

    # Assert the output is just the same
    assert p == "tmp/tmp.txt"


def test__relpath_mount(tmp_path: Path, mount_string: str):
    # Call the function on a path that is located on another mount
    p = _relpath(Path(mount_string, "tmp", "tmp.txt"), tmp_path)

    # Assert the output is just the same
    assert p == f"{mount_string}tmp/tmp.txt"


def test__relpath_other(tmp_path: Path):
    # Call the function on value that could not be paths
    p = _relpath([2, 2], tmp_path)  # E.g. a list

    # Assert that the list is returned
    assert p == [2, 2]


def test_get_mask_layer_none(mask_layer: xr.DataArray):
    # Call the function, nothing in is nothing out
    mask = get_mask_layer(None)
    # Assert mask is indeed None
    assert mask is None

    # Call the function but layer not found in dataset(s)
    mask = get_mask_layer(
        "mask",
        mask_layer.to_dataset(name="no_mask"),
        mask_layer.to_dataset(name="still_no_mask"),
    )
    # Assert None because not found
    assert mask is None


def test_get_mask_layer_from_str(mask_layer: xr.DataArray):
    # Call the function with a layer in a dataset that can be found
    mask = get_mask_layer(
        "mask",
        mask_layer.to_dataset(name="mask"),
        mask_layer.to_dataset(name="no_mask"),
    )

    # Assert the output data
    assert isinstance(mask, xr.DataArray)
    np.testing.assert_array_equal(
        mask,
        np.array([[True, True], [True, False]]),
    )


def test_get_mask_layer_from_array(mask_layer: xr.DataArray):
    # Assert the current state of the mask layer
    np.testing.assert_array_equal(mask_layer.data, np.array([[1, 1], [1, -9999]]))
    assert mask_layer.raster.nodata == -9999

    # Call the function
    mask = get_mask_layer(mask_layer)

    # Assert its now boolean data
    np.testing.assert_array_equal(
        mask,
        np.array([[True, True], [True, False]]),
    )


def test_get_mask_layer_errors():
    # Call the function but give a faulty type, like a list
    with pytest.raises(
        ValueError,
        match="Unknown type for determining mask: list",
    ):
        _ = get_mask_layer([2, 2])


def test_make_config_paths_relative(
    tmp_path: Path,
    config_dummy_document: dict,
):
    # Assert that a full path is present
    p = config_dummy_document["baz"]["file1"]
    assert Path(p).is_absolute()
    assert config_dummy_document["spooky"]["ghost"] == [1, 2, 3]
    assert config_dummy_document["baz"]["file2"] == "tmp/tmp.txt"

    # Call the function
    cfg = make_config_paths_relative(config_dummy_document, tmp_path)

    # Assert the outcome
    # Assert that a full path is present
    p = cfg["baz"]["file1"]
    assert not Path(p).is_absolute()  # Not anymore
    assert cfg["spooky"]["ghost"] == [1, 2, 3]
    assert cfg["baz"]["file2"] == "tmp/tmp.txt"


def test__equal_grid():
    # Create two identical grids
    data1 = xr.DataArray(
        np.array([[1, 2], [3, 4]]),
        dims=("y", "x"),
        coords={
            "x": ("x", [0.5, 1.5]),
            "y": ("y", [0.5, 1.5]),
        },
    )
    data1.raster.set_crs(4326)
    data1.raster.set_nodata(-1)
    data1 = data1.to_dataset(name="data1")

    # Test equality
    eq, _ = equal_grid_data(data1, data1.copy())
    assert eq

    # Check for empty
    empty_grid = xr.Dataset()
    eq, errors = equal_grid_data(empty_grid, empty_grid)
    assert eq

    eq, errors = equal_grid_data(empty_grid, data1)
    assert not eq
    assert errors["grid"] == "first grid is empty, second is not"

    # Check for crs
    data2 = data1.copy()
    data2.raster.set_crs(3857)
    eq, errors = equal_grid_data(data1, data2)
    assert not eq
    assert errors["crs"] == "the two grids have different crs"

    # Check for dims
    data2 = data1.copy()
    data2 = data1.rename({"x": "x2"})
    eq, errors = equal_grid_data(data1, data2)
    assert not eq
    assert errors["dims"] == "dim x2 not in grid"

    data2 = data1.copy()
    data2["x"] = [0.5, 2.5]
    eq, errors = equal_grid_data(data1, data2)
    assert not eq
    assert errors["dims"] == "dim x not identical"

    # Check for missing data variables
    data2 = data1.copy()
    data2["data2"] = data2["data1"]
    eq, errors = equal_grid_data(data1, data2)
    assert not eq
    assert errors["Other grid has additional maps"] == "data2"

    eq, errors = equal_grid_data(data2, data1)
    assert not eq
    assert errors["Other grid is missing maps"] == "data2"

    # Check on data variable differences: dtype, nodata, values
    data2 = data1.copy()
    data2["data1"] = data2["data1"].astype(np.float32)
    eq, errors = equal_grid_data(data1, data2)
    assert not eq
    assert errors["1 invalid maps"] == {"data1": "float32 instead of int64"}

    data2 = data1.copy()
    data2["data1"].raster.set_nodata(-9999)
    eq, errors = equal_grid_data(data1, data2)
    assert not eq
    assert errors["1 invalid maps"] == {"data1": "nodata -9999 instead of -1; "}

    data2 = data1.copy()
    data2["data1"].values = [[1, 1], [3, 6]]
    eq, errors = equal_grid_data(data1, data2)
    assert not eq
    assert errors["1 invalid maps"] == {"data1": "mean diff (2 cells): -0.5000; "}
