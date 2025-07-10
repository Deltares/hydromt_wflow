from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hydromt_wflow.components.utils import (
    _mount,
    _relpath,
    get_mask_layer,
)


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
