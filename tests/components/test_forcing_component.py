import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr
from hydromt._typing.model_mode import ModelMode

from hydromt_wflow.components import WflowForcingComponent


@pytest.fixture
def mock_model(mock_model_factory) -> MagicMock:
    """Create a mock model with a root."""
    return mock_model_factory(mode="w")


def test_wflow_forcing_component_init(mock_model: MagicMock):
    # Setup the component
    component = WflowForcingComponent(mock_model)

    # Assert that the internal data is None
    assert component._data is None

    # When asking for data property, it should return a xr.Dataset
    assert isinstance(component.data, xr.Dataset)
    assert isinstance(component._data, xr.Dataset)  # Same for internal
    assert len(component.data) == 0  # Zero data varsdef test


def test_wflow_forcing_component_set(
    mock_model_staticmaps: MagicMock,
    forcing_layer: xr.DataArray,
):
    # Setup the component
    component = WflowForcingComponent(
        mock_model_staticmaps,
        region_component="staticmaps",
    )

    # Set the data in the component
    component.set(forcing_layer, "temp")

    # Assert the data
    assert "temp" in component.data.data_vars
    assert list(component.data.dims) == ["time", "lat", "lon"]
    assert component.data.time.size == 20


def test_wflow_forcing_component_set_errors(
    mock_model: MagicMock,
    static_layer: xr.DataArray,
):
    # Setup the component
    component = WflowForcingComponent(
        mock_model,
    )

    # Static layer has no time dimension, so this should result in an error
    with pytest.raises(
        ValueError,
        match="'time' dimension not found in data",
    ):
        component.set(static_layer, "foo")


def test_wflow_forcing_component_set_warnings(
    caplog: pytest.LogCaptureFixture,
    mock_model_staticmaps: MagicMock,
    forcing_layer: xr.DataArray,
):
    caplog.set_level(logging.WARNING)
    # Setup the component
    component = WflowForcingComponent(
        mock_model_staticmaps,
        region_component="staticmaps",
    )

    # Set the lat coords to something else, which should result in it not being
    # the same as the staticmaps anymore
    forcing_layer = forcing_layer.assign_coords({"lat": [2, 3]})
    component.set(forcing_layer, "foo")
    # Assert the logging of reprojecting
    assert (
        "Forcing data differs spatially from staticmaps, \
reprojecting.."
        in caplog.text
    )
    # Funny extra assert, all data should now be nodata
    np.testing.assert_almost_equal(component.data.foo.mean().values, -9999)


def test_wflow_forcing_component_read(
    mock_model: MagicMock,
    forcing_layer_path: xr.DataArray,
):
    mock_model.root._mode = ModelMode["READ"]
    # Setup the component
    component = WflowForcingComponent(
        mock_model,
    )
    # Assert the current state
    assert len(component.data) == 0
    assert "foo" not in component.data.data_vars

    # Read some data
    component.read(filename=forcing_layer_path)

    # Assert our favourate variable is now in the data
    assert "foo" in component.data.data_vars
