import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import xarray as xr

from hydromt_wflow.components import StaticmapsComponent, WflowConfigComponent


def test_staticmaps_component_init(mock_model: MagicMock):
    # Setup the component
    component = StaticmapsComponent(mock_model)

    # Assert that the internal data is None
    assert component._data is None

    # When asking for data property, it should return a tomlkit document
    assert isinstance(component.data, xr.Dataset)
    assert isinstance(component._data, xr.Dataset)  # Same for internal
    assert len(component.data) == 0  # Zero data vars


def test_staticmaps_component_set(
    mock_model: MagicMock,
    static_layer: xr.DataArray,
):
    # Setup the component
    component = StaticmapsComponent(mock_model)

    # Set the data in the component
    component.set(static_layer, "layer1")

    # Assert the data
    assert "layer1" in component.data.data_vars
    assert list(component.data.dims) == ["lat", "lon"]


def test_staticmaps_component_set_cyclic(
    mock_model: MagicMock,
    cyclic_layer: xr.DataArray,
    cyclic_layer_large: xr.DataArray,
):
    # Setup the component
    component = StaticmapsComponent(mock_model)

    # Set the data in the component
    component.set(cyclic_layer, "layer1")

    # Assert the current state
    assert "layer1" in component.data.data_vars
    assert "time" in component.data.dims
    assert len(component.data.time) == 12

    # Add another but with a different valid time dimension
    component.set(cyclic_layer_large, "layer2")

    # Assert the data and dimensions
    assert "layer2" in component.data.data_vars
    assert "time_365" in component.data.dims
    assert len(component.data.time_365) == 365

    # Set the data in the component
    component.set(cyclic_layer * 2, "layer3")

    # Assert the current state
    assert "layer3" in component.data.data_vars
    assert "time" in component.data.dims  # Still there, not changed


def test_staticmaps_component_set_errors(
    mock_model: MagicMock,
    cyclic_layer: xr.DataArray,
):
    # Setup the component
    component = StaticmapsComponent(mock_model)

    # No naming given either via the dataarray or argument
    with pytest.raises(ValueError, match="Unable to set DataArray data without a name"):
        component.set(cyclic_layer)

    # Wrong data input type, list is of course not accepted, only xarray stuff
    with pytest.raises(
        ValueError,
        match="Cannot set data of type list",
    ):
        component.set([2, 2])

    # Wrong length of cyclic data, either monthly or daily
    with pytest.raises(
        ValueError,
        match=r"Length of cyclic dataset \(11\) is not supported",
    ):
        # Drop 1 so the length becomes 11, which is invalid
        component.set(
            cyclic_layer.where(cyclic_layer.time != 1, drop=True),
            name="layer1",
        )


def test_staticmaps_component_set_warnings(
    mock_model: MagicMock,
    static_layer: xr.DataArray,
):
    pass


def test_staticmaps_component_update_names(
    mock_model: MagicMock,
    static_layer: xr.DataArray,
):
    # Setup the component
    component = StaticmapsComponent(mock_model)

    # Set data like a dummy
    component._data = static_layer.to_dataset(name="layer1")
    component.data["layer2"] = static_layer * 2
    component.data["layer3"] = static_layer * 3

    # Rename some variables
    component.update_names(layer1="foo", layer2="bar")

    # Assert it succeded
    assert "foo" in component.data.data_vars
    assert "bar" in component.data.data_vars


def test_staticmaps_component_update_names_warming(
    caplog: pytest.LogCaptureFixture,
    mock_model: MagicMock,
    static_layer: xr.DataArray,
):
    caplog.set_level(logging.INFO)
    # Setup the component
    component = StaticmapsComponent(mock_model)

    # Set data like a dummy
    component._data = static_layer.to_dataset(name="layer1")

    # Rename but faulty
    component.update_names(layer2="foo")
    # Assert the warning message
    assert "Could not rename ['layer2'], not found in data" in caplog.text


def test_staticmaps_component_read():
    pass


def test_staticmaps_component_write(
    mock_model: MagicMock,
    static_layer: xr.DataArray,
):
    # Set a config component
    mock_model.config = WflowConfigComponent(mock_model)

    # Setup the component
    component = StaticmapsComponent(mock_model)

    # Set the data like a dummy
    component._data = static_layer.to_dataset(name="layer1")

    # Write the data
    component.write()

    # Assert the output (and config)
    assert Path(component.root.path, component._filename).is_file()
    assert "path_static" in component.model.config.data.get("input")
