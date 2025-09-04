import logging
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import xarray as xr
from hydromt.model import ModelRoot

from hydromt_wflow.components import WflowStaticmapsComponent
from hydromt_wflow.wflow_sbm import WflowSbmModel


@pytest.fixture
def mock_model(mock_model_factory) -> MagicMock:
    """Create a mock model with a root."""
    return mock_model_factory(mode="w")


def test_wflow_staticmaps_component_init(mock_model: MagicMock):
    # Setup the component
    component = WflowStaticmapsComponent(mock_model)

    # Assert that the internal data is None
    assert component._data is None

    # When asking for data property, it should return a xr.Dataset
    assert isinstance(component.data, xr.Dataset)
    assert isinstance(component._data, xr.Dataset)  # Same for internal
    assert len(component.data) == 0  # Zero data vars


def test_wflow_staticmaps_component_set(
    mock_model: MagicMock,
    static_layer: xr.DataArray,
):
    # Setup the component
    component = WflowStaticmapsComponent(mock_model)

    # Set the data in the component
    component.set(static_layer, "layer1")
    # Set with name included in the dataarray
    static_layer.name = "layer2"
    component.set(static_layer * 2)

    # Assert the data
    assert "layer1" in component.data.data_vars
    assert "layer2" in component.data.data_vars
    assert list(component.data.dims) == ["lat", "lon"]


def test_wflow_staticmaps_component_set_cyclic(
    mock_model: MagicMock,
    cyclic_layer: xr.DataArray,
    cyclic_layer_large: xr.DataArray,
):
    # Setup the component
    component = WflowStaticmapsComponent(mock_model)

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


def test_wflow_staticmaps_component_set_mask(
    mock_model: MagicMock,
    static_layer: xr.DataArray,
    mask_layer: xr.DataArray,
):
    # Setup the component
    component = WflowStaticmapsComponent(mock_model)

    # Set the mask layer in the dataset and use that as the mask in the set method
    component.set(mask_layer, name="mask")
    mock_model._MAPS = {"basins": "mask"}  # Needed as internally references like this
    component.set(static_layer, name="layer2")

    # Assert that layer2 has been masked
    np.testing.assert_array_equal(
        component.data["layer2"],
        np.array([[1, -9999], [1, 1]]),
    )


def test_wflow_staticmaps_component_set_errors(
    mock_model: MagicMock,
    cyclic_layer: xr.DataArray,
):
    # Setup the component
    component = WflowStaticmapsComponent(mock_model)

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


def test_wflow_staticmaps_component_set_warnings(
    caplog: pytest.LogCaptureFixture,
    mock_model: MagicMock,
    static_layer: xr.DataArray,
    cyclic_layer: xr.DataArray,
    cyclic_layer_large: xr.DataArray,
):
    caplog.set_level(logging.INFO)
    # Setup the component
    component = WflowStaticmapsComponent(mock_model)

    # Set the data
    component.set(static_layer, "layer1")
    component.set(cyclic_layer.rename(time="layer"), "layer2")

    # First warning when replacing a layer that is already present
    component.set(static_layer * 2, "layer1")
    # Assert the logging message
    assert "Replacing grid map: layer1" in caplog.text

    # Second warning when setting a layer with a 'layer' dimension that differs
    # from the layer dimension that is already in there
    component.set(cyclic_layer_large.rename(time="layer"), "layer3")
    # Assert the logging message
    assert (
        "Replacing 'layer' coordinate, dropping variables \
(['layer2']) associated with old coordinate"
        in caplog.text
    )


def test_wflow_staticmaps_component_update_names(
    mock_model: MagicMock,
    static_layer: xr.DataArray,
):
    # Setup the component
    component = WflowStaticmapsComponent(mock_model)

    # Set data like a dummy
    component._data = static_layer.to_dataset(name="layer1")
    component.data["layer2"] = static_layer * 2
    component.data["layer3"] = static_layer * 3

    # Rename some variables
    component.update_names(layer1="foo", layer2="bar")

    # Assert it succeded
    assert "foo" in component.data.data_vars
    assert "bar" in component.data.data_vars


def test_wflow_staticmaps_component_update_names_warming(
    caplog: pytest.LogCaptureFixture,
    mock_model: MagicMock,
    static_layer: xr.DataArray,
):
    caplog.set_level(logging.INFO)
    # Setup the component
    component = WflowStaticmapsComponent(mock_model)

    # Set data like a dummy
    component._data = static_layer.to_dataset(name="layer1")

    # Rename but faulty
    component.update_names(layer2="foo")
    # Assert the warning message
    assert "Could not rename ['layer2'], not found in data" in caplog.text


def test_wflow_staticmaps_component_read(
    mock_model_factory: Callable[[Path, str], WflowSbmModel],
    model_subbasin_cached: Path,
):
    # Set the root to model dir in read mode
    mock_model = mock_model_factory(path=model_subbasin_cached, mode="r")

    # Setup the component
    component = WflowStaticmapsComponent(mock_model)
    # Assert the internal data is None
    assert component._data is None

    # Read the data
    type(component.model.config).get_value = MagicMock(return_value="")
    component.read()

    # Assert the data
    assert isinstance(component.data, xr.Dataset)
    assert "meta_landuse" in component.data.data_vars
    assert len(component.data) == 69  # 69 layers


def test_wflow_staticmaps_component_read_empty(
    tmp_path: Path,
    mock_model: MagicMock,
):
    # Set the root to model dir in read mode
    type(mock_model).root = PropertyMock(
        side_effect=lambda: ModelRoot(tmp_path, mode="r"),
    )

    # Setup the component
    component = WflowStaticmapsComponent(mock_model)
    # Assert the internal data is None
    assert component._data is None

    # Read the data
    type(component.model.config).get_value = MagicMock(return_value="")
    component.read()
    # Assert that its an emptry Dataset
    assert isinstance(component.data, xr.Dataset)
    assert len(component.data) == 0


def test_wflow_staticmaps_component_write(
    mock_model: MagicMock,
    static_layer: xr.DataArray,
):
    # Setup the component
    component = WflowStaticmapsComponent(mock_model)
    mock_model.components = {"staticmaps": component}

    # Set the data like a dummy
    component._data = static_layer.to_dataset(name="layer1")

    # Write the data
    type(component.model.config).get_value = MagicMock(return_value="")  # dir_input
    component.write()

    # Assert the output (and config)
    assert Path(component.root.path, component._filename).is_file()
