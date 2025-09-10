import logging
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import xarray as xr

from hydromt_wflow.components import WflowOutputScalarComponent
from hydromt_wflow.wflow_sbm import WflowSbmModel


@pytest.fixture
def mock_model(mock_model_factory) -> WflowSbmModel:
    """Fixture to create a mock WflowSbmModel."""
    return mock_model_factory(mode="w")


# Define the side effect function
def get_value_side_effect(key, fallback=None):
    if key == "dir_output":
        return "run_results"
    elif key == "output.netcdf_scalar.path":
        return "output_scalar.nc"
    return fallback


def test_wflow_output_scalar_component_init(mock_model: WflowSbmModel):
    # Setup the component
    component = WflowOutputScalarComponent(mock_model)

    # Assert that the internal data is None
    assert component._data is None

    # When asking for data property, it should return a xarray.Dataset
    assert isinstance(component.data, xr.Dataset)
    assert isinstance(component._data, xr.Dataset)  # Same for internal
    assert len(component.data) == 0


def test_wflow_output_scalar_component_set(
    mock_model: WflowSbmModel, forcing_layer: xr.DataArray
):
    # Setup the component
    component = WflowOutputScalarComponent(mock_model)

    # Assert that the internal data is None, initializing will happen in set
    assert component._data is None

    # Set an entry
    # Copy else the name gets altered for the next test?
    component.set(forcing_layer.copy(), name="test_layer")

    # Assert the content
    assert isinstance(component._data, xr.Dataset)
    assert "test_layer" in component.data
    assert len(component.data) == 1


def test_wflow_output_scalar_component_set_alt(
    mock_model: WflowSbmModel, forcing_layer: xr.DataArray
):
    # Setup the component
    component = WflowOutputScalarComponent(mock_model)

    # Assert that the internal data is None, initializing will happen in set
    assert component._data is None

    # Set an entry
    # Copy else the name gets altered for the next test?
    forcing_layer.name = "precip"
    component.set(forcing_layer.copy())

    # Assert the content
    assert "precip" in component.data


def test_wflow_output_scalar_component_set_errors(
    mock_model: WflowSbmModel, forcing_layer: xr.DataArray
):
    # Setup the component
    component = WflowOutputScalarComponent(mock_model)

    # Missing name
    forcing_layer.name = None
    with pytest.raises(ValueError, match="Name required for DataArray."):
        component.set(forcing_layer)

    # Try with np.ndarray
    array = forcing_layer.values
    with pytest.raises(TypeError, match="cannot set data of type ndarray"):
        component.set(array)


def test_wflow_output_scalar_component_read(
    mock_model_factory: Callable[[Path, str], WflowSbmModel],
    model_subbasin_cached: Path,
):
    # Set it to read mode
    mock_model = mock_model_factory(path=model_subbasin_cached, mode="r")

    # Setup the component
    component = WflowOutputScalarComponent(mock_model)

    # Assert its data currently none
    assert component._data is None

    # Read the data
    type(component.model.config).get_value = MagicMock(
        side_effect=get_value_side_effect
    )
    component.read()

    # Assert the read data
    assert isinstance(component.data, xr.Dataset)
    assert len(component.data) == 2
    assert "temp_coord" in component.data


def test_wflow_output_scalar_component_read_init(
    mock_model_factory: Callable[[Path, str], WflowSbmModel],
    model_subbasin_cached: Path,
):
    # Set it to read mode
    mock_model = mock_model_factory(path=model_subbasin_cached, mode="r")

    # Setup the component
    component = WflowOutputScalarComponent(mock_model)
    type(component.model.config).get_value = MagicMock(
        side_effect=get_value_side_effect
    )
    assert component._data is None  # Assert no data or structure yet

    # Read at init
    assert len(component.data) == 2
    assert "temp_coord" in component.data


def test_wflow_output_scalar_component_write(
    mock_model: WflowSbmModel,
    grid_dummy_data: xr.DataArray,
    caplog: pytest.LogCaptureFixture,
):
    # Setup the component
    component = WflowOutputScalarComponent(mock_model)
    component._data = grid_dummy_data.to_dataset(name="test_layer")

    # Write to a file
    # Check the logger message when calling component.write()
    with caplog.at_level(logging.INFO):
        component.write()
    assert "netcdf_scalar is an output of Wflow and will not be written." in caplog.text


def test_wflow_output_scalar_component_equal(
    mock_model: WflowSbmModel,
    grid_dummy_data: xr.DataArray,
):
    # Setup the components
    component = WflowOutputScalarComponent(mock_model)
    component2 = WflowOutputScalarComponent(mock_model)

    # Update them manually
    component._data = grid_dummy_data.to_dataset()
    component2._data = grid_dummy_data.to_dataset()

    # Assert these are equal
    eq, errors = component.test_equal(component2)
    assert eq is True

    # Update component2 to make them not equal
    component2._data["test_layer"] = grid_dummy_data + 1  # Modify data

    # Assert unequal
    eq, errors = component.test_equal(component2)
    assert eq is False
    assert "Data variables only on the right object" in errors["data"]
    assert "test_layer" in errors["data"]
