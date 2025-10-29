from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from hydromt_wflow.components import WflowStatesComponent
from hydromt_wflow.wflow_sbm import WflowSbmModel


@pytest.fixture
def mock_model(mock_model_factory) -> WflowSbmModel:
    """Fixture to create a mock WflowSbmModel."""
    return mock_model_factory(mode="w")


def test_wflow_states_component_init(mock_model: WflowSbmModel):
    # Setup the component
    component = WflowStatesComponent(mock_model)

    # Assert that the internal data is None
    assert component._data is None

    # When asking for data property, it should return a xarray.Dataset
    assert isinstance(component.data, xr.Dataset)
    assert isinstance(component._data, xr.Dataset)  # Same for internal
    assert len(component.data) == 0


def test_wflow_states_component_init_with_region(mock_model_staticmaps: WflowSbmModel):
    # Setup the component with a region component
    component = WflowStatesComponent(
        mock_model_staticmaps, region_component="staticmaps"
    )

    # Check if the region component is set correctly
    assert component._region_component == "staticmaps"
    assert "basin" in component._get_grid_data()


def test_wflow_states_component_set(
    mock_model_staticmaps: WflowSbmModel, grid_dummy_data: xr.DataArray
):
    # Setup the component
    component = WflowStatesComponent(
        mock_model_staticmaps, region_component="staticmaps"
    )

    # Assert that the internal data is None, initializing will happen in set
    assert component._data is None

    # Set an entry
    # Copy else the name gets altered for the next test?
    component.set(grid_dummy_data.copy(), name="test_layer")

    # Assert the content
    assert isinstance(component._data, xr.Dataset)
    assert "test_layer" in component.data
    assert len(component.data) == 1


def test_wflow_states_component_set_alt(
    mock_model_staticmaps: WflowSbmModel, grid_dummy_data: xr.DataArray
):
    # Setup the component
    component = WflowStatesComponent(
        mock_model_staticmaps, region_component="staticmaps"
    )

    # Try with dataset
    component.set(grid_dummy_data.to_dataset())

    # Assert the content
    assert isinstance(component._data, xr.Dataset)
    assert "dummy_grid" in component.data


def test_wflow_states_component_set_errors(
    mock_model_staticmaps: WflowSbmModel, grid_dummy_data: xr.DataArray
):
    # Setup the component
    component = WflowStatesComponent(
        mock_model_staticmaps, region_component="staticmaps"
    )

    # Try with np.ndarray
    array = grid_dummy_data.values
    with pytest.raises(TypeError, match="Data must be an xarray Dataset or DataArray"):
        component.set(array)

    # Try with a different grid that is not identical
    # This should raise an error
    different_grid = grid_dummy_data.copy()
    different_grid["x"] = different_grid.x + 1  # Shift x coordinates
    with pytest.raises(
        ValueError, match="Data grid must be identical to staticmaps component"
    ):
        component.set(different_grid)


def test_wflow_states_component_read(
    mock_model_factory: Callable[[Path, str], WflowSbmModel],
    model_subbasin_cached: Path,
):
    # Set it to read mode
    mock_model = mock_model_factory(path=model_subbasin_cached, mode="r")

    # Setup the component
    component = WflowStatesComponent(mock_model)

    # Assert its data currently none
    assert component._data is None

    # Read the data
    type(component.model.config).get_value = MagicMock(return_value="")
    component.read()

    # Assert the read data
    assert isinstance(component.data, xr.Dataset)
    assert len(component.data) == 13
    assert "river_instantaneous_q" in component.data


def test_wflow_states_component_read_init(
    mock_model_factory: Callable[[Path, str], WflowSbmModel],
    model_subbasin_cached: Path,
):
    # Set it to read mode
    mock_model = mock_model_factory(path=model_subbasin_cached, mode="r")

    # Setup the component
    component = WflowStatesComponent(mock_model)
    type(component.model.config).get_value = MagicMock(return_value="")
    assert component._data is None  # Assert no data or structure yet

    # Read at init
    assert len(component.data) == 13
    assert "river_instantaneous_q" in component.data


def test_wflow_states_component_write(
    mock_model: WflowSbmModel,
    grid_dummy_data: xr.DataArray,
):
    # Setup the component
    component = WflowStatesComponent(mock_model)
    mock_model.components = {"states": component}

    component._data = grid_dummy_data.to_dataset(name="test_layer")

    # Write to a file
    type(component.model.config).get_value = MagicMock(return_value="")  # dir_input
    component.write()

    # Check if the file was created and has the expected content
    file = Path(component.root.path, component._filename)
    assert file.is_file()

    written_data = xr.open_dataset(file)
    assert "test_layer" in written_data
    # rename dims is True so x, y get renamed when writting
    assert "latitude" in written_data
    assert np.all(written_data["test_layer"].values == grid_dummy_data.values)


def test_wflow_states_component_equal(
    mock_model: WflowSbmModel,
    grid_dummy_data: xr.DataArray,
):
    # Setup the components
    component = WflowStatesComponent(mock_model)
    component2 = WflowStatesComponent(mock_model)

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
    assert "Other grid has additional maps" in errors
    assert "test_layer" in errors["Other grid has additional maps"]
