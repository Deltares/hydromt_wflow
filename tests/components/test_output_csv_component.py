import logging
from collections.abc import Callable
from pathlib import Path

import pytest
import xarray as xr

from hydromt_wflow.components import WflowOutputCsvComponent
from hydromt_wflow.wflow_sbm import WflowSbmModel


@pytest.fixture
def mock_model(mock_model_factory) -> WflowSbmModel:
    """Fixture to create a mock WflowSbmModel."""
    return mock_model_factory(mode="w")


# Define the side effect function
def get_value_side_effect(key, fallback=None):
    if key == "dir_output":
        return "run_results"
    elif key == "output.csv.path":
        return "output.csv"
    elif key == "dir_input":
        return ""
    return fallback


def test_wflow_output_csv_component_init(mock_model: WflowSbmModel):
    # Setup the component
    component = WflowOutputCsvComponent(mock_model)

    # Assert that the internal data is None
    assert component._data is None

    # When asking for data property, it should return a dict
    assert isinstance(component.data, dict)
    assert isinstance(component._data, dict)  # Same for internal
    assert len(component.data) == 0


def test_wflow_output_csv_component_init_with_locs(
    mock_model_staticmaps: WflowSbmModel,
):
    # Setup the component with a region component
    component = WflowOutputCsvComponent(
        mock_model_staticmaps, locations_component="staticmaps"
    )

    # Check if the locations component is set correctly
    assert component._locations_component == "staticmaps"
    assert "basin" in component._get_locations_data()


def test_wflow_output_csv_component_set(
    mock_model_staticmaps: WflowSbmModel, forcing_layer: xr.DataArray
):
    # Setup the component
    component = WflowOutputCsvComponent(
        mock_model_staticmaps, locations_component="staticmaps"
    )

    # Assert that the internal data is None, initializing will happen in set
    assert component._data is None

    # Set an entry
    # Copy else the name gets altered for the next test?
    component.set(forcing_layer.copy(), name="test_layer")

    # Assert the content
    assert isinstance(component._data, dict)
    assert "test_layer" in component.data
    assert len(component.data) == 1
    assert isinstance(component.data["test_layer"], xr.DataArray)


def test_wflow_output_csv_component_read(
    mock_model_staticmaps_config_factory: Callable[[Path, str], WflowSbmModel],
    model_subbasin_cached: Path,
):
    # Set it to read mode
    mock_model = mock_model_staticmaps_config_factory(
        path=model_subbasin_cached, mode="r", config_filename="wflow_sbm_results.toml"
    )
    # Here we really need the config and the staticmaps to read...
    assert "dir_output" in mock_model.config.data
    assert "subcatchment" in mock_model.staticmaps.data

    # Setup the component
    component = WflowOutputCsvComponent(mock_model, locations_component="staticmaps")

    # Assert its data currently none
    assert component._data is None

    # Read the data
    component.read()

    # Assert the read data
    assert isinstance(component.data, dict)
    assert len(component.data) == 6
    assert "river_max_q" in component.data


def test_wflow_output_csv_component_read_logging(
    mock_model_staticmaps_config_factory: Callable[[Path, str], WflowSbmModel],
    model_subbasin_cached: Path,
    caplog: pytest.LogCaptureFixture,
):
    # Set it to read mode
    mock_model = mock_model_staticmaps_config_factory(
        path=model_subbasin_cached, mode="r", config_filename="wflow_sbm_results.toml"
    )

    # Setup the component
    component = WflowOutputCsvComponent(mock_model, locations_component="staticmaps")

    # Missing map in staticmaps
    mock_model.staticmaps.drop_vars("gauges_grdc")

    with caplog.at_level(logging.DEBUG, logger="hydromt.hydromt_wflow.utils"):
        # Read the data
        component.read()

    assert "Reading csv column 'river_max_q'" in caplog.text
    assert "Map 'gauges_grdc' not found in staticmaps. Skip reading." in caplog.text
    assert len(component.data) == 0

    # Empty staticmaps
    mock_model.staticmaps._data = xr.Dataset()
    with caplog.at_level(logging.WARNING):
        component.read()

    assert "Staticmaps data is empty, skip reading." in caplog.text
    assert len(component.data) == 0


def test_wflow_output_csv_component_write(
    mock_model: WflowSbmModel,
    grid_dummy_data: xr.DataArray,
    caplog: pytest.LogCaptureFixture,
):
    # Setup the component
    component = WflowOutputCsvComponent(mock_model)
    component._data = {
        "test_layer": grid_dummy_data,
    }

    # Write to a file
    # Check the logger message when calling component.write()
    with caplog.at_level(logging.INFO):
        component.write()
    assert "output_csv is an output of Wflow and will not be written." in caplog.text
