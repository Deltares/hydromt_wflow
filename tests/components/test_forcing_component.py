from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
import xarray as xr

from hydromt_wflow.components import WflowForcingComponent
from hydromt_wflow.wflow import WflowModel


@pytest.fixture
def mock_model(mock_model_staticmaps_factory) -> MagicMock:
    """Create a mock model with a root."""
    return mock_model_staticmaps_factory()


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
    mock_model: MagicMock,
    forcing_layer: xr.DataArray,
):
    # Setup the component
    component = WflowForcingComponent(
        mock_model,
        region_component="staticmaps",
    )

    # Set the data in the component
    component.set(forcing_layer, "temp")

    # Assert the data
    assert "temp" in component.data.data_vars
    assert list(component.data.dims) == ["time", "lat", "lon"]
    assert component.data.time.size == 20


def test_wflow_forcing_component_set_errors(
    mock_model: WflowModel,
    forcing_layer: xr.DataArray,
):
    # Setup
    component = WflowForcingComponent(mock_model, region_component="staticmaps")

    # staticmaps doesnt have time dim
    # No time dim should error
    da_no_time = forcing_layer.isel(time=0, drop=True)
    with pytest.raises(
        ValueError,
        match="'time' dimension not found in data",
    ):
        component.set(da_no_time, "foo")

    # not the same grid as staticmaps should error
    da_grid_diff = forcing_layer.assign_coords({"lat": forcing_layer["lat"] + 10})
    with pytest.raises(
        ValueError,
        match="Data grid must be identical to staticmaps component",
    ):
        component.set(da_grid_diff, "foo")


def test_wflow_forcing_component_read(
    mock_model_staticmaps_factory: MagicMock,
    forcing_layer_path: xr.DataArray,
):
    # Setup the component
    mock_model = mock_model_staticmaps_factory(mode="r")
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


def test_wflow_forcing_component_write(
    mock_model: MagicMock, forcing_layer: xr.DataArray, caplog
):
    # Setup the component
    component = WflowForcingComponent(
        mock_model,
    )
    component._data = forcing_layer.to_dataset(promote_attrs=True)
    out_path = mock_model.root.path / "inmaps.nc"

    # Write once -> out_path
    filename, starttime, endtime = component.write(filename=out_path)

    # Assert the output
    assert out_path.is_file()
    assert len(list(Path(mock_model.root.path).glob("*.nc"))) == 1
    assert filename == out_path
    assert starttime.strftime("%Y-%m-%dT%H:%M:%S") == "2000-01-01T00:00:00"
    assert endtime.strftime("%Y-%m-%dT%H:%M:%S") == "2000-01-20T00:00:00"

    # Write twice -> out_path exists, so warn and create new name
    filename, starttime, endtime = component.write(filename=out_path)

    # Assert the output
    warnings = [
        f"Netcdf forcing file `{out_path}` already exists. Be careful, overwriting models partially can lead to inconsistencies."  # noqa: E501
        in " ".join(message.split())
        for message in caplog.messages
    ]
    assert warnings.count(True) == 1
    assert len(list(Path(mock_model.root.path).glob("*.nc"))) == 2
    assert filename != out_path
    assert filename.exists()
    assert starttime.strftime("%Y-%m-%dT%H:%M:%S") == "2000-01-01T00:00:00"
    assert endtime.strftime("%Y-%m-%dT%H:%M:%S") == "2000-01-20T00:00:00"

    # Write again will warn and skip writing
    filename, starttime, endtime = component.write(filename=out_path)
    assert len(list(Path(mock_model.root.path).glob("*.nc"))) == 3
    warnings = [
        f"Netcdf forcing file `{out_path}` already exists. Be careful, overwriting models partially can lead to inconsistencies."  # noqa: E501
        in " ".join(message.split())
        for message in caplog.messages
    ]
    assert warnings.count(True) == 2
    assert starttime.strftime("%Y-%m-%dT%H:%M:%S") == "2000-01-01T00:00:00"
    assert endtime.strftime("%Y-%m-%dT%H:%M:%S") == "2000-01-20T00:00:00"


def test_wflow_forcing_component_write_freq(
    mock_model: MagicMock,
    forcing_layer: xr.DataArray,
):
    # Setup the component
    component = WflowForcingComponent(
        mock_model,
    )
    component._data = forcing_layer.to_dataset(promote_attrs=True)
    out_path = mock_model.root.path / "inmaps.nc"

    # Write to the drive
    filename, _, _ = component.write(filename=out_path, output_frequency="5D")
    # I.e. 5 day output freq

    # Assert the output
    assert filename == out_path.parent / "inmaps_*.nc"
    assert len(list(out_path.parent.glob("*.nc"))) == 4


def test_wflow_forcing_component_write_time(
    mock_model: MagicMock,
    forcing_layer: xr.DataArray,
):
    # Setup the component
    component = WflowForcingComponent(
        mock_model,
    )
    out_path = mock_model.root.path / "inmaps.nc"
    component._data = forcing_layer.to_dataset(promote_attrs=True)
    expected_start = pd.Timestamp(min(forcing_layer.time).values).to_pydatetime()
    expected_end = pd.Timestamp(max(forcing_layer.time).values).to_pydatetime()

    # Write to the drive
    file_path, starttime, endtime = component.write(
        filename=out_path,
        starttime=expected_start - pd.Timedelta(weeks=2),
        endtime=expected_end + pd.Timedelta(weeks=2),
    )  # Both exceeding their respective end

    # Assert the output
    assert out_path.is_file()
    assert file_path == out_path
    assert starttime == expected_start
    assert endtime == expected_end


def test_wflow_forcing_component_write_no_filename(
    mock_model: MagicMock,
    forcing_layer: xr.DataArray,
):
    # Setup the component
    component = WflowForcingComponent(
        mock_model,
    )
    component._data = forcing_layer.to_dataset(promote_attrs=True)

    # Write to the drive
    file_path, _, _ = component.write(
        filename=None,
    )

    # Assert the output
    assert file_path.is_file()
    assert file_path == mock_model.root.path / component._filename


@pytest.mark.parametrize("rel_path", ["forcing/inmaps.nc", "../forcing/inmaps.nc"])
def test_wflow_forcing_component_write_relative_filename(
    rel_path: str,
    mock_model: MagicMock,
    forcing_layer: xr.DataArray,
):
    # Setup the component
    component = WflowForcingComponent(
        mock_model,
    )
    component._data = forcing_layer.to_dataset(promote_attrs=True)

    # Write to the drive
    file_path, _, _ = component.write(
        filename=rel_path,
    )

    # Assert the output
    assert file_path.is_file()
    assert file_path == (mock_model.root.path / rel_path).resolve()


def test_wflow_forcing_component_write_abs_file(
    mock_model: MagicMock,
    forcing_layer: xr.DataArray,
    tmp_path: Path,
):
    # Setup the component
    component = WflowForcingComponent(
        mock_model,
    )
    component._data = forcing_layer.to_dataset(promote_attrs=True)
    abs_path = tmp_path / "abs/forcing/inmaps.nc"

    # Write to the drive
    file_path, _, _ = component.write(
        filename=abs_path,
    )

    # Assert the output
    assert file_path.is_file()
    assert file_path == abs_path
