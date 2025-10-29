from pathlib import Path
from unittest.mock import MagicMock

import pytest
import xarray as xr

from hydromt_wflow.components import WflowForcingComponent
from hydromt_wflow.wflow_sbm import WflowSbmModel


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
    mock_model: WflowSbmModel,
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
    model_subbasin_cached: Path,
):
    # Setup the component
    mock_model = mock_model_staticmaps_factory(path=model_subbasin_cached, mode="r")
    component = WflowForcingComponent(
        mock_model,
    )
    # Assert the current state
    assert len(component.data) == 0
    assert "foo" not in component.data.data_vars

    # Read some data
    type(component.model.config).get_value = MagicMock(return_value="")
    component.read()

    # Assert our favourate variable is now in the data
    assert "precip" in component.data.data_vars


def test_wflow_forcing_component_write(
    mock_model: MagicMock, forcing_layer: xr.DataArray, caplog
):
    # Setup the component
    component = WflowForcingComponent(
        mock_model,
    )
    component._data = forcing_layer.to_dataset(promote_attrs=True)
    filename_in = "inmaps.nc"
    out_path = mock_model.root.path / filename_in

    # Write once -> out_path
    type(component.model.config).get_value = MagicMock(return_value="")
    component.write(filename=filename_in)

    # Assert the output
    assert out_path.is_file()
    assert len(list(Path(mock_model.root.path).glob("*.nc"))) == 1

    # Write twice -> out_path exists, so warn and create new name
    component.write(filename=filename_in)
    new_filepath = mock_model.root.path / "inmaps_None_2000_2000.nc"

    # Assert the output
    EXISTS_WARNING = "Netcdf forcing file"

    exists_warnings = [EXISTS_WARNING in message for message in caplog.messages]
    assert exists_warnings.count(True) == 1
    assert len(list(Path(mock_model.root.path).glob("*.nc"))) == 2
    assert new_filepath.exists()

    # Write 3rd time -> warn, skip writing, return none
    component.write(filename=filename_in)

    SKIP_WARNING = "Netcdf generated forcing file name"
    exists_warnings = [
        EXISTS_WARNING in " ".join(message.split()) for message in caplog.messages
    ]
    skip_warnings = [
        SKIP_WARNING in " ".join(message.split()) for message in caplog.messages
    ]
    assert len(list(Path(mock_model.root.path).glob("*.nc"))) == 2
    assert exists_warnings.count(True) == 2
    assert skip_warnings.count(True) == 1


def test_wflow_forcing_component_write_with_overwrite(
    mock_model: MagicMock, forcing_layer: xr.DataArray, caplog
):
    # Setup the component
    component = WflowForcingComponent(
        mock_model,
    )
    component._data = forcing_layer.to_dataset(promote_attrs=True)
    filename_in = "inmaps.nc"
    out_path: Path = mock_model.root.path / filename_in
    out_path.touch()
    last_modification_time = out_path.stat().st_mtime

    # Write once -> out_path
    type(component.model.config).get_value = MagicMock(return_value="")
    component.write(filename=filename_in, overwrite=True)

    # Assert the output
    DELETE_WARNING = f"Deleting existing forcing file {out_path.as_posix()}"
    deletion_warnings = [
        DELETE_WARNING in " ".join(message.split()) for message in caplog.messages
    ]

    assert deletion_warnings.count(True) == 1
    assert out_path.is_file()
    assert len(list(Path(mock_model.root.path).glob("*.nc"))) == 1
    assert last_modification_time < out_path.stat().st_mtime


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
    type(component.model.config).get_value = MagicMock(return_value="")
    component.write(filename=out_path, output_frequency="5D")
    # I.e. 5 day output freq

    # Assert the output
    assert len(list(out_path.parent.glob("*.nc"))) == 4


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
    type(component.model.config).get_value = MagicMock(return_value="")
    component.write(
        filename=None,
    )

    # Assert the output
    assert (mock_model.root.path / component._filename).is_file()


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
    type(component.model.config).get_value = MagicMock(return_value="")
    component.write(
        filename=rel_path,
    )

    # Assert the output
    assert (mock_model.root.path / rel_path).resolve().is_file()


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
    type(component.model.config).get_value = MagicMock(return_value="")
    component.write(
        filename=abs_path,
    )

    # Assert the output
    assert (mock_model.root.path / abs_path).is_file()
