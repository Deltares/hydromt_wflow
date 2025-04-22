import logging
from pathlib import Path
from unittest.mock import PropertyMock

import numpy as np
import pytest
from hydromt.model import ModelRoot
from pyproj.crs import CRS

from hydromt_wflow.components import ForcingComponent


def test_forcing_component_empty(mock_model):
    # Setup the component with the mock model
    component = ForcingComponent(model=mock_model)

    # assert that it is empty
    assert len(component.data) == 0
    assert component._filename == "inmaps/forcing.nc"


def test_forcing_component_set_default(mock_model, dummy_precipitation):
    # Setup the component with the mock model
    component = ForcingComponent(model=mock_model)

    # Set some forcing..
    component.set(data=dummy_precipitation, name="precip")

    # Assert the content
    assert len(component.data) != 0
    assert "precip" in component.data.data_vars
    assert len(component.data.time) == 7


def test_forcing_component_set_alt(mock_model, dummy_precipitation, caplog):
    caplog.set_level(logging.INFO)
    # Setup the component with the mock model
    component = ForcingComponent(model=mock_model)

    # Set a dataset for the dims
    component.set(dummy_precipitation)

    # Set a numpy ndarray
    component.set(data=dummy_precipitation, name="some_var")
    assert "some_var" in component.data.data_vars

    # Set the same variable again
    component.set(data=dummy_precipitation, name="some_var")
    assert "Replacing grid map: some_var" in caplog.text


def test_forcing_component_set_errors(mock_model, dummy_precipitation):
    # Setup the component with the mock model
    component = ForcingComponent(model=mock_model)

    # Set with no name
    no_name = dummy_precipitation.copy()
    no_name.name = None
    with pytest.raises(ValueError, match="Unable to set DataArray data without a name"):
        component.set(data=no_name)

    # Check for wrong type
    with pytest.raises(
        ValueError,
        match="Cannot set data of type int",
    ):
        component.set(data=2, name="some_var")


def test_forcing_component_read(tmp_path, mock_model, dummy_precipitation):
    # Setup the component and assert it's empty
    type(mock_model).root = PropertyMock(
        side_effect=lambda: ModelRoot(tmp_path, mode="r+"),
    )
    component = ForcingComponent(model=mock_model)
    component.read()
    assert len(component.data) == 0

    # Write the dummy dataset
    ds_path = Path(tmp_path, "inmaps", "forcing.nc")
    ds_path.parent.mkdir(parents=True)
    dummy_precipitation.to_netcdf(ds_path)

    # Read the component
    component.read()
    assert len(component.data) != 0


def test_forcing_component_write(tmp_path, mock_model, dummy_precipitation, caplog):
    caplog.set_level(logging.INFO)
    # Setup the component with the mock model
    component = ForcingComponent(model=mock_model)
    # Write empty
    component.write()
    assert "No grid data found, skip writing." in caplog.text

    # Set some forcing
    component.set(data=dummy_precipitation, name="precip")
    # Write the data
    component.write()
    # Assert it's existence
    assert Path(tmp_path, "inmaps", "forcing.nc").is_file()


def test_forcing_component_attributes(mock_model, dummy_precipitation):
    # Setup the component with the mock model
    component = ForcingComponent(model=mock_model)

    assert component.bounds is None
    assert component.crs is None
    assert component.res is None
    assert component.transform is None

    # Set some data
    component.set(dummy_precipitation)

    assert component.bounds is not None
    assert component.crs.equals(CRS.from_epsg(4326))
    assert np.isclose(component.res[0], 0.2)
    assert component.transform is not None
