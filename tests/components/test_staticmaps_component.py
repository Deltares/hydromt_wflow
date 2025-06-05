from unittest.mock import MagicMock

import xarray as xr

from hydromt_wflow.components import StaticmapsComponent


def test_staticmaps_component_init(mock_model: MagicMock):
    # Setup the component
    component = StaticmapsComponent(mock_model)

    # Assert that the internal data is None
    assert component._data is None

    # When asking for data property, it should return a tomlkit document
    assert isinstance(component.data, xr.Dataset)
    assert isinstance(component._data, xr.Dataset)  # Same for internal
    assert len(component.data) == 0  # Zero data vars


def test_staticmaps_component_set(mock_model: MagicMock):
    pass


def test_staticmaps_component_set_cyclic(
    mock_model: MagicMock,
    cyclic_layer: xr.DataArray,
):
    # Setup the component
    component = StaticmapsComponent(mock_model)

    # Set the data in the component
    component.set(cyclic_layer, "layer1")
    pass


def test_staticmaps_component_update_naming(mock_model: MagicMock):
    pass


def test_staticmaps_component_read():
    pass


def test_staticmaps_component_write():
    pass
