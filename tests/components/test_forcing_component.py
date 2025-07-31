from unittest.mock import MagicMock

import pytest
import xarray as xr

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
