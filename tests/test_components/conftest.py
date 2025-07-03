import platform
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, PropertyMock

import pytest
import xarray as xr
from hydromt import DataCatalog
from hydromt.model import ModelRoot
from hydromt.model.components import GridComponent
from pyproj.crs import CRS
from pytest_mock import MockerFixture

from hydromt_wflow import WflowModel

SUBDIR = ""
if platform.system().lower() != "windows":
    SUBDIR = "linux64"
TEST_COMPONENT_ROOT_FOLDER = Path(__file__).parent


## Data directories
@pytest.fixture(scope="session")
def cached_models() -> Path:
    p = Path(TEST_COMPONENT_ROOT_FOLDER, "..", "..", "examples", SUBDIR)
    assert p.is_dir()
    return p


@pytest.fixture(scope="session")
def model_subbasin_cached(cached_models: Path) -> Path:
    p = Path(cached_models, "wflow_piave_subbasin")
    assert p.is_dir()
    return p


## Model related fixtures
@pytest.fixture
def mock_model_factory(
    mocker: MockerFixture, tmp_path: Path
) -> Callable[[Path, str], WflowModel]:
    def _factory(path: Path = tmp_path, mode: str = "w") -> WflowModel:
        model = mocker.create_autospec(WflowModel)
        model.root = ModelRoot(path, mode=mode)
        model.data_catalog = mocker.create_autospec(DataCatalog)
        model.crs = CRS.from_epsg(4326)
        return model

    return _factory


@pytest.fixture
def mock_model_staticmaps(
    mock_model_factory: Callable[[Path, str], WflowModel],
    grid_dummy_data: xr.DataArray,
) -> WflowModel:
    # TODO: replace with WflowStaticmapsComponent when available
    # Add a GridComponent to mock model
    mock_model = mock_model_factory()
    staticmaps = GridComponent(mock_model)
    staticmaps._data = grid_dummy_data.to_dataset(name="basin")

    type(mock_model).components = PropertyMock(
        side_effect=lambda: {"staticmaps": staticmaps}
    )
    type(mock_model).staticmaps = PropertyMock(side_effect=lambda: staticmaps)
    # Mock the get_component method of mock_model to return the staticmaps component
    mock_model.get_component = MagicMock(name="staticmaps", return_value=staticmaps)

    return mock_model


## Extra data structures
@pytest.fixture(scope="session")
def config_dummy_data() -> dict:
    data = {
        "biem": "bam",
        "time": {"sometime": "now"},
        "foo": {"bar": "baz", "bip": "bop"},
    }
    return data


@pytest.fixture(scope="session")
def grid_dummy_data() -> xr.DataArray:
    """Create a dummy grid data array."""
    data = xr.DataArray(
        data=[[1, 2], [3, 4]],
        dims=["y", "x"],
        name="dummy_grid",
        coords={
            "y": [0, 1],
            "x": [0, 1],
        },
        attrs={
            "crs": "EPSG:4326",
            "grid_mapping_name": "latitude_longitude",
        },
    )
    return data
