import platform
from pathlib import Path
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
def mock_model(tmp_path: Path, mocker: MockerFixture) -> MagicMock:
    model = mocker.create_autospec(WflowModel)
    model.root = mocker.create_autospec(ModelRoot(tmp_path), instance=True)
    model.root.path.return_value = tmp_path
    model.data_catalog = mocker.create_autospec(DataCatalog)
    # Set attributes for practical use
    type(model).crs = PropertyMock(side_effect=lambda: CRS.from_epsg(4326))
    type(model).root = PropertyMock(side_effect=lambda: ModelRoot(tmp_path))
    return model


@pytest.fixture
def mock_model_staticmaps(
    mock_model: MagicMock, grid_dummy_data: xr.DataArray
) -> MagicMock:
    # TODO: replace with WflowStaticmapsComponent when available
    # Add a GridComponent to mock model
    mock_model.add_component(name="staticmaps", component=GridComponent())
    # Give it some data to get a grid definition (crs, y, x)
    mock_model.staticmaps.set(grid_dummy_data, name="basin")

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
