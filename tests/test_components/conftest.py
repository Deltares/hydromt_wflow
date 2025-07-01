import platform
from pathlib import Path
from typing import Callable

import pytest
from hydromt import DataCatalog
from hydromt.model import ModelRoot
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


## Extra data structures
@pytest.fixture(scope="session")
def config_dummy_data() -> dict:
    data = {
        "biem": "bam",
        "time": {"sometime": "now"},
        "foo": {"bar": "baz", "bip": "bop"},
    }
    return data
