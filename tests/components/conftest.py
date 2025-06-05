import platform
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import xarray as xr
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
def mock_model(tmp_path: Path, mocker: MockerFixture) -> MagicMock:
    model = mocker.create_autospec(WflowModel)
    model.root = mocker.create_autospec(ModelRoot(tmp_path), instance=True)
    model.root.path.return_value = tmp_path
    model.data_catalog = mocker.create_autospec(DataCatalog)
    # Set attributes for practical use
    type(model).crs = PropertyMock(side_effect=lambda: CRS.from_epsg(4326))
    type(model).root = PropertyMock(side_effect=lambda: ModelRoot(tmp_path))
    return model


## Extra data structures
@pytest.fixture(scope="session")
def config_dummy_data() -> dict:
    data = {
        "biem": "bam",
        "time": {"sometime": "now"},
        "foo": {"bar": "baz", "bip": "bop"},
    }
    return data


@pytest.fixture
def cyclic_layer() -> xr.DataArray:
    da = xr.DataArray(
        np.ones((12, 2, 2)),
        coords={
            "time": range(1, 13),
            "lat": range(2),
            "lon": range(2),
        },
        dims=["time", "lat", "lon"],
    )
    return da


@pytest.fixture
def cyclic_layer_large() -> xr.DataArray:
    da = xr.DataArray(
        np.ones((365, 2, 2)),
        coords={
            "time": range(1, 366),
            "lat": range(2),
            "lon": range(2),
        },
        dims=["time", "lat", "lon"],
    )
    return da


@pytest.fixture
def static_layer() -> xr.DataArray:
    da = xr.DataArray(
        np.ones((2, 2)),
        coords={
            "lat": range(2),
            "lon": range(2),
        },
        dims=["lat", "lon"],
    )
    return da
