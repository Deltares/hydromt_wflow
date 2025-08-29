import platform
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import xarray as xr
from hydromt import DataCatalog
from hydromt.model import ModelRoot
from pyproj.crs import CRS
from pytest_mock import MockerFixture

from hydromt_wflow import WflowSbmModel
from hydromt_wflow.components import WflowConfigComponent, WflowStaticmapsComponent

SUBDIR = ""
if platform.system().lower() != "windows":
    SUBDIR = "linux64"
TEST_COMPONENT_ROOT_FOLDER = Path(__file__).parent


## OS related fixture
@pytest.fixture(scope="session")
def mount_string() -> str:
    if platform.system().lower() == "windows":
        return "d:/"
    return "/d/"  # Posix paths


## Data directories
@pytest.fixture(scope="session")
def cached_models() -> Path:
    p = Path(TEST_COMPONENT_ROOT_FOLDER, "..", "..", "examples", SUBDIR)
    assert p.is_dir()
    return p.resolve()


@pytest.fixture(scope="session")
def model_subbasin_cached(cached_models: Path) -> Path:
    p = Path(cached_models, "wflow_piave_subbasin")
    assert p.is_dir()
    return p


## Model related fixtures
@pytest.fixture
def mock_model_factory(
    mocker: MockerFixture, tmp_path: Path
) -> Callable[[Path, str], WflowSbmModel]:
    def _factory(path: Path = tmp_path, mode: str = "w") -> WflowSbmModel:
        model = mocker.create_autospec(WflowSbmModel)
        model.root = ModelRoot(path, mode=mode)
        model.data_catalog = mocker.create_autospec(DataCatalog)
        model.crs = CRS.from_epsg(4326)
        model._MAPS = {}
        return model

    return _factory


@pytest.fixture
def mock_model_staticmaps(
    mock_model_factory: Callable[[Path, str], WflowSbmModel],
    grid_dummy_data: xr.DataArray,
) -> WflowSbmModel:
    # Add a GridComponent to mock model
    mock_model = mock_model_factory()
    staticmaps = WflowStaticmapsComponent(mock_model)
    staticmaps._data = grid_dummy_data.to_dataset(name="basin", promote_attrs=True)

    type(mock_model).components = PropertyMock(
        side_effect=lambda: {"staticmaps": staticmaps}
    )
    type(mock_model).staticmaps = PropertyMock(side_effect=lambda: staticmaps)
    # Mock the get_component method of mock_model to return the staticmaps component
    mock_model.get_component = MagicMock(name="staticmaps", return_value=staticmaps)

    return mock_model


@pytest.fixture
def mock_model_staticmaps_factory(
    grid_dummy_data: xr.DataArray,
    mock_model_factory: Callable[[Path, str], WflowSbmModel],
    tmp_path: Path,
) -> WflowSbmModel:
    def factory(path: Path = tmp_path, mode: str = "w"):
        mock_model = mock_model_factory(path, mode)
        staticmaps = WflowStaticmapsComponent(mock_model)
        staticmaps._data = grid_dummy_data.to_dataset(name="basin", promote_attrs=True)
        mock_model.components = {"staticmaps": staticmaps}
        type(mock_model).components = PropertyMock(
            side_effect=lambda: {"staticmaps": staticmaps}
        )
        type(mock_model).staticmaps = PropertyMock(side_effect=lambda: staticmaps)
        mock_model.get_component = MagicMock(name="staticmaps", return_value=staticmaps)
        return mock_model

    return factory


@pytest.fixture
def mock_model_staticmaps_config_factory(
    mock_model_factory: Callable[[Path, str], WflowSbmModel],
    tmp_path: Path,
) -> WflowSbmModel:
    def factory(
        path: Path = tmp_path, mode: str = "w", config_filename: str = "wflow_sbm.toml"
    ):
        mock_model = mock_model_factory(path, mode)
        staticmaps = WflowStaticmapsComponent(mock_model)
        staticmaps._data = None
        config = WflowConfigComponent(mock_model, filename=config_filename)
        config._data = None

        mock_model.components = {"staticmaps": staticmaps, "config": config}
        type(mock_model).components = PropertyMock(
            side_effect=lambda: {"staticmaps": staticmaps, "config": config}
        )
        type(mock_model).staticmaps = PropertyMock(side_effect=lambda: staticmaps)
        type(mock_model).config = PropertyMock(side_effect=lambda: config)
        mock_model.get_component = MagicMock(name="staticmaps", return_value=staticmaps)
        return mock_model

    return factory


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
def config_dummy_document(tmp_path: Path) -> dict:
    data = {}
    data.update(
        {
            "foo": "bar",
            "baz": {
                "file1": Path(tmp_path, "tmp.txt").as_posix(),
                "file2": "tmp/tmp.txt",
            },
            "spooky": {"ghost": [1, 2, 3]},
        }
    )
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
    da.raster.set_nodata(-9999)
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
    da.raster.set_nodata(-9999)
    return da


@pytest.fixture
def forcing_layer() -> xr.DataArray:
    time = np.arange(
        np.datetime64("2000-01-01"), np.datetime64("2000-01-21"), np.timedelta64(1, "D")
    ).astype("datetime64[ns]")
    da = xr.DataArray(
        np.ones((20, 2, 2)),
        coords={
            "time": time,
            "lat": range(2),
            "lon": range(2),
        },
        dims=["time", "lat", "lon"],
        attrs={
            "crs": "EPSG:4326",
            "grid_mapping_name": "latitude_longitude",
        },
    )
    da.raster.set_nodata(-9999)
    da.name = "foo"
    return da


@pytest.fixture
def forcing_layer_path(tmp_path: Path, forcing_layer: xr.DataArray):
    p = Path(tmp_path, "tmp_forcing.nc")
    forcing_layer.to_netcdf(p)
    assert p.is_file()
    return p


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


@pytest.fixture
def mask_layer() -> xr.DataArray:
    data = np.ones((2, 2))
    data[1, 1] = -9999
    da = xr.DataArray(
        data,
        coords={
            "lat": range(2),
            "lon": range(2),
        },
        dims=["lat", "lon"],
    )
    da.raster.set_nodata(-9999)
    return da


@pytest.fixture
def static_file(tmp_path: Path, static_layer: xr.DataArray) -> Path:
    p = Path(tmp_path, "tmp.nc")
    ds = static_layer.to_dataset(name="layer1")
    ds["layer2"] = static_layer.where(
        static_layer == static_layer.raster.nodata,
        static_layer * 2,
    )
    ds = ds.raster.gdal_compliant()
    ds.to_netcdf(p)
    return p
