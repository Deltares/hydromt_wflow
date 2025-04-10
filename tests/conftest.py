from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import geopandas as gpd
import pytest
from hydromt import DataCatalog
from hydromt.model.root import ModelRoot
from pyproj.crs import CRS
from pytest_mock import MockerFixture
from shapely.geometry import box

from hydromt_wflow import WflowModel
from hydromt_wflow.data import fetch_data


## Cached data and models
@pytest.fixture(scope="session")
def build_data() -> Path:
    build_dir = fetch_data("build-data")
    assert build_dir.is_dir()
    assert Path(build_dir, "region.geojson").is_file()
    return build_dir


@pytest.fixture(scope="session")
def build_region(build_data) -> Path:
    p = Path(build_data, "region.geojson")
    assert p.is_file()
    return p


@pytest.fixture(scope="session")
def build_region_gdf(build_region) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(build_region)
    assert len(gdf) != 0
    return gdf


@pytest.fixture(scope="session")
def build_region_3857(build_data) -> Path:
    p = Path(build_data, "region_3857.geojson")
    assert p.is_file()
    return p


@pytest.fixture(scope="session")
def build_region_3857_gdf(build_region_3857) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(build_region_3857)
    assert len(gdf) != 0
    return gdf


## Mocked objects
@pytest.fixture
def mock_model(tmp_path, mocker: MockerFixture) -> MagicMock:
    model = mocker.create_autospec(WflowModel)
    model.root = mocker.create_autospec(ModelRoot(tmp_path), instance=True)
    model.root.path.return_value = tmp_path
    model.data_catalog = mocker.create_autospec(DataCatalog)
    # Set attributes for practical use
    type(model).crs = PropertyMock(side_effect=lambda: CRS.from_epsg(4326))
    type(model).root = PropertyMock(side_effect=lambda: ModelRoot(tmp_path))
    return model


## More custom data structures
@pytest.fixture
def box_geometry() -> gpd.GeoDataFrame:
    geom = gpd.GeoDataFrame(
        geometry=[box(4.355, 52.035, 4.365, 52.045)],
        crs=4326,
    )
    return geom
