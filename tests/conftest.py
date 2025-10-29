"""add global fixtures."""

import platform
from os.path import abspath, dirname, join
from pathlib import Path
from typing import Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hydromt import DataCatalog
from hydromt.io import read_workflow_yaml
from pytest_mock import MockerFixture
from shapely.geometry import Point, box

from hydromt_wflow import WflowSbmModel, WflowSedimentModel
from hydromt_wflow.data.fetch import fetch_data

SUBDIR = ""
if platform.system().lower() != "windows":
    SUBDIR = "linux64"

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples", SUBDIR)
TESTCATALOGDIR = join(dirname(abspath(__file__)), "..", "examples", "data")

# This is the recommended by pandas and will become default behaviour in pandas 3.0.
# https://pandas.pydata.org/pandas-docs/stable/user_guide/copy_on_write.html#copy-on-write-chained-assignment
# https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
pd.options.mode.copy_on_write = True


## Cached data and models
@pytest.fixture(scope="session")
def build_data() -> Path:
    build_dir = fetch_data("artifact-data")
    assert build_dir.is_dir()
    assert Path(build_dir, "era5.nc").is_file()
    return build_dir


@pytest.fixture(scope="session")
def build_data_catalog(build_data) -> Path:
    p = Path(build_data, "data_catalog.yml")
    assert p.is_file()
    return p


@pytest.fixture
def example_wflow_model():
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")
    mod = WflowSbmModel(
        root=root,
        mode="r",
        data_libs=[
            "artifact_data",
            join(TESTCATALOGDIR, "demand", "data_catalog.yml"),
        ],
    )
    return mod


@pytest.fixture
def example_wflow_model_factory() -> Callable[[str, str, list[str]], WflowSbmModel]:
    def factory(
        root: str = join(EXAMPLEDIR, "wflow_piave_subbasin"),
        mode: str = "r",
        data_libs: list[str] = [
            "artifact_data",
            join(TESTCATALOGDIR, "demand", "data_catalog.yml"),
        ],
    ) -> WflowSbmModel:
        return WflowSbmModel(root=root, mode=mode, data_libs=data_libs)

    return factory


@pytest.fixture
def example_sediment_model() -> WflowSedimentModel:
    root = join(EXAMPLEDIR, "wflow_sediment_piave_subbasin")
    mod = WflowSedimentModel(
        root=root,
        mode="r",
        data_libs=["artifact_data"],
    )
    return mod


@pytest.fixture
def example_models(
    example_wflow_model: WflowSbmModel, example_sediment_model: WflowSedimentModel
):
    models = {
        "wflow": example_wflow_model,
        "wflow_sediment": example_sediment_model,
        "wflow_simple": None,
    }
    return models


@pytest.fixture
def wflow_ini():
    config = join(TESTDATADIR, "wflow_piave_build_subbasin.yml")
    _, _, steps = read_workflow_yaml(config)
    return steps


@pytest.fixture
def sediment_ini():
    config = join(TESTDATADIR, "wflow_sediment_piave_build_subbasin.yml")
    _, _, steps = read_workflow_yaml(config)
    return steps


@pytest.fixture
def wflow_simple_ini():
    config = join(dirname(abspath(__file__)), "..", "examples", "wflow_build.yml")
    _, _, steps = read_workflow_yaml(config)
    return steps


@pytest.fixture
def example_inis(wflow_ini, sediment_ini, wflow_simple_ini):
    inis = {
        "wflow": wflow_ini,
        "wflow_sediment": sediment_ini,
        "wflow_simple": wflow_simple_ini,
    }
    return inis


@pytest.fixture
def example_wflow_outputs() -> WflowSbmModel:
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")
    config_fn = join(EXAMPLEDIR, "wflow_piave_subbasin", "wflow_sbm_results.toml")
    mod = WflowSbmModel(root=root, mode="r", config_filename=config_fn)
    return mod


@pytest.fixture
def clipped_wflow_model(build_data_catalog) -> WflowSbmModel:
    root = join(EXAMPLEDIR, "wflow_piave_clip")
    mod = WflowSbmModel(
        root=root,
        mode="r",
        data_libs=[build_data_catalog],
    )
    return mod


@pytest.fixture
def floodplain1d_testdata() -> xr.Dataset:
    data = xr.load_dataset(
        join(TESTDATADIR, SUBDIR, "floodplain_layers.nc"),
        lock=False,
    )
    # Rename testdata variables to match the model
    for var in data.data_vars:
        if "hydrodem" in var:
            new_name = var.replace("hydrodem", "river_bank_elevation")
            data = data.rename({var: new_name})
    return data


@pytest.fixture
def globcover_gdf() -> gpd.GeoDataFrame:
    cat = DataCatalog("artifact_data")
    globcover = cat.get_rasterdataset("globcover_2009")
    globcover_gdf = globcover.raster.vectorize()
    globcover_gdf.rename(columns={"value": "landuse"}, inplace=True)
    return globcover_gdf


@pytest.fixture
def planted_forest_testdata() -> gpd.GeoDataFrame:
    bbox1 = [12.38, 46.12, 12.42, 46.16]
    bbox2 = [12.21, 46.07, 12.26, 46.11]
    gdf = gpd.GeoDataFrame(geometry=[box(*bbox1), box(*bbox2)], crs="EPSG:4326")
    gdf["forest_type"] = ["Pine", "Orchard"]
    return gdf


@pytest.fixture
def rivers1d() -> gpd.GeoDataFrame:
    # Also for linux the data is in the normal example folder
    data = gpd.read_file(
        join(dirname(abspath(__file__)), "..", "examples", "data", "rivers.geojson"),
    )
    return data


@pytest.fixture
def rivers1d_projected() -> gpd.GeoDataFrame:
    data = gpd.read_file(
        join(dirname(abspath(__file__)), "data", "1d-river-3857.geojson"),
    )
    return data


@pytest.fixture
def df_precip_stations() -> pd.DataFrame:
    np.random.seed(42)
    time = pd.date_range(
        start="2010-02-01T00:00:00", end="2010-09-01T00:00:00", freq="D"
    )
    data = np.random.rand(len(time), 8)
    df = pd.DataFrame(data=data, columns=[1, 2, 3, 4, 5, 6, 7, 8], index=time)
    df.index.name = "time"
    df.columns.name = "index"
    return df


@pytest.fixture
def gdf_precip_stations() -> gpd.GeoDataFrame:
    geometry = [
        # inside Piave basin
        Point(12.6, 46.6),
        Point(11.9, 46.3),
        Point(12.1, 45.9),
        # outside Piave basin
        Point(11.4, 46.9),
        Point(13.1, 46.7),
        Point(11.5, 45.7),
        Point(12.6, 46.0),
        Point(12.5, 45.6),
    ]
    gdf = gpd.GeoDataFrame(
        data=None, index=[1, 2, 3, 4, 5, 6, 7, 8], geometry=geometry, crs="EPSG:4326"
    )
    gdf.index.name = "index"
    return gdf


@pytest.fixture
def da_pet(example_wflow_model: WflowSbmModel) -> xr.DataArray:
    da = example_wflow_model.data_catalog.get_rasterdataset(
        "era5",
        geom=example_wflow_model.region,
        buffer=2,
        variables=["temp"],
    )
    da = 0.5 * (0.45 * da + 8)  # simple pet from Bradley Criddle
    da.name = "pet"
    da = da.astype("float64")

    return da


@pytest.fixture
def demda() -> xr.DataArray:
    np.random.seed(11)
    da = xr.DataArray(
        data=np.random.rand(15, 10),
        dims=("y", "x"),
        coords={"y": -np.arange(0, 1500, 100), "x": np.arange(0, 1000, 100)},
        attrs=dict(_FillValue=-9999),
    )
    # NOTE epsg 3785 is deprecated https://epsg.io/3785
    da.raster.set_crs(3857)
    return da


@pytest.fixture
def reservoir_rating() -> dict[str, pd.DataFrame]:
    """Reservoir rating curves for testing purposes."""
    rating = {
        "reservoir_sh_169986": pd.DataFrame(
            {
                "H": [100, 101, 102, 103, 104, 105],
                "S": [0, 1000, 1500, 2000, 3000, 5000],
            }
        ),
        "reservoir_hq_169986": pd.DataFrame(
            {
                "H": [102, 103, 104, 105],
                "1": [0, 1, 2.5, 4],
            }
        ),
    }
    # repeat column 1 of "reservoir_hq_169986" 365 time (1 column per day of year)
    rating["reservoir_hq_169986"] = pd.concat(
        [rating["reservoir_hq_169986"]] + [rating["reservoir_hq_169986"]["1"]] * 364,
        axis=1,
    )
    rating["reservoir_hq_169986"].columns = ["H"] + [str(i) for i in range(1, 366)]

    # Copy for reservoir id 3367
    rating["reservoir_sh_3367"] = rating["reservoir_sh_169986"].copy()
    rating["reservoir_hq_3367"] = rating["reservoir_hq_169986"].copy()

    return rating


@pytest.fixture
def mock_rasterdataset(mocker: MockerFixture) -> xr.Dataset:
    """Mock rasterdataset for testing purposes."""
    ds = mocker.create_autospec(xr.Dataset, instance=True)

    mock_raster = mocker.Mock()
    mock_raster.crs.is_geographic = True
    mock_raster.res = (1 / 120.0, 1 / 120.0)
    mock_raster.clip_geom.return_value = ds
    mock_raster.geometry_mask.return_value = xr.DataArray(
        np.array([[1, 0], [0, 1]]), dims=("y", "x")
    )
    ds.raster = mock_raster
    ds.coords = {}
    ds.__getitem__.side_effect = lambda key: ds.coords.get(key)

    return ds


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
    da.raster.set_crs(4326)
    da.raster.set_nodata(-9999)
    return da
