"""add global fixtures."""

import platform
from os.path import abspath, dirname, join

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hydromt import DataCatalog
from hydromt.cli._utils import parse_config
from shapely.geometry import Point, box

from hydromt_wflow import WflowModel, WflowSedimentModel

SUBDIR = ""
if platform.system().lower() != "windows":
    SUBDIR = "linux64"

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples", SUBDIR)
TESTCATALOGDIR = join(dirname(abspath(__file__)), "..", "examples", "data")


@pytest.fixture
def example_wflow_model():
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")
    mod = WflowModel(
        root=root,
        mode="r",
        data_libs=[
            "artifact_data",
            join(TESTCATALOGDIR, "demand", "data_catalog.yml"),
        ],
    )
    return mod


@pytest.fixture
def example_sediment_model():
    root = join(EXAMPLEDIR, "wflow_sediment_piave_subbasin")
    mod = WflowSedimentModel(
        root=root,
        mode="r",
        data_libs=["artifact_data"],
    )
    return mod


@pytest.fixture
def example_models(example_wflow_model, example_sediment_model):
    models = {
        "wflow": example_wflow_model,
        "wflow_sediment": example_sediment_model,
        "wflow_simple": None,
    }
    return models


@pytest.fixture
def wflow_ini():
    config = join(TESTDATADIR, "wflow_piave_build_subbasin.yml")
    opt = parse_config(config)
    return opt


@pytest.fixture
def sediment_ini():
    config = join(TESTDATADIR, "wflow_sediment_piave_build_subbasin.yml")
    opt = parse_config(config)
    return opt


@pytest.fixture
def wflow_simple_ini():
    config = join(dirname(abspath(__file__)), "..", "examples", "wflow_build.yml")
    opt = parse_config(config)
    return opt


@pytest.fixture
def example_inis(wflow_ini, sediment_ini, wflow_simple_ini):
    inis = {
        "wflow": wflow_ini,
        "wflow_sediment": sediment_ini,
        "wflow_simple": wflow_simple_ini,
    }
    return inis


@pytest.fixture
def example_wflow_results():
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")
    config_fn = join(EXAMPLEDIR, "wflow_piave_subbasin", "wflow_sbm_results.toml")
    mod = WflowModel(root=root, mode="r", config_fn=config_fn)
    return mod


@pytest.fixture
def clipped_wflow_model():
    root = join(EXAMPLEDIR, "wflow_piave_clip")
    mod = WflowModel(
        root=root,
        mode="r",
        data_libs=[
            "artifact_data",
            "https://github.com/Deltares/hydromt_wflow/releases/download/v0.5.0/wflow_artifacts.yml",
        ],
    )
    return mod


@pytest.fixture
def floodplain1d_testdata():
    data = xr.load_dataset(
        join(TESTDATADIR, SUBDIR, "floodplain_layers.nc"),
        lock=False,
        mode="r",
    )
    # Rename testdata variables to match the model
    for var in data.data_vars:
        if "hydrodem" in var:
            new_name = var.replace("hydrodem", "river_bank_elevation")
            data = data.rename({var: new_name})
    return data


@pytest.fixture
def globcover_gdf():
    cat = DataCatalog("artifact_data")
    globcover = cat.get_rasterdataset("globcover_2009")
    globcover_gdf = globcover.raster.vectorize()
    globcover_gdf.rename(columns={"value": "landuse"}, inplace=True)
    return globcover_gdf


@pytest.fixture
def planted_forest_testdata():
    bbox1 = [12.38, 46.12, 12.42, 46.16]
    bbox2 = [12.21, 46.07, 12.26, 46.11]
    gdf = gpd.GeoDataFrame(geometry=[box(*bbox1), box(*bbox2)], crs="EPSG:4326")
    gdf["forest_type"] = ["Pine", "Orchard"]
    return gdf


@pytest.fixture
def rivers1d():
    # Also for linux the data is in the normal example folder
    data = gpd.read_file(
        join(dirname(abspath(__file__)), "..", "examples", "data", "rivers.geojson"),
    )
    return data


@pytest.fixture
def df_precip_stations():
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
def gdf_precip_stations():
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
def da_pet(example_wflow_model):
    da = example_wflow_model.data_catalog.get_rasterdataset(
        "era5", geom=example_wflow_model.region, buffer=2, variables=["temp"]
    )
    da = 0.5 * (0.45 * da + 8)  # simple pet from Bradley Criddle
    da.name = "pet"
    da = da.astype("float64")

    return da


@pytest.fixture
def demda():
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
