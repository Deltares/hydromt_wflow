"""add global fixtures."""

import logging
import platform
from os.path import abspath, dirname, join

import geopandas as gpd
import pytest
import xarray as xr
from hydromt.cli.cli_utils import parse_config

from hydromt_wflow import WflowModel, WflowSedimentModel

SUBDIR = ""
if platform.system().lower() != "windows":
    SUBDIR = "linux64"

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples", SUBDIR)


@pytest.fixture()
def example_wflow_model():
    logger = logging.getLogger(__name__)
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)
    return mod


@pytest.fixture()
def example_sediment_model():
    logger = logging.getLogger(__name__)
    root = join(EXAMPLEDIR, "wflow_sediment_piave_subbasin")
    mod = WflowSedimentModel(
        root=root, mode="r", data_libs="artifact_data", logger=logger
    )
    return mod


@pytest.fixture()
def example_models(example_wflow_model, example_sediment_model):
    models = {"wflow": example_wflow_model, "wflow_sediment": example_sediment_model}
    return models


@pytest.fixture()
def wflow_ini():
    config = join(TESTDATADIR, "wflow_piave_build_subbasin.yml")
    opt = parse_config(config)
    return opt


@pytest.fixture()
def sediment_ini():
    config = join(TESTDATADIR, "wflow_sediment_piave_build_subbasin.yml")
    opt = parse_config(config)
    return opt


@pytest.fixture()
def example_inis(wflow_ini, sediment_ini):
    inis = {"wflow": wflow_ini, "wflow_sediment": sediment_ini}
    return inis


@pytest.fixture()
def example_wflow_results():
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")
    config_fn = join(EXAMPLEDIR, "wflow_piave_subbasin", "wflow_sbm_results.toml")
    mod = WflowModel(root=root, mode="r", config_fn=config_fn)
    return mod


@pytest.fixture()
def clipped_wflow_model():
    root = join(EXAMPLEDIR, "wflow_piave_clip")
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data")
    return mod


@pytest.fixture()
def floodplain1d_testdata():
    data = xr.load_dataset(
        join(TESTDATADIR, SUBDIR, "floodplain_layers.nc"),
        lock=False,
        mode="r",
    )
    return data


@pytest.fixture()
def rivers1d():
    data = gpd.read_file(
        join(TESTDATADIR, "rivers.geojson"),
    )
    return data


@pytest.fixture()
def da_pet(example_wflow_model):
    da = example_wflow_model.data_catalog.get_rasterdataset(
        "era5", geom=example_wflow_model.region, buffer=2, variables=["temp"]
    )
    da = 0.5 * (0.45 * da + 8)  # simple pet from Bradley Criddle
    da.name = "pet"

    return da
