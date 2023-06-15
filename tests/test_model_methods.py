"""Unit tests for hydromt_wflow methods and workflows"""

import pytest
from os.path import join, dirname, abspath
import numpy as np
import warnings
import pdb
import xarray as xr
from hydromt_wflow.wflow import WflowModel
import pandas as pd

import logging

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")


def test_setup_rootzoneclim():
    # load csv with dummy data for long timeseries of precip, pet and dummy Q data.
    test_data = pd.read_csv(
        join(TESTDATADIR, "df_sub_dummy.csv"),
        parse_dates=True,
        dayfirst=True,
        index_col=0,
        header=0,
    )

    logger = logging.getLogger(__name__)
    # read model from examples folder
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")

    # Initialize model and read results
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)

    # open too short time series of eobs from artifact data - to get example extent.
    ds_obs = mod.data_catalog.get_rasterdataset(
        "eobs",
        geom=mod.region,
        buffer=2,
        variables=["temp", "precip"],
    )

    # use long dummy timeseries of pet, precip and Q in a dataset
    zeros = np.zeros((len(test_data), len(ds_obs.latitude), len(ds_obs.longitude)))
    ds = xr.Dataset(
        data_vars=dict(
            precip=(["time", "latitude", "longitude"], zeros),
            pet=(["time", "latitude", "longitude"], zeros),
        ),
        coords=dict(
            time=test_data.index,
            latitude=ds_obs.latitude.values,
            longitude=ds_obs.longitude.values,
        ),
    )
    # fill data with dummy precip and pet data - uniform over area
    ds["pet"] = ds["pet"] + test_data["pet"].to_xarray()
    ds["precip"] = ds["precip"] + test_data["precip"].to_xarray()

    ds_cc_hist = ds.copy()
    ds_cc_hist["pet"] = ds["pet"] * 1.05

    ds_cc_fut = ds.copy()
    ds_cc_fut["pet"] = ds["pet"] * 1.3

    # make dummy dataset with dummy Q data
    indices = [1, 2, 3]
    ds_run = xr.Dataset(
        data_vars=dict(
            discharge=(
                ["time", "index"],
                np.zeros((len(test_data.index), len(indices))),
            ),
        ),
        coords=dict(
            time=test_data.index,
            index=indices,
            lon=(["index"], mod.staticgeoms["gauges_grdc"].geometry.x.values),
            lat=(["index"], mod.staticgeoms["gauges_grdc"].geometry.y.values),
        ),
    )
    # fill with dummy data
    for ind in indices:
        ds_run["discharge"].loc[dict(index=ind)] = test_data[f"Q_{ind}"]

    mod.setup_rootzoneclim(
        run_fn=ds_run,
        forcing_obs_fn=ds,
        forcing_cc_hist_fn=ds_cc_hist,
        forcing_cc_fut_fn=ds_cc_fut,
        start_hydro_year="Oct",
        start_field_capacity="Apr",
        time_tuple=("2005-01-01", "2015-12-31"),
        time_tuple_fut=("2005-01-01", "2017-12-31"),
        missing_days_threshold=330,
        return_period=[2, 5, 10, 15, 20],
        update_toml_rootingdepth="RootingDepth_obs_2",
        rootzone_storage=True,
    )

    assert "RootingDepth_obs_20" in mod.staticmaps
    assert "RootingDepth_cc_hist_20" in mod.staticmaps
    assert "RootingDepth_cc_fut_20" in mod.staticmaps

    assert "rootzone_storage_obs_15" in mod.staticmaps
    assert "rootzone_storage_cc_hist_15" in mod.staticmaps
    assert "rootzone_storage_cc_fut_15" in mod.staticmaps

    assert mod.get_config("input.vertical.rootingdepth") == "RootingDepth_obs_2"

    assert mod.staticgeoms["rootzone_storage"].loc[1][
        "rootzone_storage_obs_2"
    ] == pytest.approx(72.32875652734612, abs=0.5)
    assert mod.staticgeoms["rootzone_storage"].loc[1][
        "rootzone_storage_cc_hist_2"
    ] == pytest.approx(70.26993058321949, abs=0.5)
    assert mod.staticgeoms["rootzone_storage"].loc[1][
        "rootzone_storage_cc_fut_2"
    ] == pytest.approx(80.89081789601374, abs=0.5)

    # change settings for LAI and correct_cc_deficit
    mod.setup_rootzoneclim(
        run_fn=ds_run,
        forcing_obs_fn=ds,
        forcing_cc_hist_fn=ds_cc_hist,
        forcing_cc_fut_fn=ds_cc_fut,
        LAI=True,
        start_hydro_year="Oct",
        start_field_capacity="Apr",
        time_tuple=("2005-01-01", "2015-12-31"),
        time_tuple_fut=("2005-01-01", "2017-12-31"),
        correct_cc_deficit=True,
        missing_days_threshold=330,
        return_period=[2, 5, 10, 15, 20],
        update_toml_rootingdepth="RootingDepth_obs_2",
    )

    assert mod.staticgeoms["rootzone_storage"].loc[1][
        "rootzone_storage_obs_2"
    ] == pytest.approx(82.85684577620462, abs=0.5)
    assert mod.staticgeoms["rootzone_storage"].loc[1][
        "rootzone_storage_cc_hist_2"
    ] == pytest.approx(82.44039441508069, abs=0.5)
    assert mod.staticgeoms["rootzone_storage"].loc[1][
        "rootzone_storage_cc_fut_2"
    ] == pytest.approx(104.96931418911882, abs=0.5)
