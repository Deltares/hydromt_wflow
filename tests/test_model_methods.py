"""Unit tests for hydromt_wflow methods and workflows"""

import pytest
from os.path import join, dirname, abspath
import numpy as np
import warnings
import pdb
import pandas as pd
import xarray as xr
from hydromt_wflow.wflow import WflowModel
import pandas as pd

import logging

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")


def test_setup_staticmaps():
    logger = logging.getLogger(__name__)
    # read model from examples folder
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")

    # Initialize model and read results
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)

    # Tests on setup_staticmaps_from_raster
    mod.setup_staticmaps_from_raster(
        raster_fn="merit_hydro",
        reproject_method="average",
        variables=["elevtn"],
        wflow_variables=["input.vertical.altitude"],
        fill_method="nearest",
    )
    assert "elevtn" in mod.staticmaps
    assert mod.get_config("input.vertical.altitude") == "elevtn"

    mod.setup_staticmaps_from_raster(
        raster_fn="globcover",
        reproject_method="mode",
        wflow_variables=["input.vertical.landuse"],
    )
    assert "globcover" in mod.staticmaps
    assert mod.get_config("input.vertical.landuse") == "globcover"

    # Test on exceptions
    with pytest.raises(ValueError, match="Length of variables"):
        mod.setup_staticmaps_from_raster(
            raster_fn="merit_hydro",
            reproject_method="average",
            variables=["elevtn", "lndslp"],
            wflow_variables=["input.vertical.altitude"],
        )
    with pytest.raises(ValueError, match="variables list is not provided"):
        mod.setup_staticmaps_from_raster(
            raster_fn="merit_hydro",
            reproject_method="average",
            wflow_variables=["input.vertical.altitude"],
        )


def test_setup_lake(tmpdir):
    logger = logging.getLogger(__name__)
    # read model from examples folder
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")

    # Initialize model and read results
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)

    # Create dummy lake rating curves
    lakes = mod.staticgeoms["lakes"]
    lake_id = lakes["waterbody_id"].iloc[0]
    area = lakes["LakeArea"].iloc[0]
    dis = lakes["LakeAvgOut"].iloc[0]
    lvl = lakes["LakeAvgLevel"].iloc[0]
    elev = lakes["Elevation"].iloc[0]
    lvls = np.linspace(0, lvl)

    df = pd.DataFrame(data={"elevtn": (lvls + elev), "volume": (lvls * area)})
    df = df.join(
        pd.DataFrame(
            {"elevtn": (lvls[-5:-1] + elev), "discharge": np.linspace(0, dis, num=4)}
        ).set_index("elevtn"),
        on="elevtn",
    )
    fn_lake = join(tmpdir, f"rating_curve_{lake_id}.csv")
    df.to_csv(fn_lake, sep=",", index=False, header=True)

    # Register as new data source
    mod.data_catalog.from_dict(
        {
            "lake_rating_test_{index}": {
                "data_type": "DataFrame",
                "driver": "csv",
                "path": join(tmpdir, "rating_curve_{index}.csv"),
                "placeholders": {
                    "index": [str(lake_id)],
                },
            }
        }
    )
    # Update model with it
    mod.setup_lakes(
        lakes_fn="hydro_lakes",
        rating_curve_fns=[f"lake_rating_test_{lake_id}"],
        min_area=5,
        add_maxstorage=True,
    )

    assert f"lake_sh_{lake_id}" in mod.tables
    assert f"lake_hq_{lake_id}" in mod.tables
    assert 2 in np.unique(mod.staticmaps["LakeStorFunc"].values)
    assert 1 in np.unique(mod.staticmaps["LakeOutflowFunc"].values)
    assert "LakeMaxStorage" not in mod.staticmaps  # no Vol_max column in hydro_lakes

    # Write and read back
    mod.set_root(join(tmpdir, "wflow_lake_test"))
    mod.write_tables()
    test_table = mod.tables[f"lake_sh_{lake_id}"]
    mod._tables = dict()
    mod.read_tables()

    assert mod.tables[f"lake_sh_{lake_id}"].equals(test_table)


@pytest.mark.timeout(300)  # max 5 min
@pytest.mark.parametrize("source", ["gww", "jrc"])
def test_setup_reservoirs(source, tmpdir):
    logger = logging.getLogger(__name__)

    # Read model 'wflow_piave_subbasin' from EXAMPLEDIR
    model = "wflow"
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")
    mod1 = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)
    mod1.read()

    # Update model (reservoirs only)
    destination = str(tmpdir.join(model))
    mod1.set_root(destination, mode="w")

    config = {
        "setup_reservoirs": {
            "reservoirs_fn": "hydro_reservoirs",
            "timeseries_fn": source,
            "min_area": 0.0,
        }
    }

    mod1.update(model_out=destination, opt=config)
    mod1.write()

    # Check if all parameter maps are available
    required = [
        "ResDemand",
        "ResMaxRelease",
        "ResMaxVolume",
        "ResSimpleArea",
        "ResTargetFullFrac",
        "ResTargetMinFrac",
    ]
    assert all(
        x == True for x in [k in mod1.staticmaps.keys() for k in required]
    ), "1 or more reservoir map missing"

    # Check if all parameter maps contain x non-null values, where x equals the number of reservoirs in the model area
    staticmaps = mod1.staticmaps.where(mod1.staticmaps.wflow_reservoirlocs != -999)
    stacked = staticmaps.wflow_reservoirlocs.stack(x=["lat", "lon"])
    stacked = stacked[stacked.notnull()]
    number_of_reservoirs = stacked.size

    for i in required:
        assert (
            np.count_nonzero(
                ~np.isnan(
                    staticmaps[i].sel(lat=stacked.lat.values, lon=stacked.lon.values)
                )
            )
            == number_of_reservoirs
        ), f"Number of non-null values in {i} not equal to number of reservoirs in model area"


def test_setup_rootzoneclim():
    logger = logging.getLogger(__name__)
    # read model from examples folder
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")

    # Initialize model and read results
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)

    # load csv with dummy data for long timeseries of precip, pet and dummy Q data.
    test_data = pd.read_csv(
        join(TESTDATADIR, "df_sub_dummy.csv"),
        parse_dates=True,
        dayfirst=True,
        index_col=0,
        header=0,
    )

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


def test_setup_gauges():
    logger = logging.getLogger(__name__)
    # read model from examples folder
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")

    # Initialize model and read results
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)

    # uparea rename not in the latest artifact_data version
    mod.data_catalog["grdc"].rename = {"area": "uparea"}
    mod.setup_gauges(
        gauges_fn="grdc",
        basename="grdc_uparea",
        snap_to_river=False,
        mask=None,
        snap_uparea=True,
        wdw=5,
        rel_error=0.05,
    )
    gdf = mod.staticgeoms["gauges_grdc_uparea"]
    ds_samp = mod.staticmaps[["wflow_river", "wflow_uparea"]].raster.sample(gdf, wdw=0)
    assert np.all(ds_samp["wflow_river"].values == 1)
    assert np.allclose(ds_samp["wflow_uparea"].values, gdf["uparea"].values, rtol=0.05)

    # Test with/without snapping
    stations_fn = join(EXAMPLEDIR, "test_stations.csv")
    mod.setup_gauges(
        gauges_fn=stations_fn,
        basename="stations_snapping",
        snap_to_river=True,
        mask=None,
    )
    gdf_snap = mod.staticgeoms["gauges_stations_snapping"]

    mod.setup_gauges(
        gauges_fn=stations_fn,
        basename="stations_no_snapping",
        snap_to_river=False,
        mask=None,
    )
    gdf_no_snap = mod.staticgeoms["gauges_stations_no_snapping"]

    # Check that not all geometries of gdf_snap and gdf_no_snap are equal
    assert not gdf_snap.equals(gdf_no_snap)
    # Find wich row is identical
    equal = gdf_snap[gdf_snap["geometry"] == gdf_no_snap["geometry"]]
    assert len(equal) == 1
    assert equal.index.values[0] == 1003


@pytest.mark.parametrize("elevtn_map", ["wflow_dem", "dem_subgrid"])
def test_setup_rivers(elevtn_map):
    # load netcdf file with floodplain layers
    test_data = xr.open_dataset(join(TESTDATADIR, "floodplain_layers.nc"))

    logger = logging.getLogger(__name__)
    # read model from examples folder
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")

    # Initialize model and read results
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)

    mod.setup_rivers(
        hydrography_fn="merit_hydro",
        river_geom_fn="rivers_lin2019_v1",
        river_upa=30,
        rivdph_method="powlaw",
        min_rivdph=1,
        min_rivwth=30,
        slope_len=2000,
        smooth_len=5000,
        river_routing="local-inertial",
        elevtn_map=elevtn_map,
    )

    mapname = {"wflow_dem": "hydrodem_avg", "dem_subgrid": "hydrodem_subgrid"}[
        elevtn_map
    ]

    assert mapname in mod.staticmaps
    assert mod.get_config("model.river_routing") == "local-inertial"
    assert mod.get_config("input.lateral.river.bankfull_elevation") == mapname
    assert mod.staticmaps[mapname].raster.mask_nodata().equals(test_data[mapname])


def test_setup_floodplains_1d():
    # load netcdf file with floodplain layers
    test_data = xr.open_dataset(join(TESTDATADIR, "floodplain_layers.nc"))

    logger = logging.getLogger(__name__)
    # read model from examples folder
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")

    # Initialize model and read results
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)

    flood_depths = [0.5, 1.0, 1.5, 2.0, 2.5]

    mod.setup_rivers(
        hydrography_fn="merit_hydro",
        river_geom_fn="rivers_lin2019_v1",
        river_upa=30,
        rivdph_method="powlaw",
        min_rivdph=1,
        min_rivwth=30,
        slope_len=2000,
        smooth_len=5000,
        river_routing="local-inertial",
        elevtn_map="wflow_dem",
    )

    mod.setup_floodplains(
        hydrography_fn="merit_hydro",
        floodplain_type="1d",
        river_upa=30,
        flood_depths=flood_depths,
    )

    assert "floodplain_volume" in mod.staticmaps
    assert mod.get_config("model.floodplain_1d") == True
    assert mod.get_config("model.land_routing") == "kinematic-wave"
    assert (
        mod.get_config("input.lateral.river.floodplain.volume") == "floodplain_volume"
    )
    assert np.all(mod.staticmaps.flood_depth.values == flood_depths)
    assert mod.staticmaps.floodplain_volume.raster.mask_nodata().equals(
        test_data.floodplain_volume
    )


@pytest.mark.parametrize("elevtn_map", ["wflow_dem", "dem_subgrid"])
def test_setup_floodplains_2d(elevtn_map):
    # load netcdf file with floodplain layers
    test_data = xr.open_dataset(join(TESTDATADIR, "floodplain_layers.nc"))

    logger = logging.getLogger(__name__)
    # read model from examples folder
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")

    # Initialize model and read results
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)

    mod.setup_rivers(
        hydrography_fn="merit_hydro",
        river_geom_fn="rivers_lin2019_v1",
        river_upa=30,
        rivdph_method="powlaw",
        min_rivdph=1,
        min_rivwth=30,
        slope_len=2000,
        smooth_len=5000,
        river_routing="local-inertial",
        elevtn_map="wflow_dem",
    )

    mod.setup_floodplains(
        hydrography_fn="merit_hydro", floodplain_type="2d", elevtn_map=elevtn_map
    )

    mapname = {"wflow_dem": "hydrodem_avg", "dem_subgrid": "hydrodem_subgrid"}[
        elevtn_map
    ]

    assert f"{mapname}_D4" in mod.staticmaps
    assert mod.get_config("model.floodplain_1d") == False
    assert mod.get_config("model.land_routing") == "local-inertial"
    assert mod.get_config("input.lateral.river.bankfull_elevation") == f"{mapname}_D4"
    assert mod.get_config("input.lateral.land.elevation") == f"{mapname}_D4"
    assert (
        mod.staticmaps[f"{mapname}_D4"]
        .raster.mask_nodata()
        .equals(test_data[f"{mapname}_D4"])
    )
