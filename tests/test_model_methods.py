"""Unit tests for hydromt_wflow methods and workflows."""

import logging
from itertools import product
from os.path import abspath, dirname, join

import numpy as np

# import warnings
# import pdb
import pandas as pd
import pytest
import xarray as xr
from hydromt.raster import full_like

from hydromt_wflow.wflow import WflowModel

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")


def test_setup_basemaps(tmpdir):
    # Region
    region = {
        "basin": [12.2051, 45.8331],
        "strord": 4,
        "bounds": [11.70, 45.35, 12.95, 46.70],
    }
    mod = WflowModel(
        root=str(tmpdir.join("wflow_base")),
        mode="w",
        data_libs=["artifact_data"],
    )

    hydrography = mod.data_catalog.get_rasterdataset("merit_hydro")
    # Change dtype to uint32
    hydrography["basins"] = hydrography["basins"].astype("uint32")

    # Run setup_basemaps
    mod.setup_basemaps(
        region=region,
        hydrography_fn=hydrography,
        res=hydrography.raster.res[0],  # no upscaling
    )

    assert mod.grid["wflow_subcatch"].dtype == "int32"


def test_setup_grid(example_wflow_model):
    # Tests on setup_grid_from_raster
    example_wflow_model.setup_grid_from_raster(
        raster_fn="merit_hydro",
        reproject_method="average",
        variables=["elevtn"],
        wflow_variables=["input.vertical.altitude"],
        fill_method="nearest",
    )
    assert "elevtn" in example_wflow_model.grid
    assert example_wflow_model.get_config("input.vertical.altitude") == "elevtn"

    example_wflow_model.setup_grid_from_raster(
        raster_fn="globcover",
        reproject_method="mode",
        wflow_variables=["input.vertical.landuse"],
    )
    assert "globcover" in example_wflow_model.grid
    assert example_wflow_model.get_config("input.vertical.landuse") == "globcover"

    # Test on exceptions
    with pytest.raises(ValueError, match="Length of variables"):
        example_wflow_model.setup_grid_from_raster(
            raster_fn="merit_hydro",
            reproject_method="average",
            variables=["elevtn", "lndslp"],
            wflow_variables=["input.vertical.altitude"],
        )
    with pytest.raises(ValueError, match="variables list is not provided"):
        example_wflow_model.setup_grid_from_raster(
            raster_fn="merit_hydro",
            reproject_method="average",
            wflow_variables=["input.vertical.altitude"],
        )


def test_projected_crs(tmpdir):
    logger = logging.getLogger(__name__)

    # Instantiate wflow model
    root = str(tmpdir.join("wflow_projected"))
    mod = WflowModel(
        root=root,
        mode="w",
        data_libs=["artifact_data", join(TESTDATADIR, "merit_utm", "merit_utm.yml")],
        logger=logger,
    )

    # Setup basemaps
    with pytest.raises(
        ValueError,
        match=r"The model resolution 1000 should be smaller than 1 degree",
    ) as error:
        mod.setup_basemaps(
            region={"basin": [12.862, 45.701]},
            res=1000,
            hydrography_fn="merit_hydro",
        )
    assert str(error.value).startswith(
        "The model resolution 1000 should be smaller than 1 degree"
    )

    with pytest.raises(
        ValueError,
        match=r"The model resolution 0.01 should be larger than",
    ) as error:
        mod.setup_basemaps(
            region={"basin": [1427596.0, 5735404.0]},
            res=0.01,
            hydrography_fn="merit_hydro_1k_utm",
            basin_index_fn=None,
        )
    assert str(error.value).startswith(
        "The model resolution 0.01 should be larger than"
    )

    mod.setup_basemaps(
        region={"basin": [1427596.0, 5735404.0]},
        res=2000,
        hydrography_fn="merit_hydro_1k_utm",
        basin_index_fn=None,
    )

    # Add more data eg landuse
    mod.setup_lulcmaps("globcover")

    assert mod.grid.raster.crs == 3857
    assert np.quantile(mod.grid["wflow_landuse"], 0.95) == 190.0  # urban
    assert mod.get_config("model.sizeinmetres") == True


def test_setup_lake(tmpdir, example_wflow_model):
    # Create dummy lake rating curves
    lakes = example_wflow_model.geoms["lakes"]
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
    example_wflow_model.data_catalog.from_dict(
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
    example_wflow_model.setup_lakes(
        lakes_fn="hydro_lakes",
        rating_curve_fns=[f"lake_rating_test_{lake_id}"],
        min_area=5,
        add_maxstorage=True,
    )

    assert f"lake_sh_{lake_id}" in example_wflow_model.tables
    assert f"lake_hq_{lake_id}" in example_wflow_model.tables
    assert 2 in np.unique(example_wflow_model.grid["LakeStorFunc"].values)
    assert 1 in np.unique(example_wflow_model.grid["LakeOutflowFunc"].values)
    assert (
        "LakeMaxStorage" not in example_wflow_model.grid
    )  # no Vol_max column in hydro_lakes

    # Write and read back
    example_wflow_model.set_root(join(tmpdir, "wflow_lake_test"))
    example_wflow_model.write_tables()
    test_table = example_wflow_model.tables[f"lake_sh_{lake_id}"]
    example_wflow_model._tables = dict()
    example_wflow_model.read_tables()

    assert example_wflow_model.tables[f"lake_sh_{lake_id}"].equals(test_table)


@pytest.mark.timeout(300)  # max 5 min
@pytest.mark.parametrize("source", ["gww", "jrc"])
def test_setup_reservoirs(source, tmpdir, example_wflow_model):
    # Read model 'wflow_piave_subbasin' from EXAMPLEDIR
    model = "wflow"
    example_wflow_model.read()

    # Update model (reservoirs only)
    destination = str(tmpdir.join(model))
    example_wflow_model.set_root(destination, mode="w")

    config = {
        "setup_reservoirs": {
            "reservoirs_fn": "hydro_reservoirs",
            "timeseries_fn": source,
            "min_area": 0.0,
        }
    }

    example_wflow_model.update(model_out=destination, opt=config)
    example_wflow_model.write()

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
        x == True for x in [k in example_wflow_model.grid.keys() for k in required]
    ), "1 or more reservoir map missing"

    # Check if all parameter maps contain x non-null values, where x equals
    # the number of reservoirs in the model area
    grid = example_wflow_model.grid.where(
        example_wflow_model.grid.wflow_reservoirlocs != -999
    )
    stacked = grid.wflow_reservoirlocs.stack(x=[grid.raster.y_dim, grid.raster.x_dim])
    stacked = stacked[stacked.notnull()]
    number_of_reservoirs = stacked.size

    for i in required:
        assert (
            np.count_nonzero(
                ~np.isnan(
                    grid[i].sel(
                        {
                            grid.raster.y_dim: stacked[grid.raster.y_dim].values,
                            grid.raster.x_dim: stacked[grid.raster.x_dim].values,
                        }
                    )
                )
            )
            == number_of_reservoirs
        ), f"Number of non-null values in {i} not equal to \
number of reservoirs in model area"


def test_setup_ksathorfrac(tmpdir, example_wflow_model):
    # Read the modeldata
    model = "wflow"
    example_wflow_model.read()
    # Create dummy ksat data
    da = full_like(example_wflow_model.grid["KsatHorFrac"])
    data = np.zeros(da.shape)
    for x, y in product(*[range(item) for item in da.shape]):
        data[x, y] = 750 - ((x + y) ** 0.4 * 114.07373)
    da.values = data

    # Set the output directory
    destination = str(tmpdir.join(model))
    example_wflow_model.set_root(destination, mode="w")

    # Build the map
    example_wflow_model.setup_ksathorfrac(
        ksat_fn=da,
    )
    # Write and read the map
    example_wflow_model.write_grid()
    example_wflow_model.read_grid()

    # Check values
    values = example_wflow_model.grid.KsatHorFrac.raster.mask_nodata()
    max_val = values.max().values
    mean_val = values.mean().values
    assert int(max_val * 100) == 43175
    assert int(mean_val * 100) == 22020


def test_setup_rootzoneclim(example_wflow_model):
    # load csv with dummy data for long timeseries of precip, pet and dummy Q data.
    test_data = pd.read_csv(
        join(TESTDATADIR, "df_sub_dummy.csv"),
        parse_dates=True,
        dayfirst=True,
        index_col=0,
        header=0,
    )

    # open too short time series of eobs from artifact data - to get example extent.
    ds_obs = example_wflow_model.data_catalog.get_rasterdataset(
        "eobs",
        geom=example_wflow_model.region,
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
    ds.raster.set_crs(ds_obs.raster.crs)

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
            lon=(
                ["index"],
                example_wflow_model.geoms["gauges_grdc"].geometry.x.values,
            ),
            lat=(
                ["index"],
                example_wflow_model.geoms["gauges_grdc"].geometry.y.values,
            ),
        ),
    )
    # ds_run.raster.set_crs(ds_obs.raster.crs)
    # ds_run.raster.set_spatial_dims(x_dim="lon", y_dim="lat")
    # fill with dummy data
    for ind in indices:
        ds_run["discharge"].loc[dict(index=ind)] = test_data[f"Q_{ind}"]

    example_wflow_model.setup_rootzoneclim(
        run_fn=ds_run,
        forcing_obs_fn=ds,
        forcing_cc_hist_fn=ds_cc_hist,
        forcing_cc_fut_fn=ds_cc_fut,
        start_hydro_year="Oct",
        start_field_capacity="Apr",
        time_tuple=("2005-01-01", "2020-12-31"),
        time_tuple_fut=("2005-01-01", "2020-12-31"),
        missing_days_threshold=330,
        return_period=[2, 5, 10, 15, 20],
        update_toml_rootingdepth="RootingDepth_obs_2",
        rootzone_storage=True,
    )

    assert "RootingDepth_obs_20" in example_wflow_model.grid
    assert "RootingDepth_cc_hist_20" in example_wflow_model.grid
    assert "RootingDepth_cc_fut_20" in example_wflow_model.grid

    assert "rootzone_storage_obs_15" in example_wflow_model.grid
    assert "rootzone_storage_cc_hist_15" in example_wflow_model.grid
    assert "rootzone_storage_cc_fut_15" in example_wflow_model.grid

    assert (
        example_wflow_model.get_config("input.vertical.rootingdepth")
        == "RootingDepth_obs_2"
    )

    assert example_wflow_model.geoms["rootzone_storage"].loc[1][
        "rootzone_storage_obs_2"
    ] == pytest.approx(72.32875652734612, abs=0.5)
    assert example_wflow_model.geoms["rootzone_storage"].loc[1][
        "rootzone_storage_cc_hist_2"
    ] == pytest.approx(70.26993058321949, abs=0.5)
    assert example_wflow_model.geoms["rootzone_storage"].loc[1][
        "rootzone_storage_cc_fut_2"
    ] == pytest.approx(80.89081789601374, abs=0.5)

    # change settings for LAI and correct_cc_deficit
    example_wflow_model.setup_rootzoneclim(
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

    assert example_wflow_model.geoms["rootzone_storage"].loc[1][
        "rootzone_storage_obs_2"
    ] == pytest.approx(82.85684577620462, abs=0.5)
    assert example_wflow_model.geoms["rootzone_storage"].loc[1][
        "rootzone_storage_cc_hist_2"
    ] == pytest.approx(82.44039441508069, abs=0.5)
    assert example_wflow_model.geoms["rootzone_storage"].loc[1][
        "rootzone_storage_cc_fut_2"
    ] == pytest.approx(104.96931418911882, abs=0.5)


def test_setup_outlets(example_wflow_model):
    # Update wflow_subcatch ID
    new_subcatch = example_wflow_model.grid["wflow_subcatch"].copy()
    new_subcatch = new_subcatch.where(new_subcatch == new_subcatch.raster.nodata, 1001)
    example_wflow_model.set_grid(new_subcatch, "wflow_subcatch")

    # Derive outlets
    example_wflow_model.setup_outlets()

    # Check if the ID is indeed 1001
    val, count = np.unique(example_wflow_model.grid["wflow_gauges"], return_counts=True)
    # 0 is no data
    assert val[1] == 1001
    assert count[1] == 1


def test_setup_gauges(example_wflow_model):
    # uparea rename not in the latest artifact_data version
    example_wflow_model.data_catalog["grdc"].rename = {"area": "uparea"}
    example_wflow_model.setup_gauges(
        gauges_fn="grdc",
        basename="grdc_uparea",
        snap_to_river=False,
        mask=None,
        snap_uparea=True,
        wdw=5,
        rel_error=0.05,
    )
    gdf = example_wflow_model.geoms["gauges_grdc_uparea"]
    ds_samp = example_wflow_model.grid[["wflow_river", "wflow_uparea"]].raster.sample(
        gdf, wdw=0
    )
    assert np.all(ds_samp["wflow_river"].values == 1)
    assert np.allclose(ds_samp["wflow_uparea"].values, gdf["uparea"].values, rtol=0.05)

    # Test with/without snapping
    stations_fn = join(EXAMPLEDIR, "test_stations.csv")
    example_wflow_model.setup_gauges(
        gauges_fn=stations_fn,
        basename="stations_snapping",
        snap_to_river=True,
        mask=None,
    )
    gdf_snap = example_wflow_model.geoms["gauges_stations_snapping"]

    example_wflow_model.setup_gauges(
        gauges_fn=stations_fn,
        basename="stations_no_snapping",
        snap_to_river=False,
        mask=None,
    )
    gdf_no_snap = example_wflow_model.geoms["gauges_stations_no_snapping"]

    # Check that not all geometries of gdf_snap and gdf_no_snap are equal
    assert not gdf_snap.equals(gdf_no_snap)
    # Find wich row is identical
    equal = gdf_snap[gdf_snap["geometry"] == gdf_no_snap["geometry"]]
    assert len(equal) == 1
    assert equal.index.values[0] == 1003


@pytest.mark.parametrize("elevtn_map", ["wflow_dem", "dem_subgrid"])
def test_setup_rivers(elevtn_map, floodplain1d_testdata, example_wflow_model):
    example_wflow_model.setup_rivers(
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

    assert mapname in example_wflow_model.grid
    assert example_wflow_model.get_config("model.river_routing") == "local-inertial"
    assert (
        example_wflow_model.get_config("input.lateral.river.bankfull_elevation")
        == mapname
    )
    assert (
        example_wflow_model.grid[mapname]
        .raster.mask_nodata()
        .equals(floodplain1d_testdata[mapname])
    )


def test_setup_floodplains_1d(example_wflow_model, floodplain1d_testdata):
    flood_depths = [0.5, 1.0, 1.5, 2.0, 2.5]

    example_wflow_model.setup_rivers(
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

    example_wflow_model.setup_floodplains(
        hydrography_fn="merit_hydro",
        floodplain_type="1d",
        river_upa=30,
        flood_depths=flood_depths,
    )

    assert "floodplain_volume" in example_wflow_model.grid
    assert example_wflow_model.get_config("model.floodplain_1d") == True
    assert example_wflow_model.get_config("model.land_routing") == "kinematic-wave"
    assert (
        example_wflow_model.get_config("input.lateral.river.floodplain.volume")
        == "floodplain_volume"
    )
    assert np.all(example_wflow_model.grid.flood_depth.values == flood_depths)
    assert example_wflow_model.grid.floodplain_volume.raster.mask_nodata().equals(
        floodplain1d_testdata.floodplain_volume
    )


@pytest.mark.parametrize("elevtn_map", ["wflow_dem", "dem_subgrid"])
def test_setup_floodplains_2d(elevtn_map, example_wflow_model, floodplain1d_testdata):
    example_wflow_model.setup_rivers(
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

    example_wflow_model.setup_floodplains(
        hydrography_fn="merit_hydro", floodplain_type="2d", elevtn_map=elevtn_map
    )

    mapname = {"wflow_dem": "hydrodem_avg", "dem_subgrid": "hydrodem_subgrid"}[
        elevtn_map
    ]

    assert f"{mapname}_D4" in example_wflow_model.grid
    assert example_wflow_model.get_config("model.floodplain_1d") == False
    assert example_wflow_model.get_config("model.land_routing") == "local-inertial"
    assert (
        example_wflow_model.get_config("input.lateral.river.bankfull_elevation")
        == f"{mapname}_D4"
    )
    assert (
        example_wflow_model.get_config("input.lateral.land.elevation")
        == f"{mapname}_D4"
    )
    assert (
        example_wflow_model.grid[f"{mapname}_D4"]
        .raster.mask_nodata()
        .equals(floodplain1d_testdata[f"{mapname}_D4"])
    )


def test_setup_pet_forcing(example_wflow_model, da_pet):
    example_wflow_model.setup_pet_forcing(
        pet_fn=da_pet,
    )

    assert "pet" in example_wflow_model.forcing
    # used to be debruin before update
    assert "pet_method" not in example_wflow_model.forcing["pet"].attrs
    assert example_wflow_model.forcing["pet"].min().values == da_pet.min().values
    mean_val = example_wflow_model.forcing["pet"].mean().values
    assert int(mean_val * 1000) == 2984


def test_setup_1dmodel_connection(example_wflow_model, rivers1d):
    # test subbasin_area method with river boundaries
    example_wflow_model.setup_1dmodel_connection(
        river1d_fn=rivers1d,
        connection_method="subbasin_area",
        area_max=10.0,
        add_tributaries=True,
        include_river_boundaries=True,
        mapname="1dmodel",
        update_toml=True,
        toml_output="netcdf",
    )

    assert "gauges_1dmodel" in example_wflow_model.geoms
    assert "subcatch_1dmodel" in example_wflow_model.geoms
    assert "subcatch_riv_1dmodel" in example_wflow_model.geoms

    assert len(example_wflow_model.geoms["gauges_1dmodel"]) == 6
    assert len(example_wflow_model.geoms["subcatch_1dmodel"]) == 3
    conf_dict = {
        "name": "Q",
        "map": "gauges_1dmodel",
        "parameter": "lateral.river.q_av",
    }
    assert conf_dict in example_wflow_model.config["netcdf"]["variable"]

    # test subbasin_area method with river boundaries
    example_wflow_model.setup_1dmodel_connection(
        river1d_fn=rivers1d,
        connection_method="subbasin_area",
        area_max=10.0,
        add_tributaries=True,
        include_river_boundaries=False,
        mapname="1dmodel-nobounds",
        update_toml=True,
        toml_output="csv",
    )

    assert len(example_wflow_model.geoms["gauges_1dmodel-nobounds"]) == 3
    assert len(example_wflow_model.geoms["subcatch_1dmodel-nobounds"]) == 3
    conf_dict = {
        "header": "Q",
        "map": "gauges_1dmodel-nobounds",
        "parameter": "lateral.river.q_av",
    }
    assert conf_dict in example_wflow_model.config["csv"]["column"]
    assert np.all(
        example_wflow_model.geoms["subcatch_1dmodel"].geometry.geom_equals(
            example_wflow_model.geoms["subcatch_1dmodel-nobounds"].geometry
        )
    )

    # test nodes method without extra tributaries
    example_wflow_model.setup_1dmodel_connection(
        river1d_fn=rivers1d,
        connection_method="nodes",
        area_max=10.0,
        add_tributaries=False,
        include_river_boundaries=False,
        mapname="1dmodel-nodes",
        update_toml=False,
    )

    assert "gauges_1dmodel-nodes" not in example_wflow_model.geoms
    assert len(example_wflow_model.geoms["subcatch_1dmodel-nodes"]) == 7


def test_skip_nodata_reservoir(clipped_wflow_model):
    # Using the clipped_wflow_model as the reservoirs are not in this model
    clipped_wflow_model.setup_reservoirs(
        reservoirs_fn="hydro_reservoirs",
        min_area=0.0,
    )
    assert clipped_wflow_model.config["model"]["reservoirs"] == False
    # Get names for two reservoir layers
    for mapname in ["resareas", "reslocs"]:
        # Check if layers are indeed not present in the model
        assert (
            clipped_wflow_model._MAPS[mapname] not in clipped_wflow_model.grid.data_vars
        )
