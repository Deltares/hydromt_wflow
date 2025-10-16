"""Unit tests for hydromt_wflow methods and workflows."""

from itertools import product
from os.path import abspath, dirname, join
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import xarray.testing as xrt
from hydromt.data_catalog.sources import create_source
from hydromt.gis import GeoDataset, full_like

from hydromt_wflow import workflows
from hydromt_wflow.wflow_sbm import WflowSbmModel

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")


def test_setup_basemaps(tmpdir: Path):
    # Region
    region = {
        "basin": [12.2051, 45.8331],
        "strord": 4,
        "bounds": [11.70, 45.35, 12.95, 46.70],
    }
    mod = WflowSbmModel(
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
        hydrography_fn=hydrography.copy(),
        res=hydrography.raster.res[0],  # no upscaling
    )

    assert mod.staticmaps.data["subcatchment"].dtype == "int32"

    # Test for too small basins
    region = {"subbasin": [12.572061, 46.601984]}
    with pytest.raises(ValueError) as error:  # noqa PT011
        mod.setup_basemaps(
            region=region,
            hydrography_fn=hydrography,
        )
    assert str(error.value).startswith(
        "(Sub)basin at original resolution should at least consist of two cells"
    )

    region = {"subbasin": [12.572061, 46.600359]}
    with pytest.raises(ValueError) as error:  # noqa PT011
        mod.setup_basemaps(
            region=region,
            hydrography_fn=hydrography,
        )
    assert str(error.value).startswith(
        "The output extent at model resolution should at least consist of two cells on"
    )


def test_setup_grid(example_wflow_model):
    # Tests on setup_grid_from_raster
    example_wflow_model.setup_grid_from_raster(
        raster_fn="merit_hydro",
        reproject_method="average",
        variables=["elevtn"],
        wflow_variables=["land_surface_water_flow__ground_elevation"],
        fill_method="nearest",
    )
    assert "elevtn" in example_wflow_model.staticmaps.data
    assert (
        example_wflow_model.get_config(
            "input.static.land_surface_water_flow__ground_elevation"
        )
        == "elevtn"
    )

    example_wflow_model.setup_grid_from_raster(
        raster_fn="globcover_2009",
        reproject_method="mode",
    )
    assert "globcover" in example_wflow_model.staticmaps.data

    # Test on exceptions
    with pytest.raises(ValueError, match="Length of variables"):
        example_wflow_model.setup_grid_from_raster(
            raster_fn="merit_hydro",
            reproject_method="average",
            variables=["elevtn", "lndslp"],
            wflow_variables=["land_surface_water_flow__ground_elevation"],
        )
    with pytest.raises(ValueError, match="variables list is not provided"):
        example_wflow_model.setup_grid_from_raster(
            raster_fn="merit_hydro",
            reproject_method="average",
            wflow_variables=["land_surface_water_flow__ground_elevation"],
        )


def test_projected_crs(tmpdir: Path):
    # Instantiate wflow model
    root = str(tmpdir.join("wflow_projected"))
    mod = WflowSbmModel(
        root=root,
        mode="w",
        data_libs=["artifact_data", join(TESTDATADIR, "merit_utm", "merit_utm.yml")],
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
        match=r"Model resolution 0.01 should be larger than",
    ) as error:
        mod.setup_basemaps(
            region={"basin": [1427596.0, 5735404.0]},
            res=0.01,
            hydrography_fn="merit_hydro_1k_utm",
            basin_index_fn=None,
        )
    assert str(error.value).startswith("Model resolution 0.01 should be larger than")

    mod.setup_basemaps(
        region={"basin": [1427596.0, 5735404.0]},
        res=2000,
        hydrography_fn="merit_hydro_1k_utm",
        basin_index_fn=None,
    )

    # Add more data eg landuse
    mod.setup_lulcmaps("globcover_2009", lulc_mapping_fn="globcover_mapping_default")

    assert mod.staticmaps.data.raster.crs == 3857
    # 95 quantile is class 190 ie urban
    assert (mod.staticmaps.data["meta_landuse"] == 190).count().values == 338
    assert mod.get_config("model.cell_length_in_meter__flag") == True


@pytest.mark.parametrize("glacier_fn", ["glaciers_4326", "glaciers_3857"])
def test_projected_crs_glaciers(glacier_fn, tmpdir):
    # Instantiate wflow model
    root = str(tmpdir.join("wflow_projected"))
    mod = WflowSbmModel(
        root=root,
        mode="w",
        data_libs=[
            "artifact_data",
            join(TESTDATADIR, "merit_utm", "merit_utm.yml"),
            join(TESTDATADIR, "glacier", "glacier_utm.yml"),
        ],
    )

    mod.setup_basemaps(
        region={"basin": [1427596.0, 5735404.0]},
        res=2000,
        hydrography_fn="merit_hydro_1k_utm",
        basin_index_fn=None,
    )

    # Add glaciers
    mod.setup_glaciers(glacier_fn)

    # Confirm glacier maps exist
    assert "meta_glacier_area_id" in mod.staticmaps.data
    assert "glacier_fraction" in mod.staticmaps.data
    assert "glacier_initial_leq_depth" in mod.staticmaps.data
    assert "glaciers" in mod.geoms.data

    # Confirm glaciers have the same CRS as the grid (merit_utm is 3857)
    assert mod.staticmaps.data["glacier_fraction"].raster.crs == 3857
    assert mod.geoms.get("glaciers").crs == 3857

    # Confirm glacier fraction has values
    assert (mod.staticmaps.data["glacier_fraction"] > 0).any().item()

    # Confirm glacier IDs
    assert mod.staticmaps.data["meta_glacier_area_id"].max().item() == 1

    # Confirm config flags
    assert mod.get_config("model.glacier__flag") is True
    assert (
        mod.get_config("state.variables.glacier_ice__leq_depth") == "glacier_leq_depth"
    )


@pytest.fixture
def model_with_rating_curve_data(
    tmpdir: Path, example_wflow_model: WflowSbmModel
) -> tuple[WflowSbmModel, int]:
    # Create dummy lake rating curves
    lakes = example_wflow_model.geoms.get("meta_reservoirs_no_control")
    lake_id = lakes["waterbody_id"].iloc[0]
    area = lakes["reservoir_area"].iloc[0]
    dis = lakes["meta_reservoir_mean_outflow"].iloc[0]
    lvl = lakes["reservoir_initial_depth"].iloc[0]
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

    source = create_source(
        data={
            "name": f"lake_rating_test_{lake_id}",
            "data_type": "DataFrame",
            "driver": {
                "name": "pandas",
            },
            "uri": str(tmpdir / f"rating_curve_{lake_id}.csv"),
        }
    )

    # Register as new data source
    example_wflow_model.data_catalog.add_source(
        name=f"lake_rating_test_{lake_id}", source=source
    )
    return example_wflow_model, lake_id


def test_setup_reservoirs_no_control(
    tmpdir: Path, model_with_rating_curve_data: tuple[WflowSbmModel, int]
):
    example_wflow_model, lake_id = model_with_rating_curve_data

    # Update model with it
    example_wflow_model.setup_reservoirs_no_control(
        reservoirs_fn="hydro_lakes",
        rating_curve_fns=[f"lake_rating_test_{lake_id}"],
        min_area=5,
    )

    assert f"reservoir_sh_{lake_id}" in example_wflow_model.tables.data
    assert f"reservoir_hq_{lake_id}" in example_wflow_model.tables.data
    assert 2 in np.unique(
        example_wflow_model.staticmaps.data["reservoir_storage_curve"].values
    )
    assert 1 in np.unique(
        example_wflow_model.staticmaps.data["reservoir_rating_curve"].values
    )
    assert (
        "meta_reservoir_max_storage" not in example_wflow_model.staticmaps.data
    )  # no Vol_max column in hydro_lakes
    for reservoir_layer in [
        "reservoir_rating_curve",
        "reservoir_storage_curve",
        "reservoir_lower_id",
    ]:
        assert example_wflow_model.staticmaps.data[reservoir_layer].dtype == np.int32
        assert isinstance(
            example_wflow_model.staticmaps.data[reservoir_layer]._FillValue, np.int32
        )
    # Write and read back
    new_root = join(tmpdir, "wflow_lake_test")
    example_wflow_model.set_root(new_root, mode="w")
    example_wflow_model.write_tables()

    test_table = example_wflow_model.tables.data[f"reservoir_sh_{lake_id}"]

    example_wflow_model.tables.data.clear()
    example_wflow_model.set_root(new_root, mode="r")
    example_wflow_model.read_tables()

    assert example_wflow_model.tables.data[f"reservoir_sh_{lake_id}"].equals(test_table)


@pytest.mark.timeout(120)  # max 2 min
@pytest.mark.parametrize("source", ["gww", "jrc"])
def test_reservoirs_simple_control(source, tmpdir, example_wflow_model):
    # Read model 'wflow_piave_subbasin' from EXAMPLEDIR
    model = "wflow"
    example_wflow_model.read()

    # Update model (reservoirs only)
    destination = str(tmpdir.join(model))
    example_wflow_model.set_root(destination, mode="w")

    config = {
        "setup_reservoirs_simple_control": {
            "reservoirs_fn": "hydro_reservoirs",
            "timeseries_fn": source,
            "min_area": 0.0,
            "update_existing": False,
        }
    }

    example_wflow_model.update(model_out=destination, steps=[config], write=False)
    example_wflow_model.write()

    # Check if all parameter maps are available
    required = [
        "reservoir_demand",
        "reservoir_max_release",
        "reservoir_max_volume",
        "reservoir_area",
        "reservoir_target_full_fraction",
        "reservoir_target_min_fraction",
        "reservoir_rating_curve",
        "reservoir_storage_curve",
        "reservoir_initial_depth",
    ]
    assert all(
        x == True
        for x in [k in example_wflow_model.staticmaps.data.keys() for k in required]
    ), "1 or more reservoir map missing"

    # Check if all parameter maps contain x non-null values, where x equals
    # the number of reservoirs in the model area
    grid = example_wflow_model.staticmaps.data.where(
        example_wflow_model.staticmaps.data.reservoir_outlet_id != -999
    )
    stacked = grid.reservoir_outlet_id.stack(x=[grid.raster.y_dim, grid.raster.x_dim])
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
        ), (
            f"Number of non-null values in {i} not equal to \
number of reservoirs in model area"
        )


def test_setup_ksathorfrac(tmpdir, example_wflow_model):
    # Read the modeldata
    model = "wflow"
    example_wflow_model.read()
    # Create dummy ksat data
    da = full_like(example_wflow_model.staticmaps.data["soil_ksat_vertical"])
    data = np.zeros(da.shape)
    for x, y in product(*[range(item) for item in da.shape]):
        data[x, y] = 750 - ((x + y) ** 0.4 * 114.07373)
    da.values = data
    da.name = "subsurface_ksat_horizontal_ratio"
    # Set the output directory
    destination = str(tmpdir.join(model))
    example_wflow_model.set_root(destination, mode="w")

    # Build the map
    example_wflow_model.setup_ksathorfrac(
        ksat_fn=da,
    )

    # Check values
    values = example_wflow_model.staticmaps.data[
        "subsurface_ksat_horizontal_ratio"
    ].raster.mask_nodata()
    max_val = values.max().values
    mean_val = values.mean().values
    assert np.isclose(max_val, 431.75)
    assert np.isclose(mean_val, 220.206)


def test_setup_ksatver_vegetation(example_wflow_model):
    # Build the soil_ksat_vertical vegetation map
    example_wflow_model.setup_ksatver_vegetation(
        soil_fn="soilgrids",
        output_name="soil_ksat_vertical_vegetation_test",
    )

    # Check values
    values = example_wflow_model.staticmaps.data[
        "soil_ksat_vertical_vegetation_test"
    ].raster.mask_nodata()
    max_val = values.max().values
    mean_val = values.mean().values
    assert int(max_val) == 4247
    assert int(mean_val) == 1672


def test_setup_lai(example_wflow_model: WflowSbmModel):
    # Use vito and MODIS lai data for testing
    # Read vegetation_leaf_area_index data
    da_lai = example_wflow_model.data_catalog.get_rasterdataset(
        "modis_lai", geom=example_wflow_model.region, buffer=2
    )
    # Read landuse data
    da_landuse = example_wflow_model.data_catalog.get_rasterdataset(
        "vito_2015", geom=example_wflow_model.region, buffer=2
    )

    # Derive mapping using the method any
    df_lai_any = workflows.create_lulc_lai_mapping_table(
        da_lulc=da_landuse.copy(),
        da_lai=da_lai.copy(),
        sampling_method="any",
        lulc_zero_classes=[80, 200, 0],
    )
    # Check that all landuse classes are present in the mapping
    assert np.all(df_lai_any.index.values == np.unique(da_landuse))

    # Try with the other two methods
    df_lai_mode = workflows.create_lulc_lai_mapping_table(
        da_lulc=da_landuse.copy(),
        da_lai=da_lai.copy(),
        sampling_method="mode",
        lulc_zero_classes=[80, 200, 0],
    )
    df_lai_q3 = workflows.create_lulc_lai_mapping_table(
        da_lulc=da_landuse.copy(),
        da_lai=da_lai.copy(),
        sampling_method="q3",
        lulc_zero_classes=[80, 200, 0],
    )
    # Check the number of landuse classes in the mapping tables
    assert len(df_lai_any[df_lai_any.samples == 0]) == 1
    assert len(df_lai_mode[df_lai_mode.samples == 0]) == 3
    assert len(df_lai_q3[df_lai_q3.samples == 0]) == 3
    # Check number of samples for landuse class 20 with the different methods
    assert int(df_lai_any.loc[20].samples) == 2481
    assert int(df_lai_mode.loc[20].samples) == 59
    assert int(df_lai_q3.loc[20].samples) == 4

    # Try to use the mapping tables to setup the vegetation_leaf_area_index
    example_wflow_model.setup_laimaps_from_lulc_mapping(
        lulc_fn="vito_2015",
        lai_mapping_fn=df_lai_any,
        output_name="lai_from_vito_mapping",
    )

    assert "lai_from_vito_mapping" in example_wflow_model.staticmaps.data
    assert (
        example_wflow_model.get_config("input.cyclic.vegetation__leaf_area_index")
        == "lai_from_vito_mapping"
    )


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
                example_wflow_model.geoms.get("gauges_grdc").geometry.x.values,
            ),
            lat=(
                ["index"],
                example_wflow_model.geoms.get("gauges_grdc").geometry.y.values,
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
        time_range=("2005-01-01", "2020-12-31"),
        time_range_fut=("2005-01-01", "2020-12-31"),
        missing_days_threshold=330,
        return_period=[2, 5, 10, 15, 20],
        output_name_rootingdepth="vegetation_root_depth_obs_2",
        rootzone_storage=True,
    )

    assert "vegetation_root_depth_obs_20" in example_wflow_model.staticmaps.data
    assert "vegetation_root_depth_cc_hist_20" in example_wflow_model.staticmaps.data
    assert "vegetation_root_depth_cc_fut_20" in example_wflow_model.staticmaps.data

    assert "meta_rootzone_storage_obs_15" in example_wflow_model.staticmaps.data
    assert "meta_rootzone_storage_cc_hist_15" in example_wflow_model.staticmaps.data
    assert "meta_rootzone_storage_cc_fut_15" in example_wflow_model.staticmaps.data

    assert (
        example_wflow_model.get_config("input.static.vegetation_root__depth")
        == "vegetation_root_depth_obs_2"
    )

    assert example_wflow_model.geoms.get("rootzone_storage").loc[1][
        "rootzone_storage_obs_2"
    ] == pytest.approx(72.32875652734612, abs=0.5)
    assert example_wflow_model.geoms.get("rootzone_storage").loc[1][
        "rootzone_storage_cc_hist_2"
    ] == pytest.approx(70.26993058321949, abs=0.5)
    assert example_wflow_model.geoms.get("rootzone_storage").loc[1][
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
        time_range=("2005-01-01", "2015-12-31"),
        time_range_fut=("2005-01-01", "2015-12-31"),
        correct_cc_deficit=True,
        missing_days_threshold=330,
        return_period=[2, 5, 10, 15, 20],
        output_name_rootingdepth="vegetation_root_depth_obs_2",
    )

    assert example_wflow_model.geoms.get("rootzone_storage").loc[1][
        "rootzone_storage_obs_2"
    ] == pytest.approx(82.85684577620462, abs=0.5)
    assert example_wflow_model.geoms.get("rootzone_storage").loc[1][
        "rootzone_storage_cc_hist_2"
    ] == pytest.approx(82.44039441508069, abs=0.5)
    assert example_wflow_model.geoms.get("rootzone_storage").loc[1][
        "rootzone_storage_cc_fut_2"
    ] == pytest.approx(106.03809681174451, abs=0.5)


def test_setup_outlets(example_wflow_model):
    # Update subcatchment ID
    new_subcatch = example_wflow_model.staticmaps.data["subcatchment"].copy()
    new_subcatch = new_subcatch.where(new_subcatch == new_subcatch.raster.nodata, 1001)
    example_wflow_model.set_grid(new_subcatch, "subcatchment")

    # Derive outlets
    example_wflow_model.setup_outlets()

    # Check if the ID is indeed 1001
    val, count = np.unique(
        example_wflow_model.staticmaps.data["outlets"], return_counts=True
    )
    # 0 is no data
    assert val[1] == 1001
    assert count[1] == 1


def test_setup_gauges(example_wflow_model: WflowSbmModel):
    # 1. Test with grdc data
    # uparea rename not in the latest artifact_data version
    source = example_wflow_model.data_catalog.get_source("grdc")
    source.data_adapter.rename = {"area": "uparea"}

    example_wflow_model.setup_gauges(
        gauges_fn="grdc",
        basename="grdc_uparea",
        snap_to_river=False,
        mask=None,
        snap_uparea=True,
        wdw=5,
        rel_error=0.05,
    )
    gdf = example_wflow_model.geoms.get("gauges_grdc_uparea")
    ds_samp = example_wflow_model.staticmaps.data[
        ["river_mask", "meta_upstream_area"]
    ].raster.sample(gdf, wdw=0)
    assert np.allclose(
        ds_samp["meta_upstream_area"].values,
        gdf["uparea"].values,
        rtol=0.05,
    )

    # 2. Test with/without snapping to mask
    stations_fn = join(TESTDATADIR, "test_stations.csv")
    example_wflow_model.setup_gauges(
        gauges_fn=stations_fn,
        basename="stations_snapping",
        snap_to_river=True,
        mask=None,
    )
    gdf_snap = example_wflow_model.geoms.get("gauges_stations_snapping")

    example_wflow_model.setup_gauges(
        gauges_fn=stations_fn,
        basename="stations_no_snapping",
        snap_to_river=False,
        mask=None,
    )
    gdf_no_snap = example_wflow_model.geoms.get("gauges_stations_no_snapping")

    # Check that not all geometries of gdf_snap and gdf_no_snap are equal
    assert not gdf_snap.equals(gdf_no_snap)
    # Find which row is identical
    equal = gdf_snap[gdf_snap["geometry"] == gdf_no_snap["geometry"]]
    assert len(equal) == 1
    assert equal.index.values[0] == 1003

    # 3. Test uparea with/without river snapping
    example_wflow_model.setup_gauges(
        gauges_fn=stations_fn,
        basename="stations_uparea_no_snapping",
        snap_to_river=False,
        mask=None,
        snap_uparea=True,
        wdw=5,
        rel_error=0.05,
        fillna=False,
    )
    gdf_no_snap = example_wflow_model.geoms.get("gauges_stations_uparea_no_snapping")
    # Only two gauges have uparea values and fillna is False
    assert gdf_no_snap.index.size == 2

    example_wflow_model.setup_gauges(
        gauges_fn=stations_fn,
        basename="stations_uparea_no_snapping_fillna",
        snap_to_river=False,
        mask=None,
        snap_uparea=True,
        wdw=5,
        rel_error=0.05,
        fillna=True,
    )
    gdf_no_snap_fillna = example_wflow_model.geoms.get(
        "gauges_stations_uparea_no_snapping_fillna"
    )
    # Two gauges have uparea values and fillna is True
    assert gdf_no_snap_fillna.index.size == 3
    # Not all gauges are in the river as snap_to_river is False
    ds_samp = example_wflow_model.staticmaps.data[
        ["river_mask", "meta_upstream_area"]
    ].raster.sample(gdf_no_snap_fillna, wdw=0)
    assert not np.all(ds_samp["river_mask"].values == 1)

    example_wflow_model.setup_gauges(
        gauges_fn=stations_fn,
        basename="stations_uparea_snapping",
        snap_to_river=True,
        mask=None,
        snap_uparea=True,
        wdw=5,
        rel_error=0.05,
        abs_error=25,
        fillna=False,
    )
    gdf_snap = example_wflow_model.geoms.get("gauges_stations_uparea_snapping")
    # Only one gauge has uparea value and is in the river
    # (the one with NaN for upstream area would have ended in the river if fillna=True)
    assert gdf_snap.index.size == 1
    # Check that they are all in the river
    ds_samp = example_wflow_model.staticmaps.data[
        ["river_mask", "meta_upstream_area"]
    ].raster.sample(gdf_snap, wdw=0)
    assert np.all(ds_samp["river_mask"].values == 1)

    # 4. Test with test_stations.csv file
    stations_csv_file = join(EXAMPLEDIR, "test_stations.csv")
    example_wflow_model.setup_gauges(gauges_fn=stations_csv_file, basename="test-flow")
    test_stations_gdf = example_wflow_model.geoms.get("gauges_test-flow")
    assert len(test_stations_gdf) == 3


@pytest.mark.parametrize("elevtn_map", ["land_elevation", "meta_subgrid_elevation"])
def test_setup_rivers(elevtn_map, floodplain1d_testdata, example_wflow_model):
    example_wflow_model.setup_rivers(
        hydrography_fn="merit_hydro",
        river_geom_fn="hydro_rivers_lin",
        river_upa=30,
        rivdph_method="powlaw",
        min_rivdph=1,
        min_rivwth=30,
        slope_len=2000,
        smooth_len=5000,
        river_routing="local_inertial",
        elevtn_map=elevtn_map,
        output_names={},
    )

    mapname = {
        "land_elevation": "river_bank_elevation_avg",
        "meta_subgrid_elevation": "river_bank_elevation_subgrid",
    }[elevtn_map]

    assert mapname in example_wflow_model.staticmaps.data
    assert example_wflow_model.get_config("model.river_routing") == "local_inertial"
    assert (
        example_wflow_model.get_config("input.static.river_bank_water__elevation")
        == mapname
    )
    assert (
        example_wflow_model.staticmaps.data[mapname]
        .raster.mask_nodata()
        .equals(floodplain1d_testdata[mapname])
    )


def test_setup_rivers_depth(tmpdir: Path):
    # Instantiate new wflow model
    # Region
    region = {
        "subbasin": [12.3661, 46.3998],
        "strord": 4,
        "bounds": [11.70, 45.35, 12.95, 46.70],
    }
    mod = WflowSbmModel(
        root=str(tmpdir.join("river_mask")),
        mode="w",
        data_libs=["artifact_data"],
    )

    mod.setup_basemaps(
        region=region,
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
        res=0.016666,
    )

    # Try using manning method
    mod.setup_rivers(
        hydrography_fn="merit_hydro",
        river_geom_fn="hydro_rivers_lin",
        river_upa=30,
        rivdph_method="manning",
        min_rivdph=1,
        min_rivwth=30,
        slope_len=2000,
        smooth_len=5000,
        river_routing="local_inertial",
        elevtn_map="land_elevation",
    )

    assert "river_depth" in mod.staticmaps.data

    # Try using gvf method
    mod.setup_rivers(
        hydrography_fn="merit_hydro",
        river_geom_fn="hydro_rivers_lin",
        river_upa=30,
        rivdph_method="gvf",
        min_rivdph=1,
        min_rivwth=30,
        slope_len=2000,
        smooth_len=5000,
        river_routing="local_inertial",
        elevtn_map="meta_subgrid_elevation",
    )

    # RiverDepth iteslf doesn't matter here, this assertion
    # is just to check the method ran without errors
    # as this will error if something went wrong in the process
    assert "river_depth" in mod.staticmaps.data


def test_setup_floodplains_1d(
    example_wflow_model: WflowSbmModel, floodplain1d_testdata
):
    flood_depths = [0.5, 1.0, 1.5, 2.0, 2.5]

    example_wflow_model.setup_rivers(
        hydrography_fn="merit_hydro",
        river_geom_fn="hydro_rivers_lin",
        river_upa=30,
        rivdph_method="powlaw",
        min_rivdph=1,
        min_rivwth=30,
        slope_len=2000,
        smooth_len=5000,
        river_routing="local_inertial",
        elevtn_map="land_elevation",
    )

    example_wflow_model.setup_floodplains(
        hydrography_fn="merit_hydro",
        floodplain_type="1d",
        river_upa=30,
        flood_depths=flood_depths,
    )

    assert "floodplain_volume" in example_wflow_model.staticmaps.data
    assert example_wflow_model.get_config("model.floodplain_1d__flag") == True
    assert example_wflow_model.get_config("model.land_routing") == "kinematic_wave"
    assert (
        example_wflow_model.get_config(
            "input.static.floodplain_water__sum_of_volume_per_depth"
        )
        == "floodplain_volume"
    )
    name = example_wflow_model._MAPS["floodplain_volume"]
    assert np.all(
        example_wflow_model.staticmaps.data[name].flood_depth.values == flood_depths
    )
    xrt.assert_allclose(
        example_wflow_model.staticmaps.data[name].raster.mask_nodata(),
        floodplain1d_testdata.floodplain_volume,
    )

    # Check states in config
    assert (
        example_wflow_model.get_config(
            "state.variables.floodplain_water__instantaneous_volume_flow_rate"
        )
        == "floodplain_instantaneous_q"
    )

    assert (
        example_wflow_model.get_config("state.variables.floodplain_water__depth")
        == "floodplain_h"
    )

    assert (
        example_wflow_model.get_config(
            "state.variables.land_surface_water__instantaneous_volume_flow_rate"
        )
        == "land_instantaneous_q"
    )


@pytest.mark.parametrize("elevtn_map", ["land_elevation", "meta_subgrid_elevation"])
def test_setup_floodplains_2d(
    elevtn_map, example_wflow_model: WflowSbmModel, floodplain1d_testdata
):
    example_wflow_model.setup_rivers(
        hydrography_fn="merit_hydro",
        river_geom_fn="hydro_rivers_lin",
        river_upa=30,
        rivdph_method="powlaw",
        min_rivdph=1,
        min_rivwth=30,
        slope_len=2000,
        smooth_len=5000,
        river_routing="local_inertial",
        elevtn_map="land_elevation",
        output_names={},
    )

    example_wflow_model.setup_floodplains(
        hydrography_fn="merit_hydro", floodplain_type="2d", elevtn_map=elevtn_map
    )

    mapname = {
        "land_elevation": "river_bank_elevation_avg",
        "meta_subgrid_elevation": "river_bank_elevation_subgrid",
    }[elevtn_map]

    assert f"{mapname}_D4" in example_wflow_model.staticmaps.data
    assert example_wflow_model.get_config("model.floodplain_1d__flag") == False
    assert example_wflow_model.get_config("model.land_routing") == "local_inertial"
    assert (
        example_wflow_model.get_config("input.static.river_bank_water__elevation")
        == f"{mapname}_D4"
    )
    assert (
        example_wflow_model.get_config(
            "input.static.land_surface_water_flow__ground_elevation"
        )
        == elevtn_map
    )
    xrt.assert_allclose(
        example_wflow_model.staticmaps.data[f"{mapname}_D4"].raster.mask_nodata(),
        floodplain1d_testdata[f"{mapname}_D4"],
    )

    # Check states in config
    assert (
        example_wflow_model.get_config(
            "state.variables.land_surface_water__x_component_of_instantaneous_volume_flow_rate"
        )
        == "land_instantaneous_qx"
    )
    assert (
        example_wflow_model.get_config(
            "state.variables.land_surface_water__y_component_of_instantaneous_volume_flow_rate"
        )
        == "land_instantaneous_qy"
    )


def test_setup_precip_from_point_timeseries(
    example_wflow_model: WflowSbmModel, df_precip_stations, gdf_precip_stations
):
    # Interpolation types and the mean value to check the test
    # first value is for all 8 stations, second for 3 stations inside Piave
    # Linear interpolation can result in missing values when coverage is too low
    # such as in the test case (resulting in low value of 155).
    # Note: start_time and end_time of model are used to slice timeseries

    interp_types = {
        "nearest": [446, 435],
        "linear": [459, 155],  # note the low value for linear interpolation
        "idw": [443, 431],
        "ordinarykriging": [446, 426],
        # "externaldriftkriging" requires an additional drift variable,
    }

    for interp_type, test_val in interp_types.items():
        example_wflow_model.setup_precip_from_point_timeseries(
            precip_fn=df_precip_stations,
            precip_stations_fn=gdf_precip_stations,
            interp_type=interp_type,
            buffer=1e6,
        )
        # Check forcing and dtype
        assert "precip" in example_wflow_model.forcing.data
        assert example_wflow_model.forcing.data["precip"].dtype == "float32"

        # Compare computed value with expected value using all stations
        mean_all = example_wflow_model.forcing.data["precip"].mean().values
        assert int(mean_all * 1000) == test_val[0]

        # Do the some but only for the 3 stations inside the basin (buffer = 0)
        example_wflow_model.setup_precip_from_point_timeseries(
            precip_fn=df_precip_stations,
            precip_stations_fn=gdf_precip_stations,
            interp_type=interp_type,
            buffer=0,
        )
        mean_inside = example_wflow_model.forcing.data["precip"].mean().values
        assert int(mean_inside * 1000) == test_val[1]

    # Similar test but for GeoDataset
    for interp_type, test_val in interp_types.items():
        geodataset_precip_stations = GeoDataset.from_gdf(
            gdf=gdf_precip_stations,
            data_vars={"precip": df_precip_stations},
            index_dim="index",
            keep_cols=False,
            merge_index="gdf",
        )

        example_wflow_model.setup_precip_from_point_timeseries(
            precip_fn=geodataset_precip_stations,
            precip_stations_fn=None,
            interp_type=interp_type,
            buffer=1e6,
        )
        # Check forcing and dtype
        assert "precip" in example_wflow_model.forcing.data
        assert example_wflow_model.forcing.data["precip"].dtype == "float32"

        # Compare computed value with expected value using all stations
        mean_all = example_wflow_model.forcing.data["precip"].mean().values
        assert int(mean_all * 1000) == test_val[0]

    # Also include a test for uniform precipitation
    example_wflow_model.setup_precip_from_point_timeseries(
        precip_fn=df_precip_stations.iloc[:, 0].to_frame(),
        precip_stations_fn=None,
        interp_type="uniform",
    )
    # Check if the values per timestep are unique
    for i, _ in enumerate(example_wflow_model.forcing.data["precip"].time):
        unique_values = np.unique(
            example_wflow_model.forcing.data["precip"].isel(time=i)
        )
        assert len(unique_values[~np.isnan(unique_values)]) == 1
    # Check mean value
    mean_uniform = example_wflow_model.forcing.data["precip"].mean().values
    assert int(mean_uniform * 1000) == 274


def test_setup_pet_forcing(example_wflow_model: WflowSbmModel, da_pet: xr.DataArray):
    example_wflow_model.setup_pet_forcing(
        pet_fn=da_pet,
    )

    assert "pet" in example_wflow_model.forcing.data
    # Check dtype
    assert example_wflow_model.forcing.data["pet"].dtype == "float32"
    # used to be debruin before update
    assert "pet_method" not in example_wflow_model.forcing.data["pet"].attrs
    assert example_wflow_model.forcing.data["pet"].min().values == da_pet.min().values
    mean_val = example_wflow_model.forcing.data["pet"].mean().values
    assert int(mean_val * 1000) == 2984


def test_setup_1dmodel_connection(example_wflow_model: WflowSbmModel, rivers1d):
    # test subbasin_area method with river boundaries
    example_wflow_model.setup_1dmodel_connection(
        river1d_fn=rivers1d,
        connection_method="subbasin_area",
        area_max=10.0,
        add_tributaries=True,
        include_river_boundaries=True,
        mapname="1dmodel",
        update_toml=True,
        toml_output="netcdf_scalar",
    )

    assert "gauges_1dmodel" in example_wflow_model.geoms.data
    assert "subcatchment_1dmodel" in example_wflow_model.geoms.data
    assert "subcatchment_riv_1dmodel" in example_wflow_model.geoms.data

    assert len(example_wflow_model.geoms.get("gauges_1dmodel")) == 3
    assert len(example_wflow_model.geoms.get("subcatchment_1dmodel")) == 2
    conf_dict = {
        "name": "Q",
        "map": "gauges_1dmodel",
        "parameter": "river_water__volume_flow_rate",
    }
    assert (
        conf_dict
        in example_wflow_model.config.data["output"]["netcdf_scalar"]["variable"]  # noqa: E501
    )

    # test subbasin_area method with river boundaries
    example_wflow_model.setup_1dmodel_connection(
        river1d_fn=rivers1d,
        connection_method="subbasin_area",
        area_max=30.0,
        add_tributaries=True,
        include_river_boundaries=False,
        mapname="1dmodel-nobounds",
        update_toml=True,
        toml_output="csv",
    )

    assert len(example_wflow_model.geoms.get("gauges_1dmodel-nobounds")) == 1
    assert len(example_wflow_model.geoms.get("subcatchment_1dmodel-nobounds")) == 2
    assert np.all(
        example_wflow_model.geoms.get("subcatchment_1dmodel").geometry.geom_equals(
            example_wflow_model.geoms.get("subcatchment_1dmodel-nobounds").geometry
        )
    )

    # test nodes method without extra tributaries
    example_wflow_model.setup_1dmodel_connection(
        river1d_fn=rivers1d,
        connection_method="nodes",
        add_tributaries=False,
        include_river_boundaries=False,
        mapname="1dmodel-nodes",
        update_toml=False,
        max_dist=5000,
    )

    assert "gauges_1dmodel-nodes" not in example_wflow_model.geoms.data
    assert len(example_wflow_model.geoms.get("subcatchment_1dmodel-nodes")) == 6


def test_setup_1dmodel_connection_other(
    example_wflow_model: WflowSbmModel, rivers1d_projected: gpd.GeoDataFrame
):
    # rivers1d_projected has different crs and parts outside the model area

    # raise ValueError too small area_max
    with pytest.raises(
        ValueError, match=r"^1 subbasin\(s\) do not contain a wflow river cell."
    ):
        example_wflow_model.setup_1dmodel_connection(
            river1d_fn=rivers1d_projected,
            connection_method="subbasin_area",
            area_max=30.0,
            add_tributaries=True,
            include_river_boundaries=True,
            mapname="1dmodel",
            update_toml=True,
            toml_output="netcdf_scalar",
        )

    # Check the coupling and results
    example_wflow_model.setup_1dmodel_connection(
        river1d_fn=rivers1d_projected,
        connection_method="subbasin_area",
        area_max=10.0,
        add_tributaries=True,
        include_river_boundaries=True,
        mapname="1dmodel",
        update_toml=True,
        toml_output="netcdf_scalar",
    )

    assert "gauges_1dmodel" in example_wflow_model.geoms.data
    assert "subcatchment_1dmodel" in example_wflow_model.geoms.data
    assert "subcatchment_riv_1dmodel" in example_wflow_model.geoms.data

    assert len(example_wflow_model.geoms.data["gauges_1dmodel"]) == 1
    assert len(example_wflow_model.geoms.data["subcatchment_1dmodel"]) == 1


def test_skip_nodata_reservoir(clipped_wflow_model: WflowSbmModel):
    # Using the clipped_wflow_model as the reservoirs are not in this model
    clipped_wflow_model.setup_reservoirs_simple_control(
        reservoirs_fn="hydro_reservoirs",
        min_area=0.0,
        overwrite_existing=True,
    )
    assert clipped_wflow_model.config.data["model"]["reservoir__flag"] == False
    # Get names for two reservoir layers
    for mapname in ["reservoir_area_id", "reservoir_outlet_id"]:
        # Check if layers are indeed not present in the model
        assert (
            clipped_wflow_model._MAPS[mapname]
            not in clipped_wflow_model.staticmaps.data.data_vars
        )


def test_setup_lulc_vector(
    example_wflow_model: WflowSbmModel,
    globcover_gdf,
):
    # Test for wflow sbm
    example_wflow_model.setup_lulcmaps_from_vector(
        lulc_fn=globcover_gdf,
        lulc_mapping_fn="globcover_mapping_default",
        lulc_res=0.0025,
        save_raster_lulc=False,
    )
    assert "meta_landuse" in example_wflow_model.staticmaps.data


def test_setup_lulc_paddy(example_wflow_model: WflowSbmModel, tmpdir: Path):
    # Read the data
    example_wflow_model.read()
    example_wflow_model.set_root(Path(tmpdir), mode="w")

    layers = [50, 100, 50, 200, 800]

    # Use the method
    # Note: 11 this not NOT a rice class, but chosen just for testing purposes
    example_wflow_model.setup_lulcmaps_with_paddy(
        lulc_fn="glcnmo",
        paddy_class=11,
        wflow_thicknesslayers=layers,
    )

    # Set to shorter name to improve readability of tests
    ds = example_wflow_model.staticmaps.data.copy()

    assert "soil_ksat_vertical_factor" in ds
    assert "vegetation_crop_factor" in ds
    assert "soil_brooks_corey_c" in ds
    # Assert layers are updated
    assert example_wflow_model.config.data["model"]["soil_layer__thickness"] == layers
    # Adding +1 to the layers to also represent the last layer
    assert len(ds.layer) == len(layers) + 1
    assert ds.soil_brooks_corey_c.shape[0] == len(layers) + 1
    assert ds.soil_ksat_vertical_factor.shape[0] == len(layers) + 1
    # Assert soil_ksat_vertical_factor is written to vertical section in config
    assert (
        example_wflow_model.config.data["input"]["static"][
            "soil_water__vertical_saturated_hydraulic_conductivity_factor"
        ]
        == "soil_ksat_vertical_factor"
    )

    # Test soil_ksat_vertical_factor is not 1 at the right layer for a paddy cell
    kvfrac_values = ds.soil_ksat_vertical_factor.sel(
        latitude=45.89, longitude=12.10, method="nearest"
    ).values
    assert kvfrac_values[0] == 1.0
    assert kvfrac_values[2] != 1.0
    assert kvfrac_values[5] == 1.0

    # Test values for updated C
    c_values = ds.soil_brooks_corey_c.sel(
        latitude=45.89, longitude=12.10, method="nearest"
    ).values
    assert np.isclose(c_values[0], 9.220022)
    assert np.isclose(c_values[2], 9.553196)
    assert np.isclose(c_values[5], 9.849495)

    # Test values for crop coefficient
    assert np.isclose(
        ds["vegetation_crop_factor"].raster.mask_nodata().mean().values, 0.8869253
    )

    # Test with a separate paddy_map
    example_wflow_model.setup_lulcmaps_with_paddy(
        lulc_fn="globcover_2009",
        paddy_class=11,
        output_paddy_class=12,
        paddy_fn=ds["meta_landuse"].where(
            ds["meta_landuse"] == 11, ds["meta_landuse"].raster.nodata
        ),
        lulc_mapping_fn="globcover_mapping_default",
        wflow_thicknesslayers=layers,
    )

    ds2 = example_wflow_model.staticmaps.data.copy()

    assert np.any(ds2["meta_landuse"] == 12)


def test_setup_allocation_areas(example_wflow_model: WflowSbmModel, tmpdir: Path):
    # Read the data and set new root
    example_wflow_model.read()
    example_wflow_model.set_root(
        Path(
            tmpdir,
        ),
        mode="w",
    )

    # Use the method
    example_wflow_model.setup_allocation_areas(
        waterareas_fn="gadm_level2",
        priority_basins=True,
    )

    # Assert entries
    assert "demand_allocation_area_id" in example_wflow_model.geoms.data
    assert "demand_allocation_area_id" in example_wflow_model.staticmaps.data

    # Assert output values
    assert len(example_wflow_model.geoms.get("demand_allocation_area_id")) == 3
    # on unique values
    uni = example_wflow_model.geoms.get("demand_allocation_area_id").value.unique()
    assert np.all(np.sort(uni) == [11, 16, 17])


def test_setup_allocation_surfacewaterfrac(
    example_wflow_model: WflowSbmModel, tmpdir: Path
):
    # Read the data and set new root
    example_wflow_model.read()
    example_wflow_model.set_root(
        Path(
            tmpdir,
        ),
        mode="w",
    )
    # Add lisflood data from test
    lisflood_yml = join(TESTDATADIR, "demand", "data_catalog.yml")
    example_wflow_model.data_catalog = example_wflow_model.data_catalog.from_yml(
        lisflood_yml
    )

    # Use the method fully with lisflood
    example_wflow_model.setup_allocation_surfacewaterfrac(
        gwfrac_fn="lisflood_gwfrac",
        waterareas_fn="lisflood_waterregions",
        gwbodies_fn="lisflood_gwbodies",
        ncfrac_fn="lisflood_ncfrac",
        interpolate_nodata=False,
    )

    # Assert entries
    assert "demand_surface_water_ratio" in example_wflow_model.staticmaps.data
    assert np.isclose(
        example_wflow_model.staticmaps.data["demand_surface_water_ratio"]
        .raster.mask_nodata()
        .mean()
        .values,
        0.9411998,
    )

    # Use the method without gwbodies and ncfrac and waterareas from wflow
    example_wflow_model.setup_allocation_areas(
        waterareas_fn="gadm_level2",
    )
    example_wflow_model.setup_allocation_surfacewaterfrac(
        gwfrac_fn="lisflood_gwfrac",
        waterareas_fn=None,
        gwbodies_fn=None,
        ncfrac_fn=None,
        interpolate_nodata=False,
    )

    # Assert entries
    assert "demand_surface_water_ratio" in example_wflow_model.staticmaps.data
    assert np.isclose(
        example_wflow_model.staticmaps.data["demand_surface_water_ratio"]
        .raster.mask_nodata()
        .mean()
        .values,
        0.9865037,
    )


def test_setup_non_irrigation(example_wflow_model: WflowSbmModel, tmpdir: Path):
    # Read the data
    example_wflow_model.read()
    example_wflow_model.set_root(Path(tmpdir), mode="w")

    # Use the method
    example_wflow_model.setup_domestic_demand(
        domestic_fn="pcr_globwb",
        population_fn="ghs_pop_2015",
        domestic_fn_original_res=0.5,
    )
    example_wflow_model.setup_other_demand(
        demand_fn="pcr_globwb",
        variables=[
            "industry_gross",
            "industry_net",
            "livestock_gross",
            "livestock_net",
        ],
    )

    # Assert entries
    assert "demand_domestic_gross" in example_wflow_model.staticmaps.data
    assert "meta_population" in example_wflow_model.staticmaps.data

    # Assert some values
    dom_gross_vals = (
        example_wflow_model.staticmaps.data["demand_domestic_gross"]
        .isel(latitude=32, longitude=26)
        .values
    )
    assert int(np.mean(dom_gross_vals) * 100) == 127
    popu_val = (
        example_wflow_model.staticmaps.data["meta_population"]
        .isel(latitude=32, longitude=26)
        .values
    )
    assert int(popu_val) == 7450

    ind_mean = (
        example_wflow_model.staticmaps.data["demand_industry_gross"].mean().values
    )
    assert np.isclose(ind_mean, 0.065195)

    # test with other method
    example_wflow_model.setup_domestic_demand_from_population(
        population_fn="ghs_pop_2015",
        domestic_gross_per_capita=0.35,
        domestic_net_per_capita=0.25,
    )

    # Assert entries
    assert "demand_domestic_gross" in example_wflow_model.staticmaps.data
    assert "meta_population" in example_wflow_model.staticmaps.data

    # Assert some values
    dom_gross_vals = (
        example_wflow_model.staticmaps.data["demand_domestic_gross"]
        .isel(latitude=32, longitude=26)
        .values
    )
    assert int(dom_gross_vals * 100) == 109

    # Check other combination
    with pytest.raises(ValueError, match="The provided domestic demand data is "):
        example_wflow_model.setup_domestic_demand_from_population(
            population_fn="ghs_pop_2015",
            domestic_gross_per_capita=[0.35, 0.35, 0.34, 0.36],
            domestic_net_per_capita=[0.25, 0.25, 0.24, 0.26],
        )

    # Check cyclic
    example_wflow_model.setup_domestic_demand_from_population(
        population_fn="ghs_pop_2015",
        domestic_gross_per_capita=[
            0.35,
            0.35,
            0.34,
            0.34,
            0.36,
            0.36,
            0.35,
            0.35,
            0.34,
            0.34,
            0.36,
            0.36,
        ],
        domestic_net_per_capita=[
            0.25,
            0.25,
            0.24,
            0.24,
            0.26,
            0.26,
            0.25,
            0.25,
            0.24,
            0.24,
            0.26,
            0.26,
        ],
    )
    assert "time" in example_wflow_model.staticmaps.data["demand_domestic_gross"].dims
    assert "time" in example_wflow_model.staticmaps.data["demand_domestic_net"].dims


def test_setup_irrigation_nopaddy(
    example_wflow_model: WflowSbmModel, tmpdir: Path, globcover_gdf: gpd.GeoDataFrame
):
    # Read the data
    example_wflow_model.read()
    example_wflow_model.set_root(Path(tmpdir), mode="w")

    # Use the method
    example_wflow_model.setup_irrigation(
        irrigated_area_fn="irrigated_area",
        irrigation_value=[1],
        cropland_class=[11, 14, 20, 30],
        paddy_class=[],
        area_threshold=0.6,
        lai_threshold=0.2,
    )

    # Set to shorter name to improve readability of tests
    ds = example_wflow_model.staticmaps.data

    # Assert entries
    assert "demand_paddy_irrigated_mask" not in ds
    assert "demand_nonpaddy_irrigated_mask" in ds
    assert "demand_nonpaddy_irrigation_trigger" in ds

    # Assert the irrigation_trigger map has the same shape as LAI
    assert (
        ds["demand_nonpaddy_irrigation_trigger"].shape[0]
        == ds["vegetation_leaf_area_index"].shape[0]
    )

    # There is no paddy in this region
    assert ds["demand_nonpaddy_irrigated_mask"].raster.mask_nodata().sum().values == 5
    # Check if more irrigation is allowed during summer than winter
    assert (
        ds["demand_nonpaddy_irrigation_trigger"]
        .raster.mask_nodata()
        .sel(time=2)
        .sum()
        .values
        < ds["demand_nonpaddy_irrigation_trigger"]
        .raster.mask_nodata()
        .sel(time=8)
        .sum()
        .values
    )

    # Test with geodataframe
    # Use globcover gdf class 11
    irrigation_gdf = globcover_gdf.loc[globcover_gdf["landuse"].isin([14])]
    example_wflow_model.setup_irrigation_from_vector(
        irrigated_area_fn=irrigation_gdf,
        cropland_class=[11, 14, 20, 30],
        paddy_class=[],
        area_threshold=0.6,
        lai_threshold=0.2,
    )

    # Set to shorter name to improve readability of tests
    ds = example_wflow_model.staticmaps.data

    # Assert entries
    assert "demand_paddy_irrigated_mask" not in ds
    assert "demand_nonpaddy_irrigated_mask" in ds
    assert "demand_nonpaddy_irrigation_trigger" in ds

    assert ds["demand_nonpaddy_irrigated_mask"].raster.mask_nodata().sum().values == 8


def test_setup_irrigation_withpaddy(example_wflow_model: WflowSbmModel, tmpdir: Path):
    # Read the data
    example_wflow_model.read()
    example_wflow_model.set_root(
        Path(
            tmpdir,
        ),
        mode="w",
    )

    # First update landuse to add rice
    layers = [50, 100, 50, 200, 800]
    # Note: 11 this not NOT a rice class, but chosen just for testing purposes
    example_wflow_model.setup_lulcmaps_with_paddy(
        lulc_fn="glcnmo",
        paddy_class=11,
        wflow_thicknesslayers=layers,
    )

    example_wflow_model.setup_irrigation(
        irrigated_area_fn="irrigated_area",
        irrigation_value=[1],
        cropland_class=[11, 13],
        paddy_class=[11],
        area_threshold=0.6,
        lai_threshold=0.2,
    )

    # Set to shorter name to improve readability of tests
    ds = example_wflow_model.staticmaps.data

    # Assert entries
    assert "demand_paddy_irrigated_mask" in ds
    assert "demand_paddy_irrigation_trigger" in ds


def test_setup_cold_states(example_wflow_model: WflowSbmModel, tmpdir: Path):
    # Create states
    example_wflow_model.setup_cold_states()
    states = example_wflow_model.states.data.copy()

    assert "land_instantaneous_q" in example_wflow_model.states.data
    assert "layer" in example_wflow_model.states.data["soil_unsaturated_depth"].dims
    assert np.isclose(
        example_wflow_model.states.data["soil_saturated_depth"]
        .raster.mask_nodata()
        .mean()
        .values,
        648.43677,
    )
    assert np.isclose(
        example_wflow_model.states.data["subsurface_q"]
        .raster.mask_nodata()
        .mean()
        .values,
        67.45569,
    )

    # test write
    example_wflow_model.set_root(str(tmpdir.join("wflow_cold_states")), mode="w")
    example_wflow_model.write_states()

    assert tmpdir.join("wflow_cold_states", "instate", "instates.nc").exists()

    # test read
    example_wflow_model.set_root(str(tmpdir.join("wflow_cold_states")), mode="r+")
    example_wflow_model.read_states()

    xrt.assert_equal(
        xr.merge(states.values(), compat="override"),
        xr.merge(example_wflow_model.states.data.values(), compat="override"),
    )
