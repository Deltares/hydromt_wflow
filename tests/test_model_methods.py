"""Unit tests for WflowModel methods and workflows."""

from pathlib import Path
from unittest import mock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hydromt.gis import GeoDataArray
from hydromt.model.processes.rivers import river_depth

from hydromt_wflow.wflow_sbm import WflowSbmModel


def test_setup_basemaps(tmp_path: Path):
    # Region
    region = {
        "basin": [12.2051, 45.8331],
        "strord": 4,
        "bounds": [11.70, 45.35, 12.95, 46.70],
    }
    mod = WflowSbmModel(
        root=tmp_path / "wflow_base",
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
        example_wflow_model.config.get_value(
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


def test_projected_crs(tmp_path: Path, test_data_dir: Path):
    # Instantiate wflow model
    root = str(tmp_path / "wflow_projected")
    mod = WflowSbmModel(
        root=root,
        mode="w",
        data_libs=["artifact_data", str(test_data_dir / "merit_utm" / "merit_utm.yml")],
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
    assert mod.config.get_value("model.cell_length_in_meter__flag") == True


@pytest.mark.parametrize("glacier_fn", ["glaciers_4326", "glaciers_3857"])
def test_projected_crs_glaciers(glacier_fn, tmp_path: Path, test_data_dir: Path):
    # Instantiate wflow model
    root = str(tmp_path / "wflow_projected")
    mod = WflowSbmModel(
        root=root,
        mode="w",
        data_libs=[
            "artifact_data",
            str(test_data_dir / "merit_utm" / "merit_utm.yml"),
            str(test_data_dir / "glacier" / "glacier_utm.yml"),
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
    assert mod.config.get_value("model.glacier__flag") is True
    assert (
        mod.config.get_value("state.variables.glacier_ice__leq_depth")
        == "glacier_leq_depth"
    )


def test_setup_outlets(example_wflow_model):
    # Update subcatchment ID
    new_subcatch = example_wflow_model.staticmaps.data["subcatchment"].copy()
    new_subcatch = new_subcatch.where(new_subcatch == new_subcatch.raster.nodata, 1001)
    example_wflow_model.staticmaps.set(new_subcatch, "subcatchment")

    # Derive outlets
    example_wflow_model.setup_outlets()

    # Check if the ID is indeed 1001
    val, count = np.unique(
        example_wflow_model.staticmaps.data["outlets"], return_counts=True
    )
    # 0 is no data
    assert val[1] == 1001
    assert count[1] == 1


def test_setup_gauges(example_wflow_model: WflowSbmModel, test_data_dir: Path):
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
    stations_fn = str(test_data_dir / "test_stations.csv")
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
    stations_csv_file = str(test_data_dir / "test_stations.csv")
    example_wflow_model.setup_gauges(gauges_fn=stations_csv_file, basename="test-flow")
    test_stations_gdf = example_wflow_model.geoms.get("gauges_test-flow")
    assert len(test_stations_gdf) == 3


def test_setup_subbasins(example_wflow_model: WflowSbmModel):
    # 1. Stream order method
    example_wflow_model.setup_subbasins(
        method="streamorder",
        threshold=4,
    )

    assert "subbasins_streamorder_4" in example_wflow_model.staticmaps.data
    assert "subbasins_streamorder_4_outlets" not in example_wflow_model.geoms.data
    assert "subbasins_streamorder_4" in example_wflow_model.geoms.data
    gdf = example_wflow_model.geoms.get("subbasins_streamorder_4")
    assert (gdf["value"].values > 0).all()
    assert gdf.index.is_unique
    assert len(gdf) > 1
    assert len(gdf) == 7

    # 2. Pfafstetter method
    example_wflow_model.setup_subbasins(
        method="pfafstetter",
        threshold=1,
    )

    assert "subbasins_pfafstetter_1" in example_wflow_model.staticmaps.data
    assert "subbasins_pfafstetter_1" in example_wflow_model.geoms.data
    gdf = example_wflow_model.geoms.get("subbasins_pfafstetter_1")
    assert len(gdf) > 1
    # this number should be fixed (linked to pfafstetter definition)
    assert len(gdf) == 9

    example_wflow_model.setup_subbasins(
        method="pfafstetter",
        threshold=2,
    )

    assert "subbasins_pfafstetter_2" in example_wflow_model.staticmaps.data
    assert "subbasins_pfafstetter_2" in example_wflow_model.geoms.data
    gdf2 = example_wflow_model.geoms.get("subbasins_pfafstetter_2")
    assert len(gdf2) > len(gdf)
    assert len(gdf2) == 72

    # 3. Area method
    example_wflow_model.setup_subbasins(
        method="area",
        threshold=500,
        add_outlets_map=True,
    )
    assert "subbasins_area_500" in example_wflow_model.staticmaps.data
    assert "subbasins_area_500_outlets" in example_wflow_model.staticmaps.data
    assert "subbasins_area_500" in example_wflow_model.geoms.data
    gdf = example_wflow_model.geoms.get("subbasins_area_500")
    assert len(gdf) > 1
    assert len(gdf) == 6


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
    assert (
        example_wflow_model.config.get_value("model.river_routing") == "local_inertial"
    )
    assert (
        example_wflow_model.config.get_value("input.static.river_bank_water__elevation")
        == mapname
    )
    assert (
        example_wflow_model.staticmaps.data[mapname]
        .raster.mask_nodata()
        .equals(floodplain1d_testdata[mapname])
    )


def test_setup_rivers_smooth_len_affects_result(example_wflow_model: WflowSbmModel):
    """Verify that different smooth_len values produce different river widths.

    Before the fix (GH#740), ``min(1, ...)`` was used instead of ``max(1, ...)``,
    causing nsmooth to always be 1 regardless of smooth_len.  With the fix,
    a larger smooth_len yields more smoothing and a visibly different result.
    """
    common_kwargs = dict(
        hydrography_fn="merit_hydro",
        river_geom_fn="hydro_rivers_lin",
        river_upa=30,
        rivdph_method=None,
        min_rivwth=30,
        slope_len=2000,
        river_routing="local_inertial",
        elevtn_map="land_elevation",
        output_names={},
    )

    example_wflow_model.setup_rivers(smooth_len=5_000, **common_kwargs)
    rivwth_small = example_wflow_model.staticmaps.data["river_width"].copy()

    example_wflow_model.setup_rivers(smooth_len=50_000, **common_kwargs)
    rivwth_large = example_wflow_model.staticmaps.data["river_width"]

    # With the bug both would be identical (nsmooth=1 in both cases).
    assert not rivwth_small.equals(rivwth_large), (
        "river_width should differ for smooth_len=5000 vs 50000; "
        "if equal, nsmooth is likely clamped to 1 (the old min bug)"
    )


def test_setup_rivers_no_subgrid(tmp_path: Path):
    # Instantiate new wflow model
    # Region
    region = {
        "subbasin": [12.5032, 46.5327],
        "strord": 4,
        "bounds": [11.70, 45.35, 12.95, 46.70],
    }
    mod = WflowSbmModel(
        root=str(tmp_path / "river_no_subgrid"),
        mode="w",
        data_libs=["artifact_data"],
    )
    hydrography = mod.data_catalog.get_rasterdataset("merit_hydro_ihu")
    # Run setup_basemaps
    mod.setup_basemaps(
        region=region,
        hydrography_fn=hydrography.copy(),
        res=hydrography.raster.res[0],  # no upscaling
    )

    assert "x_out" not in mod.staticmaps.data

    # Setup rivers with different hydrography (with subgrid = false)
    with pytest.raises(ValueError, match="It seems model grid was not upscaled"):
        mod.setup_rivers(
            hydrography_fn="merit_hydro",
            river_geom_fn="hydro_rivers_lin",
            river_upa=30,
        )

    # Now with correct hydrography and some additional kwargs for river depth
    with mock.patch(
        "hydromt_wflow.workflows.river.river_depth",
        wraps=river_depth,
    ) as mock_river_depth:
        mod.setup_rivers(
            hydrography_fn="merit_hydro_ihu",
            river_geom_fn="hydro_rivers_lin",
            river_upa=30,
            river_depth_kwargs={"hc": 0.4, "hp": 0.9},
        )
        mock_river_depth.assert_called_once()
        called_kwargs = mock_river_depth.call_args.kwargs

        assert "hc" in called_kwargs
        assert called_kwargs["hc"] == pytest.approx(0.4)

        assert "hp" in called_kwargs
        assert called_kwargs["hp"] == pytest.approx(0.9)

    assert "river_mask" in mod.staticmaps.data


def test_setup_rivers_depth(tmp_path: Path):
    # Instantiate new wflow model
    # Region
    region = {
        "subbasin": [12.3661, 46.3998],
        "strord": 4,
        "bounds": [11.70, 45.35, 12.95, 46.70],
    }
    mod = WflowSbmModel(
        root=str(tmp_path / "river_mask"),
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


def test_setup_lulc_vector(
    example_wflow_model: WflowSbmModel, globcover_gdf, data_dir: Path
):
    # Test for wflow sbm
    # Use a file directly for lulc_mapping_fn
    mapping_fn = data_dir / "lulc" / "v0.8" / "globcover_mapping.csv"
    example_wflow_model.setup_lulcmaps_from_vector(
        lulc_fn=globcover_gdf,
        lulc_mapping_fn=mapping_fn,
        lulc_res=0.0025,
        save_raster_lulc=False,
    )
    assert "meta_landuse" in example_wflow_model.staticmaps.data


def test_setup_grid_from_geodataset_static(example_wflow_model: WflowSbmModel):
    # Create static inflows for GRDC gauges
    gauges = example_wflow_model.geoms.get("gauges_grdc")
    gauges_ids = gauges["grdc_no"].values

    # Create a dataframe with gauges_ids as columns and a random static value for inflow
    df = pd.DataFrame(
        data=np.random.rand(1, len(gauges_ids)),
        index=[0],
        columns=gauges_ids,
        dtype=np.float32,
    )

    # Call the method
    example_wflow_model.setup_grid_from_geodataset(
        geodataset_fn=df,
        locations_fn=gauges,
        index_col="grdc_no",
        variable="inflow",
        fill_value=0,
        output_names={"river_water__external_inflow_volume_flow_rate": "river_inflow"},
    )

    # Checks
    assert "river_inflow" in example_wflow_model.staticmaps.data

    da = example_wflow_model.staticmaps.data["river_inflow"]
    assert np.all(da.raster.sample(gauges).values == df.values)
    # Values for the rest of the basin should be 0
    assert 0 in np.unique(da.values)
    assert da.raster.nodata == -9999
    assert da.dtype == np.float32

    assert (
        example_wflow_model.config.get_value(
            "input.static.river_water__external_inflow_volume_flow_rate"
        )
        == "river_inflow"
    )


def test_setup_grid_from_geodataset_cyclic(
    example_wflow_model: WflowSbmModel, reservoir_outlets: gpd.GeoDataFrame
):
    # Create cyclic demand for reservoirs
    reservoirs = example_wflow_model.geoms.get("meta_reservoirs_simple_control")
    reservoirs_ids = reservoirs["waterbody_id"].values

    # Create a dataframe with reservoirs_ids as columns and random value for inflow
    df = pd.DataFrame(
        data=np.random.rand(12, len(reservoirs_ids)),
        index=np.arange(1, 13),
        columns=reservoirs_ids,
        dtype=np.float32,
    )

    # Call the method
    example_wflow_model.setup_grid_from_geodataset(
        geodataset_fn=df,
        locations_fn="reservoir_outlet_id",
        variable="reservoir_demand_cyclic",
        fill_value=-1,
        nodata_value=-999,
        mask="reservoir_outlet_id",
        output_names={
            "reservoir_water_demand__required_downstream_volume_flow_rate": "reservoir_demand_cyclic"  # noqa E501
        },
    )

    # Checks
    assert "reservoir_demand_cyclic" in example_wflow_model.staticmaps.data

    da = example_wflow_model.staticmaps.data["reservoir_demand_cyclic"]

    samples = da.raster.sample(reservoir_outlets)
    # Select the first two columns of samples (controlled reservoirs)
    assert np.all(samples[:, : len(reservoirs_ids)] == df.values)
    # Third column is a lake and should be -1
    assert np.all(samples[:, len(reservoirs_ids) :] == -1)
    assert da.raster.nodata == -999
    assert da.dtype == np.float32

    # Check masking worked: eg value at the outlet should be -999 and not -1
    outlets = example_wflow_model.geoms.get("outlets")
    outlet_samples = da.raster.sample(outlets)
    assert np.all(outlet_samples == -999)

    # Check config
    assert (
        example_wflow_model.config.get_value(
            "input.cyclic.reservoir_water_demand__required_downstream_volume_flow_rate"
        )
        == "reservoir_demand_cyclic"
    )
    assert (
        example_wflow_model.config.get_value(
            "input.static.reservoir_water_demand__required_downstream_volume_flow_rate"
        )
        is None
    )


def test_setup_grid_from_geodataset_forcing(
    example_wflow_model: WflowSbmModel, reservoir_outlets: gpd.GeoDataFrame
):
    # Create forcing outflows for all reservoirs
    reservoirs_ids = reservoir_outlets.index.values

    # Get the wflow forcing start and length
    forcing = example_wflow_model.forcing.data
    # Create hourly timesteps for the length of forcing to check resampling
    hourly_times = pd.date_range(
        start=forcing.time.values[0], end=forcing.time.values[-1], freq="h"
    )

    outflows = xr.DataArray(
        data=np.random.rand(len(hourly_times), len(reservoirs_ids)),
        dims=["time", "id"],
        coords={
            "time": hourly_times,
            "id": reservoirs_ids,
            "lon": ("id", reservoir_outlets.geometry.x),
            "lat": ("id", reservoir_outlets.geometry.y),
        },
        attrs={"units": "m3/s"},
    )
    outflows = outflows.astype(np.float32)
    outflows = GeoDataArray.from_netcdf(outflows)
    outflows.vector.set_nodata(np.nan)
    outflows.vector.set_crs(reservoir_outlets.crs)

    # Call the method
    example_wflow_model.setup_grid_from_geodataset(
        geodataset_fn=outflows,
        variable="reservoir_outflow",
        nodata_value=np.nan,
        mask="reservoir_outlet_id",
        output_names={
            "reservoir_water__outgoing_observed_volume_flow_rate": "reservoir_outflow"
        },
        resample_time_kwargs={"downsampling": "mean"},
    )

    # Checks
    assert "reservoir_outflow" in example_wflow_model.forcing.data

    da = example_wflow_model.forcing.data["reservoir_outflow"]
    # Check that the time dimension has been resampled to daily
    assert np.all(da.time == forcing.time)
    assert np.isnan(da.raster.nodata)

    samples = da.raster.sample(reservoir_outlets)
    # Assert that the samples are not all nan
    assert not np.all(np.isnan(samples))

    assert (
        example_wflow_model.config.get_value(
            "input.forcing.reservoir_water__outgoing_observed_volume_flow_rate"
        )
        == "reservoir_outflow"
    )
