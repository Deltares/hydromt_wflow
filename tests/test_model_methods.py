"""Unit tests for hydromt_wflow methods and workflows"""

import pytest
from os.path import join, dirname, abspath
import numpy as np
import warnings
import pdb
import xarray as xr
from hydromt import flw
from hydromt_wflow.wflow import WflowModel

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
    with pytest.raises(ValueError) as error:
        mod.setup_basemaps(
            region={"basin": [12.862, 45.701]},
            res=1000,
            hydrography_fn="merit_hydro",
        )
    assert str(error.value).startswith(
        "The model resolution 1000 should be smaller than 1 degree"
    )

    with pytest.raises(ValueError) as error:
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

    assert mod.staticmaps.raster.crs == 3857
    assert np.quantile(mod.staticmaps["wflow_landuse"], 0.95) == 190.0  # urban
    assert mod.get_config("model.sizeinmetres") == True


@pytest.mark.timeout(300)  # max 5 min
@pytest.mark.parametrize("source", ["gww", "jrc"])
def test_setup_reservoirs(source, tmpdir):
    logger = logging.getLogger(__name__)

    # Read model 'wflow_piave_subbasin' from EXAMPLEDIR
    model = "wflow"
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")
    mod1 = WflowModel(root=root, mode="r", logger=logger)
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
