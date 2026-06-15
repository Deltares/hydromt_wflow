"""Tests for the utils module."""

import numpy as np

from hydromt_wflow.utils import get_grid_from_config


def test_grid_from_config(demda):
    # Create a couple of variables in grid
    grid = demda.to_dataset(name="dem")
    grid["slope"] = demda * 0.1
    grid["mask"] = demda > 0

    # Create config with all options
    config = {
        "input": {
            "dem": "dem",
            "static": {
                "slope": "slope",
                "altitude": {
                    "netcdf_variable_name": "slope",
                    "scale": 10,
                },
            },
            "cyclic": {
                "subsurface_ksat_horizontal_ratio": {"value": 500},
                "ksathorfrac2": {
                    "netcdf_variable_name": "dem",
                    "scale": 0,
                    "offset": 500,
                },
            },
        },
    }

    # Tests
    dem = get_grid_from_config("dem", config=config, grid=grid)
    assert dem.equals(demda)

    slope = get_grid_from_config("slope", config=config, grid=grid)
    assert slope.equals(grid["slope"])

    altitude = get_grid_from_config("altitude", config=config, grid=grid)
    assert altitude.equals(grid["slope"] * 10)

    subsurface_ksat_horizontal_ratio = get_grid_from_config(
        "subsurface_ksat_horizontal_ratio",
        config=config,
        grid=grid,
    )
    assert np.unique(subsurface_ksat_horizontal_ratio.raster.mask_nodata()) == [500]

    ksathorfrac2 = get_grid_from_config(
        "ksathorfrac2",
        config=config,
        grid=grid,
    )
    assert ksathorfrac2.equals(subsurface_ksat_horizontal_ratio)
