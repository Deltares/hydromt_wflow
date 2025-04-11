import hydromt  # noqa: F401
import numpy as np
import pytest
import xarray as xr

## Cached data

## Mocked objects


## In-situ data structures
# TODO replace this dummy forcing when testdata have been made
@pytest.fixture(scope="session")
def dummy_precipitation() -> xr.DataArray:
    factors = np.array([1, 2, 5, 10, 2, 1, 0])
    temp = np.ones((10, 10))
    data = temp[np.newaxis, :, :] * factors[:, np.newaxis, np.newaxis]
    da = xr.DataArray(
        data=data,
        coords={
            "time": range(len(factors)),
            "y": np.arange(46.4, 44.4, -0.2),
            "x": np.arange(6.6, 8.6, 0.2),
        },
        dims=["time", "y", "x"],
    )
    da.name = "precip"
    da.raster.set_crs(4326)
    return da
