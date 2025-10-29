import numpy as np
import xarray as xr

from hydromt_wflow.workflows.reservoirs import set_rating_curve_layer_data_type


def test_set_rating_curve_layer_data_type():
    # create some test data
    data = np.random.randint(4, 5)
    coords = {"x": np.arange(4), "y": np.arange(5)}
    attrs = {"description": "Test data", "_FillValue": -999.0}
    da = xr.DataArray(data, coords=coords, dims=["x", "y"], attrs=attrs).astype(
        np.float32
    )
    da[0, 0] = np.nan
    da2 = da.copy(deep=True)
    da2 = da2.fillna(-999.0)
    ds = xr.Dataset({"reservoir_rating_curve": da})
    ds2 = xr.Dataset({"reservoir_rating_curve": da2})

    ds = set_rating_curve_layer_data_type(ds)
    ds2 = set_rating_curve_layer_data_type(ds2)

    assert ds["reservoir_rating_curve"].dtype == np.int32
    assert ds2["reservoir_rating_curve"].dtype == np.int32
    assert ds["reservoir_rating_curve"].raster.nodata == -999
    assert ds2["reservoir_rating_curve"].attrs["_FillValue"] == -999
    assert ds["reservoir_rating_curve"][0, 0].values == -999
