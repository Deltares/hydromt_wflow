import numpy as np
import xarray as xr

from hydromt_wflow.workflows import reservoirs


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

    ds = reservoirs.set_rating_curve_layer_data_type(ds)
    ds2 = reservoirs.set_rating_curve_layer_data_type(ds2)

    assert ds["reservoir_rating_curve"].dtype == np.int32
    assert ds2["reservoir_rating_curve"].dtype == np.int32
    assert ds["reservoir_rating_curve"].raster.nodata == -999
    assert ds2["reservoir_rating_curve"].attrs["_FillValue"] == -999
    assert ds["reservoir_rating_curve"][0, 0].values == -999


def test_exclude_reservoirs_outside_rivers():
    """Test that reservoirs outside the river network are excluded."""
    # Create a river mask river, cells = 1, non-river = 0
    river_mask = xr.DataArray(
        [
            [0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    # Create reservoir IDs, overlaps river = 10, no overlap = 20
    reservoir_ids = xr.DataArray(
        [
            [-999, 10, 10, -999, -999],
            [-999, 10, -999, -999, -999],
            [-999, -999, -999, -999, -999],
            [-999, -999, -999, 20, 20],
            [-999, -999, -999, 20, 20],
        ]
    )

    exclude_reservoirs = reservoirs.exclude_reservoirs_outside_rivers(
        river_mask, reservoir_ids, exclude_outside_reservoirs=True
    )
    include_reservoirs = reservoirs.exclude_reservoirs_outside_rivers(
        river_mask, reservoir_ids
    )

    assert 10 in exclude_reservoirs.values
    assert 20 not in exclude_reservoirs.values
    assert 10 in include_reservoirs.values
    assert 20 in include_reservoirs.values
