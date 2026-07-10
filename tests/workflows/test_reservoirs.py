from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydromt_wflow.workflows import reservoirs


@pytest.fixture
def gdf():
    return pd.DataFrame({"waterbody_id": [10, 20, 30, 40]})


@pytest.fixture
def ds_like():
    """Fixture for a template dataset with a river mask."""
    river_mask = xr.DataArray(
        np.array([[1, 1, 0]], dtype=np.int32),
        dims=("y", "x"),
        name="river_mask",
    )
    return xr.Dataset({"river_mask": river_mask})


@pytest.fixture
def ds_initial_area_id():
    da_initial = xr.DataArray(
        np.array([[10, 20, 30]], dtype=np.int32),
        dims=("y", "x"),
        name="reservoir_area_id",
        attrs={"_FillValue": -999},
    )
    return da_initial.to_dataset()


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


def test__exclude_reservoirs_outside_rivers():
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

    exclude_reservoirs = reservoirs._exclude_reservoirs_outside_rivers(
        river_mask, reservoir_ids, exclude_outside_reservoirs=True
    )
    include_reservoirs = reservoirs._exclude_reservoirs_outside_rivers(
        river_mask, reservoir_ids
    )

    assert 10 in exclude_reservoirs.values
    assert 20 not in exclude_reservoirs.values
    assert 10 in include_reservoirs.values
    assert 20 in include_reservoirs.values


def test__rasterize_reservoir_area_id(gdf, ds_like):
    """Test that _rasterize_reservoir_area_id returns the expected dataset."""
    nodata = -999
    expected_data = np.array([[1, np.nan]])

    # Mock rasterize to return a DataArray with expected values
    mock_da_wbmask = xr.DataArray(
        expected_data, dims=("y", "x"), attrs={"_FillValue": nodata}
    ).rename("reservoir_area_id")

    # Mock rasterize_geometry to return a DataArray with fraction values
    mock_da_fraction = xr.DataArray(
        np.array([[1.0, 0.0]]),
        dims=("y", "x"),
    )

    with (
        mock.patch.object(ds_like.raster, "rasterize", return_value=mock_da_wbmask),
        mock.patch.object(
            ds_like.raster, "rasterize_geometry", return_value=mock_da_fraction
        ),
    ):
        ds_out = reservoirs._rasterize_reservoir_area_id(
            gdf=gdf,
            ds_like=ds_like,
            nodata=nodata,
        )

    # Output assertions
    assert "reservoir_area_id" in ds_out
    assert ds_out["reservoir_area_id"].attrs["_FillValue"] == nodata
    assert ds_out["reservoir_area_id"].values[0, 0] == 1
    assert ds_out["reservoir_area_id"].values[0, 1] == nodata


def mock_exclude_reservoirs(river_mask, reservoir_ids, exclude_outside_reservoirs):
    """Mock for _exclude_reservoirs_outside_rivers: replaces some IDs with nodata."""
    out = reservoir_ids.copy()
    out.values = np.array([[10, -999, 30]], dtype=np.int32)
    return out


@mock.patch.object(
    reservoirs,
    "_exclude_reservoirs_outside_rivers",
    side_effect=mock_exclude_reservoirs,
)
@mock.patch.object(
    reservoirs,
    "_rasterize_reservoir_area_id",
)
def test__build_reservoir_area_id_map(
    mock_rasterize_reservoir_area_id,
    mock_exclude_reservoirs,
    gdf,
    ds_like,
    ds_initial_area_id,
):
    """Test that reservoir ids are rasterized and filtered correctly."""
    # Call the function under test
    nodata = -999
    mock_rasterize_reservoir_area_id.return_value = ds_initial_area_id
    ds_out, gdf_out = reservoirs._build_reservoir_area_id_map(
        gdf=gdf,
        ds_like=ds_like,
        nodata=nodata,
        exclude_outside_reservoirs=True,
        fraction=0.1,
    )

    # Assert the output dataset contains the expected reservoir area IDs
    expected_reservoir_ids = np.array([[10, nodata, 30]])
    assert np.array_equal(ds_out["reservoir_area_id"].values, expected_reservoir_ids)

    # Assert the output GeoDataFrame contains only the expected waterbody IDs
    expected_waterbody_ids = [10, 30]
    assert gdf_out["waterbody_id"].tolist() == expected_waterbody_ids
