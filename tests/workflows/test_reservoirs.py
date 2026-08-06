from unittest import mock

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr

from hydromt_wflow.workflows import reservoirs


@pytest.fixture
def gdf():
    """Fixture for a GeoDataFrame with waterbody ids."""
    return gpd.GeoDataFrame(
        {"waterbody_id": [10, 20, 30, 40]},
        geometry=gpd.points_from_xy([0, 1, 2, 3], [0, 1, 2, 3]),
    )


@pytest.fixture
def ds_like():
    """Fixture for a template dataset with a river mask and upstream area."""
    coords = {"y": [0], "x": [0, 1, 2]}
    ds_like = xr.Dataset(
        {
            "river_mask": xr.DataArray(
                np.array([[1, 1, 0]], dtype=np.int32),
                dims=("y", "x"),
                coords=coords,
            ),
            "uparea": xr.DataArray(
                np.array([[100, 200, 300]]),
                dims=("y", "x"),
                coords=coords,
            ),
        }
    )
    ds_like.raster.set_crs("EPSG:4326")
    return ds_like


@pytest.fixture
def ds_initial_area_id():
    """Fixture a dataset containing rasterized reservoir area ids."""
    return xr.Dataset(
        {
            "reservoir_area_id": xr.DataArray(
                np.array([[10, 20, 30]], dtype=np.int32),
                dims=("y", "x"),
                coords={"y": [0], "x": [0, 1, 2]},
                attrs={"_FillValue": -999},
            ),
            "reservoir_outlet_id": xr.DataArray(
                np.full((1, 3), -999, dtype=np.int32),
                dims=("y", "x"),
                coords={"y": [0], "x": [0, 1, 2]},
            ),
        }
    )


def test_set_rating_curve_layer_data_type():
    """Test that the rating curve layer data type is set correctly."""
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


@pytest.mark.parametrize(
    ("fraction", "expected_data", "expected_fraction_calls"),
    [
        (0.1, np.array([[1, -999]]), 1),
        (None, np.array([[1, 2]]), 0),
    ],
)
def test__rasterize_reservoir_area_id(
    gdf, ds_like, fraction, expected_data, expected_fraction_calls
):
    """Test that _rasterize_reservoir_area_id behaves correctly for both fraction and no-fraction cases."""
    nodata = -999

    # Mock rasterize to return a DataArray with expected values
    mock_da_wbmask = xr.DataArray(
        expected_data, dims=("y", "x"), attrs={"_FillValue": nodata}
    ).rename("reservoir_area_id")

    # Mock rasterize_geometry to return a DataArray with fraction values
    mock_da_fraction = xr.DataArray(np.array([[1.0, 0.0]]), dims=("y", "x"))

    with (
        mock.patch.object(ds_like.raster, "rasterize", return_value=mock_da_wbmask),
        mock.patch.object(
            ds_like.raster, "rasterize_geometry", return_value=mock_da_fraction
        ) as mock_rasterize_geometry,
    ):
        ds_out = reservoirs._rasterize_reservoir_area_id(
            gdf=gdf,
            ds_like=ds_like,
            nodata=nodata,
            fraction=fraction,
        )

    # Assertions
    assert "reservoir_area_id" in ds_out
    assert ds_out["reservoir_area_id"].attrs["_FillValue"] == nodata
    assert ds_out["reservoir_area_id"].values[0, 0] == 1
    assert ds_out["reservoir_area_id"].values[0, 1] == expected_data[0, 1]
    assert mock_rasterize_geometry.call_count == expected_fraction_calls


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


def test_build_reservoir_outlets_from_uparea(gdf, ds_like, ds_initial_area_id):
    """Test that outlet coordinates are correctly calculated from upstream area."""
    test_gdf = gdf[gdf["waterbody_id"].isin([10, 20, 30])].copy()
    outgdf = reservoirs._build_reservoir_outlets_from_uparea(
        gdf=test_gdf,
        ds_out=ds_initial_area_id,
        ds_like=ds_like,
        uparea_name="uparea",
    )

    expected_coords = {
        10: {"xout": 0, "yout": 0},
        20: {"xout": 1, "yout": 0},
        30: {"xout": 2, "yout": 0},
    }

    for res_id, expected in expected_coords.items():
        row = outgdf[outgdf["waterbody_id"] == res_id].iloc[0]
        assert row["xout"] == expected["xout"]
        assert row["yout"] == expected["yout"]

    assert outgdf.geometry.type.tolist() == ["Point", "Point", "Point"]


def test__build_reservoir_outlet_id(gdf, ds_like, ds_initial_area_id):
    """Test outlet ID map creation from provided xout/yout coordinates."""
    nodata = -999
    gdf_with_outlets = gdf[gdf["waterbody_id"].isin([10, 20, 30])].copy()
    gdf_with_outlets["xout"] = [0.0, 1.0, 2.0]
    gdf_with_outlets["yout"] = [0.0, 0.0, 0.0]

    rasterized_outlets = xr.DataArray(
        np.array([[10, 20, 30]], dtype=np.int32),
        dims=("y", "x"),
        coords={"y": [0], "x": [0, 1, 2]},
        attrs={"_FillValue": nodata},
    )

    with mock.patch.object(
        ds_like.raster, "rasterize", return_value=rasterized_outlets
    ) as mock_rasterize:
        ds_out, gdf_out = reservoirs._build_reservoir_outlet_id_map(
            gdf=gdf_with_outlets,
            ds_out=ds_initial_area_id.copy(deep=True),
            ds_like=ds_like,
            nodata=nodata,
            uparea_name=None,
        )

    assert mock_rasterize.call_count == 1
    assert np.array_equal(ds_out["reservoir_outlet_id"].values, [[10, 20, 30]])
    assert ds_out["reservoir_outlet_id"].attrs["_FillValue"] == nodata
    assert np.allclose(gdf_out["xout"].to_numpy(), [0.0, 1.0, 2.0])
    assert np.allclose(gdf_out["yout"].to_numpy(), [0.0, 0.0, 0.0])


@mock.patch.object(reservoirs, "_build_reservoir_outlet_id_map")
@mock.patch.object(reservoirs, "_build_reservoir_area_id_map")
def test_reservoir_id_maps(
    mock_build_area,
    mock_build_outlet,
    gdf,
    ds_like,
    ds_initial_area_id,
):
    """Test reservoir_id_maps area filtering and helper call orchestration."""
    # gdf with which contains a waterbody that is below the min_area threshold.
    gdf_with_area = gdf[gdf["waterbody_id"].isin([10, 20, 30])].copy()
    gdf_with_area["Area_avg"] = [1.0e6, 2.0e6, 3.0e6]

    # gdf after filtering by min_area, which should only include waterbody_ids 20 and 30.
    gdf_after_area = gdf_with_area[gdf_with_area["waterbody_id"].isin([20, 30])].copy()

    # gdf after adding outlet coordinates, xout/yout
    gdf_after_outlet = gdf_after_area.copy()
    gdf_after_outlet["xout"] = [1.0, 2.0]
    gdf_after_outlet["yout"] = [0.0, 0.0]

    # Mock the return values
    mock_build_area.return_value = (ds_initial_area_id.copy(deep=True), gdf_after_area)
    mock_build_outlet.return_value = (
        ds_initial_area_id.copy(deep=True),
        gdf_after_outlet,
    )

    # Call the function under test
    ds_out, gdf_out = reservoirs.reservoir_id_maps(
        gdf=gdf_with_area,
        ds_like=ds_like.drop_vars("uparea"),
        min_area=2.0,
        uparea_name="uparea",
        exclude_outside_reservoirs=True,
        fraction=0.25,
    )

    # Assertions
    assert mock_build_area.call_count == 1
    assert mock_build_outlet.call_count == 1

    area_call = mock_build_area.call_args.kwargs
    assert area_call["gdf"]["waterbody_id"].tolist() == [20, 30]
    assert area_call["exclude_outside_reservoirs"] is True
    assert area_call["fraction"] == 0.25

    assert ds_out is mock_build_outlet.return_value[0]
    assert gdf_out is mock_build_outlet.return_value[1]
