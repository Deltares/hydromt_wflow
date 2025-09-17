import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
from shapely.geometry import box

from hydromt_wflow.workflows import glaciermaps


@pytest.fixture
def sample_gdf():
    """Return a simple GeoDataFrame with glacier polygons and AREA attribute."""
    geometries = [box(0, 0, 1, 1), box(1, 1, 2, 2)]
    data = {"simple_id": [1, 2], "AREA": [1.0, 2.0]}
    return gpd.GeoDataFrame(data, geometry=geometries, crs="EPSG:4326")


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    """Return a dummy xr.Dataset with raster attributes and a fake raster accessor."""
    # Create a simple 10x10 grid
    data = np.zeros((10, 10), dtype=np.float32)
    transform = rasterio.transform.from_origin(0, 10, 1, 1)

    # Create an xarray DataArray with rioxarray accessor
    da = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": np.arange(10, 0, -1), "x": np.arange(10)},
    )
    da.rio.write_crs("EPSG:4326", inplace=True)
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_nodata(-9999, inplace=True)

    return da.to_dataset(name="elevtn")


def test_glaciermaps_output_structure(sample_gdf, sample_dataset):
    """Test that glaciermaps returns expected variables and correct shapes."""
    ds_out = glaciermaps(
        gdf=sample_gdf,
        ds_like=sample_dataset,
        id_column="simple_id",
        elevtn_name="elevtn",
    )

    # Check output type
    assert isinstance(ds_out, xr.Dataset)

    # Check expected variables
    for var in ["glacareas", "glacstore", "glacfracs"]:
        assert var in ds_out

    # Check shapes
    expected_shape = sample_dataset["elevtn"].shape
    for var in ["glacareas", "glacstore", "glacfracs"]:
        assert ds_out[var].shape == expected_shape

    # Check data types
    assert ds_out["glacareas"].dtype == np.int32
    assert ds_out["glacstore"].dtype == np.float32
    assert ds_out["glacfracs"].dtype == np.float32


def test_glaciermaps_nodata_and_values(sample_gdf, sample_dataset):
    """Test that nodata values are set and maps are non-negative."""
    ds_out = glaciermaps(
        gdf=sample_gdf,
        ds_like=sample_dataset,
        id_column="simple_id",
        elevtn_name="elevtn",
    )

    # Nodata should be 0
    for var in ["glacareas", "glacstore", "glacfracs"]:
        nodata_value = ds_out[var].rio.nodata
        assert nodata_value == 0

    # Values should be >= 0
    for var in ["glacareas", "glacstore", "glacfracs"]:
        data = ds_out[var].values
        assert np.all(data >= 0)


def test_glaciermaps_respects_id_column(sample_gdf, sample_dataset):
    """Test that changing id_column works."""
    sample_gdf["custom_id"] = sample_gdf["simple_id"] + 100
    ds_out = glaciermaps(
        gdf=sample_gdf,
        ds_like=sample_dataset,
        id_column="custom_id",
        elevtn_name="elevtn",
    )
    assert "glacareas" in ds_out

    expected_ids = set(sample_gdf["custom_id"]) | {0}  # 0 for nodata
    actual_ids = set(np.unique(ds_out["glacareas"].values))
    assert actual_ids <= expected_ids, (
        "All elements of actual_ids should be in expected_ids"
    )


def test_glaciermaps_with_projected_crs(sample_gdf, sample_dataset):
    """Test glaciermaps using a projected CRS (EPSG:3857) in ds_like."""
    # Reproject ds_like to EPSG:3857 (Web Mercator)
    ds_proj = sample_dataset.rio.reproject("EPSG:3857")

    # Ensure sample_gdf is still in EPSG:4326 to test reprojection inside glaciermaps
    assert sample_gdf.crs.to_string() == "EPSG:4326"
    assert ds_proj.rio.crs.to_string() == "EPSG:3857"

    ds_out = glaciermaps(
        gdf=sample_gdf, ds_like=ds_proj, id_column="simple_id", elevtn_name="elevtn"
    )

    # Check CRS of output
    assert ds_out.rio.crs.to_string() == "EPSG:3857"

    # Check variable existence and types
    for var in ["glacareas", "glacstore", "glacfracs"]:
        assert var in ds_out
        assert ds_out[var].shape == ds_proj["elevtn"].shape

    # Check no NaNs or negative values
    assert np.all(ds_out["glacfracs"].values >= 0)
    assert np.all(ds_out["glacstore"].values >= 0)
