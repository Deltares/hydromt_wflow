from unittest import mock

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from hydromt import DataCatalog
from shapely.geometry import box

from hydromt_wflow import WflowModel
from hydromt_wflow.components import WflowGeomsComponent


@pytest.fixture
def mock_wflow_model():
    model = mock.MagicMock(spec=WflowModel)
    model.data_catalog = mock.MagicMock(spec=DataCatalog)

    # Basic mock of raster dataset
    ds = mock.MagicMock(spec=xr.Dataset)
    mock_raster = mock.Mock()
    mock_raster.res = (1 / 120.0, 1 / 120.0)
    mock_raster.crs.is_geographic = True
    mock_raster.clip_geom.return_value = ds
    mock_raster.geometry_mask.return_value = xr.DataArray(
        np.array([[1, 0], [0, 1]]), dims=("y", "x")
    )

    ds.raster = mock_raster
    ds.coords = {}  # mock.coords["mask"] will be valid if dict-like
    ds.__getitem__.side_effect = lambda key: ds.coords.get(key)

    model.data_catalog.get_rasterdataset.return_value = ds
    model.data_catalog.get_source.return_value = "mock_basin_index"

    return model


@mock.patch("hydromt_wflow.components.geoms.get_basin_geometry")
def test_parse_region(mock_get_basin_geometry, mock_wflow_model):
    """Test the parsing of a basin region."""
    component = WflowGeomsComponent(model=mock_wflow_model)
    x, y = 12.2051, 45.8331
    test_geom = gpd.GeoDataFrame(
        geometry=[box(x - 1, y - 1, x + 1, y + 1)], crs="EPSG:4326"
    )
    test_xy = np.array([[x, y]])

    mock_get_basin_geometry.return_value = (test_geom, test_xy)
    region = {
        "basin": [x, y],
        "strord": 4,
        "bounds": [x - 1, y - 1, x + 1, y + 1],
    }

    # Act
    geom, xy, ds_org = component.parse_region(
        region=region.copy(),
    )

    # Assert
    assert isinstance(geom, gpd.GeoDataFrame), (
        "Parsed geometry should be a GeoDataFrame."
    )
    assert geom.crs.to_string() == "EPSG:4326", "Geometry should have CRS EPSG:4326."
    assert not geom.empty, "Returned geometry should not be empty."
    bounds = geom.total_bounds
    assert bounds[0] <= x <= bounds[2], (
        "X coordinate should fall within geometry bounds."
    )
    assert bounds[1] <= y <= bounds[3], (
        "Y coordinate should fall within geometry bounds."
    )

    assert isinstance(xy, np.ndarray), "XY should be a NumPy array."
    assert xy.shape == (1, 2), "XY should be a single (x, y) pair."
    assert np.allclose(xy[0], [x, y]), f"XY values should match input ({x}, {y})."

    assert isinstance(ds_org, xr.Dataset), (
        "Returned object should be an xarray Dataset."
    )
    assert "mask" in ds_org.coords, "'mask' should be present in dataset coordinates."
    mask = ds_org.coords["mask"].values
    assert mask.shape == (2, 2), "Mask should have expected shape (2, 2)."
    assert np.array_equal(mask, np.array([[1, 0], [0, 1]])), (
        "Mask values should match mocked output."
    )
