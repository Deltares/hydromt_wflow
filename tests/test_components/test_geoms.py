import logging
import re
from unittest import mock

import geopandas as gpd
import hydromt
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


@pytest.fixture
def mock_xy() -> tuple[float, float]:
    return (12.2051, 45.8331)


@pytest.fixture
def mock_geometry(mock_xy) -> gpd.GeoDataFrame:
    x, y = mock_xy
    return gpd.GeoDataFrame(geometry=[box(x - 1, y - 1, x + 1, y + 1)], crs="EPSG:4326")


@pytest.fixture
def mock_region(mock_xy) -> dict:
    x, y = mock_xy
    return {
        "basin": [x, y],
        "strord": 4,
        "bounds": [x - 1, y - 1, x + 1, y + 1],
    }


@mock.patch("hydromt_wflow.components.geoms.get_basin_geometry")
def test_parse_region(
    mock_get_basin_geometry, mock_wflow_model, mock_geometry, mock_xy, mock_region
):
    """Test the parsing of a basin region."""
    component = WflowGeomsComponent(model=mock_wflow_model)
    x, y = mock_xy
    test_xy = np.array([[x, y]])
    mock_get_basin_geometry.return_value = (mock_geometry, test_xy)
    # Act
    geom, xy, ds_org = component.parse_region(
        region=mock_region.copy(),
        hydrography_fn="mock_hydrography",
        # datacatalog is mocked so no need for a real file
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


def test_parse_region_errors(
    mock_wflow_model, mock_region, mocker, mock_geometry, mock_xy
):
    component = WflowGeomsComponent(model=mock_wflow_model)

    # Mock the datacatalog.get_rasterdataset method
    with mock.patch.object(
        component.data_catalog, "get_rasterdataset", return_value=None
    ):
        with pytest.raises(
            ValueError,
            match="hydrography_fn hydrography_file not found in data catalog.",
        ):
            component.parse_region(mock_region, hydrography_fn="hydrography_file")

    bad_resolution = 1 / 240
    err_msg = (
        f"The model resolution {bad_resolution} should be larger than the "
        f"hydrography_fn resolution {1 / 120.0}"
    )
    with pytest.raises(ValueError, match=err_msg):
        component.parse_region(
            region=mock_region,
            hydrography_fn="hydrography_fn",
            resolution=bad_resolution,
        )

    bad_resolution = 2
    err_msg = (
        f"The model resolution {bad_resolution} should be smaller than 1 degree "
        "(111km) for geographic coordinate systems. "
        "Make sure you provided res in degree rather than in meters."
    )
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        component.parse_region(
            region=mock_region, hydrography_fn="hydrography_region", resolution=2
        )

    interbasin_region = {"interbasin": mock_region["basin"]}
    err_msg = (
        "wflow region kind not understood or supported: interbasin. "
        "Use 'basin', 'subbasin', 'bbox' or 'geom'."
    )

    with pytest.raises(ValueError, match=err_msg):
        component.parse_region(interbasin_region, hydrography_fn="hydrography_fn")

    mock_get_basin_geometry = mocker.patch(
        "hydromt_wflow.components.geoms.get_basin_geometry"
    )
    x, y = mock_xy
    test_xy = np.array([[x, y]])
    mock_geometry.crs = None
    mock_get_basin_geometry.return_value = (mock_geometry, test_xy)
    with pytest.raises(ValueError, match="wflow region geometry has no CRS"):
        component.parse_region(mock_region, hydrography_fn="test")


def test_get(mock_wflow_model, caplog, mocker):
    component = WflowGeomsComponent(model=mock_wflow_model)

    # Mock the data property of the parent geoms component
    mocker.patch.object(hydromt.model.components.geoms.GeomsComponent, "data", {})
    caplog.set_level(logging.WARNING)
    geom = component.get("not_a_geom")
    assert "Geometry 'not_a_geom' not found in geoms." in caplog.text
    assert geom is None

    mocker.patch.object(
        hydromt.model.components.geoms.GeomsComponent,
        "data",
        {"geom": gpd.GeoDataFrame()},
    )
    caplog.set_level(logging.INFO)
    geom = component.get("geom")
    assert "Retrieved geometry 'geom' from geoms." in caplog.text
    assert isinstance(geom, gpd.GeoDataFrame)


def test_pop(mock_wflow_model, caplog, mocker):
    component = WflowGeomsComponent(model=mock_wflow_model)

    # Mock the data property of the parent geoms component
    mocker.patch.object(hydromt.model.components.geoms.GeomsComponent, "data", {})
    caplog.set_level(logging.WARNING)
    geom = component.pop("not_a_geom")
    assert (
        "Geometry 'not_a_geom' not found in geoms, returning default value : None."
    ) in caplog.text
    assert geom is None
    mocker.patch.object(
        hydromt.model.components.geoms.GeomsComponent,
        "data",
        {"geom": gpd.GeoDataFrame()},
    )
    caplog.at_level(logging.INFO)
    geom = component.pop("geom")
    assert "Removed geometry 'geom' from geoms."
    assert isinstance(geom, gpd.GeoDataFrame)
    assert component.data == {}
