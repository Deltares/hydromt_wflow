import logging
from pathlib import Path
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
    model.root = mock.MagicMock()
    model.root.is_reading_mode.return_value = False
    model.crs = "EPSG:4326"

    # Basic mock of raster dataset
    ds = mock.MagicMock(spec=xr.Dataset)
    mock_raster = mock.Mock()
    mock_raster.crs.is_geographic = True
    mock_raster.res = (1 / 120.0, 1 / 120.0)
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
def mock_geometry(mock_xy) -> gpd.GeoDataFrame:
    x, y = mock_xy
    return gpd.GeoDataFrame(geometry=[box(x - 1, y - 1, x + 1, y + 1)], crs="EPSG:4326")


def test_get_success(mock_wflow_model, caplog, mocker):
    component = WflowGeomsComponent(model=mock_wflow_model)
    caplog.set_level(logging.INFO)
    mocker.patch.object(
        hydromt.model.components.geoms.GeomsComponent,
        "data",
        {"geom": gpd.GeoDataFrame()},
    )

    geom = component.get("geom")

    assert isinstance(geom, gpd.GeoDataFrame)


def test_get_failure(mock_wflow_model, caplog, mocker):
    component = WflowGeomsComponent(model=mock_wflow_model)
    caplog.set_level(logging.INFO)

    mocker.patch.object(hydromt.model.components.geoms.GeomsComponent, "data", {})

    geom = component.get("not_a_geom")

    assert "Geometry 'not_a_geom' not found in geoms." in caplog.text
    assert geom is None


def test_pop_failure(mock_wflow_model, caplog, mocker):
    component = WflowGeomsComponent(model=mock_wflow_model)
    mocker.patch.object(hydromt.model.components.geoms.GeomsComponent, "data", {})
    caplog.set_level(logging.WARNING)

    geom = component.pop("not_a_geom")

    assert (
        "Geometry 'not_a_geom' not found in geoms, returning default value : None."
    ) in caplog.text
    assert geom is None


def test_pop_success(mock_wflow_model, caplog, mocker):
    mocker.patch.object(
        hydromt.model.components.geoms.GeomsComponent,
        "data",
        {"geom": gpd.GeoDataFrame()},
    )
    component = WflowGeomsComponent(model=mock_wflow_model)
    caplog.at_level(logging.INFO)

    geom = component.pop("geom")

    assert "Removed geometry 'geom' from geoms."
    assert isinstance(geom, gpd.GeoDataFrame)
    assert component.data == {}


def test_write_and_read(tmp_path: Path, mock_wflow_model, mock_geometry):
    # Initialize component
    comp = WflowGeomsComponent(model=mock_wflow_model)

    # Add geometry and write it to disk
    comp.set(mock_geometry, name="test_geom")
    comp.write(dir_out=tmp_path)

    # Confirm file was written
    out_file = tmp_path / "test_geom.geojson"
    assert out_file.exists()

    # Create new instance and read
    new_comp = WflowGeomsComponent(model=mock_wflow_model)
    new_comp.read(read_dir=tmp_path)

    # Check that the geometry matches
    gdf_read = new_comp.get("test_geom")
    assert gdf_read is not None
    assert gdf_read.equals(mock_geometry), (
        "Read geometry does not match the original geometry."
    )


def test_read_merge_behavior(tmp_path: Path, mock_wflow_model, mock_geometry):
    # First write a geom file
    comp = WflowGeomsComponent(model=mock_wflow_model)
    comp.set(mock_geometry, name="geom1")
    comp.write(dir_out=tmp_path)

    # Second component with one item in memory
    comp2 = WflowGeomsComponent(model=mock_wflow_model)
    comp2.set(mock_geometry, name="in_memory")

    # Read with merge_data=False (should clear previous data)
    comp2.read(read_dir=tmp_path, merge_data=False)
    assert comp2.get("in_memory") is None
    assert comp2.get("geom1") is not None

    # Read with merge_data=True (should preserve previous data)
    comp3 = WflowGeomsComponent(model=mock_wflow_model)
    comp3.set(mock_geometry, name="in_memory")
    comp3.read(read_dir=tmp_path, merge_data=True)
    assert comp3.get("in_memory") is not None
    assert comp3.get("geom1") is not None


@pytest.mark.skip("move tests to where ever parse region ends up")
class TestParseRegion:
    """Tests for the parse_region method of WflowGeomsComponent."""

    @pytest.fixture
    def mock_xy(self) -> tuple[float, float]:
        """Mock coordinates for testing."""
        return (12.2051, 45.8331)

    @pytest.fixture
    def mock_region(self, mock_xy) -> dict:
        """Mock a region dictionary."""
        x, y = mock_xy
        return {
            "basin": [x, y],
            "strord": 4,
            "bounds": [x - 1, y - 1, x + 1, y + 1],
        }

    @pytest.fixture
    def mock_get_basin_geometry(self, mock_geometry, mock_xy, mocker):
        """Mock the get_basin_geometry function."""
        mock_get_basin_geometry = mocker.patch(
            "hydromt_wflow.components.geoms.get_basin_geometry"
        )
        x, y = mock_xy
        test_xy = np.array([[x, y]])
        mock_get_basin_geometry.return_value = (mock_geometry, test_xy)
        return mock_get_basin_geometry

    def test_parse_region(
        self, mock_get_basin_geometry, mock_wflow_model, mock_xy, mock_region
    ):
        """Test the parsing of a basin region."""
        component = WflowGeomsComponent(model=mock_wflow_model)
        x, y = mock_xy

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
        assert geom.crs.to_string() == "EPSG:4326", (
            "Geometry should have CRS EPSG:4326."
        )
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
        assert "mask" in ds_org.coords, (
            "'mask' should be present in dataset coordinates."
        )
        mask = ds_org.coords["mask"].values
        assert mask.shape == (2, 2), "Mask should have expected shape (2, 2)."
        assert np.array_equal(mask, np.array([[1, 0], [0, 1]])), (
            "Mask values should match mocked output."
        )

    def test_parse_regions_different_kinds(
        self, mock_geometry, mock_region, mock_xy, mock_wflow_model, caplog, mocker
    ):
        """Test parsing regions of different kinds."""
        component = WflowGeomsComponent(model=mock_wflow_model)
        caplog.set_level(logging.WARNING)
        x, y = mock_xy
        bbox_region = {"bbox": [x - 1, y - 1, x + 1, y + 1]}

        mock_parse_region_bbox = mocker.patch(
            "hydromt_wflow.components.geoms.parse_region_bbox"
        )
        mock_parse_region_bbox.return_value = mock_geometry
        parsed_region = component.parse_region(
            region=bbox_region, hydrography_fn="test"
        )

        assert (
            "Kind 'bbox' for the region is not recommended as it can lead "
            "to mistakes in the catchment delineation. Use carefully."
        ) in caplog.text
        assert isinstance(parsed_region[0], gpd.GeoDataFrame)
        assert parsed_region[1] is None

        geom_region = {"geom": mock_geometry.geometry}
        mock_parse_region_geom = mocker.patch(
            "hydromt_wflow.components.geoms.parse_region_geom"
        )
        mock_parse_region_geom.return_value = mock_geometry
        parsed_region = component.parse_region(
            region=geom_region, hydrography_fn="test"
        )

        assert (
            "Kind 'geom' for the region is not recommended as it can lead "
            "to mistakes in the catchment delineation. Use carefully."
        ) in caplog.text
        assert isinstance(parsed_region[0], gpd.GeoDataFrame)
        assert parsed_region[1] is None

    def test_parse_region_hydrography_fn_not_found(self, mock_wflow_model, mock_region):
        """Test that an error is raised if hydrography_fn is not found in the data catalog."""  # noqa: E501
        component = WflowGeomsComponent(model=mock_wflow_model)
        with mock.patch.object(
            component.data_catalog, "get_rasterdataset", return_value=None
        ):
            with pytest.raises(
                ValueError,
                match="hydrography_fn hydrography_file not found in data catalog.",
            ):
                component.parse_region(
                    mock_region.copy(), hydrography_fn="hydrography_file"
                )

    def test_parse_region_invalid_region_kind(self, mock_wflow_model, mock_region):
        """Test that an error is raised if the region kind is not supported."""
        component = WflowGeomsComponent(model=mock_wflow_model)
        interbasin_region = {"interbasin": mock_region.copy()["basin"]}
        err_msg = (
            "wflow region kind not understood or supported: interbasin. "
            "Use 'basin', 'subbasin', 'bbox' or 'geom'."
        )
        mock_raster_ds = mock_wflow_model.data_catalog.get_rasterdataset.return_value
        mock_raster_ds.raster.crs.is_geographic = False  # reset
        with pytest.raises(ValueError, match=err_msg):
            component.parse_region(interbasin_region, hydrography_fn="hydrography_fn")

    def test_parse_region_geometry_without_crs(
        self, mock_wflow_model, mock_get_basin_geometry, mock_region
    ):
        """Test that an error is raised if the basin geometry has no CRS."""
        component = WflowGeomsComponent(model=mock_wflow_model)
        basin_geom, xy = mock_get_basin_geometry.return_value
        basin_geom.crs = None
        mock_get_basin_geometry.return_value = (basin_geom, xy)
        with pytest.raises(ValueError, match="wflow region geometry has no CRS"):
            component.parse_region(mock_region.copy(), hydrography_fn="hydrography_fn")

    @pytest.mark.parametrize(
        ("resolution", "is_geographic", "expected_err_msg"),
        [
            (
                1 / 240,
                False,
                "Model resolution 0.004166666666666667 should be larger than the "
                "hydrography_fn resolution 0.008333333333333333.",
            ),  # Case 1: scale ratio << 1
            (
                2,
                True,
                r"The model resolution 2 should be smaller than 1 degree \(111km\) for "
                "geographic coordinate systems. Make sure you provided res in degree rather"  # noqa: E501
                " than in meters.",
            ),  # Case 2: scale ratio >> 1 (geographic crs)
        ],
    )
    def test_parse_region_scale_ratio_errors(
        self, mock_wflow_model, mock_region, resolution, is_geographic, expected_err_msg
    ):
        """Test that an error is raised if the model resolution is not compatible with the hydrography resolution."""  # noqa: E501
        component = WflowGeomsComponent(model=mock_wflow_model)
        mock_raster_ds = mock_wflow_model.data_catalog.get_rasterdataset.return_value
        mock_raster_ds.raster.crs.is_geographic = is_geographic
        with pytest.raises(ValueError, match=expected_err_msg):
            component.parse_region(
                region=mock_region.copy(),
                hydrography_fn="hydrography_fn",
                resolution=resolution,
            )

    def test_parse_region_scale_ratio_close_to_one_no_error(
        self,
        mock_wflow_model,
        mock_get_basin_geometry,
        mock_xy,
        mock_region,
        mock_geometry,
    ):
        """Test that no error is raised if the model resolution is close to the hydrography resolution."""  # noqa: E501
        component = WflowGeomsComponent(model=mock_wflow_model)
        mock_geometry.crs = "EPSG:4326"  # restore valid CRS
        mock_get_basin_geometry.return_value = (mock_geometry, mock_xy)
        resolution = (
            mock_wflow_model.data_catalog.get_rasterdataset.return_value.raster.res[0]
        )
        # Should not raise any errors or warnings
        component.parse_region(
            mock_region.copy(), hydrography_fn="test", resolution=resolution
        )

    def test_parse_region_scale_ratio_warning(
        self,
        mock_get_basin_geometry,
        mock_wflow_model,
        mock_region,
        caplog,
    ):
        """Test that a warning is logged if the model resolution is slightly smaller than the hydrography resolution."""  # noqa: E501
        component = WflowGeomsComponent(model=mock_wflow_model)
        hydrography_resolution = (
            mock_wflow_model.data_catalog.get_rasterdataset.return_value.raster.res[0]
        )
        slightly_smaller_resolution = hydrography_resolution * 0.9  # scale_ratio ~ 1.11
        with caplog.at_level(logging.WARNING):
            component.parse_region(
                mock_region.copy(),
                hydrography_fn="test",
                resolution=slightly_smaller_resolution,
            )
        assert any(
            "Model resolution" in rec.message
            and "does not match the hydrography resolution" in rec.message
            for rec in caplog.records
        )
