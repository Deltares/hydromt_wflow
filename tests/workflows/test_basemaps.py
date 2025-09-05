import logging

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr
from hydromt import DataCatalog
from pytest_mock import MockerFixture
from shapely.geometry import box

from hydromt_wflow.workflows.basemaps import parse_region


class TestParseRegion:
    """Tests for the parse_region method."""

    @pytest.fixture
    def mock_datacatalog(
        self, mock_rasterdataset: xr.Dataset, mocker: MockerFixture
    ) -> DataCatalog:
        """Mock DataCatalog, a raster dataset with a geographic CRS and resolution."""
        mock_data_catalog = mocker.create_autospec(DataCatalog, instance=True)
        mock_data_catalog.get_rasterdataset.return_value = mock_rasterdataset
        mock_data_catalog.get_source.return_value = "mock_basin_index"
        return mock_data_catalog

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
            "hydromt_wflow.workflows.basemaps.get_basin_geometry"
        )
        x, y = mock_xy
        test_xy = np.array([[x, y]])
        mock_get_basin_geometry.return_value = (mock_geometry, test_xy)
        return mock_get_basin_geometry

    @pytest.fixture
    def mock_geometry(self, mock_xy) -> gpd.GeoDataFrame:
        """Mock a geometry GeoDataFrame."""
        x, y = mock_xy
        return gpd.GeoDataFrame(
            geometry=[box(x - 1, y - 1, x + 1, y + 1)], crs="EPSG:4326"
        )

    def test_parse_region(
        self,
        mock_get_basin_geometry,
        mock_datacatalog,
        mock_xy,
        mock_region,
    ):
        """Test the parsing of a basin region."""
        x, y = mock_xy

        # Act
        geoms, xy, ds_org = parse_region(
            data_catalog=mock_datacatalog,
            region=mock_region.copy(),
            hydrography_fn="mock_hydrography",
            # datacatalog is mocked so no need for a real file
        )

        # Assert
        for name, geom in geoms.items():
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
        self,
        mock_geometry,
        mock_region,
        mock_xy,
        mock_datacatalog,
        caplog,
        mocker,
    ):
        """Test parsing regions of different kinds."""
        caplog.set_level(logging.WARNING)
        x, y = mock_xy
        bbox_region = {"bbox": [x - 1, y - 1, x + 1, y + 1]}

        mock_parse_region_bbox = mocker.patch(
            "hydromt_wflow.workflows.basemaps.parse_region_bbox"
        )
        mock_parse_region_bbox.return_value = mock_geometry

        parsed_region = parse_region(
            data_catalog=mock_datacatalog,
            region=bbox_region,
            hydrography_fn="test",
        )

        assert (
            "Kind 'bbox' for the region is not recommended as it can lead "
            "to mistakes in the catchment delineation. Use carefully."
        ) in caplog.text

        assert isinstance(parsed_region[0], dict)
        assert parsed_region[1] is None

        geom_region = {"geom": mock_geometry.geometry}
        mock_parse_region_geom = mocker.patch(
            "hydromt_wflow.workflows.basemaps.parse_region_geom"
        )
        mock_parse_region_geom.return_value = mock_geometry

        parsed_region = parse_region(
            mock_datacatalog, region=geom_region, hydrography_fn="test"
        )

        assert (
            "Kind 'geom' for the region is not recommended as it can lead "
            "to mistakes in the catchment delineation. Use carefully."
        ) in caplog.text
        assert isinstance(parsed_region[0], dict)
        assert parsed_region[1] is None

    def test_parse_region_invalid_region_kind(self, mock_datacatalog, mock_region):
        """Test that an error is raised if the region kind is not supported."""
        interbasin_region = {"interbasin": mock_region.copy()["basin"]}
        err_msg = (
            "wflow region kind not understood or supported: interbasin. "
            "Use 'basin', 'subbasin', 'bbox' or 'geom'."
        )
        mock_raster_ds = mock_datacatalog.get_rasterdataset.return_value
        mock_raster_ds.raster.crs.is_geographic = False  # reset
        with pytest.raises(ValueError, match=err_msg):
            parse_region(
                mock_datacatalog,
                interbasin_region,
                hydrography_fn="hydrography_fn",
            )

    def test_parse_region_geometry_without_crs(
        self, mock_datacatalog, mock_get_basin_geometry, mock_region
    ):
        """Test that an error is raised if the basin geometry has no CRS."""
        basin_geom, xy = mock_get_basin_geometry.return_value
        basin_geom_no_crs = basin_geom.set_crs(None, allow_override=True)
        mock_get_basin_geometry.return_value = (basin_geom_no_crs, xy)

        with pytest.raises(ValueError, match="wflow region geometry has no CRS"):
            parse_region(
                mock_datacatalog,
                mock_region.copy(),
                hydrography_fn="hydrography_fn",
            )

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
        self,
        mock_datacatalog,
        mock_region,
        resolution,
        is_geographic,
        expected_err_msg,
    ):
        """Test that an error is raised if the model resolution is not compatible with the hydrography resolution."""  # noqa: E501
        mock_raster_ds = mock_datacatalog.get_rasterdataset.return_value
        mock_raster_ds.raster.crs.is_geographic = is_geographic
        with pytest.raises(ValueError, match=expected_err_msg):
            parse_region(
                mock_datacatalog,
                region=mock_region.copy(),
                hydrography_fn="hydrography_fn",
                resolution=resolution,
            )

    def test_parse_region_scale_ratio_close_to_one_no_error(
        self,
        mock_datacatalog,
        mock_get_basin_geometry,
        mock_xy,
        mock_region,
        mock_geometry,
    ):
        """Test that no error is raised if the model resolution is close to the hydrography resolution."""  # noqa: E501
        mock_geometry.set_crs(crs="EPSG:4326")  # restore valid CRS
        mock_get_basin_geometry.return_value = (mock_geometry, mock_xy)
        resolution = mock_datacatalog.get_rasterdataset.return_value.raster.res[0]
        # Should not raise any errors or warnings
        parse_region(
            mock_datacatalog,
            mock_region.copy(),
            hydrography_fn="test",
            resolution=resolution,
        )

    def test_parse_region_scale_ratio_warning(
        self,
        mock_get_basin_geometry,
        mock_datacatalog,
        mock_region,
        caplog,
    ):
        """Test that a warning is logged if the model resolution is slightly smaller than the hydrography resolution."""  # noqa: E501
        hydrography_resolution = (
            mock_datacatalog.get_rasterdataset.return_value.raster.res[0]
        )
        slightly_smaller_resolution = hydrography_resolution * 0.9  # scale_ratio ~ 1.11

        with caplog.at_level(logging.WARNING):
            parse_region(
                mock_datacatalog,
                mock_region.copy(),
                hydrography_fn="test",
                resolution=slightly_smaller_resolution,
            )
        assert any(
            "Model resolution" in rec.message
            and "does not match the hydrography resolution" in rec.message
            for rec in caplog.records
        )
