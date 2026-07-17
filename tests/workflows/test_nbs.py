import numpy as np
import pytest
import xarray as xr

from hydromt_wflow.workflows.nbs import (
    _compute_hand,
    _compute_landslope,
    _compute_nbs_coverage,
    _hydrography_suitability,
    _landuse_suitability,
    nbs_suitability_from_thresholds,
    ponding_level_from_suitability,
)


def _make_da(
    data: np.ndarray,
    name: str = "da",
    nodata: float = -9999,
    epsg: int = 4326,
) -> xr.DataArray:
    """Create a DataArray with the given data, name, nodata value, and CRS."""
    coords = {"y": [2.0, 1.0, 0.0], "x": [0.0, 1.0, 2.0]}
    da = xr.DataArray(
        data.astype(np.float32), coords=coords, dims=["y", "x"], name=name
    )
    da.raster.set_nodata(nodata)
    da.raster.set_crs(epsg)
    return da


@pytest.fixture
def landuse() -> xr.DataArray:
    """Create a simple 3x3 land-use map with three classes and a nodata value of -1."""
    data = np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1]], dtype=np.int16)
    return _make_da(data, name="landuse", nodata=-1)


@pytest.fixture
def hydro_ds() -> xr.Dataset:
    """Create a 3x3 hydrography dataset with elevation, slope, HAND, and flwdir."""
    elevtn_data = np.array(
        [[300.0, 400.0, 500.0], [200.0, 300.0, 400.0], [100.0, 200.0, 300.0]],
        dtype=np.float32,
    )
    slope_data = np.array(
        [[0.05, 0.10, 0.15], [0.02, 0.05, 0.10], [0.00, 0.02, 0.05]],
        dtype=np.float32,
    )
    hand_data = np.array(
        [[10.0, 20.0, 30.0], [5.0, 10.0, 20.0], [0.0, 5.0, 10.0]],
        dtype=np.float32,
    )
    flwdir_data = np.full((3, 3), 2, dtype=np.uint8)

    coords = {"y": [2.0, 1.0, 0.0], "x": [0.0, 1.0, 2.0]}

    def _da(data: np.ndarray, name: str, nodata: float = -9999) -> xr.DataArray:
        da = xr.DataArray(data, coords=coords, dims=["y", "x"], name=name)
        da.raster.set_nodata(nodata)
        return da

    ds = xr.Dataset(
        {
            "elevtn": _da(elevtn_data, "elevtn"),
            "lndslp": _da(slope_data, "lndslp"),
            "hand": _da(hand_data, "hand"),
            "flwdir": _da(flwdir_data, "flwdir", nodata=0),
        }
    )
    ds.raster.set_crs(4326)
    return ds


@pytest.fixture
def hydro_ds_no_flwdir(hydro_ds: xr.Dataset) -> xr.Dataset:
    """Create a version of the hydro dataset with the flwdir variable removed."""
    return hydro_ds.drop_vars("flwdir")


@pytest.fixture
def hydro_ds_no_slope(hydro_ds: xr.Dataset) -> xr.Dataset:
    """Create a version of the hydro dataset with the lndslp variable removed."""
    return hydro_ds.drop_vars("lndslp")


@pytest.fixture
def basin_mask() -> xr.DataArray:
    """Create a simple 3x3 basin mask with all pixels valid (value 1)."""
    data = np.ones((3, 3), dtype=np.float32)
    return _make_da(data, name="basins", nodata=-9999)


@pytest.fixture
def suitability(hydro_ds: xr.Dataset) -> xr.DataArray:
    """Create a suitability DataArray with hydro_data with elevation range criteria."""
    return nbs_suitability_from_thresholds(
        hydro_data=hydro_ds, elevtn_range=(200.0, 400.0)
    )


@pytest.fixture
def ds_like(hydro_ds: xr.Dataset) -> xr.Dataset:
    """Coarser wflow-like staticmaps grid for reprojection target."""
    coords = {"y": [1.5, 0.5], "x": [0.5, 1.5]}
    basin_data = np.ones((2, 2), dtype=np.float32)
    basin = xr.DataArray(basin_data, coords=coords, dims=["y", "x"], name="basins")
    basin.raster.set_nodata(-9999)
    ds = xr.Dataset({"basins": basin})
    ds.raster.set_crs(4326)
    return ds


def test_landuse_single_class_is_suitable(landuse: xr.DataArray):
    """Pixels matching the specified class are marked suitable, others are not."""
    result = _landuse_suitability(landuse, lulc_classes=[1])
    assert result.values[0, 0] == 1  # class 1 : suitable
    assert result.values[0, 1] == 0  # class 2 : not suitable


def test_landuse_multiple_classes(landuse: xr.DataArray):
    """All listed classes are marked suitable; unlisted classes are not."""
    result = _landuse_suitability(landuse, lulc_classes=[1, 2])
    assert result.values[0, 0] == 1  # class 1
    assert result.values[0, 1] == 1  # class 2
    assert result.values[0, 2] == 0  # class 3


def test_landuse_no_matching_class_returns_zeros(landuse: xr.DataArray):
    """A class absent from the map produces an all-zero suitability result."""
    result = _landuse_suitability(landuse, lulc_classes=[99])
    assert (result.values <= 0).all()


def test_landuse_nodata_is_minus_one(landuse: xr.DataArray):
    """Nodata must be -1 (not 0) so downstream mode-resampling works correctly."""
    result = _landuse_suitability(landuse, lulc_classes=[1])
    assert result.raster.nodata == -1


def test_landuse_output_dtype_int8(landuse: xr.DataArray):
    """Output array uses int8 to keep memory footprint small."""
    result = _landuse_suitability(landuse, lulc_classes=[1])
    assert result.dtype == np.int8


def test_hydro_elevation_filter(hydro_ds: xr.Dataset):
    """Only pixels whose elevation falls within the given range are marked suitable."""
    result = _hydrography_suitability(hydro_ds, elevtn_range=(200.0, 400.0))
    assert result.values[0, 0] == 1  # elevtn 300 : in range
    assert result.values[0, 2] == 0  # elevtn 500 : out of range
    assert result.values[2, 0] == 0  # elevtn 100 : out of range


def test_hydro_slope_filter(hydro_ds: xr.Dataset):
    """Only pixels whose slope falls within the given range are marked suitable."""
    result = _hydrography_suitability(hydro_ds, slope_range=(0.04, 0.06))
    assert result.values[0, 0] == 1  # slope 0.05 : in range
    assert result.values[0, 2] == 0  # slope 0.15 : out of range


def test_hydro_hand_filter(hydro_ds: xr.Dataset):
    """Only pixels whose HAND value falls within the given range are marked suitable."""
    result = _hydrography_suitability(hydro_ds, hand_range=(5.0, 15.0))
    assert result.values[1, 0] == 1  # hand 5 : in range
    assert result.values[0, 2] == 0  # hand 30 : out of range


def test_hydro_combined_filters_are_anded(hydro_ds: xr.Dataset):
    """A pixel must satisfy all active criteria simultaneously to be marked suitable."""
    result = _hydrography_suitability(
        hydro_ds,
        elevtn_range=(200.0, 400.0),
        slope_range=(0.04, 0.06),
    )
    assert result.values[0, 0] == 1  # elevtn 300 AND slope 0.05
    assert result.values[0, 2] == 0  # no elevtn 500


def test_hydro_no_criteria_returns_all_ones(hydro_ds: xr.Dataset):
    """When no filter criteria are given the entire grid is considered suitable."""
    result = _hydrography_suitability(hydro_ds)
    assert (result.values == 1).all()


def test_hydro_derives_slope_when_missing(hydro_ds_no_slope: xr.Dataset):
    """Slope is computed on the fly from elevation when not present in the dataset."""
    result = _hydrography_suitability(hydro_ds_no_slope, slope_range=(0.0, 1.0))
    assert result.dtype == np.int8


def test_hydro_output_nodata_is_minus_one(hydro_ds: xr.Dataset):
    """Nodata value is -1 to stay consistent with the land-use suitability output."""
    result = _hydrography_suitability(hydro_ds, elevtn_range=(0.0, 9999.0))
    assert result.raster.nodata == -1


def test_landslope_is_non_negative():
    """Slope values are clamped to zero; no negative values are returned."""
    elevtn = _make_da(
        np.array([[100.0, 200.0, 150.0], [80.0, 120.0, 90.0], [50.0, 60.0, 70.0]])
    )
    result = _compute_landslope(elevtn)
    assert (result.values >= 0).all()


def test_landslope_flat_dem_is_zero():
    """A perfectly flat DEM produces zero slope everywhere."""
    elevtn = _make_da(np.full((3, 3), 100.0))
    result = _compute_landslope(elevtn)
    np.testing.assert_array_equal(result.values, 0.0)


def test_landslope_output_name():
    """Output DataArray is named 'lndslp' as expected by downstream consumers."""
    elevtn = _make_da(np.ones((3, 3)))
    assert _compute_landslope(elevtn).name == "lndslp"


def test_landslope_unit_attr():
    """Output carries the correct unit attribute for slope in m/m."""
    elevtn = _make_da(np.ones((3, 3)))
    assert _compute_landslope(elevtn).attrs["unit"] == "m.m-1"


def test_hand_raises_without_flwdir(hydro_ds_no_flwdir: xr.Dataset):
    """A ValueError is raised when flow direction data is absent from the dataset."""
    with pytest.raises(ValueError, match="flwdir"):
        _compute_hand(hydro_ds_no_flwdir)


def test_hand_output_name(hydro_ds: xr.Dataset):
    """Output DataArray is named 'hand' as expected by downstream consumers."""
    assert _compute_hand(hydro_ds).name == "hand"


def test_hand_long_name_attr(hydro_ds: xr.Dataset):
    """Output carries a long_name attribute describing height above nearest drainage."""
    assert "height above nearest drainage" in _compute_hand(hydro_ds).attrs["long_name"]


def test_hand_units_attr(hydro_ds: xr.Dataset):
    """Output carries the correct units attribute in metres."""
    assert _compute_hand(hydro_ds).attrs["units"] == "m"


def test_hand_shape_matches_input(hydro_ds: xr.Dataset):
    """Output shape matches the input elevation grid exactly."""
    assert _compute_hand(hydro_ds).shape == hydro_ds["elevtn"].shape


def test_hand_valid_values_non_negative(hydro_ds: xr.Dataset):
    """All non-fill HAND values are zero or positive."""
    hand = _compute_hand(hydro_ds)
    valid = hand.values[~np.isclose(hand.values, hand.raster.nodata)]
    assert (valid >= 0).all()


def test_nbs_coverage_zero_basin_area_raises(basin_mask: xr.DataArray):
    """A basin mask with no valid pixels raises ValueError."""
    empty_basin = basin_mask.where(basin_mask == 0, other=-9999)
    nbs_map = basin_mask.copy()
    with pytest.raises(ValueError, match="Basin mask contains no valid pixels"):
        _compute_nbs_coverage(nbs_map, empty_basin)


def test_nbs_coverage_runs_without_raising(basin_mask: xr.DataArray):
    """Coverage calculation completes without raising for a valid basin and nbs map."""
    nbs_map = basin_mask.copy()
    _compute_nbs_coverage(nbs_map, basin_mask)


def test_nbs_lulc_only_without_hydro_data_does_not_crash(landuse: xr.DataArray):
    """lulc-only path must not crash when hydro_data is None (CRS fallback bug)."""
    result = nbs_suitability_from_thresholds(
        landuse=landuse,
        lulc_classes=[1],
    )
    assert result.name == "nbs_suitability"


def test_nbs_lulc_only_without_hydro_data_preserves_crs(landuse: xr.DataArray):
    """CRS from landuse is preserved when no hydro_data is provided."""
    result = nbs_suitability_from_thresholds(
        landuse=landuse,
        lulc_classes=[1],
    )
    assert result.raster.crs is not None


def test_ponding_suitable_pixels_get_pond_level(
    suitability: xr.DataArray, ds_like: xr.Dataset
):
    """Pixels that are suitable receive a non-zero ponding level after reprojection."""
    result = ponding_level_from_suitability(suitability, ds_like, pond_level=0.2)
    # assert all values between 0 and the specified pond_level,
    # and that all non-zero values are close to the pond_level
    assert (result > 0).all()
    assert (result <= 0.2).all()


def test_ponding_unsuitable_pixels_are_zero(
    suitability: xr.DataArray, ds_like: xr.Dataset
):
    """Pixels that are not suitable receive a ponding level of zero."""
    all_unsuitable = xr.zeros_like(suitability)
    all_unsuitable.raster.set_crs(suitability.raster.crs)
    result = ponding_level_from_suitability(all_unsuitable, ds_like, pond_level=0.2)
    assert (result.values == 0.0).all()


def test_ponding_output_name(suitability: xr.DataArray, ds_like: xr.Dataset):
    """Output DataArray is named 'ponding_level'."""
    result = ponding_level_from_suitability(suitability, ds_like)
    assert result.name == "ponding_level"


def test_ponding_shape_matches_ds_like(suitability: xr.DataArray, ds_like: xr.Dataset):
    """Output shape matches the wflow staticmaps reprojection target."""
    result = ponding_level_from_suitability(suitability, ds_like)
    assert result.shape == ds_like["basins"].shape


def test_ponding_nodata_is_set(suitability: xr.DataArray, ds_like: xr.Dataset):
    """Output carries a nodata value."""
    result = ponding_level_from_suitability(suitability, ds_like)
    assert result.raster.nodata is not None


def test_ponding_custom_pond_level(suitability: xr.DataArray, ds_like: xr.Dataset):
    """A custom pond_level value is reflected in the output for suitable pixels."""
    result = ponding_level_from_suitability(suitability, ds_like, pond_level=0.5)
    assert (result > 0).all()
    assert (result <= 0.5).all()


def test_nbs_raises_with_no_inputs():
    """Calling with no arguments raises a ValueError."""
    with pytest.raises(
        ValueError,
        match="At least one of landuse or hydro_data with criteria must be provided.",
    ):
        nbs_suitability_from_thresholds()


def test_nbs_raises_with_landuse_but_no_classes(landuse: xr.DataArray):
    """Passing a landuse map without lulc_classes raises a ValueError."""
    with pytest.raises(
        ValueError,
        match="At least one of landuse or hydro_data with criteria must be provided.",
    ):
        nbs_suitability_from_thresholds(landuse=landuse)


def test_nbs_raises_with_hydro_but_no_criteria(hydro_ds: xr.Dataset):
    """Passing hydro_data without any range criteria raises a ValueError."""
    with pytest.raises(
        ValueError,
        match="At least one of landuse or hydro_data with criteria must be provided.",
    ):
        nbs_suitability_from_thresholds(hydro_data=hydro_ds)


def test_nbs_lulc_only(landuse: xr.DataArray, hydro_ds: xr.Dataset):
    """Land-use criteria alone produce a correctly named suitability map."""
    result = nbs_suitability_from_thresholds(
        landuse=landuse,
        hydro_data=hydro_ds,
        lulc_classes=[1],
    )
    assert result.name == "nbs_suitability"
    assert result.values[0, 0] == 1  # class 1 : suitable
    assert result.values[0, 1] == 0  # class 2 : not suitable


def test_nbs_hydro_only(hydro_ds: xr.Dataset):
    """Hydrographic criteria alone produce a correctly named suitability map."""
    result = nbs_suitability_from_thresholds(
        hydro_data=hydro_ds,
        elevtn_range=(200.0, 400.0),
    )
    assert result.name == "nbs_suitability"
    assert result.values[0, 0] == 1
    assert result.values[0, 2] == 0


def test_nbs_combined_lulc_and_hydro(landuse: xr.DataArray, hydro_ds: xr.Dataset):
    """Both criteria sets are applied and their results intersected."""
    result = nbs_suitability_from_thresholds(
        landuse=landuse,
        hydro_data=hydro_ds,
        lulc_classes=[1],
        elevtn_range=(200.0, 500.0),
    )
    assert result.values[0, 0] == 1  # class 1 AND elevtn 300
    assert result.values[0, 1] == 0  # class 2 not suitable


def test_nbs_combined_is_strict_intersection(
    landuse: xr.DataArray, hydro_ds: xr.Dataset
):
    """Pixels that pass lulc but fail elevation must be 0."""
    result = nbs_suitability_from_thresholds(
        landuse=landuse,
        hydro_data=hydro_ds,
        lulc_classes=[1],
        elevtn_range=(350.0, 600.0),  # all class-1 pixels are ≤ 300 m
    )
    lulc_one_mask = landuse.values == 1
    assert (result.values[lulc_one_mask] == 0).all()


def test_nbs_output_name(hydro_ds: xr.Dataset):
    """Output DataArray is always named 'nbs_suitability'."""
    result = nbs_suitability_from_thresholds(
        hydro_data=hydro_ds, elevtn_range=(0.0, 9999.0)
    )
    assert result.name == "nbs_suitability"


def test_nbs_long_name_attr_present(hydro_ds: xr.Dataset):
    """Output carries a long_name attribute describing the applied criteria."""
    result = nbs_suitability_from_thresholds(
        hydro_data=hydro_ds, elevtn_range=(0.0, 9999.0)
    )
    assert "long_name" in result.attrs


def test_nbs_long_name_records_criteria(hydro_ds: xr.Dataset):
    """The long_name attribute contains the numeric values of all applied criteria."""
    result = nbs_suitability_from_thresholds(
        hydro_data=hydro_ds,
        elevtn_range=(100.0, 400.0),
        slope_range=(0.0, 0.5),
    )
    long_name = result.attrs["long_name"]
    assert "100" in long_name
    assert "400" in long_name
    assert "0.0" in long_name
    assert "0.5" in long_name


def test_nbs_nodata_is_zero(hydro_ds: xr.Dataset):
    """Nodata is set to 0, unsuitable pixels and nodata are indistinguishable."""
    result = nbs_suitability_from_thresholds(
        hydro_data=hydro_ds, elevtn_range=(0.0, 9999.0)
    )
    assert result.raster.nodata == 0


def test_nbs_with_basin_mask_does_not_raise(
    hydro_ds: xr.Dataset, basin_mask: xr.DataArray
):
    """Providing a basin mask triggers coverage logging without raising."""
    result = nbs_suitability_from_thresholds(
        hydro_data=hydro_ds,
        elevtn_range=(0.0, 9999.0),
        basin_mask=basin_mask,
    )
    assert result is not None
