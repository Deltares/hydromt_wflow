import logging

import geopandas as gpd
import pytest
from shapely.geometry import box, mapping

from hydromt_wflow.components import WflowGeomsComponent
from hydromt_wflow.wflow import WflowModel


@pytest.fixture
def mock_xy() -> tuple[float, float]:
    """Mock coordinates in the Netherlands, suitable for EPSG:28992 and EPSG:4326."""
    return (5.3872, 52.1552)  # near Amersfoort, valid for both CRSs


@pytest.fixture
def mock_geometry(mock_xy) -> gpd.GeoDataFrame:
    x, y = mock_xy
    diff = 0.123456789  # Some small value with high precision
    return gpd.GeoDataFrame(
        geometry=[box(x - diff, y - diff, x + diff, y + diff)], crs="EPSG:4326"
    )


def test_get_success(mock_model_factory, mock_geometry):
    model: WflowModel = mock_model_factory(mode="w")
    component = WflowGeomsComponent(model=model)
    component.set(geom=mock_geometry, name="geom")

    geom = component.get("geom")

    assert isinstance(geom, gpd.GeoDataFrame)
    assert geom.equals(mock_geometry), "Retrieved geometry does not match the original."


def test_get_failure(mock_model_factory):
    model: WflowModel = mock_model_factory(mode="w")
    component = WflowGeomsComponent(model=model)

    with pytest.raises(KeyError) as excinfo:
        component.get("not_a_geom")
    assert "Geometry 'not_a_geom' not found in geoms." in str(excinfo.value)


def test_pop_failure(mock_model_factory, caplog):
    model: WflowModel = mock_model_factory(mode="w")
    component = WflowGeomsComponent(model=model)
    caplog.set_level(logging.WARNING)

    with pytest.raises(KeyError) as excinfo:
        component.pop("not_a_geom")

    assert "Geometry 'not_a_geom' not found in geoms." in str(excinfo.value)


def test_pop_success(mock_model_factory, mock_geometry, caplog):
    model: WflowModel = mock_model_factory(mode="w")
    component = WflowGeomsComponent(model=model)
    caplog.at_level(logging.INFO)
    component.set(geom=mock_geometry, name="geom")

    geom = component.pop("geom")

    assert "Removed geometry 'geom' from geoms."
    assert isinstance(geom, gpd.GeoDataFrame)
    assert component.data == {}


def test_set_write_read(
    mock_model_factory,
    mock_geometry,
):
    # Initialize component
    model: WflowModel = mock_model_factory(mode="w")
    comp = WflowGeomsComponent(model=model)

    # Add geometry and write it to disk
    comp.set(mock_geometry, name="test_geom")
    comp.write(dir_out=model.root.path)

    # Confirm file was written
    out_file = model.root.path / "test_geom.geojson"
    assert out_file.exists()

    # Create new instance and read
    model: WflowModel = mock_model_factory(path=model.root.path, mode="r")
    new_comp = WflowGeomsComponent(model=model)
    new_comp.read(filename=str(out_file.with_name("{name}.geojson")))

    # Check that the geometry matches
    gdf_read = new_comp.get("test_geom")
    assert gdf_read is not None
    assert gdf_read.geom_equals_exact(mock_geometry, tolerance=1e-7).all(), (
        "Read geometry does not match the original geometry."
    )


def test_read_with_pattern(mock_model_factory, mock_geometry):
    # Write multiple geometries to disk
    model: WflowModel = mock_model_factory(mode="w")
    comp = WflowGeomsComponent(model=model)
    comp.set(mock_geometry, name="geom1")
    comp.set(mock_geometry, name="geom2")
    comp.write(dir_out=model.root.path)

    # Confirm files were written
    outfiles = [model.root.path / f"{name}.geojson" for name in ["geom1", "geom2"]]
    for out_file in outfiles:
        assert out_file.exists(), f"File {out_file} was not created."

    # Read using a pattern
    model: WflowModel = mock_model_factory(model.root.path, mode="r")
    new_comp = WflowGeomsComponent(model=model)
    new_comp.read(filename=str(model.root.path / "{name}.geojson"))

    # Check that both geometries are read
    assert new_comp.get("geom1") is not None
    assert new_comp.get("geom2") is not None


def test_write_to_wgs84(mock_model_factory, mock_geometry):
    # Initialize component
    model: WflowModel = mock_model_factory(mode="w")
    comp = WflowGeomsComponent(model=model)
    geom = mock_geometry.to_crs("EPSG:28992")

    # Add geometry and write it to disk in WGS84
    comp.set(geom, name="test_geom")
    comp.write(dir_out=model.root.path, to_wgs84=True)

    # Confirm file was written
    out_file = model.root.path / "test_geom.geojson"
    assert out_file.exists()

    # Read back the geometry
    gdf_read = gpd.read_file(out_file)

    # Check if CRS is WGS84
    assert gdf_read.crs == "EPSG:4326", "Geometry was not written in WGS84."


def check_precision(coord, precision: int, tolerance: float):
    if isinstance(coord[0], (float, int)):
        for val in coord:
            rounded = round(val, precision)
            assert abs(val - rounded) < tolerance, (
                f"Value {val} exceeds tolerance {tolerance} for precision {precision}"  # noqa: E501
            )
    else:
        for sub_coord in coord:
            check_precision(sub_coord, precision, tolerance)


@pytest.mark.parametrize(
    ("crs", "expected_precision"), [("EPSG:4326", 6), ("EPSG:28992", 1)]
)
def test_write_precision_defaults(
    crs: str,
    expected_precision: int,
    mock_geometry: gpd.GeoDataFrame,
    mock_model_factory,
):
    # Convert mock geometry to the target CRS
    geometry = mock_geometry.to_crs(crs)

    # Initialize, write and read geometry
    model: WflowModel = mock_model_factory(mode="w")
    comp = WflowGeomsComponent(model=model)
    comp.set(geometry, name="test_geom")
    comp.write(dir_out=model.root.path)
    out_file = model.root.path / "test_geom.geojson"
    assert out_file.exists()
    gdf_read = gpd.read_file(out_file)

    # Set allowable tolerance based on expected precision
    tolerance = 10 ** (-expected_precision)
    for geom in gdf_read.geometry:
        coords = mapping(geom)["coordinates"]
        check_precision(coords, expected_precision, tolerance)


@pytest.mark.parametrize(
    "precision",
    [
        1,
        3,
        6,
        8,
        10,
    ],
)
def test_write_precision_manual(
    precision: int,
    mock_geometry: gpd.GeoDataFrame,
    mock_model_factory,
):
    # Initialize, write and read geometry
    model: WflowModel = mock_model_factory(mode="w")
    comp = WflowGeomsComponent(model=model)
    comp.set(mock_geometry, name="test_geom")
    comp.write(dir_out=model.root.path, precision=precision)
    out_file = model.root.path / "test_geom.geojson"
    assert out_file.exists()
    gdf_read = gpd.read_file(out_file)

    # Set allowable tolerance based on expected precision
    tolerance = 10 ** (-precision)

    for geom in gdf_read.geometry:
        coords = mapping(geom)["coordinates"]
        check_precision(coords, precision, tolerance)
