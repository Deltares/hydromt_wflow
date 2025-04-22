import logging
from pathlib import Path
from unittest.mock import PropertyMock

import geopandas as gpd
from hydromt.model import ModelRoot
from pyproj.crs import CRS
from shapely.geometry import MultiPolygon, Polygon

from hydromt_wflow.components import RegionComponent


def test_region_component_empty(mock_model):
    # Setup the component with the mock model
    component = RegionComponent(model=mock_model)

    # assert that it is empty
    assert len(component.data) == 0
    assert component._filename == "region.geojson"


def test_region_component_set(build_region_gdf, build_region_3857_gdf, mock_model):
    # Setup the component with the mock model
    component = RegionComponent(model=mock_model)

    # Add a geometry
    component.set(build_region_gdf)

    # Assert that there is data
    assert "region" in component.data
    assert len(component.data) == 1
    assert component.region is not None
    assert len(component.region.columns) == 1

    # Empty the component and assert that crs is adjusted based on the model
    component._data = {}
    assert build_region_3857_gdf.crs.to_epsg() == 3857
    component.set(build_region_3857_gdf)
    assert component.region.crs.to_epsg() == 4326

    # Assert that a GeoSeries is sufficient as input
    component._data = {}
    assert component.region is None
    component.set(build_region_gdf.geometry)
    assert component.region is not None


def test_region_component_append(box_geometry, build_region_gdf, mock_model):
    # Setup the component with the mock model
    component = RegionComponent(model=mock_model)
    component.set(build_region_gdf)
    assert isinstance(component.region.geometry[0], Polygon)

    # Add a polygon that will enter a union with the current region
    component.set(box_geometry)

    # Assert
    assert len(component.data) == 1
    assert isinstance(component.region.geometry[0], MultiPolygon)


def test_region_component_read(tmp_path, build_region_gdf, mock_model):
    # Setup the component
    type(mock_model).root = PropertyMock(
        side_effect=lambda: ModelRoot(tmp_path, mode="r"),
    )
    component = RegionComponent(model=mock_model)
    component.read()
    assert len(component.data) == 0
    assert component.region is None

    # Write the region gdf to the tmp directory
    build_region_gdf.to_file(Path(tmp_path, "region.geojson"))

    # Re-read
    component.read()
    assert len(component.data) == 1
    assert component.region is not None


def test_region_component_write_empty(mock_model, caplog):
    caplog.set_level(logging.DEBUG)

    # Setup component and a region
    component = RegionComponent(model=mock_model)

    # Empty write
    component.write()
    assert "No geoms data found, skip writing." in caplog.text

    # Write empty region GeoDataFrame
    component._data = {"region": gpd.GeoDataFrame()}
    component.write()
    assert "region is empty. Skipping..." in caplog.text


def test_region_component_write_default(tmp_path, build_region_gdf, mock_model):
    # Setup the component
    component = RegionComponent(model=mock_model)

    # Set something in the component
    component.set(build_region_gdf)
    # Write the data
    component.write()
    assert Path(tmp_path, "region.geojson").is_file()

    # Write to separate directory
    component.write(filename="geom/region.geojson")
    assert Path(tmp_path, "geom").is_dir()
    component = None


def test_region_component_write_crs(tmp_path, build_region_3857_gdf, mock_model):
    # Create new component
    # Adjust the model crs to test the write capabilities
    type(mock_model).crs = PropertyMock(side_effect=lambda: CRS.from_epsg(3857))
    component = RegionComponent(model=mock_model)

    # Set the component with a geometry in a local projection
    component.set(build_region_3857_gdf)
    assert component.region.crs.to_epsg() == 3857

    # Write and assert output
    component.write()
    gdf = gpd.read_file(Path(tmp_path, "region.geojson"))
    assert gdf.crs.to_epsg() == 3857
    gdf = None
    # Check that the to_wgs84 works
    component.write(to_wgs84=True)
    gdf = gpd.read_file(Path(tmp_path, "region.geojson"))
    assert gdf.crs.to_epsg() == 4326
