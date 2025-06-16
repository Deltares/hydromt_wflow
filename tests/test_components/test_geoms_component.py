import logging
from pathlib import Path

import geopandas as gpd
import hydromt
import pytest
from shapely.geometry import box

from hydromt_wflow import WflowModel
from hydromt_wflow.components import WflowGeomsComponent


@pytest.fixture
def mock_wflow_model(
    mock_model: WflowModel, mock_rasterdataset: hydromt.DataCatalog
) -> WflowModel:
    mock_model.data_catalog.get_rasterdataset.return_value = mock_rasterdataset
    return mock_model


@pytest.fixture
def mock_xy() -> tuple[float, float]:
    """Mock coordinates for testing."""
    return (12.2051, 45.8331)


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
