import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hydromt.gis import flw
from shapely.geometry import LineString

from hydromt_wflow.workflows import connect


@pytest.fixture
def ds_model():
    """Fixture for a 3x6 model where every row flows east to a pit in column 4."""
    coords = {"y": [2.5, 1.5, 0.5], "x": np.arange(6) + 0.5}
    # ldd: 6 is east, 5 is a pit and 255 is nodata (column 5 is outside the basin)
    flwdir = np.full((3, 6), 6, dtype=np.uint8)
    flwdir[:, 4] = 5
    flwdir[:, 5] = 255
    basins = np.ones((3, 6), dtype=np.int32)
    basins[:, 5] = 0
    # the river is the middle row from column 1 to the pit
    rivmsk = np.zeros((3, 6), dtype=np.int32)
    rivmsk[1, 1:5] = 1
    ds = xr.Dataset(
        {
            name: xr.DataArray(data, dims=("y", "x"), coords=coords)
            for name, data in [
                ("flwdir", flwdir),
                ("basins", basins),
                ("rivmsk", rivmsk),
            ]
        }
    )
    ds["flwdir"].attrs["_FillValue"] = 255
    ds.raster.set_crs("EPSG:4326")
    return ds


@pytest.fixture
def flwdir(ds_model):
    """Fixture for the flow direction raster of the model."""
    return flw.flwdir_from_da(ds_model["flwdir"])


@pytest.fixture
def gdf_points():
    """Fixture for two nodes in the middle row, the first one is next to the river."""
    return gpd.GeoDataFrame(
        {"node_id": [10, 20]},
        geometry=gpd.points_from_xy([0.5, 3.5], [1.5, 1.5]),
        crs="EPSG:4326",
    )


def test_derive_river_nodes():
    """Test that the unique start and end points of the river lines are the nodes."""
    # two lines that share a point
    gdf_riv = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        crs="EPSG:4326",
    )

    nodes = connect.derive_river_nodes(gdf_riv)

    assert list(nodes.index) == [1, 2, 3]
    assert list(nodes.geometry.x) == [0, 1, 2]


def test_derive_point_nodes(ds_model, gdf_points):
    """Test that the node ids are taken from id_column or numbered from 1."""
    nodes = connect.derive_point_nodes(ds_model, gdf_points, id_column="node_id")
    nodes_numbered = connect.derive_point_nodes(ds_model, gdf_points)

    assert list(nodes.index) == [10, 20]
    assert list(nodes_numbered.index) == [1, 2]


def test_derive_point_nodes_outside_basin(ds_model, gdf_points):
    """Test that nodes outside the basin or the model grid are skipped."""
    # column 5 is outside the basin and x = 10.5 is outside the grid
    outside = gpd.GeoDataFrame(
        {"node_id": [30, 40]},
        geometry=gpd.points_from_xy([5.5, 10.5], [1.5, 1.5]),
        crs="EPSG:4326",
    )

    nodes = connect.derive_point_nodes(
        ds_model, pd.concat([gdf_points, outside]), id_column="node_id"
    )
    assert list(nodes.index) == [10, 20]

    with pytest.raises(ValueError, match="None of the nodes are within"):
        connect.derive_point_nodes(ds_model, outside, id_column="node_id")


@pytest.mark.parametrize("node_ids", [[1, 1], [0, 1], [1.5, 2.5]])
def test_derive_point_nodes_invalid_ids(ds_model, gdf_points, node_ids):
    """Test that node ids must be unique positive integers."""
    gdf_points["node_id"] = node_ids

    with pytest.raises(ValueError, match="unique positive integers"):
        connect.derive_point_nodes(ds_model, gdf_points, id_column="node_id")


def test_connect_nodes(ds_model, flwdir, gdf_points):
    """Test that the nodes are snapped to the river and get their own subbasin."""
    nodes = connect.derive_point_nodes(ds_model, gdf_points, id_column="node_id")

    da_subbasins, gdf_snapped = connect.connect_nodes(ds_model, nodes, flwdir, flwdir)

    # node 10 is snapped downstream to the first river cell
    assert list(gdf_snapped.index) == [10, 20]
    assert list(gdf_snapped.geometry.x) == [1.5, 3.5]
    # node 20 only gets the area that is not drained by the upstream node 10
    assert (da_subbasins == 10).sum() == 2
    assert (da_subbasins == 20).sum() == 2
    assert da_subbasins.raster.crs == ds_model.raster.crs


def test_connect_nodes_unique_cells(ds_model, flwdir, gdf_points):
    """Test that nodes snapped to the same river cell can be rejected."""
    gdf_points.geometry = gpd.points_from_xy([3.5, 3.5], [1.5, 1.5])
    nodes = connect.derive_point_nodes(ds_model, gdf_points, id_column="node_id")

    connect.connect_nodes(ds_model, nodes, flwdir, flwdir)

    with pytest.raises(ValueError, match="snapped to the same wflow river cell"):
        connect.connect_nodes(ds_model, nodes, flwdir, flwdir, unique_cells=True)
