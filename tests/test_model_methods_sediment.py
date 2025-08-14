"""Unit tests for hydromt_wflow sediment methods and workflows."""

from os.path import abspath, dirname, join

import geopandas as gpd
import numpy as np

from hydromt_wflow.utils import planar_operation_in_utm

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")


def test_setup_lulc_sed(
    example_sediment_model, planted_forest_testdata: gpd.GeoDataFrame
):
    example_sediment_model.setup_lulcmaps(
        lulc_fn="globcover_2009",
        lulc_mapping_fn="globcover_mapping_default",
        planted_forest_fn=planted_forest_testdata,
        lulc_vars={"erosion_usle_c": "soil_erosion__usle_c_factor"},
        planted_forest_c=0.0881,
        orchard_name="Orchard",
        orchard_c=0.2188,
    )

    centroid = planar_operation_in_utm(
        planted_forest_testdata, lambda geom: geom.centroid
    )
    da = example_sediment_model.staticmaps.data["erosion_usle_c"].raster.sample(
        centroid
    )

    # Strict equality checking is okay here because no processing is actually happening
    # and we want to make sure we don't add any rounding errors
    assert np.all(da.values == np.array([0.0881, 0.2188]))


def test_setup_lulc_vector(
    example_sediment_model,
    globcover_gdf,
    planted_forest_testdata,
):
    # Test for sediment model
    example_sediment_model.setup_lulcmaps_from_vector(
        lulc_fn=globcover_gdf,
        lulc_mapping_fn="globcover_mapping_default",
        planted_forest_fn=planted_forest_testdata,
        lulc_res=None,
        save_raster_lulc=False,
        planted_forest_c=0.0881,
    )
    assert "erosion_usle_c" in example_sediment_model.staticmaps.data


def test_setup_soilmaps_sed(
    example_sediment_model,
):
    values = example_sediment_model.staticmaps.data[
        "erosion_usle_k"
    ].raster.mask_nodata()
    mean_val = values.mean().values
    assert np.isclose(mean_val, 0.022215, atol=1e-6)

    example_sediment_model.setup_soilmaps(
        soil_fn="soilgrids",
        usle_k_method="epic",
        add_aggregates=False,
    )
    da = example_sediment_model.staticmaps.data

    values = da["erosion_usle_k"].raster.mask_nodata()
    mean_val = values.mean().values
    assert np.isclose(mean_val, 0.0307964, atol=1e-6)

    assert "soil_sediment_d50" in da
    assert "soil_clay_fraction" in da

    soil_composition = (
        da["soil_clay_fraction"].raster.mask_nodata()
        + da["soil_silt_fraction"].raster.mask_nodata()
        + da["soil_sand_fraction"].raster.mask_nodata()
    )
    mask = ~np.isnan(soil_composition.values)
    assert np.allclose(soil_composition.values[mask], 1, atol=1e-6)
