"""Unit tests for hydromt_wflow sediment methods and workflows."""

from os.path import abspath, dirname, join

import numpy as np

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")


def test_setup_lulc_sed(example_sediment_model, planted_forest_testdata):
    example_sediment_model.setup_lulcmaps(
        lulc_fn="globcover_2009",
        lulc_mapping_fn="globcover_mapping_default",
        planted_forest_fn=planted_forest_testdata,
        lulc_vars={"USLE_C": "soil_erosion__usle_c_factor"},
        planted_forest_c=0.0881,
        orchard_name="Orchard",
        orchard_c=0.2188,
    )
    da = example_sediment_model.grid["USLE_C"].raster.sample(
        planted_forest_testdata.geometry.centroid
    )

    # Strict equality checking is okay here because no processing is actually happening
    # and we want to make sure we don't add any roudning errors
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
    assert "USLE_C" in example_sediment_model.grid


def test_setup_soilmaps_sed(
    example_sediment_model,
):
    values = example_sediment_model.grid["usle_k"].raster.mask_nodata()
    mean_val = values.mean().values
    assert np.isclose(mean_val, 0.022215, atol=1e-6)

    example_sediment_model.setup_soilmaps(
        soil_fn="soilgrids",
        usleK_method="epic",
        add_aggregates=False,
    )
    da = example_sediment_model.grid

    values = da["usle_k"].raster.mask_nodata()
    mean_val = values.mean().values
    assert np.isclose(mean_val, 0.0307964, atol=1e-6)

    assert "d50_soil" in da
    assert "fclay_soil" in da

    soil_composition = (
        da["fclay_soil"].raster.mask_nodata()
        + da["fsilt_soil"].raster.mask_nodata()
        + da["fsand_soil"].raster.mask_nodata()
    )
    mask = ~np.isnan(soil_composition.values)
    assert np.allclose(soil_composition.values[mask], 1, atol=1e-6)
