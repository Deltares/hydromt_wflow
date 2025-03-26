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
        lulc_vars=["USLE_C"],
        planted_forest_c=0.0881,
        orchard_name="Orchard",
        orchard_c=0.2188,
    )
    da = example_sediment_model.grid["USLE_C"].raster.sample(
        planted_forest_testdata.geometry.centroid
    )
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
    assert int(mean_val * 1000000) == 22215

    example_sediment_model.setup_soilmaps(
        soil_fn="soilgrids",
        usleK_method="epic",
        add_aggregates=False,
    )
    da = example_sediment_model.grid

    values = da["usle_k"].raster.mask_nodata()
    mean_val = values.mean().values
    assert np.isclose(mean_val , 0.031182)

    assert "d50_soil" in da
    assert "fclay_soil" in da

    soil_composition = da["fclay_soil"] + da["fsilt_soil"] + da["fsand_soil"]
    assert np.all(np.isclose(soil_composition.values, 1))
