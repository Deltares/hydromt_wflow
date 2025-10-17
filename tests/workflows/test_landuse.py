from hydromt_wflow.workflows.landuse import _get_resample_method


def test_get_resample_method():
    """Test _get_resample_method function."""
    # Default
    assert _get_resample_method("landuse", None) == "nearest"
    assert _get_resample_method("lai", None) == "average"
    assert _get_resample_method("vegetation_feddes_alpha_h1", None) == "mode"
    assert _get_resample_method("other", None) == "average"

    # Custom
    resample_method = {
        "landuse": "mode",
        "lai": "nearest",
        "vegetation_feddes_alpha_h1": "nearest",
        "other": "mode",
    }
    assert _get_resample_method("landuse", resample_method) == "mode"
    assert _get_resample_method("lai", resample_method) == "nearest"
    assert (
        _get_resample_method("vegetation_feddes_alpha_h1", resample_method) == "nearest"
    )
    assert _get_resample_method("other", resample_method) == "mode"
    assert _get_resample_method("unknown", resample_method) == "average"
