import pytest

import hydromt_wflow.workflows as workflows


def test_validate_lulc_vars_wrong_values():
    with pytest.raises(ValueError, match="Invalid lulc_vars: {'invalid_var'}"):
        workflows.validate_lulc_vars(["landuse", "invalid_var"])


def test_validate_lulc_vars_correct_not_all_values():
    # Should not raise
    workflows.validate_lulc_vars(["landuse", "erosion_usle_c"])


def test_validate_lulc_vars_all_correct_values():
    workflows.validate_lulc_vars(workflows.LULC_VARS_MAPPING.keys())
