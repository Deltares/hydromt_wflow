"""Tests for the Wflow standard names in naming.py and version_upgrade.py."""

import pytest

from hydromt_wflow import naming, version_upgrade
from hydromt_wflow.naming import WFLOW_NAMES

_NAME_TABLES = {
    "WFLOW_NAMES": naming.WFLOW_NAMES,
    "WFLOW_STATES_NAMES": naming.WFLOW_STATES_NAMES,
    "WFLOW_SEDIMENT_NAMES": naming.WFLOW_SEDIMENT_NAMES,
    "WFLOW_SEDIMENT_STATES_NAMES": naming.WFLOW_SEDIMENT_STATES_NAMES,
}

_UPGRADE_TABLES = {
    "_V0_TO_V1_INPUT_SBM": version_upgrade._V0_TO_V1_INPUT_SBM,
    "_V0_TO_V1_STATES_SBM": version_upgrade._V0_TO_V1_STATES_SBM,
    "_V0_TO_V1_INPUT_SEDIMENT": version_upgrade._V0_TO_V1_INPUT_SEDIMENT,
    "_V0_TO_V1_STATES_SEDIMENT": version_upgrade._V0_TO_V1_STATES_SEDIMENT,
}


@pytest.mark.parametrize("table", sorted(_NAME_TABLES))
def test_wflow_names_have_no_surrounding_whitespace(table):
    bad = {
        key: entry["wflow_name"]
        for key, entry in _NAME_TABLES[table].items()
        if entry.get("wflow_name")
        and entry["wflow_name"] != entry["wflow_name"].strip()
    }
    assert bad == {}


@pytest.mark.parametrize("table", sorted(_UPGRADE_TABLES))
def test_upgraded_names_have_no_surrounding_whitespace(table):
    bad = {
        key: name
        for key, name in _UPGRADE_TABLES[table].items()
        if isinstance(name, str) and name != name.strip()
    }
    assert bad == {}


def test_pits_standard_name():
    assert WFLOW_NAMES["pits"]["wflow_name"] == "basin_pit_location__mask"
    assert version_upgrade._V0_TO_V1_INPUT_SBM["pits"] == "basin_pit_location__mask"
