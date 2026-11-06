import numpy as np
import pytest

from hydromt_wflow.workflows.ptf import (
    air_entry_pressure_brakensiek,
    air_entry_pressure_clapp_hornberger,
)


@pytest.mark.parametrize(
    ("clay", "sand", "expected"),
    [
        (50.0, 30.0, -40.5),  # clay
        (45.0, 10.0, -49.0),  # silty clay
        (30.0, 30.0, -63.0),  # clay loam
        (40.0, 50.0, -15.3),  # sandy clay
        (30.0, 10.0, -35.6),  # silty clay loam
        (5.0, 10.0, -78.6),  # silt
        (15.0, 40.0, -47.8),  # loam
        (25.0, 55.0, -29.9),  # sandy clay loam
        (10.0, 60.0, -21.8),  # sandy loam
        (5.0, 80.0, -9.0),  # loamy sand
        (2.0, 95.0, -12.1),  # sand
    ],
)
def test_air_entry_pressure_clapp_hornberger(clay, sand, expected):
    assert air_entry_pressure_clapp_hornberger(clay, sand) == expected


@pytest.mark.parametrize(
    ("clay", "sand"),
    [(np.nan, 40.0), (20.0, np.nan)],
)
def test_air_entry_pressure_clapp_hornberger_propagates_nan(clay, sand):
    assert np.isnan(air_entry_pressure_clapp_hornberger(clay, sand))


def test_air_entry_pressure_brakensiek():
    result = air_entry_pressure_brakensiek(20.0, 40.0, 0.45)

    assert np.isclose(result, -26.57336805)


@pytest.mark.parametrize(
    ("clay", "sand"),
    [(5.0, 40.0), (60.0, 20.0), (20.0, 5.0), (20.0, 70.0)],
)
def test_air_entry_pressure_brakensiek_falls_back_at_boundaries(clay, sand):
    expected = air_entry_pressure_clapp_hornberger(clay, sand)

    assert air_entry_pressure_brakensiek(clay, sand, 0.45) == expected


def test_air_entry_pressure_brakensiek_propagates_nan():
    assert np.isnan(air_entry_pressure_brakensiek(20.0, 40.0, np.nan))
