from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from hydromt_wflow.workflows.soilgrids import (
    air_entry_pressure,
    do_curve_fit,
    soilgrids,
)


def generate_data(b: float = 0.2, n: int = 50):
    x = np.linspace(0, 10, n)
    y = np.exp(-b * x)
    return x, y, b


@pytest.mark.parametrize("b_true", [0.05, 0.1, 0.2, 0.5])
def test_curve_fit_recovers_parameter(b_true):
    x, y, _ = generate_data(b_true)

    b_est = do_curve_fit(x, y)

    assert np.isclose(b_est, b_true, atol=1e-3)


@pytest.mark.parametrize(
    "y_mod",
    [
        lambda y: y,  # clean data
        lambda y: np.where(np.arange(len(y)) == 3, np.nan, y),
        lambda y: np.where(np.arange(len(y)) == 5, -1.0, y),
        lambda y: np.where(np.arange(len(y)) == 7, np.inf, y),
    ],
)
def test_invalid_values_are_filtered(y_mod):
    x, y, b_true = generate_data(0.3)
    y = y_mod(y)

    b_est = do_curve_fit(x, y)

    assert np.isfinite(b_est)
    assert np.isclose(b_est, b_true, atol=1e-3)


def test_returns_nan_if_no_valid_data():
    x = np.arange(5)
    y = np.array([-1, np.nan, -5, np.inf, -3])

    result = do_curve_fit(x, y)

    assert np.isnan(result)


def test_lstsq_fallback_when_curve_fit_fails():
    x, y, b_true = generate_data(0.15)

    with patch("hydromt_wflow.workflows.soilgrids.curve_fit", side_effect=RuntimeError):
        b_est = do_curve_fit(x, y)

    assert np.isclose(b_est, b_true, atol=1e-3)


def test_air_entry_pressure():
    coords = {"sl": [1], "y": [0.0], "x": [0.0]}
    ds = xr.Dataset(
        {
            "clyppt": (("sl", "y", "x"), [[[20.0]]]),
            "sndppt": (("sl", "y", "x"), [[[40.0]]]),
        },
        coords=coords,
    )
    ds.raster.set_spatial_dims(x_dim="x", y_dim="y")
    thetas = xr.DataArray(
        [[[0.45]]],
        dims=("sl", "y", "x"),
        coords=coords,
    )
    thetas.raster.set_spatial_dims(x_dim="x", y_dim="y")

    result = air_entry_pressure(ds, thetas, "brakensiek")

    assert result.name == "soil_hb"
    assert np.isnan(result.raster.nodata)
    assert np.isclose(result.values[0, 0, 0], -26.57336805)


def test_soilgrids_rejects_unknown_air_entry_pressure_ptf():
    with pytest.raises(ValueError, match="expected 'brakensiek', 'clapp', or None"):
        soilgrids(xr.Dataset(), xr.Dataset(), ptf_hb="unknown")
