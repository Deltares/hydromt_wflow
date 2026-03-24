"""Tests for WflowSbmModel.setup_temp_pet_forcing."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hydromt.error import NoDataException, NoDataStrategy

from hydromt_wflow.wflow_sbm import WflowSbmModel


def _make_time_index(n=3, freq="D"):
    return pd.date_range("2020-01-01", periods=n, freq=freq)


def _make_spatial_coords():
    return {
        "y": ("y", [52.0, 52.1]),
        "x": ("x", [4.0, 4.1]),
    }


def _make_var(name, time_index=None, value=1.0):
    """Return a trivial DataArray for use in a forcing Dataset."""
    if time_index is None:
        time_index = _make_time_index()
    coords = {"time": time_index, **_make_spatial_coords()}
    data = np.full((len(time_index), 2, 2), value, dtype="float32")
    return xr.DataArray(data, coords=coords, dims=["time", "y", "x"], name=name)


def _make_ds(*var_names, freq="D"):
    """Build a minimal Dataset containing the named variables."""
    time_index = _make_time_index(freq=freq)
    return xr.Dataset({name: _make_var(name, time_index) for name in var_names})


def _make_model(ds_override=None):
    """Return a WflowSbmModel with all heavy dependencies mocked out.

    Only the data_catalog, config, staticmaps, forcing, and region are set up.
    """
    model = MagicMock(spec=WflowSbmModel)

    # --- config ---
    model.config.get_value.side_effect = lambda key: {
        "time.starttime": "2020-01-01T00:00:00",
        "time.endtime": "2020-01-03T00:00:00",
        "time.timestepsecs": 86400,
    }[key]

    # --- staticmaps ---
    elevtn = _make_var("elevtn").isel(time=0).drop_vars("time")
    basins = xr.ones_like(elevtn).rename("basins")
    model._MAPS = {
        "basins": "wflow_subcatch",
        "elevtn": "wflow_dem",
        "pet": "pet",
        "temp": "temp",
    }
    # Make _MAPS["basins"] and _MAPS["elevtn"] work
    model.staticmaps.data = {
        "wflow_subcatch": basins,
        "wflow_dem": elevtn,
    }

    # --- region ---
    model.region = MagicMock()
    model.data_catalog = MagicMock()
    model.forcing = MagicMock()

    # --- data_catalog ---
    if ds_override is not None:
        model.data_catalog.get_rasterdataset.return_value = ds_override

    return model


# ---------------------------------------------------------------------------
# Fixtures for typical complete datasets per method
# ---------------------------------------------------------------------------


@pytest.fixture
def ds_debruin():
    return _make_ds("temp", "press_msl", "kin", "kout", "wind")


@pytest.fixture
def ds_makkink():
    return _make_ds("temp", "press_msl", "kin", "wind")


@pytest.fixture
def ds_pm_rh_simple():
    return _make_ds("temp", "temp_min", "temp_max", "rh", "kin", "wind")


@pytest.fixture
def ds_pm_tdew():
    return _make_ds(
        "temp", "temp_min", "temp_max", "temp_dew", "kin", "press_msl", "wind"
    )


class TestSkipPet:
    """When skip_pet=True the method should only need 'temp' (+ a wind var)."""

    def test_skip_pet_selects_only_temp(self, ds_debruin):
        """
        Variables list should be built as ['temp'] when skip_pet=True.

        No pet-specific variables (press_msl, kin, kout) should be required.
        """
        # A dataset with only temp + wind is sufficient
        ds_minimal = _make_ds("temp", "wind")
        model = _make_model(ds_override=ds_minimal)

        # We patch the downstream processing so we can isolate variable selection
        with (
            patch("hydromt.model.processes.meteo.temp", return_value=_make_var("temp")),
            patch(
                "hydromt.model.processes.meteo.resample_time",
                return_value=_make_var("temp"),
            ),
        ):
            WflowSbmModel.setup_temp_pet_forcing(
                model,
                temp_pet_fn="dummy_source",
                skip_pet=True,
            )

        # pet should NOT have been written
        pet_calls = [
            c
            for c in model.forcing.set.call_args_list
            if c.kwargs.get("name") == "pet" or (c.args and "pet" in str(c.args[-1]))
        ]
        assert len(pet_calls) == 0, "pet should not be set when skip_pet=True"

        # temp SHOULD have been written
        temp_calls = [c for c in model.forcing.set.call_args_list if "temp" in str(c)]
        assert len(temp_calls) > 0

    def test_skip_pet_does_not_require_kin(self):
        """Dataset without 'kin' should succeed when skip_pet=True."""
        ds_no_kin = _make_ds("temp", "wind")  # no kin, kout, press_msl
        model = _make_model(ds_override=ds_no_kin)

        with (
            patch("hydromt.model.processes.meteo.temp", return_value=_make_var("temp")),
            patch(
                "hydromt.model.processes.meteo.resample_time",
                return_value=_make_var("temp"),
            ),
        ):
            # Should NOT raise due to missing kin/kout
            WflowSbmModel.setup_temp_pet_forcing(
                model,
                temp_pet_fn="dummy_source",
                skip_pet=True,
            )


class TestPetMethodVariableSelection:
    """Variable selection: known pet_method → correct variable set chosen."""

    @pytest.mark.parametrize(
        ("pet_method", "expected_vars"),
        [
            ("debruin", ["temp", "press_msl", "kin", "kout"]),
            ("makkink", ["temp", "press_msl", "kin"]),
            (
                "penman-monteith_rh_simple",
                ["temp", "temp_min", "temp_max", "rh", "kin"],
            ),
            (
                "penman-monteith_tdew",
                ["temp", "temp_min", "temp_max", "temp_dew", "kin", "press_msl"],
            ),
        ],
    )
    def test_method_requires_correct_vars(self, pet_method, expected_vars):
        """For a given pet_method, the expected variables should be required."""
        # Build a dataset that has wind but is intentionally missing the last
        # required variable for this method.
        missing_var = expected_vars[-1]
        present_vars = [v for v in expected_vars if v != missing_var] + ["wind"]
        ds_incomplete = _make_ds(*present_vars)
        model = _make_model(ds_override=ds_incomplete)

        with pytest.raises(NoDataException):
            WflowSbmModel.setup_temp_pet_forcing(
                model,
                temp_pet_fn="dummy_source",
                pet_method=pet_method,
                nodata_strategy=NoDataStrategy.RAISE,
            )

    def test_unknown_pet_method_raises_value_error(self):
        """An unrecognised pet_method should raise ValueError immediately."""
        ds = _make_ds("temp", "wind")
        model = _make_model(ds_override=ds)

        with pytest.raises(ValueError, match="Unknown pet_method"):
            WflowSbmModel.setup_temp_pet_forcing(
                model,
                temp_pet_fn="dummy_source",
                pet_method="nonexistent_method",
            )


class TestWindVariableResolution:
    """Logic for selecting wind variable(s) when both total and components are present."""

    def test_total_wind_selected_when_present(self, ds_debruin):
        """If 'wind' is present, it should be added to the variables list."""
        assert "wind" in ds_debruin.data_vars
        # When all required vars including 'wind' are present, no error should fire
        model = _make_model(ds_override=ds_debruin)

        with (
            patch("hydromt.model.processes.meteo.temp", return_value=_make_var("temp")),
            patch("hydromt.model.processes.meteo.pet", return_value=_make_var("pet")),
            patch(
                "hydromt.model.processes.meteo.resample_time",
                return_value=_make_var("temp"),
            ),
        ):
            WflowSbmModel.setup_temp_pet_forcing(
                model,
                temp_pet_fn="dummy_source",
                pet_method="debruin",
            )

    def test_wind_components_accepted_when_no_total_wind(self):
        """If 'wind' is absent but 'wind10_u' and 'wind10_v' are present, no error."""
        ds = _make_ds("temp", "press_msl", "kin", "kout", "wind10_u", "wind10_v")
        model = _make_model(ds_override=ds)

        with (
            patch("hydromt.model.processes.meteo.temp", return_value=_make_var("temp")),
            patch("hydromt.model.processes.meteo.pet", return_value=_make_var("pet")),
            patch(
                "hydromt.model.processes.meteo.resample_time",
                return_value=_make_var("temp"),
            ),
        ):
            WflowSbmModel.setup_temp_pet_forcing(
                model,
                temp_pet_fn="dummy_source",
                pet_method="debruin",
            )

    def test_no_wind_at_all_raises_with_nodata_raise(self):
        """If neither 'wind' nor wind components are present, strategy=RAISE errors."""
        ds = _make_ds("temp", "press_msl", "kin", "kout")  # no wind at all
        model = _make_model(ds_override=ds)

        with pytest.raises(NoDataException, match="wind"):
            WflowSbmModel.setup_temp_pet_forcing(
                model,
                temp_pet_fn="dummy_source",
                pet_method="debruin",
                nodata_strategy=NoDataStrategy.RAISE,
            )

    def test_no_wind_warns_with_nodata_warn(self, caplog: pytest.LogCaptureFixture):
        """If neither wind variable is found and strategy=WARN, no exception is raised."""
        ds = _make_ds("temp", "press_msl", "kin", "kout")  # no wind
        model = _make_model(ds_override=ds)

        # Should return early (returns None) without raising
        with caplog.at_level("WARNING"):
            result = WflowSbmModel.setup_temp_pet_forcing(
                model,
                temp_pet_fn="dummy_source",
                pet_method="debruin",
                nodata_strategy=NoDataStrategy.WARN,
            )
            assert result is None
            assert "not find wind variables" in caplog.text, (
                "Expected warning about missing wind variable"
            )

    def test_wind_total_takes_priority_over_components(self):
        """When both 'wind' and wind components exist, 'wind' should be preferred."""
        ds = _make_ds(
            "temp",
            "press_msl",
            "kin",
            "kout",
            "wind",
            "wind10_u",
            "wind10_v",
        )
        model = _make_model(ds_override=ds)

        # Capture what variables ends up being selected by inspecting the pet() call
        pet_mock = MagicMock(return_value=_make_var("pet"))
        with (
            patch("hydromt.model.processes.meteo.temp", return_value=_make_var("temp")),
            patch("hydromt.model.processes.meteo.pet", pet_mock),
            patch(
                "hydromt.model.processes.meteo.resample_time",
                return_value=_make_var("temp"),
            ),
        ):
            WflowSbmModel.setup_temp_pet_forcing(
                model,
                temp_pet_fn="dummy_source",
                pet_method="debruin",
            )

        # The first positional arg to pet() is the slice ds[variables[1:]]
        pet_ds_arg = pet_mock.call_args[0][0]
        assert "wind" in pet_ds_arg.data_vars
        assert "wind10_u" not in pet_ds_arg.data_vars
        assert "wind10_v" not in pet_ds_arg.data_vars


class TestMissingRequiredVariables:
    """If a required variable (other than wind) is missing, trigger nodata strategy."""

    @pytest.mark.parametrize(
        ("pet_method", "all_vars", "drop_var"),
        [
            ("debruin", ["temp", "press_msl", "kin", "kout", "wind"], "press_msl"),
            ("debruin", ["temp", "press_msl", "kin", "kout", "wind"], "kin"),
            ("debruin", ["temp", "press_msl", "kin", "kout", "wind"], "kout"),
            ("makkink", ["temp", "press_msl", "kin", "wind"], "kin"),
            ("makkink", ["temp", "press_msl", "kin", "wind"], "press_msl"),
        ],
    )
    def test_missing_var_raises(self, pet_method, all_vars, drop_var):
        """When a required variable is missing and strategy=RAISE, raise NoDataException."""
        ds = _make_ds(*[v for v in all_vars if v != drop_var])
        model = _make_model(ds_override=ds)

        with pytest.raises(NoDataException):
            WflowSbmModel.setup_temp_pet_forcing(
                model,
                temp_pet_fn="dummy_source",
                pet_method=pet_method,
                nodata_strategy=NoDataStrategy.RAISE,
            )

    @pytest.mark.parametrize(
        ("pet_method", "all_vars", "drop_var"),
        [
            ("debruin", ["temp", "press_msl", "kin", "kout", "wind"], "press_msl"),
            ("makkink", ["temp", "press_msl", "kin", "wind"], "kin"),
        ],
    )
    def test_missing_var_warn_returns_none(
        self, pet_method, all_vars, drop_var, caplog
    ):
        """When a required variable is missing and strategy=WARN, return None and log."""
        ds = _make_ds(*[v for v in all_vars if v != drop_var])
        model = _make_model(ds_override=ds)

        with caplog.at_level("WARNING"):
            result = WflowSbmModel.setup_temp_pet_forcing(
                model,
                temp_pet_fn="dummy_source",
                pet_method=pet_method,
                nodata_strategy=NoDataStrategy.WARN,
            )
            assert result is None
            assert f"Variables ['{drop_var}'] are required" in caplog.text


class TestPenmanMonteithSubdaily:
    """Penman-Monteith methods require daily data. Sub-daily data should raise RuntimeError."""

    @pytest.mark.parametrize(
        "pm_method",
        ["penman-monteith_rh_simple", "penman-monteith_tdew"],
    )
    def test_subdaily_input_raises(self, pm_method):
        """Penman-Monteith - Sub-daily source frequency must raise RuntimeError."""
        vars_rh = ["temp", "temp_min", "temp_max", "rh", "kin", "wind"]
        vars_tdew = [
            "temp",
            "temp_min",
            "temp_max",
            "temp_dew",
            "kin",
            "press_msl",
            "wind",
        ]
        var_map = {
            "penman-monteith_rh_simple": vars_rh,
            "penman-monteith_tdew": vars_tdew,
        }
        ds = _make_ds(*var_map[pm_method], freq="h")  # hourly = sub-daily
        model = _make_model(ds_override=ds)

        with (
            patch("hydromt.model.processes.meteo.temp", return_value=_make_var("temp")),
        ):
            with pytest.raises(RuntimeError, match="sub-daily"):
                WflowSbmModel.setup_temp_pet_forcing(
                    model,
                    temp_pet_fn="dummy_source",
                    pet_method=pm_method,
                )

    @pytest.mark.parametrize(
        "pm_method",
        ["penman-monteith_rh_simple", "penman-monteith_tdew"],
    )
    def test_daily_input_is_accepted(self, pm_method):
        """Penman-Monteith with daily data should not raise RuntimeError."""
        vars_rh = ["temp", "temp_min", "temp_max", "rh", "kin", "wind"]
        vars_tdew = [
            "temp",
            "temp_min",
            "temp_max",
            "temp_dew",
            "kin",
            "press_msl",
            "wind",
        ]
        var_map = {
            "penman-monteith_rh_simple": vars_rh,
            "penman-monteith_tdew": vars_tdew,
        }
        ds = _make_ds(*var_map[pm_method], freq="D")
        model = _make_model(ds_override=ds)

        temp_da = _make_var("temp")
        temp_ds = xr.Dataset(
            {"temp": temp_da, "temp_max": temp_da, "temp_min": temp_da}
        )

        with (
            patch("hydromt.model.processes.meteo.temp", return_value=temp_ds),
            patch("hydromt.model.processes.meteo.pet", return_value=_make_var("pet")),
            patch("hydromt.model.processes.meteo.resample_time", return_value=temp_da),
        ):
            # Should NOT raise
            WflowSbmModel.setup_temp_pet_forcing(
                model,
                temp_pet_fn="dummy_source",
                pet_method=pm_method,
            )
