"""System regression tests for Wflow model outputs.

Compares model output at a single outlet for a defined period against a baseline.
Invoked via: pixi run test-regression <MODEL_ROOT>
"""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Only run these tests when explicitly requested via pytest -m regression
pytest_mark = pytest.mark.regression


def pytest_addoption(parser):
    parser.addoption(
        "--model-root",
        action="store",
        required=True,
        help="Path to the model root directory containing output.nc",
    )


@pytest.fixture
def model_root(request):
    return request.config.getoption("--model-root")


@pytest.fixture
def model_output(model_root):
    output_path = Path(model_root) / "run_default" / "output.nc"
    if not output_path.exists():
        pytest.skip(f"Output file not found: {output_path}")
    return xr.open_dataset(output_path)


@pytest.fixture
def baseline(model_root):
    baseline_path = Path(model_root) / "baseline" / "output.nc"
    if not baseline_path.exists():
        pytest.skip(f"Baseline file not found: {baseline_path}")
    return xr.open_dataset(baseline_path)


class TestSBMRegression:
    """Regression tests for SBM model output at the outlet."""

    OUTLET_INDEX = -1  # last gauge / outlet
    RTOL = 1e-5  # relative tolerance
    ATOL = 1e-8  # absolute tolerance

    def test_discharge_outlet(self, model_output, baseline):
        """Compare discharge at the outlet between run and baseline."""
        var = "Q"
        if var not in model_output:
            pytest.skip(f"Variable '{var}' not in model output")

        actual = model_output[var].isel(wflow_id=self.OUTLET_INDEX).values
        expected = baseline[var].isel(wflow_id=self.OUTLET_INDEX).values

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=self.RTOL,
            atol=self.ATOL,
            err_msg=f"Discharge at outlet diverged from baseline (rtol={self.RTOL})",
        )

    def test_discharge_rmse(self, model_output, baseline):
        """Report single-number RMSE at the outlet for release notes."""
        var = "Q"
        if var not in model_output or var not in baseline:
            pytest.skip(f"Variable '{var}' not available for RMSE check")

        actual = model_output[var].isel(wflow_id=self.OUTLET_INDEX).values
        expected = baseline[var].isel(wflow_id=self.OUTLET_INDEX).values

        rmse = float(np.sqrt(np.mean((actual - expected) ** 2)))
        # TeamCity service message for reporting
        print(f"##teamcity[buildStatisticValue key='rmse_outlet_Q' value='{rmse:.6e}']")
        # Fail if RMSE exceeds threshold (absolute discharge units)
        assert rmse < 0.01, f"RMSE at outlet ({rmse:.6e}) exceeds threshold 0.01"
