from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hydromt.data_catalog.sources import create_source
from shapely import box

from hydromt_wflow.wflow_sbm import WflowSbmModel
from hydromt_wflow.workflows import reservoirs


@pytest.fixture
def example_wflow_model_extra_reservoir(
    example_wflow_model: WflowSbmModel,
):
    mod = example_wflow_model
    mod.read()

    # Get an existing reservoir table and clone one row
    gdf_res = mod.data_catalog.get_geodataframe(
        "hydro_reservoirs",
        geom=mod.region,
        predicate="intersects",
    )
    extra = gdf_res.iloc[[0]].copy()
    extra_id = int(gdf_res["waterbody_id"].max()) + 1
    extra["waterbody_id"] = extra_id

    # Pick a non-river grid cell inside model domain
    river = mod.staticmaps.data["river_mask"]
    x, y = 12.1, 46.45  # Hard coded for current example model region
    dx = abs(float(river.raster.res[0])) * 0.4
    dy = abs(float(river.raster.res[1])) * 0.4
    extra.geometry = [box(x - dx, y - dy, x + dx, y + dy)]
    extra["xout"], extra["yout"] = x, y

    gdf_aug = pd.concat([gdf_res, extra], ignore_index=True)
    tests_data = Path(__file__).parent.parent / "data"
    fn = tests_data / "additional_reservoir.geojson"
    gdf_aug.to_file(fn, driver="GeoJSON")

    source = create_source(
        data={
            "name": "extra_hydro_reservoir",
            "data_type": "GeoDataFrame",
            "driver": {"name": "pyogrio"},
            "uri": str(fn),
        }
    )
    mod.data_catalog.add_source("extra_hydro_reservoir", source)
    return mod, extra_id


def test_outside_reservoir_is_excluded(
    example_wflow_model_extra_reservoir: tuple[WflowSbmModel, int],
    caplog: pytest.LogCaptureFixture,
):
    """Test that a reservoir outside the river network is excluded."""
    mod, extra_id = example_wflow_model_extra_reservoir
    gdf_in = mod.data_catalog.get_geodataframe(
        "extra_hydro_reservoir",
        geom=mod.basins_highres,
    )
    assert extra_id in gdf_in["waterbody_id"].values

    with caplog.at_level("WARNING"):
        mod.setup_reservoirs_no_control(
            reservoirs_fn="extra_hydro_reservoir",
            min_area=0.0,
        )

    assert extra_id not in np.unique(mod.staticmaps.data["reservoir_area_id"].values)
    reservoir_warnings = [
        r.message for r in caplog.records if r.name.endswith("workflows.reservoirs")
    ]
    assert any(
        "were excluded because no cells were found within the river network." in msg
        for msg in reservoir_warnings
    )


def test_set_rating_curve_layer_data_type():
    # create some test data
    data = np.random.randint(4, 5)
    coords = {"x": np.arange(4), "y": np.arange(5)}
    attrs = {"description": "Test data", "_FillValue": -999.0}
    da = xr.DataArray(data, coords=coords, dims=["x", "y"], attrs=attrs).astype(
        np.float32
    )
    da[0, 0] = np.nan
    da2 = da.copy(deep=True)
    da2 = da2.fillna(-999.0)
    ds = xr.Dataset({"reservoir_rating_curve": da})
    ds2 = xr.Dataset({"reservoir_rating_curve": da2})

    ds = reservoirs.set_rating_curve_layer_data_type(ds)
    ds2 = reservoirs.set_rating_curve_layer_data_type(ds2)

    assert ds["reservoir_rating_curve"].dtype == np.int32
    assert ds2["reservoir_rating_curve"].dtype == np.int32
    assert ds["reservoir_rating_curve"].raster.nodata == -999
    assert ds2["reservoir_rating_curve"].attrs["_FillValue"] == -999
    assert ds["reservoir_rating_curve"][0, 0].values == -999
