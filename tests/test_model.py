from pathlib import Path

import geopandas as gpd
import pytest

from hydromt_wflow import WflowModel


def test_empty_model(tmp_path):
    # Setup an empty fiat model
    model = WflowModel(tmp_path)

    # Assert some basic statements
    assert "config" in model.components
    assert "region" in model.components
    assert model.region is None
    assert len(model.components) == 7


def test_basic_read_write(tmp_path):
    # Setup the model
    model = WflowModel(tmp_path, mode="w")

    # Call the necessary setup methods
    model.setup_config(some_var="some_value")
    # Write the model
    model.write()
    model = None
    assert Path(tmp_path, "wflow_sbm.toml").is_file()

    # Model in read mode
    model = WflowModel(tmp_path, mode="r")
    model.read()

    assert len(model.config.data) != 0


def test_setup_config(tmp_path):
    # Setup the model
    model = WflowModel(tmp_path, mode="w")

    # Setup some config variables
    # TODO Make sure this complies with wflow v1 toml entries for real case
    model.setup_config(
        **{
            "input.static": "file1",
            "input.forcing": "file2",
            "time.timestepsecs": 86400,
            "vertical.ksat.value": 2,
        }
    )

    # Assert the config component
    assert model.config.data["time"] == {"timestepsecs": 86400}
    assert model.config.get_value("input.forcing") == "file2"
    assert len(model.config.get_value("input")) == 2
    assert model.config.get_value("vertical.ksat") == {"value": 2}


def test_setup_region(tmp_path, build_region):
    # Setup the model
    model = WflowModel(tmp_path, mode="w")
    assert model.region is None

    # Setup the region
    model.setup_region(region=build_region)
    assert model.region is not None
    assert isinstance(model.region, gpd.GeoDataFrame)
    assert len(model.region) == 1


def test_setup_region_error(tmp_path):
    # Setup the model
    model = WflowModel(tmp_path, mode="w")

    # Setup the region
    region_no = Path(tmp_path, "region.geojson")
    with pytest.raises(FileNotFoundError, match=region_no.as_posix()):
        model.setup_region(region=region_no)
