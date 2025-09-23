import logging
from copy import deepcopy
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest
from hydromt.model import ModelRoot

from hydromt_wflow.components import WflowConfigComponent
from hydromt_wflow.utils import DATADIR


@pytest.fixture
def mock_model(mock_model_factory) -> MagicMock:
    """Create a mock model with a root."""
    return mock_model_factory(mode="w")


def test_wflow_config_component_init(mock_model: MagicMock):
    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Assert that the internal data is None
    assert component._data is None

    # When asking for data property, it should return a dictionary
    assert isinstance(component.data, dict)
    assert isinstance(component._data, dict)  # Same for internal
    assert len(component.data) == 0


def test_wflow_config_component_get(
    mock_model: MagicMock,
    config_dummy_data: dict,
):
    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Update it like a dummy to request
    component.data.update(config_dummy_data)

    # Assert asking for entry
    assert component.get_value("biem") == "bam"
    assert component.get_value("time") == {"sometime": "now"}
    assert isinstance(component.get_value("foo"), dict)
    assert isinstance(component.data["foo"], dict)
    assert component.get_value("foo.bar") == "baz"
    assert component.get_value("foo.bip") == "bop"
    assert component.get_value("no") is None


def test_wflow_config_component_set(mock_model: MagicMock):
    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Assert that the internal data is None, initializing will happen in set
    assert component._data is None

    # Set an entry
    component.set("foo.bar", "baz")
    # Assert the content
    assert isinstance(component._data, dict)
    assert component.data["foo"] == {"bar": "baz"}
    assert len(component.data) == 1


def test_wflow_config_component_update(mock_model: MagicMock):
    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Update the config
    component.update(
        {
            "foo.bar": "baz",
            "time": "now",
        }
    )
    # Assert the content
    assert component.data["foo"] == {"bar": "baz"}
    assert len(component.data) == 2


def test_wflow_config_component_remove(mock_model: MagicMock):
    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Update the config
    component._data = {
        "model": {
            "river_routing": "kinematic_wave",
            "land_routing": "kinematic_wave",
        },
        "time": "now",
    }

    assert component.get_value("model.river_routing") == "kinematic_wave"
    # Remove a config entry
    popped = component.remove("model.river_routing")
    assert popped == "kinematic_wave"

    # Check if it is removed
    assert component.get_value("model.river_routing") is None
    assert component.get_value("model") is not None

    with pytest.raises(KeyError):
        component.remove("model", "river_routing")

    with pytest.raises(KeyError):
        component.remove("model", "non_existing_key", "some_stuff")

    assert component.remove("model.non_existing_key", errors="ignore") is None


def test_wflow_config_component_read(
    mock_model: MagicMock,
    model_subbasin_cached: Path,
):
    # Set it to read mode
    type(mock_model).root = PropertyMock(
        side_effect=lambda: ModelRoot(model_subbasin_cached, mode="r"),
    )

    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Assert its data currently none
    assert component._data is None

    # Read the data
    component.read()

    # Assert the read data
    assert isinstance(component.data, dict)
    assert len(component.data) == 7
    assert component.data["dir_output"] == "run_default"
    assert component.data["input"]


def test_wflow_config_component_read_init(
    mock_model: MagicMock,
    model_subbasin_cached: Path,
):
    # Set it to read mode
    type(mock_model).root = PropertyMock(
        side_effect=lambda: ModelRoot(model_subbasin_cached, mode="r"),
    )

    # Setup the component
    component = WflowConfigComponent(mock_model)
    assert component._data is None  # Assert no data or structure yet

    # Read at init
    assert len(component.data) == 7
    assert component.data["dir_output"] == "run_default"


def test_wflow_config_component_read_default_read_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    mock_model: MagicMock,
):
    # Set it to read mode
    type(mock_model).root = PropertyMock(
        side_effect=lambda: ModelRoot(tmp_path, mode="r"),
    )

    # Setup the component
    component = WflowConfigComponent(
        model=mock_model,
        default_template_filename=str(DATADIR / "wflow_sbm" / "wflow_sbm.toml"),
    )
    assert component._data is None  # Assert no data or structure yet

    # Read at init
    assert len(component.data) == 0


def test_wflow_config_component_read_default_write_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    mock_model: MagicMock,
):
    caplog.set_level(logging.INFO)
    # Reading the template only happens in w and w+ modes
    # Set it to write mode
    type(mock_model).root = PropertyMock(
        side_effect=lambda: ModelRoot(tmp_path, mode="w"),
    )

    # Setup the component
    component = WflowConfigComponent(
        model=mock_model,
        default_template_filename=str(DATADIR / "wflow_sbm" / "wflow_sbm.toml"),
    )
    assert component._data is None  # Assert no data or structure yet

    # Read at init
    assert len(component.data) == 7
    assert component.data["dir_output"] == "run_default"
    assert "Reading default config file from " in caplog.text


def test_wflow_config_component_read_warnings(
    caplog: pytest.LogCaptureFixture,
    mock_model: MagicMock,
    model_subbasin_cached: Path,
):
    caplog.set_level(logging.INFO)
    # Set it to read mode
    type(mock_model).root = PropertyMock(
        side_effect=lambda: ModelRoot(model_subbasin_cached, mode="r"),
    )

    # Setup the component
    component = WflowConfigComponent(mock_model, filename="unknown.toml")

    # Read the content
    component.read()

    # Check for the warning
    assert "No config was found at" in caplog.text
    assert len(component.data) == 0


def test_wflow_config_component_write(
    mock_model: MagicMock,
    config_dummy_data: dict,
):
    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Update it like a dummy to request
    component.data.update(config_dummy_data)

    # Write the data
    component.write()

    # Assert the output
    file = Path(component.root.path, component._filename)
    assert file.is_file()
    with open(file) as f:
        data = f.read()
        assert data.startswith('biem = "bam"\n\n[time]')


def test_wflow_config_component_write_warnings(
    caplog: pytest.LogCaptureFixture,
    mock_model: MagicMock,
):
    caplog.set_level(logging.DEBUG)
    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Write empty
    component.write()

    # Assert the warning
    assert "Model config has no data, skip writing." in caplog.text


def test_wflow_config_component_equal(mock_model: MagicMock, config_dummy_data: dict):
    # Setup the components
    component = WflowConfigComponent(mock_model)
    component2 = WflowConfigComponent(mock_model)

    # Update them like a dummy to request
    component.update(config_dummy_data)
    config_dummy_data2 = deepcopy(config_dummy_data)
    component2.update(config_dummy_data2)

    # Assert these are equal
    eq, errors = component.test_equal(component2)
    assert eq
    assert len(errors) == 0

    # Update component2 to make them not equal
    component2.set("time.spooky", "ghost")

    # Assert unequal
    eq, errors = component.test_equal(component2)
    assert not eq
    assert errors == {"config": "Configs are not equal"}


def test_wflow_config_component_equal_error(mock_model: MagicMock):
    # Setup the components
    component = WflowConfigComponent(mock_model)

    # Error as wrong type is compared
    eq, errors = component.test_equal(1)
    assert errors == {
        "__class__": "other does not inherit from <class 'hydromt_wflow.components.config.WflowConfigComponent'>."  # noqa: E501
    }
