import logging
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest
from hydromt.model import ModelRoot
from tomlkit import TOMLDocument
from tomlkit.items import Table

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

    # When asking for data property, it should return a tomlkit document
    assert isinstance(component.data, TOMLDocument)
    assert isinstance(component._data, TOMLDocument)  # Same for internal
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
    assert component.get("biem") == "bam"
    assert component.get("time") == {"sometime": "now"}
    assert isinstance(component.get("foo"), dict)
    assert isinstance(component.data["foo"], Table)
    assert component.get("foo.bar") == "baz"
    assert component.get("foo", "bip") == "bop"
    assert component.get("no") is None


def test_wflow_config_component_set(mock_model: MagicMock):
    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Assert that the internal data is None, initializing will happen in set
    assert component._data is None

    # Set an entry
    component.set("foo.bar", "baz")
    # Assert the content
    assert isinstance(component._data, TOMLDocument)
    assert component.data["foo"] == {"bar": "baz"}
    assert len(component.data) == 1


def test_wflow_config_component_set_alt(mock_model: MagicMock):
    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Set an entry
    component.set("foo", "bar", "baz")
    # Assert the content
    assert component.data["foo"] == {"bar": "baz"}
    assert len(component.data) == 1


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
    assert isinstance(component.data, TOMLDocument)
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


def test_wflow_config_component_read_default(
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
        default_template_filename=Path(DATADIR, "wflow", "wflow_sbm.toml"),
    )
    assert component._data is None  # Assert no data or structure yet

    # Read at init
    assert len(component.data) == 7
    assert component.data["dir_output"] == "run_default"
    assert (
        f"No config file found at {Path(tmp_path, component._filename).as_posix()} \
defaulting to"
        in caplog.text
    )


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
    assert "No default model config was found at" in caplog.text
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
    caplog.set_level(logging.INFO)
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

    # Updatethem like a dummy to request
    component.data.update(config_dummy_data)
    component2.data.update(config_dummy_data)

    # Assert these are equal
    assert component == component2

    # Update component2 to make them not equal
    component2.set("time.spooky", "ghost")

    # Assert unequal
    assert component != component2


def test_wflow_config_component_equal_error(mock_model: MagicMock):
    # Setup the components
    component = WflowConfigComponent(mock_model)

    # Error as wrong type is compared
    with pytest.raises(
        ValueError,
        match="Can't compare WflowConfigComponent with type int",
    ):
        _ = component == 1
