from pathlib import Path
from unittest.mock import PropertyMock

from hydromt.model import ModelRoot
from pytest_mock import MockerFixture
from tomlkit import TOMLDocument

from hydromt_wflow.components import WflowConfigComponent


def test_wflow_config_component_init(mock_model: MockerFixture):
    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Assert that the internal data is None
    assert component._data is None

    # When asking for data property, it should return a tomlkit document
    assert isinstance(component.data, TOMLDocument)
    assert isinstance(component._data, TOMLDocument)  # Same for internal
    assert len(component.data) == 0


def test_wflow_config_component_get(mock_model: MockerFixture):
    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Update it like a dummy to request
    component.data.update(
        {
            "biem": "bam",
            "time": {"sometime": "now"},
            "foo": {"bar": "baz", "bip": "bop"},
        }
    )

    # Assert asking for entry
    assert component.get("biem") == "bam"
    assert component.get("time") == {"sometime": "now"}
    assert isinstance(component.get("foo"), dict)
    assert component.get("foo.bar") == "baz"
    assert component.get("foo", "bip") == "bop"
    assert component.get("no") is None


def test_wflow_config_component_set(mock_model: MockerFixture):
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


def test_wflow_config_component_set_alt(mock_model: MockerFixture):
    # Setup the component
    component = WflowConfigComponent(mock_model)

    # Set an entry
    component.set("foo", "bar", "baz")
    # Assert the content
    assert component.data["foo"] == {"bar": "baz"}
    assert len(component.data) == 1


def test_wflow_component_read(model_subbasin_cached: Path, mock_model: MockerFixture):
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
    pass
