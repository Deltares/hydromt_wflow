"""Tests for the hydromt_wflow CLI."""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from hydromt_wflow.cli import _parse_upgrade_config, main


@pytest.fixture
def upgrade_data_dir() -> Path:
    """Root of all upgrade test data: tests/data/upgrade/."""
    return Path(__file__).parent / "data" / "upgrade"


@pytest.fixture
def runner():
    return CliRunner()


class TestUpgradeCLI:
    def test_help(self, runner):
        result = runner.invoke(main, ["upgrade", "--help"])
        assert result.exit_code == 0
        assert "MODEL_ROOT" in result.output

    def test_requires_output_dir(self, runner, upgrade_data_dir):
        result = runner.invoke(
            main,
            [
                "upgrade",
                str(upgrade_data_dir / "sbm" / "v0x"),
            ],
        )
        assert result.exit_code != 0
        assert "Missing option" in result.output

    def test_output_must_not_exist(self, runner, tmp_path, upgrade_data_dir):
        existing = tmp_path / "existing"
        existing.mkdir()
        result = runner.invoke(
            main,
            [
                "upgrade",
                str(upgrade_data_dir / "sbm" / "v0x"),
                "-o",
                str(existing),
            ],
        )
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_sbm_upgrade(self, runner, tmp_path, upgrade_data_dir):
        source = str(upgrade_data_dir / "sbm" / "v0x")
        output = str(tmp_path / "sbm_upgraded")
        result = runner.invoke(
            main,
            [
                "upgrade",
                source,
                "-o",
                output,
                "-v",
            ],
        )
        assert result.exit_code == 0, result.output
        assert Path(output, "wflow_sbm.toml").is_file()

    def test_sediment_upgrade_with_config(self, runner, tmp_path, upgrade_data_dir):
        config_content = {
            "0.8_1.0": {"soil_fn": "soilgrids"},
        }
        config_path = tmp_path / "upgrade_opts.yml"
        with open(config_path, "w") as f:
            yaml.dump(config_content, f)

        source = str(upgrade_data_dir / "sediment" / "v0x")
        output = str(tmp_path / "sediment_upgraded")
        result = runner.invoke(
            main,
            [
                "upgrade",
                source,
                "-o",
                output,
                "--model-type",
                "wflow_sediment",
                "-i",
                str(config_path),
                "-d",
                "artifact_data",
                "-v",
            ],
        )
        assert result.exit_code == 0, result.output
        assert Path(output, "wflow_sediment.toml").is_file()

    def test_already_latest(self, runner, tmp_path, upgrade_data_dir):
        source = str(upgrade_data_dir / "sbm" / "v1_0")
        output = str(tmp_path / "already_latest")
        result = runner.invoke(
            main,
            [
                "upgrade",
                source,
                "-o",
                output,
                "-v",
            ],
        )
        assert result.exit_code == 0

    def test_default_model_type_is_sbm(self, runner):
        result = runner.invoke(main, ["upgrade", "--help"])
        assert "wflow_sbm" in result.output


class TestParseUpgradeConfig:
    def test_extracts_config_filename(self, tmp_path):
        config = {"config_filename": "my_config.toml", "0.8_1.0": {"k": "v"}}
        path = tmp_path / "cfg.yml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        filename, options = _parse_upgrade_config(path)
        assert filename == "my_config.toml"
        assert options == {"0.8_1.0": {"k": "v"}}

    def test_no_config_filename(self, tmp_path):
        config = {"0.8_1.0": {"soil_fn": "soilgrids"}}
        path = tmp_path / "cfg.yml"
        with open(path, "w") as f:
            yaml.dump(config, f)

        filename, options = _parse_upgrade_config(path)
        assert filename is None
        assert options == {"0.8_1.0": {"soil_fn": "soilgrids"}}

    def test_empty_config(self, tmp_path):
        path = tmp_path / "cfg.yml"
        path.write_text("")

        filename, options = _parse_upgrade_config(path)
        assert filename is None
        assert options is None
