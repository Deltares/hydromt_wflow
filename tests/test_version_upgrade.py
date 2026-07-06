"""Tests for the version_upgrade module."""

import logging
import shutil
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from hydromt.readers import read_toml
from packaging.version import Version

from hydromt_wflow import WflowSbmModel, WflowSedimentModel
from hydromt_wflow.components.config import WflowConfigComponent
from hydromt_wflow.version_upgrade import (
    _UPGRADES,
    WFLOW_LATEST_VERSION,
    _convert_sbm_config_v0_to_v1,
    _detect_version_from_config,
    _is_v1_schema,
    _upgrade_components_v0_to_v1_sbm,
    _upgrade_components_v0_to_v1_sediment,
    _upgrade_config_v0_to_v1,
    _upgrade_config_v1_to_v1_1,
    _validate_options,
    upgrade_model,
)


@pytest.fixture
def upgrade_data_dir(test_data_dir: Path) -> Path:
    """Root of all upgrade test data: tests/data/upgrade/."""
    return test_data_dir / "upgrade"


def assert_configs_equal(
    upgraded: dict | Path,
    reference: dict | Path,
) -> None:
    """Assert two model configs are equal."""
    model = WflowSbmModel(
        root=Path("dummy"), mode="w"
    )  # dummy model to use WflowConfigComponent
    _upgraded = WflowConfigComponent(model=model)
    _reference = WflowConfigComponent(model=model)

    upgrade_data = upgraded if isinstance(upgraded, dict) else read_toml(upgraded)
    reference_data = reference if isinstance(reference, dict) else read_toml(reference)

    _upgraded._data = upgrade_data
    _reference._data = reference_data
    _, errors = _upgraded.test_equal(_reference)
    _compare = [
        f"{key}: (upgraded: {_upgraded.get_value(key)}), (reference: {_reference.get_value(key)})\n"
        for key in errors
    ]
    assert not errors, _compare


class V0ToV1Assertions:
    """Assertions that belong to the v0 → v1 upgrade step."""

    @staticmethod
    def assert_sbm_config(
        upgraded: dict | Path,
        reference: dict | Path,
    ) -> None:
        assert_configs_equal(upgraded, reference)

    @staticmethod
    def assert_sbm_staticmaps(static_maps: xr.Dataset) -> None:
        sm = static_maps

        res_ids = np.unique(sm["reservoir_outlet_id"].raster.mask_nodata())
        assert np.all(np.isin([3349.0, 3367.0, 169986.0], res_ids))
        assert np.all(
            np.isin([3.0, 4.0], sm["reservoir_rating_curve"].raster.mask_nodata())
        )
        assert np.all(np.isin([-1.0, 2.0], sm["reservoir_e"].raster.mask_nodata()))
        assert np.all(
            np.isin(
                [-1.0, 1.0], sm["reservoir_target_full_fraction"].raster.mask_nodata()
            )
        )

    @staticmethod
    def assert_sbm_cold_start_warning(upgraded: WflowSbmModel, caplog) -> None:
        """reinit=False: upgrade should warn and force cold_start__flag=True."""
        caplog.set_level(logging.WARNING)
        _convert_sbm_config_v0_to_v1(upgraded.config.data)
        assert (
            "Converting states is not supported by this conversion code" in caplog.text
        )
        assert upgraded.config.get_value("model.cold_start__flag") is True

    @staticmethod
    def assert_sediment_config(
        upgraded: dict | Path,
        reference: dict | Path,
    ) -> None:
        assert_configs_equal(upgraded, reference)

    @staticmethod
    def assert_sediment_staticmaps(static_maps: xr.Dataset) -> None:
        sm = static_maps
        assert "soil_sagg_fraction" in sm
        assert "land_govers_c" in sm
        assert "river_kodatie_a" in sm
        assert "reservoir_outlet_id" in sm

        res_ids = np.unique(sm["reservoir_outlet_id"].raster.mask_nodata())
        assert np.all(np.isin([3349.0, 3367.0, 169986.0], res_ids))
        assert np.all(
            np.isin(
                [0.0, 1.0],
                sm["reservoir_trapping_efficiency"].raster.mask_nodata(),
            )
        )


class V1ToV1_1Assertions:
    """Assertions that belong to the v1 → v1.1 upgrade step."""

    @staticmethod
    def assert_sbm_config(
        upgraded: dict | Path,
        reference: dict | Path,
    ) -> None:
        assert_configs_equal(upgraded, reference)

    @staticmethod
    def assert_sediment_config(
        upgraded: dict | Path,
        reference: dict | Path,
    ) -> None:
        assert_configs_equal(upgraded, reference)


class TestUpgradeV0ToV1:
    """Tests _upgrade_v0_to_v1 in complete isolation."""

    def test_sbm_config(self, tmp_path: Path, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sbm" / "v0x"
        target = tmp_path / "sbm" / "v1_0"
        shutil.copytree(source, target)

        config_v0 = read_toml(target / "wflow_sbm.toml")
        _upgrade_config_v0_to_v1(
            model_root=target,
            config_v0=config_v0,
            model_type="wflow_sbm",
            config_filename="wflow_sbm.toml",
        )

        # Also run component upgrade to get reservoir config changes
        wflow = WflowSbmModel(target, config_filename="wflow_sbm.toml", mode="r")
        _upgrade_components_v0_to_v1_sbm(model=wflow, config_v0=config_v0)

        V0ToV1Assertions.assert_sbm_config(
            upgraded=wflow.config.data,
            reference=upgrade_data_dir / "sbm" / "v1_0" / "wflow_sbm.toml",
        )

    def test_sbm_staticmaps(self, upgrade_data_dir: Path, tmp_path: Path):
        source = upgrade_data_dir / "sbm" / "v0x"
        target = tmp_path / "sbm" / "v1_0"
        shutil.copytree(source, target)

        # Save config_v0 BEFORE the upgrade overwrites the file
        config_v0 = read_toml(target / "wflow_sbm.toml")
        _upgrade_config_v0_to_v1(
            model_root=target,
            config_v0=config_v0,
            model_type="wflow_sbm",
            config_filename="wflow_sbm.toml",
        )

        wflow = WflowSbmModel(target, config_filename="wflow_sbm.toml", mode="r")
        _upgrade_components_v0_to_v1_sbm(model=wflow, config_v0=config_v0)

        V0ToV1Assertions.assert_sbm_staticmaps(wflow.staticmaps.data)

    def test_sbm_reinit_false_warns(
        self, caplog, upgrade_data_dir: Path, tmp_path: Path
    ):
        """reinit=False should warn and force cold_start."""
        source = upgrade_data_dir / "sbm" / "v0x"
        target = tmp_path / "sbm_reinit_false"
        shutil.copytree(source, target)
        _upgrade_config_v0_to_v1(
            model_root=target,
            config_v0=read_toml(target / "wflow_sbm.toml"),
            model_type="wflow_sbm",
            config_filename="wflow_sbm.toml",
        )
        wflow = WflowSbmModel(target, config_filename="wflow_sbm.toml", mode="r")
        wflow.config.set("model.reinit", False)
        V0ToV1Assertions.assert_sbm_cold_start_warning(wflow, caplog)

    def test_sbm_unicode_key_aliases(self, upgrade_data_dir: Path, tmp_path: Path):
        """Upgrade handles legacy unicode key names (θₛ, kv₀, …) transparently."""
        source = upgrade_data_dir / "sbm" / "v0x"
        target = tmp_path / "sbm_unicode"
        shutil.copytree(source, target)
        wflow = WflowSbmModel(target, config_filename="wflow_sbm.toml", mode="r")

        for ascii_key, unicode_key in [
            ("input.vertical.theta_s", "input.vertical.θₛ"),
            ("input.vertical.theta_r", "input.vertical.θᵣ"),
            ("input.vertical.g_ttm", "input.vertical.g_tt"),
            ("input.vertical.kv_0", "input.vertical.kv₀"),
        ]:
            wflow.config.set(unicode_key, wflow.config.remove(ascii_key))

        config_v0 = wflow.config.data
        _upgrade_config_v0_to_v1(
            model_root=target,
            config_v0=config_v0,
            model_type="wflow_sbm",
            config_filename="wflow_sbm.toml",
        )

        # Also run component upgrade for reservoir config changes
        wflow = WflowSbmModel(target, config_filename="wflow_sbm.toml", mode="r")
        _upgrade_components_v0_to_v1_sbm(model=wflow, config_v0=config_v0)

        V0ToV1Assertions.assert_sbm_config(
            wflow.config.data,
            upgrade_data_dir / "sbm" / "v1_0" / "wflow_sbm.toml",
        )

    def test_sediment_config(self, upgrade_data_dir: Path, tmp_path: Path):
        source = upgrade_data_dir / "sediment" / "v0x"
        target = tmp_path / "sediment" / "v1_0"
        shutil.copytree(source, target)

        config_v0 = read_toml(target / "wflow_sediment.toml")
        _upgrade_config_v0_to_v1(
            model_root=target,
            config_v0=config_v0,
            model_type="wflow_sediment",
            config_filename="wflow_sediment.toml",
        )

        wflow = WflowSedimentModel(
            target,
            config_filename="wflow_sediment.toml",
            data_libs=["artifact_data"],
            mode="r",
        )
        _upgrade_components_v0_to_v1_sediment(model=wflow, config_v0=config_v0)

        V0ToV1Assertions.assert_sediment_config(
            wflow.config.data,
            upgrade_data_dir / "sediment" / "v1_0" / "wflow_sediment.toml",
        )

    def test_sediment_staticmaps(self, upgrade_data_dir: Path, tmp_path: Path):
        source = upgrade_data_dir / "sediment" / "v0x"
        target = tmp_path / "sediment" / "v1_0"
        shutil.copytree(source, target)

        config_v0 = read_toml(target / "wflow_sediment.toml")
        _upgrade_config_v0_to_v1(
            model_root=target,
            config_v0=config_v0,
            model_type="wflow_sediment",
            config_filename="wflow_sediment.toml",
        )

        wflow = WflowSedimentModel(
            target,
            config_filename="wflow_sediment.toml",
            data_libs=["artifact_data"],
            mode="r",
        )
        _upgrade_components_v0_to_v1_sediment(model=wflow, config_v0=config_v0)
        V0ToV1Assertions.assert_sediment_staticmaps(wflow.staticmaps.data)

    def test_config_toml_overwrite(self, tmp_path: Path):
        """Config.set allows overwriting an existing key, including top-level."""
        dummy = WflowSbmModel(root=tmp_path, mode="w")
        dummy.config.read()

        dummy.config.set("input.forcing.khorfrac.value", 100)
        dummy.config.set("input.forcing.khorfrac.value", 200)
        assert dummy.config.get_value("input.forcing.khorfrac.value") == 200

        dummy.config.set("path_log", "log_file.log")
        dummy.config.set("path_log", "log_file2.log")
        assert dummy.config.get_value("path_log") == "log_file2.log"

    @pytest.mark.integration
    def test_sbm_lake_files(self, tmp_path: Path, upgrade_data_dir: Path):
        """Lake CSV files are written to disk after upgrade.

        The lake CSV files (reservoir_hq_*.csv) live in v0x/ alongside the
        rest of the model data.  This test verifies they are correctly renamed
        and written when the upgraded model is saved.  Writing to tmp_path is
        unavoidable here, which means wflow.root shifts away from v0x/ and
        input.path_static will no longer match v1_0/ — so we skip the config
        equality check and only assert the files were produced.  The config
        correctness of this upgrade path is already covered by test_sbm_config.
        """
        source = upgrade_data_dir / "sbm" / "v0x"
        target = tmp_path / "sbm" / "v1_0"
        config_v0 = read_toml(source / "wflow_sbm.toml")
        shutil.copytree(source, target)
        _upgrade_config_v0_to_v1(
            model_root=target,
            config_v0=config_v0,
            model_type="wflow_sbm",
            config_filename="wflow_sbm.toml",
        )

        wflow = WflowSbmModel(target, config_filename="wflow_sbm.toml", mode="r+")

        _upgrade_components_v0_to_v1_sbm(
            model=wflow,
            config_v0=config_v0,
        )
        wflow.write()

        assert (target / "reservoir_hq_1.csv").is_file()
        assert (target / "reservoir_hq_2.csv").is_file()


class TestUpgradeV1ToV1_1:
    """Tests _upgrade_v1_to_v1_1 in complete isolation."""

    def test_sbm_config(self, upgrade_data_dir: Path, tmp_path: Path):
        source = upgrade_data_dir / "sbm" / "v1_0"
        target = tmp_path / "sbm" / "v1_1"
        shutil.copytree(source, target)
        _upgrade_config_v1_to_v1_1(
            target, model_type="wflow_sbm", config_filename="wflow_sbm.toml"
        )
        V1ToV1_1Assertions.assert_sbm_config(
            upgraded=target / "wflow_sbm.toml",
            reference=upgrade_data_dir / "sbm" / "v1_1" / "wflow_sbm.toml",
        )

    def test_sediment_config(self, upgrade_data_dir: Path, tmp_path: Path):
        source = upgrade_data_dir / "sediment" / "v1_0"
        target = tmp_path / "sediment" / "v1_1"
        shutil.copytree(source, target)
        _upgrade_config_v1_to_v1_1(
            target, model_type="wflow_sediment", config_filename="wflow_sediment.toml"
        )
        V1ToV1_1Assertions.assert_sediment_config(
            upgraded=target / "wflow_sediment.toml",
            reference=upgrade_data_dir / "sediment" / "v1_1" / "wflow_sediment.toml",
        )


class TestUpgradeToLatest:
    """Tests upgrade_to_latest() for each model type end-to-end."""

    def test_sbm(self, tmp_path: Path, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sbm" / "v0x"
        target = tmp_path / "v0x"
        wflow = WflowSbmModel(source, mode="r")
        upgraded_dir = wflow.upgrade_to_latest(target, data_libs=["artifact_data"])

        upgraded = WflowSbmModel(upgraded_dir, mode="r")
        V1ToV1_1Assertions.assert_sbm_config(
            upgraded.config.data,
            upgrade_data_dir / "sbm" / "v1_1" / "wflow_sbm.toml",
        )

    def test_sediment(self, tmp_path: Path, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sediment" / "v0x"
        target = tmp_path / "v0x"

        wflow = WflowSedimentModel(
            source,
            config_filename="wflow_sediment.toml",
            mode="r",
            data_libs=["artifact_data"],
        )
        upgraded_dir = wflow.upgrade_to_latest(target, data_libs=["artifact_data"])

        upgraded = WflowSedimentModel(
            upgraded_dir, config_filename="wflow_sediment.toml", mode="r"
        )
        V1ToV1_1Assertions.assert_sediment_config(
            upgraded.config.data,
            upgrade_data_dir / "sediment" / "v1_1" / "wflow_sediment.toml",
        )

    def test_skips_if_already_latest(
        self, upgrade_data_dir: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """upgrade_to_latest() should skip if the model is already at the latest version."""
        wflow = WflowSbmModel(
            root=upgrade_data_dir / "sbm" / "v1_1",
            mode="r",
        )
        wflow.read()
        wflow.root.set(tmp_path / "v1_1", mode="w")
        wflow.write()
        assert wflow.config.get_value("wflow_version") == str(WFLOW_LATEST_VERSION)
        with caplog.at_level(logging.INFO):
            wflow.upgrade_to_latest(
                tmp_path / "v1_1_duplicate", data_libs=["artifact_data"]
            )
        assert (
            "Model is already at the latest version, no upgrade needed." in caplog.text
        )

    def test_skips_already_applied_steps(
        self, tmp_path: Path, upgrade_data_dir: Path, caplog: pytest.LogCaptureFixture
    ):
        """upgrade_to_latest() should skip steps already applied based on detected version."""
        source = upgrade_data_dir / "sbm" / "v1_0"
        target = tmp_path / "v1_0"

        wflow = WflowSbmModel(
            source,
            mode="r",
            data_libs=["artifact_data"],
        )
        # Should only apply v1.0 -> v1.1 step, not v0.x -> v1.0
        with caplog.at_level(logging.INFO):
            upgraded_dir = wflow.upgrade_to_latest(target, data_libs=["artifact_data"])
        assert "Upgrading config from v0.x to v1.0 format" not in caplog.text

        V1ToV1_1Assertions.assert_sbm_config(
            upgraded_dir / "wflow_sbm.toml",
            upgrade_data_dir / "sbm" / "v1_1" / "wflow_sbm.toml",
        )

    def test_raises_on_invalid_options(self, tmp_path: Path, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sbm" / "v0x"
        target = tmp_path / "v0x"

        wflow = WflowSbmModel(
            source,
            config_filename="wflow_sbm.toml",
            mode="r",
            data_libs=["artifact_data"],
        )
        with pytest.raises(ValueError, match="Unknown upgrade versions"):
            wflow.upgrade_to_latest(target, options={"9.9_10.0": {}})


@pytest.mark.parametrize(
    ("config_data", "expected"),
    [
        # v1.0 schema: has input.static section
        (
            {"input": {"static": {"land_surface__elevation": "land_elevation"}}},
            True,
        ),
        # v1.0 schema: has model section with wflow-v1 keys
        ({"model": {"snow__flag": True}}, True),
        # v0 schema: has input.vertical
        ({"input": {"vertical": {"ksat": "ksat"}}}, False),
        # v0 schema: has starttime at top level
        (
            {"starttime": "2010-01-01T00:00:00", "endtime": "2010-12-31T00:00:00"},
            False,
        ),
        # empty config
        ({}, False),
    ],
)
def test_schema_detection(config_data: dict, expected: bool):
    assert _is_v1_schema(config_data) == expected


class TestDetectWflowVersion:
    """Tests for _detect_wflow_version()."""

    @pytest.mark.parametrize("version_str", ["1.1", "1.0", "2.0"])
    def test_reads_version_from_config(self, version_str: str):
        wflow = WflowSbmModel(root=Path("dummy"), mode="w")
        wflow.config.set("wflow_version", version_str)
        assert _detect_version_from_config(wflow.config.data) == Version(version_str)

    def test_detects_v1_schema_when_no_version_key(self):
        # Model is initialized with v1.1 by default
        wflow = WflowSbmModel(root=Path("dummy"), mode="w")
        wflow.config.data.pop("wflow_version", None)
        # inject a v1.0-only key
        wflow.config.set("input.static.land_surface__elevation", "land_elevation")
        assert _detect_version_from_config(wflow.config.data) == Version("1.0")

    def test_falls_back_to_v0_when_no_hints(self, caplog):
        wflow = WflowSbmModel(root=Path("dummy"), mode="w")
        wflow.config.data.clear()
        with caplog.at_level(logging.WARNING):
            version = _detect_version_from_config(wflow.config.data)
        assert version == Version("0.8")
        assert "Assuming pre-v1.0 model" in caplog.text


class TestValidateOptions:
    """Tests for _validate_options()."""

    @pytest.mark.parametrize(
        "key",
        ["0.8_1.0", "1.0_1.1"],
    )
    def test_valid_keys_coerced_to_version_tuples(self, key: str):
        options = {key: {}}
        result = _validate_options(options)
        _from, _to = key.split("_")
        assert result == {(Version(_from), Version(_to)): {}}

    def test_values_preserved(self):
        options = {"0.8_1.0": {"soil_fn": "soilgrids", "usle_k_method": "renard"}}
        result = _validate_options(options)
        assert result[(Version("0.8"), Version("1.0"))] == {
            "soil_fn": "soilgrids",
            "usle_k_method": "renard",
        }

    def test_raises_on_non_dict_input(self):
        with pytest.raises(TypeError, match="Expected 'options' to be a dict"):
            _validate_options("not_a_dict")

    def test_raises_on_non_dict_value(self):
        with pytest.raises(TypeError, match="to be a dict"):
            _validate_options({"0.8_1.0": "not_a_dict"})

    def test_raises_on_invalid_key_format(self):
        with pytest.raises(TypeError, match="format 'from_to'"):
            _validate_options({"0.8to1.0": {}})

    def test_raises_on_unknown_version_tuple(self):
        with pytest.raises(ValueError, match="Unknown upgrade versions"):
            _validate_options({"9.9_10.0": {}})

    @pytest.mark.parametrize(
        "key",
        [f"{v_from}_{v_to}" for v_from, v_to in _UPGRADES],
    )
    def test_valid_for_all_upgrade_steps(self, key: str):
        result = _validate_options({key: {}})
        _from, _to = key.split("_")
        assert (Version(_from), Version(_to)) in result


class TestUpgradeModelPathBased:
    """Tests for upgrade_model() — the path-based upgrade API."""

    def test_sbm_v0_to_latest(self, tmp_path: Path, upgrade_data_dir: Path):
        """Full v0→latest upgrade via path-based API produces correct config."""
        source = upgrade_data_dir / "sbm" / "v0x"
        model_dir = tmp_path / "sbm"
        shutil.copytree(source, model_dir)

        upgrade_model(model_dir, "wflow_sbm")

        # Verify config matches the v1.1 reference
        V1ToV1_1Assertions.assert_sbm_config(
            model_dir / "wflow_sbm.toml",
            upgrade_data_dir / "sbm" / "v1_1" / "wflow_sbm.toml",
        )

    def test_sbm_v0_to_latest_writes_staticmaps(
        self, tmp_path: Path, upgrade_data_dir: Path
    ):
        """Full v0→latest upgrade via path-based API writes a staticmaps.nc."""
        source = upgrade_data_dir / "sbm" / "v0x"
        model_dir = tmp_path / "sbm"
        shutil.copytree(source, model_dir)

        upgrade_model(model_dir, "wflow_sbm")

        assert (model_dir / "staticmaps.nc").is_file()

    def test_sbm_v1_0_to_latest(self, tmp_path: Path, upgrade_data_dir: Path):
        """v1.0→latest upgrade via path-based API (config-only, no grid changes)."""
        source = upgrade_data_dir / "sbm" / "v1_0"
        model_dir = tmp_path / "sbm"
        shutil.copytree(source, model_dir)

        upgrade_model(model_dir, "wflow_sbm")

        V1ToV1_1Assertions.assert_sbm_config(
            model_dir / "wflow_sbm.toml",
            upgrade_data_dir / "sbm" / "v1_1" / "wflow_sbm.toml",
        )

    def test_sediment_v0_to_latest(self, tmp_path: Path, upgrade_data_dir: Path):
        """Full v0→latest upgrade for sediment model."""
        source = upgrade_data_dir / "sediment" / "v0x"
        model_dir = tmp_path / "sediment"
        shutil.copytree(source, model_dir)

        upgrade_model(model_dir, "wflow_sediment", data_libs=["artifact_data"])

        upgraded = WflowSedimentModel(
            model_dir, config_filename="wflow_sediment.toml", mode="r"
        )
        V1ToV1_1Assertions.assert_sediment_config(
            upgraded.config.data,
            upgrade_data_dir / "sediment" / "v1_1" / "wflow_sediment.toml",
        )

    def test_sediment_v1_0_to_latest(self, tmp_path: Path, upgrade_data_dir: Path):
        """v1.0→latest upgrade for sediment model (config-only)."""
        source = upgrade_data_dir / "sediment" / "v1_0"
        model_dir = tmp_path / "sediment"
        shutil.copytree(source, model_dir)

        upgrade_model(model_dir, "wflow_sediment")

        V1ToV1_1Assertions.assert_sediment_config(
            model_dir / "wflow_sediment.toml",
            upgrade_data_dir / "sediment" / "v1_1" / "wflow_sediment.toml",
        )

    def test_skips_if_already_latest(
        self, tmp_path: Path, upgrade_data_dir: Path, caplog: pytest.LogCaptureFixture
    ):
        """upgrade_model() skips when config is already at latest version."""
        source = upgrade_data_dir / "sbm" / "v1_1"
        model_dir = tmp_path / "sbm"
        shutil.copytree(source, model_dir)

        with caplog.at_level(logging.INFO):
            upgrade_model(model_dir, "wflow_sbm")

        assert "Model is already at the latest version" in caplog.text

    def test_custom_config_filename(self, tmp_path: Path, upgrade_data_dir: Path):
        """upgrade_model() respects a custom config_filename."""
        source = upgrade_data_dir / "sbm" / "v1_0"
        model_dir = tmp_path / "sbm"
        shutil.copytree(source, model_dir)

        # Rename the config file
        (model_dir / "wflow_sbm.toml").rename(model_dir / "custom.toml")
        upgrade_model(model_dir, "wflow_sbm", config_filename="custom.toml")
        V1ToV1_1Assertions.assert_sbm_config(
            model_dir / "custom.toml",
            upgrade_data_dir / "sbm" / "v1_1" / "wflow_sbm.toml",
        )

    def test_raises_on_invalid_model_type(self, tmp_path: Path):
        """upgrade_model() raises ValueError on unknown model type."""
        # Create a minimal TOML so the file exists
        config_path = tmp_path / "wflow_unknown.toml"
        config_path.write_text("[model]\nsnow__flag = true\n")

        with pytest.raises(ValueError, match="Unknown model type"):
            upgrade_model(tmp_path, "wflow_unknown")

    def test_options_passed_to_grid_upgrade(
        self, tmp_path: Path, upgrade_data_dir: Path
    ):
        """Options dict is forwarded to the grid upgrade step."""
        source = upgrade_data_dir / "sbm" / "v0x"
        model_dir = tmp_path / "sbm"
        shutil.copytree(source, model_dir)

        # Valid options for v0→v1 step (should not raise)
        upgrade_model(model_dir, "wflow_sbm", options={"0.8_1.0": {}})
        V1ToV1_1Assertions.assert_sbm_config(
            model_dir / "wflow_sbm.toml",
            upgrade_data_dir / "sbm" / "v1_1" / "wflow_sbm.toml",
        )

    def test_raises_on_invalid_options(self, tmp_path: Path, upgrade_data_dir: Path):
        """upgrade_model() raises on invalid options keys."""
        source = upgrade_data_dir / "sbm" / "v0x"
        model_dir = tmp_path / "sbm"
        shutil.copytree(source, model_dir)

        with pytest.raises(ValueError, match="Unknown upgrade versions"):
            upgrade_model(model_dir, "wflow_sbm", options={"9.9_10.0": {}})
