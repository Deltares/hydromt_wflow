"""Tests for the version_upgrade module."""

import logging
from pathlib import Path

import numpy as np
import pytest
from packaging.version import Version

from hydromt_wflow import WflowSbmModel, WflowSedimentModel
from hydromt_wflow.version_upgrade import (
    WFLOW_LATEST_VERSION,
    _detect_wflow_version,
    _is_v1_schema,
    _upgrade_sbm_v0_to_v1,
    _upgrade_sbm_v1_to_v1_1,
    _upgrade_sediment_v0_to_v1,
    _upgrade_sediment_v1_to_v1_1,
    _validate_options,
)


@pytest.fixture
def upgrade_data_dir(test_data_dir: Path) -> Path:
    """Root of all upgrade test data: tests/data/upgrade/."""
    return test_data_dir / "upgrade"


def assert_configs_equal(
    upgraded: WflowSbmModel | WflowSedimentModel,
    reference: WflowSbmModel | WflowSedimentModel,
) -> None:
    """Assert two model configs are equal."""
    _, errors = upgraded.config.test_equal(reference.config)
    _compare = [
        f"{key}: (upgraded: {upgraded.config.get_value(key)}), (reference: {reference.config.get_value(key)})\n"
        for key in errors
    ]
    assert not errors, _compare


class V0ToV1Assertions:
    """Assertions that belong to the v0 → v1 upgrade step."""

    @staticmethod
    def assert_sbm_config(
        upgraded: WflowSbmModel,
        reference_root: Path,
    ) -> None:
        reference = WflowSbmModel(
            reference_root, config_filename="wflow_sbm.toml", mode="r"
        )
        assert_configs_equal(upgraded, reference)

    @staticmethod
    def assert_sbm_staticmaps(upgraded: WflowSbmModel) -> None:
        sm = upgraded.staticmaps.data

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
        _upgrade_sbm_v0_to_v1(upgraded)
        assert (
            "Converting states is not supported by this conversion code" in caplog.text
        )
        assert upgraded.config.get_value("model.cold_start__flag") is True

    @staticmethod
    def assert_sediment_config(
        upgraded: WflowSedimentModel,
        reference_root: Path,
    ) -> None:
        reference = WflowSedimentModel(
            reference_root, config_filename="wflow_sediment.toml", mode="r"
        )
        assert_configs_equal(upgraded, reference)

    @staticmethod
    def assert_sediment_staticmaps(upgraded: WflowSedimentModel) -> None:
        sm = upgraded.staticmaps.data
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
        upgraded: WflowSbmModel,
        reference_root: Path,
    ) -> None:
        reference = WflowSbmModel(
            reference_root, config_filename="wflow_sbm.toml", mode="r"
        )
        assert_configs_equal(upgraded, reference)

    @staticmethod
    def assert_sediment_config(
        upgraded: WflowSedimentModel,
        reference_root: Path,
    ) -> None:
        reference = WflowSedimentModel(
            reference_root, config_filename="wflow_sediment.toml", mode="r"
        )
        assert_configs_equal(upgraded, reference)


class TestUpgradeV0ToV1:
    """Tests _upgrade_v0_to_v1 in complete isolation."""

    def test_sbm_config(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sbm" / "v0x"
        wflow = WflowSbmModel(source, config_filename="wflow_sbm.toml", mode="r")
        _upgrade_sbm_v0_to_v1(wflow)
        V0ToV1Assertions.assert_sbm_config(wflow, upgrade_data_dir / "sbm" / "v1_0")

    def test_sbm_staticmaps(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sbm" / "v0x"
        wflow = WflowSbmModel(source, config_filename="wflow_sbm.toml", mode="r")
        _upgrade_sbm_v0_to_v1(wflow)

        V0ToV1Assertions.assert_sbm_staticmaps(wflow)

    def test_sbm_reinit_false_warns(self, caplog, upgrade_data_dir: Path):
        """reinit=False should warn and force cold_start."""
        source = upgrade_data_dir / "sbm" / "v0x"
        wflow = WflowSbmModel(source, config_filename="wflow_sbm.toml", mode="r")
        wflow.config.set("model.reinit", False)
        _upgrade_sbm_v0_to_v1(wflow)

        V0ToV1Assertions.assert_sbm_cold_start_warning(wflow, caplog)

    def test_sbm_unicode_key_aliases(self, upgrade_data_dir: Path):
        """Upgrade handles legacy unicode key names (θₛ, kv₀, …) transparently."""
        source = upgrade_data_dir / "sbm" / "v0x"
        wflow = WflowSbmModel(source, config_filename="wflow_sbm.toml", mode="r")

        for ascii_key, unicode_key in [
            ("input.vertical.theta_s", "input.vertical.θₛ"),
            ("input.vertical.theta_r", "input.vertical.θᵣ"),
            ("input.vertical.g_ttm", "input.vertical.g_tt"),
            ("input.vertical.kv_0", "input.vertical.kv₀"),
        ]:
            wflow.config.set(unicode_key, wflow.config.remove(ascii_key))

        _upgrade_sbm_v0_to_v1(wflow)

        V0ToV1Assertions.assert_sbm_config(wflow, upgrade_data_dir / "sbm" / "v1_0")

    def test_sediment_config(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sediment" / "v0x"
        wflow = WflowSedimentModel(
            source,
            config_filename="wflow_sediment.toml",
            data_libs=["artifact_data"],
            mode="r",
        )
        _upgrade_sediment_v0_to_v1(wflow)

        V0ToV1Assertions.assert_sediment_config(
            wflow, upgrade_data_dir / "sediment" / "v1_0"
        )

    def test_sediment_staticmaps(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sediment" / "v0x"
        wflow = WflowSedimentModel(
            source,
            config_filename="wflow_sediment.toml",
            data_libs=["artifact_data"],
            mode="r",
        )
        _upgrade_sediment_v0_to_v1(wflow)

        V0ToV1Assertions.assert_sediment_staticmaps(wflow)

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
        wflow = WflowSbmModel(source, config_filename="wflow_sbm.toml", mode="r")

        cyclic = wflow.config.get_value("input.cyclic", [])
        cyclic.append("lateral.river.reservoir.targetfullfrac")
        wflow.config.set("input.cyclic", cyclic)

        _upgrade_sbm_v0_to_v1(wflow)
        wflow.root.set(tmp_path, mode="w")
        wflow.write()

        assert (tmp_path / "reservoir_hq_1.csv").is_file()
        assert (tmp_path / "reservoir_hq_2.csv").is_file()


class TestUpgradeV1ToV1_1:
    """Tests _upgrade_v1_to_v1_1 in complete isolation."""

    def test_sbm_config(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sbm" / "v1_0"
        wflow = WflowSbmModel(source, config_filename="wflow_sbm.toml", mode="r")
        _upgrade_sbm_v1_to_v1_1(wflow)

        V1ToV1_1Assertions.assert_sbm_config(wflow, upgrade_data_dir / "sbm" / "v1_1")

    def test_sediment_config(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sediment" / "v1_0"
        wflow = WflowSedimentModel(
            source, config_filename="wflow_sediment.toml", mode="r"
        )
        _upgrade_sediment_v1_to_v1_1(wflow)

        V1ToV1_1Assertions.assert_sediment_config(
            wflow, upgrade_data_dir / "sediment" / "v1_1"
        )

    def test_skips_if_already_latest(self, caplog: pytest.LogCaptureFixture):
        """_upgrade_v1_to_v1_1 should skip if the model is already at the latest version."""
        wflow = WflowSbmModel(
            root=Path("dummy"),
            config_filename="wflow_sbm.toml",
            mode="w",
            data_libs=["artifact_data"],
        )
        wflow.config.set("wflow_version", "1.1")

        with caplog.at_level(logging.INFO):
            _upgrade_sbm_v1_to_v1_1(wflow)
        assert "Config is already at v1.1, no upgrade needed." in caplog.text

    def test_raises_on_unexpected_version(self):
        """_upgrade_v1_to_v1_1 should raise if the model version is undefined."""
        dummy = WflowSbmModel(root=Path("dummy"), mode="w")
        dummy.config.data  # initalize
        dummy.config.set("wflow_version", "0.8")

        with pytest.raises(
            ValueError,
            match="Expected a v1.0 model but got v0.8. Run the v0 to v1 upgrade first.",
        ):
            _upgrade_sbm_v1_to_v1_1(dummy)


class TestUpgradeToLatest:
    """Tests upgrade_to_latest() for each model type end-to-end."""

    def test_sbm(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sbm" / "v0x"
        wflow = WflowSbmModel(
            source,
            config_filename="wflow_sbm.toml",
            mode="r",
            data_libs=["artifact_data"],
        )

        wflow.upgrade_to_latest()

        V0ToV1Assertions.assert_sbm_staticmaps(wflow)
        V1ToV1_1Assertions.assert_sbm_config(wflow, upgrade_data_dir / "sbm" / "v1_1")

    def test_sediment(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sediment" / "v0x"
        wflow = WflowSedimentModel(
            source,
            config_filename="wflow_sediment.toml",
            mode="r",
            data_libs=["artifact_data"],
        )

        wflow.upgrade_to_latest()

        V0ToV1Assertions.assert_sediment_staticmaps(wflow)
        V1ToV1_1Assertions.assert_sediment_config(
            wflow, upgrade_data_dir / "sediment" / "v1_1"
        )

    def test_skips_if_already_latest(self, caplog: pytest.LogCaptureFixture):
        """upgrade_to_latest() should skip if the model is already at the latest version."""
        wflow = WflowSbmModel(
            root=Path("dummy"),
            config_filename="wflow_sbm.toml",
            mode="w",
            data_libs=["artifact_data"],
        )
        assert wflow.config.get_value("wflow_version") == str(WFLOW_LATEST_VERSION)
        with caplog.at_level(logging.INFO):
            wflow.upgrade_to_latest()
        assert (
            "Model is already at the latest version, no upgrade needed." in caplog.text
        )

    def test_skips_already_applied_steps(
        self, upgrade_data_dir: Path, caplog: pytest.LogCaptureFixture
    ):
        """upgrade_to_latest() should skip steps already applied based on detected version."""
        source = upgrade_data_dir / "sbm" / "v1_0"
        wflow = WflowSbmModel(
            source,
            config_filename="wflow_sbm.toml",
            mode="r",
            data_libs=["artifact_data"],
        )
        # Should only apply v1.0 -> v1.1 step, not v0.x -> v1.0
        with caplog.at_level(logging.INFO):
            wflow.upgrade_to_latest()
        assert "Upgrading config from v0.x to v1.0 format" not in caplog.text
        assert "Upgrading config from v1.0 to v1.1 format" in caplog.text

        V1ToV1_1Assertions.assert_sbm_config(wflow, upgrade_data_dir / "sbm" / "v1_1")

    def test_raises_on_invalid_options(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sbm" / "v0x"
        wflow = WflowSbmModel(
            source,
            config_filename="wflow_sbm.toml",
            mode="r",
            data_libs=["artifact_data"],
        )
        with pytest.raises(ValueError, match="Unknown upgrade versions"):
            wflow.upgrade_to_latest(options={"9.9_10.0": {}})


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
        assert _detect_wflow_version(wflow) == Version(version_str)

    def test_detects_v1_schema_when_no_version_key(self):
        # Model is initialized with v1.1 by default
        wflow = WflowSbmModel(root=Path("dummy"), mode="w")
        wflow.config.data.pop("wflow_version", None)
        # inject a v1.0-only key
        wflow.config.set("input.static.land_surface__elevation", "land_elevation")
        assert _detect_wflow_version(wflow) == Version("1.0")

    def test_falls_back_to_v0_when_no_hints(self, caplog):
        wflow = WflowSbmModel(root=Path("dummy"), mode="w")
        wflow.config.data.clear()
        with caplog.at_level(logging.WARNING):
            version = _detect_wflow_version(wflow)
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
        result = _validate_options(options, "wflow_sbm")
        _from, _to = key.split("_")
        assert result == {(Version(_from), Version(_to)): {}}

    def test_values_preserved(self):
        options = {"0.8_1.0": {"soil_fn": "soilgrids", "usle_k_method": "renard"}}
        result = _validate_options(options, "wflow_sbm")
        assert result[(Version("0.8"), Version("1.0"))] == {
            "soil_fn": "soilgrids",
            "usle_k_method": "renard",
        }

    def test_raises_on_non_dict_input(self):
        with pytest.raises(TypeError, match="Expected 'options' to be a dict"):
            _validate_options("not_a_dict", "wflow_sbm")

    def test_raises_on_non_dict_value(self):
        with pytest.raises(TypeError, match="to be a dict"):
            _validate_options({"0.8_1.0": "not_a_dict"}, "wflow_sbm")

    def test_raises_on_invalid_key_format(self):
        with pytest.raises(TypeError, match="format 'from_to'"):
            _validate_options({"0.8to1.0": {}}, "wflow_sbm")

    def test_raises_on_unknown_version_tuple(self):
        with pytest.raises(ValueError, match="Unknown upgrade versions"):
            _validate_options({"9.9_10.0": {}}, "wflow_sbm")

    @pytest.mark.parametrize(
        ("model_name", "key"),
        [
            ("wflow_sbm", "0.8_1.0"),
            ("wflow_sbm", "1.0_1.1"),
            ("wflow_sediment", "0.8_1.0"),
            ("wflow_sediment", "1.0_1.1"),
        ],
    )
    def test_valid_for_both_model_types(self, model_name: str, key: str):
        result = _validate_options({key: {}}, model_name)
        _from, _to = key.split("_")
        assert (Version(_from), Version(_to)) in result
