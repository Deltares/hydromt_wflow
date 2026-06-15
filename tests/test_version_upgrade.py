"""Tests for the version_upgrade module."""

import logging
from pathlib import Path

import numpy as np
import pytest

from hydromt_wflow import WflowSbmModel, WflowSedimentModel


@pytest.fixture
def upgrade_data_dir(test_data_dir: Path) -> Path:
    """Root of all upgrade test data: tests/data/upgrade/."""
    return test_data_dir / "upgrade"


def assert_configs_equal(
    upgraded: WflowSbmModel | WflowSedimentModel,
    reference: WflowSbmModel | WflowSedimentModel,
    ignore_keys: set[str] | None = None,
) -> None:
    """Assert two model configs are equal, optionally ignoring specific keys.

    Both models must share the same root so that path-relative config values
    (e.g. input.path_static) are trivially equal without needing ignore_keys.
    ignore_keys is kept as an escape hatch for the rare cases where that is
    not achievable (e.g. the lake-files test that must write to a tmp_path).
    """
    _, errors = upgraded.config.test_equal(reference.config)
    for key in ignore_keys or []:
        errors.pop(key, None)
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
        ignore_keys: set[str] | None = None,
    ) -> None:
        reference = WflowSbmModel(
            reference_root, config_filename="wflow_sbm.toml", mode="r"
        )
        assert_configs_equal(upgraded, reference, ignore_keys)

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
        upgraded._upgrade_v0_to_v1()
        assert (
            "Converting states is not supported by this conversion code" in caplog.text
        )
        assert upgraded.config.get_value("model.cold_start__flag") is True

    @staticmethod
    def assert_sediment_config(
        upgraded: WflowSedimentModel,
        reference_root: Path,
        ignore_keys: set[str] | None = None,
    ) -> None:
        reference = WflowSedimentModel(
            reference_root, config_filename="wflow_sediment.toml", mode="r"
        )
        assert_configs_equal(upgraded, reference, ignore_keys)

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
        ignore_keys: set[str] | None = None,
    ) -> None:
        reference = WflowSbmModel(
            reference_root, config_filename="wflow_sbm.toml", mode="r"
        )
        assert_configs_equal(upgraded, reference, ignore_keys)

    @staticmethod
    def assert_sediment_config(
        upgraded: WflowSedimentModel,
        reference_root: Path,
        ignore_keys: set[str] | None = None,
    ) -> None:
        reference = WflowSedimentModel(
            reference_root, config_filename="wflow_sediment.toml", mode="r"
        )
        assert_configs_equal(upgraded, reference, ignore_keys)


class TestUpgradeV0ToV1:
    """Tests _upgrade_v0_to_v1 in complete isolation.

    Each test loads a fresh model from vX/, applies only this one upgrade, and
    asserts against the vX+1/ reference directory — same root, so path_static
    is equal by construction and no path-related ignore_keys are needed.
    """

    def test_sbm_config(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sbm" / "v0x"
        wflow = WflowSbmModel(source, config_filename="wflow_sbm.toml", mode="r")
        wflow._upgrade_v0_to_v1()

        V0ToV1Assertions.assert_sbm_config(wflow, upgrade_data_dir / "sbm" / "v1_0")

    def test_sbm_staticmaps(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sbm" / "v0x"
        wflow = WflowSbmModel(source, config_filename="wflow_sbm.toml", mode="r")
        wflow._upgrade_v0_to_v1()

        V0ToV1Assertions.assert_sbm_staticmaps(wflow)

    def test_sbm_reinit_false_warns(self, caplog, upgrade_data_dir: Path):
        """reinit=False should warn and force cold_start."""
        source = upgrade_data_dir / "sbm" / "v0x"
        wflow = WflowSbmModel(source, config_filename="wflow_sbm.toml", mode="r")
        wflow.config.set("model.reinit", False)

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

        wflow._upgrade_v0_to_v1()

        V0ToV1Assertions.assert_sbm_config(wflow, upgrade_data_dir / "sbm" / "v1_0")

    def test_sediment_config(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sediment" / "v0x"
        wflow = WflowSedimentModel(
            source,
            config_filename="wflow_sediment.toml",
            data_libs=["artifact_data"],
            mode="r",
        )
        wflow._upgrade_v0_to_v1(soil_fn="soilgrids")

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
        wflow._upgrade_v0_to_v1(soil_fn="soilgrids")

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

        wflow._upgrade_v0_to_v1()
        wflow.root.set(tmp_path, mode="w")
        wflow.write()

        assert (tmp_path / "reservoir_hq_1.csv").is_file()
        assert (tmp_path / "reservoir_hq_2.csv").is_file()


class TestUpgradeV1ToV1_1:
    """Tests _upgrade_v1_to_v1_1 in complete isolation."""

    def test_sbm_config(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sbm" / "v1_0"
        wflow = WflowSbmModel(source, config_filename="wflow_sbm.toml", mode="r")
        wflow._upgrade_v1_to_v1_1()

        V1ToV1_1Assertions.assert_sbm_config(wflow, upgrade_data_dir / "sbm" / "v1_1")

    def test_sediment_config(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sediment" / "v1_0"
        wflow = WflowSedimentModel(
            source, config_filename="wflow_sediment.toml", mode="r"
        )
        wflow._upgrade_v1_to_v1_1()

        V1ToV1_1Assertions.assert_sediment_config(
            wflow, upgrade_data_dir / "sediment" / "v1_1"
        )


class TestUpgradeToLatest:
    """Tests upgrade_to_latest() for each model type end-to-end.

    Config is only asserted against the FINAL version directory (v1_1).
    Intermediate config assertions are intentionally omitted: if a key is
    modified by more than one upgrade step, asserting an intermediate config
    state after the full chain has run would produce a false failure.
    The isolated step tests (TestUpgradeV0ToV1, TestUpgradeV1ToV1_1) already
    own those intermediate assertions.

    What belongs here:
    - Final config equality (always the last version directory).
    - Additive staticmap checks from earlier steps: these assert presence of
      data that is added but never removed, so they remain valid after any
      subsequent upgrade step.

    What does NOT belong here:
    - Intermediate config assertions (V0ToV1Assertions.assert_sbm_config etc.).
    """

    def test_sbm(self, upgrade_data_dir: Path):
        source = upgrade_data_dir / "sbm" / "v0x"
        wflow = WflowSbmModel(
            source,
            config_filename="wflow_sbm.toml",
            mode="r",
            data_libs=["artifact_data"],
        )

        wflow.upgrade_to_latest()

        # Staticmap checks from v0→v1: additive, safe to check after full chain.
        V0ToV1Assertions.assert_sbm_staticmaps(wflow)
        # Final config: always assert only against the latest version directory.
        V1ToV1_1Assertions.assert_sbm_config(wflow, upgrade_data_dir / "sbm" / "v1_1")

        # When adding vN → vN+1, replace the line above with:
        # VNToVN1Assertions.assert_sbm_config(wflow, upgrade_data_dir / "sbm" / "vN_1")
        # and append any new additive staticmap checks below it.

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
