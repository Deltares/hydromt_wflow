from pathlib import Path

import pandas as pd

from hydromt_wflow.wflow_sbm import WflowSbmModel


def test_write_and_read_with_custom_staticmaps_folder(tmp_path: Path):
    model = WflowSbmModel(tmp_path)
    model.write_grid("staticmaps/staticmaps.nc")
    test_df = pd.DataFrame({"a": [1, 2, 3]})
    model.set_tables(test_df, name="test")
    model.write_tables()

    assert (tmp_path / "staticmaps" / "test.csv").is_file()

    model2 = WflowSbmModel(tmp_path, mode="r")
    model2.set_config("input.path_static", "staticmaps/staticmaps.nc")
    model2.read_tables()
    pd.testing.assert_frame_equal(model2.tables.data["test"], test_df)


def test_write_and_read_without_staticmaps_folder(tmp_path: Path):
    model = WflowSbmModel(tmp_path)
    test_df = pd.DataFrame({"a": [1, 2, 3]})
    model.set_tables(test_df, name="test")
    model.write_tables()

    assert (tmp_path / "test.csv").is_file()

    model2 = WflowSbmModel(tmp_path, mode="r")
    model2.read_tables()
    pd.testing.assert_frame_equal(model2.tables.data["test"], test_df)
