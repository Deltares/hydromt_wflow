import logging
import shutil
from pathlib import Path

from hydromt_wflow import WflowSbmModel

tmp_dir = Path(__file__).parents[2] / "temp"
path = tmp_dir / "wflow_model"
dst = tmp_dir / "wflow_model_upgraded"
logger = logging.getLogger("hydromt")

if __name__ == "__main__":
    shutil.rmtree(dst, ignore_errors=True)
    model = WflowSbmModel(root=str(path), mode="r")
    model.read()
    model.upgrade_to_v1_wflow()
    model.set_root(dst)
    model.write_grid()
    model.close()

    model = WflowSbmModel(root=str(dst), mode="r")
    model.read()
