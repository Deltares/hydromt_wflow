"""Test plugin model class against hydromt.models.model_api."""

import copy
from pathlib import Path
from uuid import uuid4

from hydromt_wflow.wflow_sbm import WflowSbmModel


def inner_benchmark_loop(tmp_path: Path, wflow_ini):
    # don't interfere with output of other rounds
    root = str(tmp_path / str(uuid4()))
    mod1 = WflowSbmModel(root=root, mode="w", data_libs="artifact_data")

    # Build model
    mod1.build(steps=copy.deepcopy(wflow_ini))


def test_outer_benchmark(tmp_path: Path, wflow_ini, benchmark):
    benchmark.pedantic(
        inner_benchmark_loop, (tmp_path, wflow_ini), iterations=1, rounds=10
    )
