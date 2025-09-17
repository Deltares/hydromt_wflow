"""Test plugin model class against hydromt.models.model_api."""

from uuid import uuid4

from hydromt_wflow.wflow_sbm import WflowSbmModel


def inner_benchmark_loop(tmpdir, wflow_ini):
    # don't interfere with output of other rounds
    root = tmpdir.join(str(uuid4()))
    mod1 = WflowSbmModel(root=root, mode="w", data_libs="artifact_data")
    # Build method options
    region = {
        "subbasin": [12.2051, 45.8331],
        "strord": 4,
        "bounds": [11.70, 45.35, 12.95, 46.70],
    }
    # get ini file
    opt = wflow_ini

    # Build model
    mod1.build(region=region, opt=opt)


def test_outer_benchmark(tmpdir, wflow_ini, benchmark):
    benchmark.pedantic(
        inner_benchmark_loop, (tmpdir, wflow_ini), iterations=1, rounds=10
    )
