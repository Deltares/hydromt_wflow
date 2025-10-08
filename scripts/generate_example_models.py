import os
from pathlib import Path

from hydromt.io.readers import read_workflow_yaml

from hydromt_wflow import WflowSbmModel

if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    if os.name == "posix":
        model_root = repo_root / "examples" / "linux64" / "wflow_piave_subbasin"
    else:
        model_root = repo_root / "examples" / "wflow_piave_subbasin"

    # Remove staticmaps file if exists
    staticmaps_file = model_root / "staticmaps.nc"
    if staticmaps_file.exists():
        staticmaps_file.unlink()

    param_path = repo_root / "hydromt_wflow" / "data" / "parameters_data.yml"

    model = WflowSbmModel(
        root=model_root.as_posix(),
        mode="w",
        data_libs=["artifact_data", param_path.as_posix()],
    )
    workflow_yaml = repo_root / "tests" / "data" / "wflow_piave_build_subbasin.yml"
    _, _, steps = read_workflow_yaml(workflow_yaml.as_posix())
    model.build(steps=steps)

    # Remove files that should not be committed
    files = [
        "hydromt_data.yml",
        "inmaps_era5_era5d_debruin_None_2010_2010.nc",  # Probably bug
        "reservoir_accuracy.csv",
        "reservoir_timeseries_None.csv",  # Probably bug
    ]
    for file in files:
        p = model_root / file
        if p.exists():
            p.unlink()
