import argparse
import os
import shutil
from pathlib import Path

from hydromt.readers import read_workflow_yaml

from hydromt_wflow import WflowSbmModel, WflowSedimentModel


def build_model(
    repo_root: Path,
    model_root: Path,
    model: type[WflowSbmModel] | type[WflowSedimentModel],
    workflow_yaml: Path,
) -> None:
    """Build example Wflow SBM model."""
    param_path = repo_root / "hydromt_wflow" / "data" / "parameters_data.yml"

    mod = model(
        root=model_root.as_posix(),
        mode="w+",
        data_libs=["artifact_data", param_path.as_posix()],
    )
    _, _, steps = read_workflow_yaml(workflow_yaml.as_posix())
    mod.build(steps=steps)


def clip_model(examples_dir: Path) -> None:
    """Clip example Wflow SBM model."""
    model_root = examples_dir / "wflow_piave_subbasin"
    destination = examples_dir / "wflow_piave_clip"

    # Remove destination if it exists
    if destination.exists():
        shutil.rmtree(destination.as_posix())

    region = {
        "subbasin": [12.3006, 46.4324],
        "meta_streamorder": 4,
    }
    model = WflowSbmModel(
        root=model_root.as_posix(),
        mode="r",
    )
    model.read()
    model.root.set(destination.as_posix(), mode="w")
    model.clip(region)
    model.write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate example Wflow models.")
    parser.add_argument(
        "model",
        type=str,
        help="Model to generate, options are: sbm, clip, sediment, all",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    if os.name == "posix":
        examples_dir = repo_root / "examples" / "linux64"
    else:
        examples_dir = repo_root / "examples"

    models = {
        "sbm": {
            "model": WflowSbmModel,
            "model_root": examples_dir / "wflow_piave_subbasin",
            "workflow_yaml": repo_root
            / "tests"
            / "data"
            / "wflow_piave_build_subbasin.yml",
        },
        "sediment": {
            "model": WflowSedimentModel,
            "model_root": examples_dir / "wflow_sediment_piave_subbasin",
            "workflow_yaml": repo_root
            / "tests"
            / "data"
            / "wflow_sediment_piave_build_subbasin.yml",
        },
    }

    if args.model in ["sbm", "sediment"]:
        model_root = models[args.model]["model_root"]
        model = models[args.model]["model"]
        workflow_yaml = models[args.model]["workflow_yaml"]
        build_model(repo_root, model_root, model, workflow_yaml)
    elif args.model == "clip":
        clip_model(examples_dir)
    elif args.model == "all":
        for key in models:
            model_root = models[key]["model_root"]
            model = models[key]["model"]
            workflow_yaml = models[key]["workflow_yaml"]
            build_model(repo_root, model_root, model, workflow_yaml)
        clip_model(examples_dir)
    else:
        raise ValueError(
            "Invalid model option. Choose from: sbm, clipped, sediment, all."
        )
