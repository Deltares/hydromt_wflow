"""HydroMT-Wflow command line interface."""

import shutil
from pathlib import Path

import click
import yaml
from hydromt import __version__ as hydromt_version
from hydromt import log

from hydromt_wflow import __version__
from hydromt_wflow.version_upgrade import upgrade_model


def _parse_upgrade_config(config_path: Path) -> tuple[str | None, dict | None]:
    """Parse a simplified upgrade YAML config.

    Expected format::

        config_filename: wflow_sbm_v0.toml
        0.8_1.0:
          soil_fn: soilgrids

    Returns
    -------
    config_filename : str or None
        Override for the config filename, if specified.
    options : dict or None
        Version-keyed options dict for ``upgrade_model``.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    config_filename = raw.pop("config_filename", None)
    options = raw if raw else None
    return config_filename, options


@click.group()
@click.version_option(
    version=__version__,
    message=f"hydromt_wflow version: %(version)s (hydromt version: {hydromt_version})",
)
def main():
    """HydroMT-Wflow command line interface."""


@main.command()
@click.argument(
    "model_root",
    type=click.Path(exists=True, resolve_path=True, dir_okay=True, file_okay=False),
)
@click.option(
    "-o",
    "--output",
    required=True,
    type=click.Path(resolve_path=True, dir_okay=True, file_okay=False),
    help="Output directory for the upgraded model. Must not already exist.",
)
@click.option(
    "--model-type",
    type=click.Choice(["wflow_sbm", "wflow_sediment"]),
    default="wflow_sbm",
    show_default=True,
    help="Type of wflow model.",
)
@click.option(
    "-i",
    "--config",
    type=click.Path(exists=True, resolve_path=True, dir_okay=False),
    default=None,
    help="Path to a simplified upgrade config YAML with config_filename and options.",
)
@click.option(
    "-d",
    "--data-catalog",
    multiple=True,
    help="Path(s) to data catalog file(s).",
)
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet", count=True)
def upgrade(model_root, output, model_type, config, data_catalog, verbose, quiet):
    """Upgrade a wflow model to the latest Wflow.jl version.

    Copies MODEL_ROOT to the output directory and applies all necessary
    upgrade steps based on the wflow_version detected in the config.

    Example usage:
    'hydromt_wflow upgrade ./my_model -o ./my_model_v1 --model-type wflow_sbm -v'
    or
    'hydromt_wflow upgrade ./mymodel -o ./upgraded -i upgrade_opts.yml -d artifact_data'
    """
    log.initialize_logging(level=log.flags_to_level(verbose, quiet))
    output_path = Path(output)
    if output_path.exists():
        raise click.BadParameter(
            f"Output directory '{output}' already exists. "
            "Please provide a path that does not exist.",
            param_hint="'-o'",
        )

    config_filename = None
    options = None
    if config:
        config_filename, options = _parse_upgrade_config(Path(config))

    data_libs = list(data_catalog) if data_catalog else None

    shutil.copytree(model_root, output_path)
    with log.to_file(output_path / "upgrade.log"):
        upgrade_model(
            model_root=output_path,
            model_type=model_type,
            config_filename=config_filename,
            data_libs=data_libs,
            options=options,
        )
