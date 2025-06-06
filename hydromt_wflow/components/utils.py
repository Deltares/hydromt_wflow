"""Utility of the wflow components."""

from os.path import relpath
from pathlib import Path
from typing import Any

from tomlkit import TOMLDocument
from tomlkit.items import Table


def _relpath(
    value: Any,
    root: Path,
):
    """Generate a relative path."""
    if isinstance(value, str) and Path(value).is_absolute():
        value = Path(value)
    if isinstance(value, Path):
        try:
            value = relpath(value, root)
        except ValueError:
            pass  # `value` path is not relative to root
        finally:
            return value.as_posix()
    return value


def make_config_paths_relative(
    data: TOMLDocument,
    root: Path,
):
    """_summary_.

    Parameters
    ----------
    data : TOMLDocument
        _description_
    root : Path
        _description_
    """
    for key, val in data.items():
        if isinstance(val, Table):
            data.update({key: make_config_paths_relative(val, root)})
        else:
            data.update({key: _relpath(val, root)})
    return data
