"""add global fixtures."""

import logging
import sys
from pathlib import Path

from packaging.version import Version
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Settings for testing purposes, loaded from environment variables or defaults if not set, case-insensitive."""

    plot_on_error: bool = False
    """Whether to save debug plots when ``_compare_wflow_models`` fails. If True, will save debug plots to the directory specified by ``plots_dir``."""

    plots_dir: Path | None = None
    """Directory to save debug plots when ``_compare_wflow_models`` fails.
    If not set and ``plot_on_error`` is True, will default to a "debug_plots" directory in the current working directory."""

    log_level: int = None
    """Log level for hydromt_wflow logs. If not set, will default to INFO."""

    log_level_hydromt: int = None
    """Log level for hydromt logs. If not set, will use the same log level as log_level."""

    log_file: Path | None = None
    """Log file path. If not set, logs will only be printed to the console."""

    python_version: Version = None
    """Setting for ``_compare_wflow_models``. We only do exact comparisons for versions greater than or equal to this version, as older versions may have different dependency versions and/or behavior."""

    @field_validator("log_file", "plots_dir", mode="before")
    @classmethod
    def _coerce_str_to_path(cls, v):
        if v is not None:
            if isinstance(v, str):
                v = Path(v).resolve()
        return v

    @field_validator("log_level", "log_level_hydromt", mode="before")
    @classmethod
    def _coerce_log_level_to_int(cls, v) -> int:
        if v is None:
            return logging.INFO
        elif isinstance(v, int):
            if v in logging._levelToName:
                return v
            else:
                raise ValueError(
                    f"log_level must be one of {list(logging._levelToName.keys())}, got {v}"
                )
        elif isinstance(v, str):
            if v.upper() not in logging._nameToLevel:
                raise ValueError(
                    f"log_level must be one of {list(logging._nameToLevel.keys())}, got {v}"
                )
            return logging._nameToLevel[v.upper()]
        else:
            raise ValueError(f"log_level must be a string or integer, got {type(v)}")

    @field_validator("python_version", mode="before")
    @classmethod
    def _coerce_python_version(cls, v):
        if v is None:
            return Version(
                f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            )
        elif isinstance(v, str):
            return Version(v)
        elif isinstance(v, Version):
            return v
        else:
            raise ValueError(
                f"python_version must be a string or Version instance, got {type(v)}"
            )

    @model_validator(mode="after")
    def _ensure_plots_dir_if_debug(self):
        if self.plot_on_error and self.plots_dir is None:
            self.plots_dir = Path.cwd() / "debug_plots"
        return self

    def should_assert(self, min_version: Version = Version("3.13")) -> bool:
        """Whether to do exact comparisons and assertions in ``_compare_wflow_models``.

        We only assert on the exact comparisons for versions greater than or equal to
        the specified version, as older versions may have different dependency versions and/or behavior.
        """
        return self.python_version >= min_version
