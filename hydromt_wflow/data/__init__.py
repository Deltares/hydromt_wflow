"""Module for accessing hydromt_wflow data."""

from pathlib import Path
from typing import Literal

import pandas as pd
from hydromt.data_catalog import DataCatalog
from hydromt.readers import read_toml

DATA_DIR = Path(__file__).parent


def default_config_headers() -> dict:
    """Load the v1 config template."""
    return read_toml(DATA_DIR / "default_config_headers.toml")


def default_config(model_type: Literal["wflow_sbm", "wflow_sediment"]) -> dict:
    """Load the default config for the specified model type."""
    return read_toml(DATA_DIR / f"{model_type}" / f"{model_type}.toml")


def parameters_datacatalog(data_catalog: DataCatalog | None = None) -> DataCatalog:
    """Load the parameters data, if provided, add it to the given data catalog."""
    dc = data_catalog or DataCatalog()
    return dc.from_yml(DATA_DIR / "parameters_data.yml")


def regr_chelsa() -> pd.DataFrame:
    """Load the CHELSA climate regridded data."""
    return pd.read_csv(DATA_DIR / "rivwth" / "regr_chelsa.csv", index_col="source")


def koppen_geiger() -> pd.DataFrame:
    """Load the Koppen-Geiger climate classification data."""
    return pd.read_csv(DATA_DIR / "rivwth" / "koppen_geiger.csv", index_col="class")
