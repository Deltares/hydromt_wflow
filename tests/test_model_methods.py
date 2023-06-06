"""Unit tests for hydromt_wflow methods and workflows"""

import pytest
from os.path import join, dirname, abspath
import warnings
import pdb
import numpy as np
import pandas as pd
from hydromt_wflow.wflow import WflowModel

import logging

TESTDATADIR = join(dirname(abspath(__file__)), "data")
EXAMPLEDIR = join(dirname(abspath(__file__)), "..", "examples")


def test_setup_staticmaps():
    logger = logging.getLogger(__name__)
    # read model from examples folder
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")

    # Initialize model and read results
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)

    # Tests on setup_staticmaps_from_raster
    mod.setup_staticmaps_from_raster(
        raster_fn="merit_hydro",
        reproject_method="average",
        variables=["elevtn"],
        wflow_variables=["input.vertical.altitude"],
        fill_method="nearest",
    )
    assert "elevtn" in mod.staticmaps
    assert mod.get_config("input.vertical.altitude") == "elevtn"

    mod.setup_staticmaps_from_raster(
        raster_fn="globcover",
        reproject_method="mode",
        wflow_variables=["input.vertical.landuse"],
    )
    assert "globcover" in mod.staticmaps
    assert mod.get_config("input.vertical.landuse") == "globcover"

    # Test on exceptions
    with pytest.raises(ValueError, match="Length of variables"):
        mod.setup_staticmaps_from_raster(
            raster_fn="merit_hydro",
            reproject_method="average",
            variables=["elevtn", "lndslp"],
            wflow_variables=["input.vertical.altitude"],
        )
    with pytest.raises(ValueError, match="variables list is not provided"):
        mod.setup_staticmaps_from_raster(
            raster_fn="merit_hydro",
            reproject_method="average",
            wflow_variables=["input.vertical.altitude"],
        )


def test_setup_lake(tmpdir):
    logger = logging.getLogger(__name__)
    # read model from examples folder
    root = join(EXAMPLEDIR, "wflow_piave_subbasin")

    # Initialize model and read results
    mod = WflowModel(root=root, mode="r", data_libs="artifact_data", logger=logger)

    # Create dummy lake rating curves
    lakes = mod.staticgeoms["lakes"]
    lake_id = lakes["waterbody_id"].iloc[0]
    area = lakes["LakeArea"].iloc[0]
    dis = lakes["LakeAvgOut"].iloc[0]
    lvl = lakes["LakeAvgLevel"].iloc[0]
    elev = lakes["Elevation"].iloc[0]
    lvls = np.linspace(0, lvl)

    df = pd.DataFrame(data={"elevtn": (lvls + elev), "volume": (lvls * area)})
    df = df.join(
        pd.DataFrame(
            {"elevtn": (lvls[-5:-1] + elev), "discharge": np.linspace(0, dis, num=4)}
        ).set_index("elevtn"),
        on="elevtn",
    )
    fn_lake = join(tmpdir, f"rating_curve_{lake_id}.csv")
    df.to_csv(fn_lake, sep=",", index=False, header=True)

    # Register as new data source
    mod.data_catalog.from_dict(
        {
            "lake_rating_test_{index}": {
                "data_type": "DataFrame",
                "driver": "csv",
                "path": join(tmpdir, "rating_curve_{index}.csv"),
                "placeholders": {
                    "index": [str(lake_id)],
                },
            }
        }
    )
    # Update model with it
    mod.setup_lakes(
        lakes_fn="hydro_lakes",
        rating_curve_fns=[f"lake_rating_test_{lake_id}"],
        min_area=5,
    )

    assert f"lake_sh_{lake_id}" in mod.tables
    assert f"lake_hq_{lake_id}" in mod.tables
    assert 2 in np.unique(mod.staticmaps["LakeStorFunc"].values)
    assert 1 in np.unique(mod.staticmaps["LakeOutflowFunc"].values)

    # Write and read back
    mod.set_root(join(tmpdir, "wflow_lake_test"))
    mod.write_tables()
    test_table = mod.tables[f"lake_sh_{lake_id}"]
    mod._tables = dict()
    mod.read_tables()

    assert mod.tables[f"lake_sh_{lake_id}"].equals(test_table)
