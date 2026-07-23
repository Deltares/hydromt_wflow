import platform
from pathlib import Path

from hydromt.writers import write_nc

from hydromt_wflow import WflowSbmModel


def main() -> None:
    """Generate floodplain layer test data from the example wflow model."""
    repo_root = Path(__file__).parent.parent
    examples_dir = repo_root / "examples"
    output_path = repo_root / "tests" / "data" / "floodplain_layers.nc"
    if platform.system().lower() == "linux":
        examples_dir = examples_dir / "linux64"
        output_path = repo_root / "tests" / "data" / "linux64" / "floodplain_layers.nc"

    demand_data_catalog_path = (
        repo_root / "examples" / "data" / "demand" / "data_catalog.yml"
    )
    example_wflow_model = WflowSbmModel(
        root=str(examples_dir / "wflow_piave_subbasin"),
        mode="r",
        data_libs=["artifact_data", str(demand_data_catalog_path)],
    )

    flood_depths = [0.5, 1.0, 1.5, 2.0, 2.5]

    # Setup rivers for both land_elevation and meta_subgrid_elevation to test both cases
    example_wflow_model.setup_rivers(
        hydrography_fn="merit_hydro",
        river_geom_fn="hydro_rivers_lin",
        river_upa=30,
        rivdph_method="powlaw",
        min_rivdph=1,
        min_rivwth=30,
        slope_len=2000,
        smooth_len=5000,
        river_routing="local_inertial",
        elevtn_map="land_elevation",
        output_names={},
    )

    example_wflow_model.setup_rivers(
        hydrography_fn="merit_hydro",
        river_geom_fn="hydro_rivers_lin",
        river_upa=30,
        rivdph_method="powlaw",
        min_rivdph=1,
        min_rivwth=30,
        slope_len=2000,
        smooth_len=5000,
        river_routing="local_inertial",
        elevtn_map="meta_subgrid_elevation",
        output_names={},
    )

    # Setup floodplains 1d and 2d for both land_elevation and meta_subgrid_elevation
    example_wflow_model.setup_floodplains(
        hydrography_fn="merit_hydro",
        floodplain_type="1d",
        river_upa=30,
        flood_depths=flood_depths,
    )

    example_wflow_model.setup_floodplains(
        hydrography_fn="merit_hydro",
        floodplain_type="2d",
        elevtn_map="land_elevation",
    )

    example_wflow_model.setup_floodplains(
        hydrography_fn="merit_hydro",
        floodplain_type="2d",
        elevtn_map="meta_subgrid_elevation",
    )

    # Grab all relevant floodplain variables and write to a netcdf file for testing
    floodplain_vars = [
        "river_bank_elevation_avg",
        "river_bank_elevation_subgrid",
        "floodplain_volume",
        "river_bank_elevation_avg_D4",
        "river_bank_elevation_subgrid_D4",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_nc(
        example_wflow_model.staticmaps.data[floodplain_vars],
        file_path=output_path,
        force_overwrite=True,
    )


if __name__ == "__main__":
    main()
