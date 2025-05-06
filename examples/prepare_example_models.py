from hydromt_wflow import WflowModel
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent

def create_wflow_piave_subbasin() -> WflowModel:
    """Create a Wflow model for the Piave subbasin example.

    All parameters and function calls are taken from `wflow_build.yml` and the example `build_model.ipynb`

    """
    model = WflowModel(
        # To be changed to `wflow_piave_subbasin`, but leave for now to be able to verify the produced model
        root=EXAMPLES_DIR / "wflow_piave_example",
        mode="w",
        data_libs=["artifact_data"]
    )

    model.setup_config(
        **{
            "time": {
                "starttime": "2010-01-01T00:00:00",
                "endtime": "2010-03-31T00:00:00",
                "timestepsecs": 86400,
            },
            "input": {
                "path_forcing": "inmaps-era5-2010.nc",
            },
            "output": {
                "netcdf_grid": {
                    "path": "output.nc",
                    "compressionlevel": 1,
                    "variables": {
                        "river_water__volume_flow_rate": "q_av",
                    },
                },
            }
        }
    )

    model.setup_basemaps(
        region={'subbasin': [12.2051, 45.8331], 'strord': 4, 'bounds': [11.70, 45.35, 12.95, 46.70]},
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
        res=0.00833,
        upscale_method="ihu",
    )

    model.setup_rivers(
        hydrography_fn="merit_hydro",
        river_geom_fn="hydro_rivers_lin",
        river_upa=30,
        rivdph_method="powlaw",
        min_rivdph=1,
        min_rivwth=30,
        slope_len=2000,
        smooth_len=5000,
        river_routing="kinematic-wave",
    )

    model.setup_reservoirs(
        reservoirs_fn="hydro_reservoirs",
        timeseries_fn="gww",
        min_area=1.0,
    )

    model.setup_lakes(
        lakes_fn="hydro_lakes",
        min_area=10.0,
    )

    model.setup_glaciers(
        glaciers_fn="rgi",
        min_area=0.0,
    )

    model.setup_lulcmaps(
        lulc_fn="globcover_2009",
        lulc_mapping_fn="globcover_mapping_default",
    )

    model.setup_laimaps(
        lai_fn="modis_lai",
    )

    model.setup_soilmaps(
        soil_fn="soilgrids",
        ptf_ksatver="brakensiek",
    )

    model.setup_outlets(
        river_only=True,
    )

    model.setup_gauges(
        gauges_fn="grdc",
        snap_to_river=True,
        derive_subcatch=False,
    )

    model.setup_precip_forcing(
        precip_fn="era5",
        precip_clim_fn=None,
    )

    model.setup_temp_pet_forcing(
        temp_pet_fn="era5",
        press_correction=True,
        temp_correction=True,
        dem_forcing_fn="era5_orography",
        pet_method="debruin",
        skip_pet=False,
    )

    model.setup_constant_pars(
        **{
            "subsurface_water__horizontal-to-vertical_saturated_hydraulic_conductivity_ratio": 100,
            "snowpack__degree-day_coefficient": 3.75653,
            "soil_surface_water__infiltration_reduction_parameter": 0.038,
            "vegetation_canopy_water__mean_evaporation-to-mean_precipitation_ratio": 0.11,
            "soil~compacted_surface_water__infiltration_capacity": 5,
            "soil_water_sat-zone_bottom__max_leakage_volume_flux": 0,
            "soil_root~wet__sigmoid_function_shape_parameter": -500,
            "atmosphere_air__snowfall_temperature_threshold": 0,
            "atmosphere_air__snowfall_temperature_interval": 2,
            "snowpack__melting_temperature_threshold": 0,
            "snowpack__liquid_water_holding_capacity": 0.1,
            "glacier_ice__degree-day_coefficient": 5.3,
            "glacier_firn_accumulation__snowpack~dry_leq-depth_fraction": 0.002,
            "glacier_ice__melting_temperature_threshold": 1.3,
        }
    )

    model.write()

    return model

def create_clipped_model(non_clipped_root: Path) -> WflowModel:
    model = WflowModel(
        root=str(non_clipped_root),
        mode="r",
        data_libs=["artifact_data"],
    )
    model.set_root(
        root=str(EXAMPLES_DIR / "wflow_test_clip_example"),
        mode="w",
    )
    model.clip_grid(
        region={'subbasin': [12.3006, 46.4324], 'strord': 4},
    )
    model.clip_forcing()
    model.clip_states()
    model.write()
    return model



if __name__ == "__main__":
    # model = create_wflow_piave_subbasin()
    create_clipped_model(non_clipped_root=EXAMPLES_DIR / "wflow_piave_example")
