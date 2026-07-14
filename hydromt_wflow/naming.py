"""Mapping dictionaries from hydromt to hydromt_wflow to Wflow.jl names."""

# Names that cannot be read from TOML but that HydroMT needs for model building
# {hydromt_name: staticmap_name}
HYDROMT_NAMES_DEFAULT: dict[str, str] = {
    "glacareas": "meta_glacier_area_id",
}

HYDROMT_NAMES_DEFAULT_SEDIMENT: dict[str, str] = {
    "elevtn": "land_elevation",
}

HYDROMT_NAMES_COMMON: dict[str, str] = {
    "subelv": "meta_subgrid_elevation",
    "uparea": "meta_upstream_area",
    "subare": "meta_subgrid_area",
    "strord": "meta_streamorder",
    "x_out": "meta_subgrid_outlet_x",
    "y_out": "meta_subgrid_outlet_y",
    "landuse": "meta_landuse",
    "soil_texture": "meta_soil_texture",
}

HYDROMT_NAMES_DEFAULT.update(HYDROMT_NAMES_COMMON)
HYDROMT_NAMES_DEFAULT_SEDIMENT.update(HYDROMT_NAMES_COMMON)

# Variables in the input section rather than input.static
WFLOW_VARS_IN_INPUT = [
    "basin__local_drain_direction",
    "subbasin_location__count",
    "river_location__mask",
    "reservoir_area__count",
    "reservoir_location__count",
    "reservoir_lower_location__count",
]

# Link between staticmap names, hydromt name (if any)
# and Wflow.jl v1 variables (if not present, None)
# {staticmap_name: {wflow_name, hydromt_name}}
WFLOW_NAMES: dict[str, dict[str, str | None]] = {
    # general input
    "subcatchment": {
        "wflow_name": "subbasin_location__count",
        "hydromt_name": "basins",
    },
    "local_drain_direction": {
        "wflow_name": "basin__local_drain_direction",
        "hydromt_name": "flwdir",
    },
    "lake_area_id": {
        "wflow_name": None,
        "hydromt_name": "lake_area_id",
    },
    "lake_outlet_id": {
        "wflow_name": None,
        "hydromt_name": "lake_outlet_id",
    },
    "reservoir_area_id": {
        "wflow_name": "reservoir_area__count",
        "hydromt_name": "reservoir_area_id",
    },
    "reservoir_outlet_id": {
        "wflow_name": "reservoir_location__count",
        "hydromt_name": "reservoir_outlet_id",
    },
    "river_mask": {
        "wflow_name": "river_location__mask",
        "hydromt_name": "rivmsk",
    },
    "pits": {
        "wflow_name": "basin_pit_location__mask ",
    },
    # atmosphere / forcing
    "precip": {
        "wflow_name": "atmosphere_water__precipitation_volume_flux",
        "hydromt_name": "precip",
    },
    "temp": {
        "wflow_name": "atmosphere_air__temperature",
        "hydromt_name": "temp",
    },
    "pet": {
        "wflow_name": "land_surface_water__potential_evaporation_volume_flux",
        "hydromt_name": "pet",
    },
    # snow
    "snow_degree_day_coefficient": {
        "wflow_name": "snowpack__degree_day_coefficient",
    },
    "snow_tt": {
        "wflow_name": "atmosphere_air__snowfall_temperature_threshold",
    },
    "snow_tti": {
        "wflow_name": "atmosphere_air__snowfall_temperature_interval",
    },
    "snow_ttm": {
        "wflow_name": "snowpack__melting_temperature_threshold",
    },
    "snow_whc": {
        "wflow_name": "snowpack__liquid_water_holding_capacity",
    },
    # glacier
    "glacier_fraction": {
        "wflow_name": "glacier_surface__area_fraction",
        "hydromt_name": "glacfracs",
    },
    "glacier_initial_leq_depth": {
        "wflow_name": "glacier_ice__initial_leq_depth",
        "hydromt_name": "glacstore",
    },
    "glacier_ttm": {
        "wflow_name": "glacier_ice__melting_temperature_threshold",
    },
    "glacier_degree_day_coefficient": {
        "wflow_name": "glacier_ice__degree_day_coefficient",
    },
    "glacier_snow_to_ice_fraction": {
        "wflow_name": "glacier_firn_accumulation__snowpack_dry_snow_leq_depth_fraction",
    },
    # vegetation
    "vegetation_gash_e_r": {
        "wflow_name": "vegetation_canopy_water__mean_evaporation_to_mean_precipitation_ratio",  # noqa: E501
    },
    "vegetation_kext": {
        "wflow_name": "vegetation_canopy__light_extinction_coefficient",
    },
    "vegetation_leaf_area_index": {
        "wflow_name": "vegetation__leaf_area_index",
        "hydromt_name": "LAI",
    },
    "vegetation_root_depth": {
        "wflow_name": "vegetation_root__depth",
    },
    "vegetation_leaf_storage": {
        "wflow_name": "vegetation__specific_leaf_storage",
        "hydromt_name": "leaf_storage",
    },
    "vegetation_wood_storage": {
        "wflow_name": "vegetation_wood_water__storage_capacity",
        "hydromt_name": "wood_storage",
    },
    "vegetation_crop_factor": {
        "wflow_name": "vegetation__crop_factor",
    },
    "vegetation_feddes_alpha_h1": {
        "wflow_name": "vegetation_root__feddes_critical_pressure_head_h1_reduction_coefficient",  # noqa: E501
    },
    "vegetation_feddes_h1": {
        "wflow_name": "vegetation_root__feddes_critical_pressure_head_h1",
    },
    "vegetation_feddes_h2": {
        "wflow_name": "vegetation_root__feddes_critical_pressure_head_h2",
    },
    "vegetation_feddes_h3_high": {
        "wflow_name": "vegetation_root__feddes_critical_pressure_head_h3_high",
    },
    "vegetation_feddes_h3_low": {
        "wflow_name": "vegetation_root__feddes_critical_pressure_head_h3_low",
    },
    "vegetation_feddes_h4": {
        "wflow_name": "vegetation_root__feddes_critical_pressure_head_h4",
    },
    # soil
    "soil_brooks_corey_c": {
        "wflow_name": "soil_layer_water__brooks_corey_exponent",
        "hydromt_name": "soil_brooks_corey_c",
    },
    "soil_cf": {
        "wflow_name": "soil_surface_water__infiltration_reduction_parameter",
    },
    "soil_ksat_vertical": {
        "wflow_name": "soil_surface_water__vertical_saturated_hydraulic_conductivity",
        "hydromt_name": "ksat_vertical",
    },
    "soil_ksat_vertical_factor": {
        "wflow_name": "soil_layer_water__vertical_saturated_hydraulic_conductivity_factor",  # noqa: E501
    },
    "subsurface_ksat_horizontal_ratio": {
        "wflow_name": "subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio",  # noqa: E501
    },
    "soil_f": {
        "wflow_name": "soil_water__vertical_saturated_hydraulic_conductivity_scale_parameter",  # noqa: E501
        "hydromt_name": "f",
    },
    "soil_compacted_infiltration_capacity": {
        "wflow_name": "compacted_soil_surface_water__infiltration_capacity",
    },
    "InfiltCapSoil": {
        "wflow_name": None,
    },
    "soil_theta_r": {
        "wflow_name": "soil_water__residual_volume_fraction",
        "hydromt_name": "theta_r",
    },
    "soil_theta_s": {
        "wflow_name": "soil_water__saturated_volume_fraction",
        "hydromt_name": "theta_s",
    },
    "soil_max_leakage": {
        "wflow_name": "soil_water_saturated_zone_bottom__max_leakage_volume_flux",
    },
    "soil_compacted_fraction": {
        "wflow_name": "compacted_soil__area_fraction",
    },
    "soil_rootdistpar": {
        "wflow_name": "soil_wet_root__sigmoid_function_shape_parameter",
    },
    "soil_thickness": {
        "wflow_name": "soil__thickness",
    },
    # land
    "land_water_fraction": {
        "wflow_name": "land_water_covered__area_fraction",
    },
    "demand_allocation_area_id": {
        "wflow_name": "land_water_allocation_area__count",
        "hydromt_name": "allocation_areas",
    },
    "demand_surface_water_ratio": {
        "wflow_name": "land_surface_water__withdrawal_fraction",
    },
    "demand_domestic_gross": {
        "wflow_name": "domestic__gross_water_demand_volume_flux",
        "hydromt_name": "domestic_gross",
    },
    "demand_domestic_net": {
        "wflow_name": "domestic__net_water_demand_volume_flux",
        "hydromt_name": "domestic_net",
    },
    "demand_industry_gross": {
        "wflow_name": "industry__gross_water_demand_volume_flux",
        "hydromt_name": "industry_gross",
    },
    "demand_industry_net": {
        "wflow_name": "industry__net_water_demand_volume_flux",
        "hydromt_name": "industry_net",
    },
    "demand_livestock_gross": {
        "wflow_name": "livestock__gross_water_demand_volume_flux",
        "hydromt_name": "livestock_gross",
    },
    "demand_livestock_net": {
        "wflow_name": "livestock__net_water_demand_volume_flux",
        "hydromt_name": "livestock_net",
    },
    "demand_paddy_h_min": {
        "wflow_name": "irrigated_paddy__min_depth",
    },
    "demand_paddy_h_opt": {
        "wflow_name": "irrigated_paddy__optimal_depth",
    },
    "demand_paddy_h_max": {
        "wflow_name": "irrigated_paddy__max_depth",
    },
    "demand_paddy_irrigated_mask": {
        "wflow_name": "irrigated_paddy_area__count",
    },
    "demand_paddy_irrigation_trigger": {
        "wflow_name": "irrigated_paddy__irrigation_trigger_flag",
    },
    "demand_nonpaddy_irrigated_mask": {
        "wflow_name": "irrigated_non_paddy_area__count",
    },
    "demand_nonpaddy_irrigation_trigger": {
        "wflow_name": "irrigated_non_paddy__irrigation_trigger_flag",
    },
    # land surface water flow
    "land_manning_n": {
        "wflow_name": "land_surface_water_flow__manning_n_parameter",
    },
    "land_elevation_D4": {
        "wflow_name": "land_surface_water_flow__ground_elevation",
        "hydromt_name": "hydrodem_D4",
    },
    "land_elevation": {
        "wflow_name": "land_surface__elevation",
        "hydromt_name": "elevtn",
    },
    "land_slope": {
        "wflow_name": "land_surface__slope",
        "hydromt_name": "lndslp",
    },
    # river
    "floodplain_volume": {
        "wflow_name": "floodplain_water__sum_of_volume_per_depth",
        "hydromt_name": "floodplain_volume",
    },
    "floodplain_manning_n": {
        "wflow_name": "floodplain_water_flow__manning_n_parameter",
    },
    "river_bank_elevation": {
        "wflow_name": "river_bank_water__elevation",
        "hydromt_name": "hydrodem",
    },
    "river_inflow": {
        "wflow_name": "river_water__external_inflow_volume_flow_rate",
    },
    "river_depth": {
        "wflow_name": "river_bank_water__depth",
        "hydromt_name": "rivdph",
    },
    "river_length": {
        "wflow_name": "river__length",
        "hydromt_name": "rivlen",
    },
    "river_manning_n": {
        "wflow_name": "river_water_flow__manning_n_parameter",
    },
    "river_slope": {
        "wflow_name": "river__slope",
        "hydromt_name": "rivslp",
    },
    "river_width": {
        "wflow_name": "river__width",
        "hydromt_name": "rivwth",
    },
    # lakes
    "lake_area": {
        "wflow_name": None,
        "hydromt_name": "lake_area",
    },
    "reservoir_initial_depth": {
        "wflow_name": "reservoir_water_surface__initial_elevation",
        "hydromt_name": "reservoir_initial_depth",
    },
    "reservoir_outflow_threshold": {
        "wflow_name": "reservoir_water_flow_threshold_level__elevation",
        "hydromt_name": "reservoir_outflow_threshold",
    },
    "reservoir_b": {
        "wflow_name": "reservoir_water__rating_curve_coefficient",
        "hydromt_name": "reservoir_b",
    },
    "reservoir_e": {
        "wflow_name": "reservoir_water__rating_curve_exponent",
        "hydromt_name": "reservoir_e",
    },
    "reservoir_rating_curve": {
        "wflow_name": "reservoir_water__rating_curve_type_count",
        "hydromt_name": "reservoir_rating_curve",
    },
    "reservoir_storage_curve": {
        "wflow_name": "reservoir_water__storage_curve_type_count",
        "hydromt_name": "reservoir_storage_curve",
    },
    "reservoir_lower_id": {
        "wflow_name": "reservoir_lower_location__count",
        "hydromt_name": "reservoir_lower_id",
    },
    # reservoirs
    "reservoir_area": {
        "wflow_name": "reservoir_surface__area",
        "hydromt_name": "reservoir_area",
    },
    "reservoir_demand": {
        "wflow_name": "reservoir_water_demand__required_downstream_volume_flow_rate",
        "hydromt_name": "reservoir_demand",
    },
    "reservoir_max_release": {
        "wflow_name": "reservoir_water_release_below_spillway__max_volume_flow_rate",
        "hydromt_name": "reservoir_max_release",
    },
    "reservoir_max_volume": {
        "wflow_name": "reservoir_water__max_volume",
        "hydromt_name": "reservoir_max_volume",
    },
    "reservoir_target_full_fraction": {
        "wflow_name": "reservoir_water__target_full_volume_fraction",
        "hydromt_name": "reservoir_target_full_fraction",
    },
    "reservoir_target_min_fraction": {
        "wflow_name": "reservoir_water__target_min_volume_fraction",
        "hydromt_name": "reservoir_target_min_fraction",
    },
    # gwf
    "groundwater_constant_head": {
        "wflow_name": "model_constant_boundary_condition__hydraulic_head",
    },
    "groundwater_ksat_horizontal": {
        "wflow_name": "subsurface_surface_water__horizontal_saturated_hydraulic_conductivity",  # noqa: E501
    },
    "river_bed_elevation": {
        "wflow_name": "river_bottom__elevation",
    },
    "river_bed_exfiltration_conductance": {
        "wflow_name": "river_water__infiltration_conductance",
    },
    "river_bed_infiltration_conductance": {
        "wflow_name": "river_water__exfiltration_conductance",
    },
    "groundwater_specific_yield": {
        "wflow_name": "subsurface_water__specific_yield",
    },
    "groundwater_f": {
        "wflow_name": "subsurface__horizontal_saturated_hydraulic_conductivity_scale_parameter",  # noqa: E501
    },
    "drain": {
        "wflow_name": "land_drain_location__mask",
    },
    "drain_conductance": {
        "wflow_name": "land_drain__conductance",
    },
    "drain_elevation": {
        "wflow_name": "land_drain__elevation",
    },
}

WFLOW_STATES_NAMES: dict[str, dict[str, str | None]] = {
    "vegetation_water_depth": {
        "wflow_name": "vegetation_canopy_water__depth",
    },
    "soil_saturated_depth": {
        "wflow_name": "soil_water_saturated_zone__depth",
    },
    "soil_unsaturated_depth": {
        "wflow_name": "soil_layer_water_unsaturated_zone__depth",
    },
    "soil_temp": {
        "wflow_name": "soil_surface__temperature",
    },
    "snow_leq_depth": {
        "wflow_name": "snowpack_dry_snow__leq_depth",
    },
    "snow_water_depth": {
        "wflow_name": "snowpack_liquid_water__depth",
    },
    "glacier_leq_depth": {
        "wflow_name": "glacier_ice__leq_depth",
    },
    "land_instantaneous_q": {
        "wflow_name": "land_surface_water__instantaneous_volume_flow_rate",
    },
    "land_instantaneous_qx": {
        "wflow_name": "land_surface_water__x_component_of_instantaneous_volume_flow_rate",  # noqa: E501
    },
    "land_instantaneous_qy": {
        "wflow_name": "land_surface_water__y_component_of_instantaneous_volume_flow_rate",  # noqa: E501
    },
    "land_h": {
        "wflow_name": "land_surface_water__depth",
    },
    "h_av_land": {
        "wflow_name": None,
    },
    "groundwater_head": {
        "wflow_name": "subsurface_water__hydraulic_head",
    },
    "subsurface_q": {
        "wflow_name": "subsurface_water__instantaneous_volume_flow_rate",
    },
    "river_instantaneous_q": {
        "wflow_name": "river_water__instantaneous_volume_flow_rate",
    },
    "river_h": {
        "wflow_name": "river_water__depth",
    },
    "h_av_river": {
        "wflow_name": None,
    },
    "floodplain_instantaneous_q": {
        "wflow_name": "floodplain_water__instantaneous_volume_flow_rate",
    },
    "floodplain_h": {
        "wflow_name": "floodplain_water__depth",
    },
    "lake_water_level": {
        "wflow_name": "reservoir_water_surface__elevation",
    },
    "reservoir_volume": {
        "wflow_name": None,
    },
    "demand_paddy_h": {
        "wflow_name": "paddy_surface_water__depth",
    },
}

WFLOW_SEDIMENT_NAMES: dict[str, dict[str, str | None]] = {
    # general input
    "subcatchment": {
        "wflow_name": "subbasin_location__count",
        "hydromt_name": "basins",
    },
    "local_drain_direction": {
        "wflow_name": "basin__local_drain_direction",
        "hydromt_name": "flwdir",
    },
    "lake_area_id": {
        "wflow_name": None,
        "hydromt_name": "lake_area_id",
    },
    "lake_outlet_id": {
        "wflow_name": None,
        "hydromt_name": "lake_outlet_id",
    },
    "reservoir_area_id": {
        "wflow_name": "reservoir_area__count",
        "hydromt_name": "reservoir_area_id",
    },
    "reservoir_outlet_id": {
        "wflow_name": "reservoir_location__count",
        "hydromt_name": "reservoir_outlet_id",
    },
    "river_mask": {
        "wflow_name": "river_location__mask",
        "hydromt_name": "rivmsk",
    },
    # forcing
    "precip": {
        "wflow_name": "atmosphere_water__precipitation_volume_flux",
        "hydromt_name": "precip",
    },
    "vegetation_interception": {
        "wflow_name": "vegetation_canopy_water__interception_volume_flux",
        "hydromt_name": "interception",
    },
    "land_h": {
        "wflow_name": "land_surface_water__depth",
    },
    "land_q": {
        "wflow_name": "land_surface_water__volume_flow_rate",
    },
    "river_h": {
        "wflow_name": "river_water__depth",
    },
    "river_q": {
        "wflow_name": "river_water__volume_flow_rate",
    },
    # land properties
    "land_elevation": {
        "wflow_name": "land_surface__elevation",
        "hydromt_name": "elevtn",
    },
    "land_slope": {
        "wflow_name": "land_surface__slope",
        "hydromt_name": "lndslp",
    },
    # river properties
    "river_length": {
        "wflow_name": "river__length",
        "hydromt_name": "rivlen",
    },
    "river_slope": {
        "wflow_name": "river__slope",
        "hydromt_name": "rivslp",
    },
    "river_width": {
        "wflow_name": "river__width",
        "hydromt_name": "rivwth",
    },
    # reservoirs
    "reservoir_area": {
        "wflow_name": "reservoir_surface__area",
        "hydromt_name": "reservoir_area",
    },
    "lake_area": {
        "wflow_name": None,
        "hydromt_name": "lake_area",
    },
    "reservoir_trapping_efficiency": {
        "wflow_name": "reservoir_water_sediment__bedload_trapping_efficiency",
        "hydromt_name": "reservoir_trapping_efficiency",
    },
    # soil erosion
    "vegetation_gap_fraction": {
        "wflow_name": "vegetation_canopy__gap_fraction",
    },
    "vegetation_height": {
        "wflow_name": "vegetation_canopy__height",
    },
    "vegetation_kext": {
        "wflow_name": None,
    },
    "leaf_storage": {
        "wflow_name": None,
        "hydromt_name": "leaf_storage",
    },
    "vegetation_wood_storage": {
        "wflow_name": None,
        "hydromt_name": "wood_storage",
    },
    "soil_compacted_fraction": {
        "wflow_name": "compacted_soil__area_fraction",
    },
    "land_water_fraction": {
        "wflow_name": "land_water_covered__area_fraction",
    },
    "erosion_soil_detachability": {
        "wflow_name": "soil_erosion__rainfall_soil_detachability_factor",
    },
    "erosion_eurosem_exp": {
        "wflow_name": "soil_erosion__eurosem_exponent",
    },
    "erosion_usle_k": {
        "wflow_name": "soil_erosion__usle_k_factor",
        "hydromt_name": "usle_k",
    },
    "erosion_usle_c": {
        "wflow_name": "soil_erosion__usle_c_factor",
        "hydromt_name": "usle_c",
    },
    "erosion_answers_sheet_factor": {
        "wflow_name": "soil_erosion__answers_overland_flow_factor",
    },
    "erosion_answers_rain_factor": {
        "wflow_name": "soil_erosion__answers_rainfall_factor",
    },
    # soil particles
    "soil_clay_fraction": {
        "wflow_name": "soil_clay__mass_fraction",
    },
    "soil_silt_fraction": {
        "wflow_name": "soil_silt__mass_fraction",
    },
    "soil_sand_fraction": {
        "wflow_name": "soil_sand__mass_fraction",
    },
    "soil_sagg_fraction": {
        "wflow_name": "soil_small_aggregates__mass_fraction",
    },
    "soil_lagg_fraction": {
        "wflow_name": "soil_large_aggregates__mass_fraction",
    },
    # land transport
    "land_govers_c": {
        "wflow_name": "land_surface_water_sediment__govers_transport_capacity_coefficient",  # noqa: E501
    },
    "land_govers_n": {
        "wflow_name": "land_surface_water_sediment__govers_transport_capacity_exponent",
    },
    "soil_sediment_d50": {
        "wflow_name": "land_surface_sediment__median_diameter",
    },
    # river transport
    "sediment_density": {
        "wflow_name": "sediment__particle_density",
    },
    "river_sediment_d50": {
        "wflow_name": None,
    },
    "river_bagnold_c": {
        "wflow_name": "river_water_sediment__bagnold_transport_capacity_coefficient",
    },
    "river_bagnold_exp": {
        "wflow_name": "river_water_sediment__bagnold_transport_capacity_exponent",
    },
    "river_kodatie_a": {
        "wflow_name": "river_water_sediment__kodatie_transport_capacity_a_coefficient",
    },
    "river_kodatie_b": {
        "wflow_name": "river_water_sediment__kodatie_transport_capacity_b_coefficient",
    },
    "river_kodatie_c": {
        "wflow_name": "river_water_sediment__kodatie_transport_capacity_c_coefficient",
    },
    "river_kodatie_d": {
        "wflow_name": "river_water_sediment__kodatie_transport_capacity_d_coefficient",
    },
    "river_bed_sediment_d50": {
        "wflow_name": "river_bottom_and_bank_sediment__median_diameter",
    },
    "river_bed_clay_fraction": {
        "wflow_name": "river_bottom_and_bank_clay__mass_fraction",
    },
    "river_bed_silt_fraction": {
        "wflow_name": "river_bottom_and_bank_silt__mass_fraction",
    },
    "river_bed_sand_fraction": {
        "wflow_name": "river_bottom_and_bank_sand__mass_fraction",
    },
    "river_bed_gravel_fraction": {
        "wflow_name": "river_bottom_and_bank_gravel__mass_fraction",
    },
    "sediment_clay_dm": {
        "wflow_name": "clay__mean_diameter",
    },
    "sediment_silt_dm": {
        "wflow_name": "silt__mean_diameter",
    },
    "sediment_sand_dm": {
        "wflow_name": "sand__mean_diameter",
    },
    "sediment_sagg_dm": {
        "wflow_name": "sediment_small_aggregates__mean_diameter",
    },
    "sediment_lagg_dm": {
        "wflow_name": "sediment_large_aggregates__mean_diameter",
    },
    "sediment_gravel_dm": {
        "wflow_name": "gravel__mean_diameter",
    },
}

WFLOW_SEDIMENT_STATES_NAMES: dict[str, dict[str, str | None]] = {
    "river_clay_load": {
        "wflow_name": "river_water_clay__mass",
    },
    "river_bed_clay_store": {
        "wflow_name": "river_bed_clay__mass",
    },
    "river_clay_flux": {
        "wflow_name": "river_water_clay__mass_flow_rate",
    },
    "river_gravel_load": {
        "wflow_name": "river_water_gravel__mass",
    },
    "river_bed_gravel_store": {
        "wflow_name": "river_bed_gravel__mass",
    },
    "river_gravel_flux": {
        "wflow_name": "river_water_gravel__mass_flow_rate",
    },
    "river_lagg_load": {
        "wflow_name": "river_water_large_aggregates__mass",
    },
    "river_bed_lagg_store": {
        "wflow_name": "river_bed_large_aggregates__mass",
    },
    "river_lagg_flux": {
        "wflow_name": "river_water_large_aggregates__mass_flow_rate",
    },
    "river_sagg_load": {
        "wflow_name": "river_water_small_aggregates__mass",
    },
    "river_bed_sagg_store": {
        "wflow_name": "river_bed_small_aggregates__mass",
    },
    "river_sagg_flux": {
        "wflow_name": "river_water_small_aggregates__mass_flow_rate",
    },
    "river_sand_load": {
        "wflow_name": "river_water_sand__mass",
    },
    "river_bed_sand_store": {
        "wflow_name": "river_bed_sand__mass",
    },
    "river_sand_flux": {
        "wflow_name": "river_water_sand__mass_flow_rate",
    },
    "river_silt_load": {
        "wflow_name": "river_water_silt__mass",
    },
    "river_bed_silt_store": {
        "wflow_name": "river_bed_silt__mass",
    },
    "river_silt_flux": {
        "wflow_name": "river_water_silt__mass_flow_rate",
    },
}


def _create_hydromt_wflow_mapping(
    hydromt_dict: dict,
    model_dict: dict,
    config_dict: dict,
) -> tuple[dict, dict]:
    """
    Create mappings between hydromt names, staticmap layers, and Wflow TOML variables.

    Builds two lookup dictionaries from the naming registry and then patches them
    with any layer-name overrides found in the model config.

    Parameters
    ----------
    hydromt_dict : dict
        {hydromt_name: staticmap_name} for variables HydroMT needs but Wflow doesn't.
    model_dict : dict
        {staticmap_name: {"wflow_name": ..., "hydromt_name": ...}} canonical registry.
    config_dict : dict
        The model TOML config (latest version).

    Returns
    -------
    hydromt_to_staticmap : dict
        {hydromt_name: staticmap_name} for renaming HydroMT variables to staticmap
        layers.
    staticmap_to_wflow : dict
        {staticmap_name: wflow_variable} for linking staticmap layers to Wflow
        variables.
    """
    # 1. Build default {hydromt_name: staticmap_name} from both sources
    hydromt_to_staticmap = dict(hydromt_dict)
    for staticmap_name, entry in model_dict.items():
        if not isinstance(entry, dict):
            continue
        hydromt_name = entry.get("hydromt_name", staticmap_name)
        hydromt_to_staticmap[hydromt_name] = staticmap_name

    # 2. Build default {wflow_variable: staticmap_name}
    wflow_to_staticmap: dict[str, str] = {}
    for staticmap_name, entry in model_dict.items():
        if not isinstance(entry, dict):
            continue
        wflow_var = entry.get("wflow_name")
        if wflow_var is not None:
            wflow_to_staticmap[wflow_var] = staticmap_name

    # 3. Override staticmap names from config (users can customize which staticmap_name
    #    a variable reads from in any of: input.<var>, input.static.<var>,
    #    input.forcing.<var>, input.cyclic.<var>)
    if "input" in config_dict:
        for var_type in ("", "forcing", "cyclic", "static"):
            if var_type == "":
                vars_dict = {
                    k: config_dict["input"][k]
                    for k in WFLOW_VARS_IN_INPUT
                    if k in config_dict["input"]
                }
            else:
                vars_dict = config_dict["input"].get(var_type, {})
                if not isinstance(vars_dict, dict):
                    continue

            for wflow_var, custom_name in vars_dict.items():
                # Resolve the actual layer name from possible dict forms
                if isinstance(custom_name, dict):
                    if "value" in custom_name:
                        continue  # constant value, no layer involved
                    custom_name = custom_name.get("netcdf_variable_name")
                if custom_name is None:
                    continue

                default_layer = wflow_to_staticmap.get(wflow_var)
                if default_layer == custom_name:
                    continue  # no change

                # Update hydromt mapping: find the entry pointing to the old layer
                if default_layer is not None:
                    for _hydromt_name, _staticmap_name in hydromt_to_staticmap.items():
                        if _staticmap_name == default_layer:
                            hydromt_to_staticmap[_hydromt_name] = custom_name
                            break

                # Update wflow: staticmap mapping
                wflow_to_staticmap[wflow_var] = custom_name

    # 4. Invert wflow_to_staticmap to get {staticmap_name: wflow_variable}
    staticmap_to_wflow = {
        staticmap_name: wflow_var
        for wflow_var, staticmap_name in wflow_to_staticmap.items()
    }

    return hydromt_to_staticmap, staticmap_to_wflow


def _create_hydromt_wflow_mapping_sbm(config: dict) -> tuple[dict, dict]:
    """
    Create a dictionary to convert from hydromt names to wflow sbm input names.

    These names will be used in staticmaps/forcing and linked to the right wflow
    internal variables in the toml.
    """
    return _create_hydromt_wflow_mapping(HYDROMT_NAMES_DEFAULT, WFLOW_NAMES, config)


def _create_hydromt_wflow_mapping_sediment(config: dict) -> tuple[dict, dict]:
    """
    Create a dictionary to convert from hydromt names to wflow sediment input names.

    These names will be used in staticmaps/forcing and linked to the right wflow
    internal variables in the toml.
    """
    return _create_hydromt_wflow_mapping(
        HYDROMT_NAMES_DEFAULT_SEDIMENT, WFLOW_SEDIMENT_NAMES, config
    )
