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
# and Wflow.jl variables for v0x and v1x (if not present, None)
# {staticmap_name: (wflow_v0, wflow_v1, hydromt_name)} or
# {staticmap_name: (wflow_v0, wflow_v1)}
WFLOW_NAMES: dict[str, dict[str, str | None]] = {
    # general input
    "subcatchment": {
        "wflow_v0": "subcatchment",
        "wflow_v1": "subbasin_location__count",
        "hydromt_name": "basins",
    },
    "local_drain_direction": {
        "wflow_v0": "ldd",
        "wflow_v1": "basin__local_drain_direction",
        "hydromt_name": "flwdir",
    },
    "lake_area_id": {
        "wflow_v0": "lateral.river.lake.areas",
        "wflow_v1": None,
        "hydromt_name": "lake_area_id",
    },
    "lake_outlet_id": {
        "wflow_v0": "lateral.river.lake.locs",
        "wflow_v1": None,
        "hydromt_name": "lake_outlet_id",
    },
    "reservoir_area_id": {
        "wflow_v0": "lateral.river.reservoir.areas",
        "wflow_v1": "reservoir_area__count",
        "hydromt_name": "reservoir_area_id",
    },
    "reservoir_outlet_id": {
        "wflow_v0": "lateral.river.reservoir.locs",
        "wflow_v1": "reservoir_location__count",
        "hydromt_name": "reservoir_outlet_id",
    },
    "river_mask": {
        "wflow_v0": "river_location",
        "wflow_v1": "river_location__mask",
        "hydromt_name": "rivmsk",
    },
    "pits": {
        "wflow_v0": "pits",
        "wflow_v1": "basin_pit_location__mask ",
    },
    # atmosphere / forcing
    "precip": {
        "wflow_v0": "vertical.precipitation",
        "wflow_v1": "atmosphere_water__precipitation_volume_flux",
        "hydromt_name": "precip",
    },
    "temp": {
        "wflow_v0": "vertical.temperature",
        "wflow_v1": "atmosphere_air__temperature",
        "hydromt_name": "temp",
    },
    "pet": {
        "wflow_v0": "vertical.potential_evaporation",
        "wflow_v1": "land_surface_water__potential_evaporation_volume_flux",
        "hydromt_name": "pet",
    },
    # snow
    "snow_degree_day_coefficient": {
        "wflow_v0": "vertical.cfmax",
        "wflow_v1": "snowpack__degree_day_coefficient",
    },
    "snow_tt": {
        "wflow_v0": "vertical.tt",
        "wflow_v1": "atmosphere_air__snowfall_temperature_threshold",
    },
    "snow_tti": {
        "wflow_v0": "vertical.tti",
        "wflow_v1": "atmosphere_air__snowfall_temperature_interval",
    },
    "snow_ttm": {
        "wflow_v0": "vertical.ttm",
        "wflow_v1": "snowpack__melting_temperature_threshold",
    },
    "snow_whc": {
        "wflow_v0": "vertical.water_holding_capacity",
        "wflow_v1": "snowpack__liquid_water_holding_capacity",
    },
    # glacier
    "glacier_fraction": {
        "wflow_v0": "vertical.glacierfrac",
        "wflow_v1": "glacier_surface__area_fraction",
        "hydromt_name": "glacfracs",
    },
    "glacier_initial_leq_depth": {
        "wflow_v0": "vertical.glacierstore",
        "wflow_v1": "glacier_ice__initial_leq_depth",
        "hydromt_name": "glacstore",
    },
    "glacier_ttm": {
        "wflow_v0": ["vertical.g_ttm", "vertical.g_tt"],
        "wflow_v1": "glacier_ice__melting_temperature_threshold",
    },
    "glacier_degree_day_coefficient": {
        "wflow_v0": "vertical.g_cfmax",
        "wflow_v1": "glacier_ice__degree_day_coefficient",
    },
    "glacier_snow_to_ice_fraction": {
        "wflow_v0": "vertical.g_sifrac",
        "wflow_v1": "glacier_firn_accumulation__snowpack_dry_snow_leq_depth_fraction",
    },
    # vegetation
    "vegetation_gash_e_r": {
        "wflow_v0": "vertical.e_r",
        "wflow_v1": "vegetation_canopy_water__mean_evaporation_to_mean_precipitation_ratio",  # noqa: E501
    },
    "vegetation_kext": {
        "wflow_v0": "vertical.kext",
        "wflow_v1": "vegetation_canopy__light_extinction_coefficient",
    },
    "vegetation_leaf_area_index": {
        "wflow_v0": "vertical.leaf_area_index",
        "wflow_v1": "vegetation__leaf_area_index",
        "hydromt_name": "LAI",
    },
    "vegetation_root_depth": {
        "wflow_v0": "vertical.rootingdepth",
        "wflow_v1": "vegetation_root__depth",
    },
    "vegetation_leaf_storage": {
        "wflow_v0": "vertical.specific_leaf",
        "wflow_v1": "vegetation__specific_leaf_storage",
        "hydromt_name": "leaf_storage",
    },
    "vegetation_wood_storage": {
        "wflow_v0": "vertical.storage_wood",
        "wflow_v1": "vegetation_wood_water__storage_capacity",
        "hydromt_name": "wood_storage",
    },
    "vegetation_crop_factor": {
        "wflow_v0": ["vertical.kc", "vertical.etreftopot"],
        "wflow_v1": "vegetation__crop_factor",
    },
    "vegetation_feddes_alpha_h1": {
        "wflow_v0": "vertical.alpha_h1",
        "wflow_v1": "vegetation_root__feddes_critical_pressure_head_h1_reduction_coefficient",  # noqa: E501
    },
    "vegetation_feddes_h1": {
        "wflow_v0": "vertical.h1",
        "wflow_v1": "vegetation_root__feddes_critical_pressure_head_h1",
    },
    "vegetation_feddes_h2": {
        "wflow_v0": "vertical.h2",
        "wflow_v1": "vegetation_root__feddes_critical_pressure_head_h2",
    },
    "vegetation_feddes_h3_high": {
        "wflow_v0": "vertical.h3_high",
        "wflow_v1": "vegetation_root__feddes_critical_pressure_head_h3_high",
    },
    "vegetation_feddes_h3_low": {
        "wflow_v0": "vertical.h3_low",
        "wflow_v1": "vegetation_root__feddes_critical_pressure_head_h3_low",
    },
    "vegetation_feddes_h4": {
        "wflow_v0": "vertical.h4",
        "wflow_v1": "vegetation_root__feddes_critical_pressure_head_h4",
    },
    # soil
    "soil_brooks_corey_c": {
        "wflow_v0": "vertical.c",
        "wflow_v1": "soil_layer_water__brooks_corey_exponent",
        "hydromt_name": "soil_brooks_corey_c",
    },
    "soil_cf": {
        "wflow_v0": "vertical.cf_soil",
        "wflow_v1": "soil_surface_water__infiltration_reduction_parameter",
    },
    "soil_ksat_vertical": {
        "wflow_v0": ["vertical.kv_0", "vertical.kv₀"],
        "wflow_v1": "soil_surface_water__vertical_saturated_hydraulic_conductivity",
        "hydromt_name": "ksat_vertical",
    },
    "soil_ksat_vertical_factor": {
        "wflow_v0": "vertical.kvfrac",
        "wflow_v1": "soil_water__vertical_saturated_hydraulic_conductivity_factor",
    },
    "subsurface_ksat_horizontal_ratio": {
        "wflow_v0": "lateral.subsurface.ksathorfrac",
        "wflow_v1": "subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio",  # noqa: E501
    },
    "soil_f": {
        "wflow_v0": "vertical.f",
        "wflow_v1": "soil_water__vertical_saturated_hydraulic_conductivity_scale_parameter",  # noqa: E501
        "hydromt_name": "f",
    },
    "soil_compacted_infiltration_capacity": {
        "wflow_v0": "vertical.infiltcappath",
        "wflow_v1": "compacted_soil_surface_water__infiltration_capacity",
    },
    "InfiltCapSoil": {
        "wflow_v0": "vertical.infiltcapsoil",
        "wflow_v1": None,
    },
    "soil_theta_r": {
        "wflow_v0": ["vertical.theta_r", "vertical.θᵣ"],
        "wflow_v1": "soil_water__residual_volume_fraction",
        "hydromt_name": "theta_r",
    },
    "soil_theta_s": {
        "wflow_v0": ["vertical.theta_s", "vertical.θₛ"],
        "wflow_v1": "soil_water__saturated_volume_fraction",
        "hydromt_name": "theta_s",
    },
    "soil_max_leakage": {
        "wflow_v0": "vertical.maxleakage",
        "wflow_v1": "soil_water_saturated_zone_bottom__max_leakage_volume_flux",
    },
    "soil_compacted_fraction": {
        "wflow_v0": "vertical.pathfrac",
        "wflow_v1": "compacted_soil__area_fraction",
    },
    "soil_rootdistpar": {
        "wflow_v0": "vertical.rootdistpar",
        "wflow_v1": "soil_wet_root__sigmoid_function_shape_parameter",
    },
    "soil_thickness": {
        "wflow_v0": "vertical.soilthickness",
        "wflow_v1": "soil__thickness",
    },
    # land
    "land_water_fraction": {
        "wflow_v0": "vertical.waterfrac",
        "wflow_v1": "land_water_covered__area_fraction",
    },
    "demand_allocation_area_id": {
        "wflow_v0": "vertical.allocation.areas",
        "wflow_v1": "land_water_allocation_area__count",
        "hydromt_name": "allocation_areas",
    },
    "demand_surface_water_ratio": {
        "wflow_v0": "vertical.allocation.frac_sw_used",
        "wflow_v1": "land_surface_water__withdrawal_fraction",
    },
    "demand_domestic_gross": {
        "wflow_v0": "vertical.domestic.demand_gross",
        "wflow_v1": "domestic__gross_water_demand_volume_flux",
        "hydromt_name": "domestic_gross",
    },
    "demand_domestic_net": {
        "wflow_v0": "vertical.domestic.demand_net",
        "wflow_v1": "domestic__net_water_demand_volume_flux",
        "hydromt_name": "domestic_net",
    },
    "demand_industry_gross": {
        "wflow_v0": "vertical.industry.demand_gross",
        "wflow_v1": "industry__gross_water_demand_volume_flux",
        "hydromt_name": "industry_gross",
    },
    "demand_industry_net": {
        "wflow_v0": "vertical.industry.demand_net",
        "wflow_v1": "industry__net_water_demand_volume_flux",
        "hydromt_name": "industry_net",
    },
    "demand_livestock_gross": {
        "wflow_v0": "vertical.livestock.demand_gross",
        "wflow_v1": "livestock__gross_water_demand_volume_flux",
        "hydromt_name": "livestock_gross",
    },
    "demand_livestock_net": {
        "wflow_v0": "vertical.livestock.demand_net",
        "wflow_v1": "livestock__net_water_demand_volume_flux",
        "hydromt_name": "livestock_net",
    },
    "demand_paddy_h_min": {
        "wflow_v0": "vertical.paddy.h_min",
        "wflow_v1": "irrigated_paddy__min_depth",
    },
    "demand_paddy_h_opt": {
        "wflow_v0": "vertical.paddy.h_opt",
        "wflow_v1": "irrigated_paddy__optimal_depth",
    },
    "demand_paddy_h_max": {
        "wflow_v0": "vertical.paddy.h_max",
        "wflow_v1": "irrigated_paddy__max_depth",
    },
    "demand_paddy_irrigated_mask": {
        "wflow_v0": "vertical.paddy.irrigation_areas",
        "wflow_v1": "irrigated_paddy_area__count",
    },
    "demand_paddy_irrigation_trigger": {
        "wflow_v0": "vertical.paddy.irrigation_trigger",
        "wflow_v1": "irrigated_paddy__irrigation_trigger_flag",
    },
    "demand_nonpaddy_irrigated_mask": {
        "wflow_v0": "vertical.nonpaddy.irrigation_areas",
        "wflow_v1": "land~irrigated-non-paddy_area__count",
    },
    "demand_nonpaddy_irrigation_trigger": {
        "wflow_v0": "vertical.nonpaddy.irrigation_trigger",
        "wflow_v1": "irrigated_non_paddy__irrigation_trigger_flag",
    },
    # land surface water flow
    "land_manning_n": {
        "wflow_v0": "lateral.land.n",
        "wflow_v1": "land_surface_water_flow__manning_n_parameter",
    },
    "land_elevation": {
        "wflow_v0": "lateral.land.elevation",
        "wflow_v1": "land_surface_water_flow__ground_elevation",
        "hydromt_name": "elevtn",
    },
    "land_slope": {
        "wflow_v0": "lateral.land.slope",
        "wflow_v1": "land_surface__slope",
        "hydromt_name": "lndslp",
    },
    # river
    "floodplain_volume": {
        "wflow_v0": "lateral.river.floodplain.volume",
        "wflow_v1": "floodplain_water__sum_of_volume_per_depth",
        "hydromt_name": "floodplain_volume",
    },
    "floodplain_manning_n": {
        "wflow_v0": "lateral.river.floodplain.n",
        "wflow_v1": "floodplain_water_flow__manning_n_parameter",
    },
    "river_bank_elevation": {
        "wflow_v0": "lateral.river.bankfull_elevation",
        "wflow_v1": "river_bank_water__elevation",
        "hydromt_name": "hydrodem",
    },
    "river_inflow": {
        "wflow_v0": "lateral.river.inflow",
        "wflow_v1": "river_water__external_inflow_volume_flow_rate",
    },
    "river_depth": {
        "wflow_v0": "lateral.river.bankfull_depth",
        "wflow_v1": "river_bank_water__depth",
        "hydromt_name": "rivdph",
    },
    "river_length": {
        "wflow_v0": "lateral.river.length",
        "wflow_v1": "river__length",
        "hydromt_name": "rivlen",
    },
    "river_manning_n": {
        "wflow_v0": "lateral.river.n",
        "wflow_v1": "river_water_flow__manning_n_parameter",
    },
    "river_slope": {
        "wflow_v0": "lateral.river.slope",
        "wflow_v1": "river__slope",
        "hydromt_name": "rivslp",
    },
    "river_width": {
        "wflow_v0": "lateral.river.width",
        "wflow_v1": "river__width",
        "hydromt_name": "rivwth",
    },
    # lakes
    "lake_area": {
        "wflow_v0": "lateral.river.lake.area",
        "wflow_v1": None,
        "hydromt_name": "lake_area",
    },
    "reservoir_initial_depth": {
        "wflow_v0": "lateral.river.lake.waterlevel",
        "wflow_v1": "reservoir_water_surface__initial_elevation",
        "hydromt_name": "reservoir_initial_depth",
    },
    "reservoir_outflow_threshold": {
        "wflow_v0": "lateral.river.lake.threshold",
        "wflow_v1": "reservoir_water_flow_threshold_level__elevation",
        "hydromt_name": "reservoir_outflow_threshold",
    },
    "reservoir_b": {
        "wflow_v0": "lateral.river.lake.b",
        "wflow_v1": "reservoir_water__rating_curve_coefficient",
        "hydromt_name": "reservoir_b",
    },
    "reservoir_e": {
        "wflow_v0": "lateral.river.lake.e",
        "wflow_v1": "reservoir_water__rating_curve_exponent",
        "hydromt_name": "reservoir_e",
    },
    "reservoir_rating_curve": {
        "wflow_v0": "lateral.river.lake.outflowfunc",
        "wflow_v1": "reservoir_water__rating_curve_type_count",
        "hydromt_name": "reservoir_rating_curve",
    },
    "reservoir_storage_curve": {
        "wflow_v0": "lateral.river.lake.storfunc",
        "wflow_v1": "reservoir_water__storage_curve_type_count",
        "hydromt_name": "reservoir_storage_curve",
    },
    "reservoir_lower_id": {
        "wflow_v0": "lateral.river.lake.linkedlakelocs",
        "wflow_v1": "reservoir_lower_location__count",
        "hydromt_name": "reservoir_lower_id",
    },
    # reservoirs
    "reservoir_area": {
        "wflow_v0": "lateral.river.reservoir.area",
        "wflow_v1": "reservoir_surface__area",
        "hydromt_name": "reservoir_area",
    },
    "reservoir_demand": {
        "wflow_v0": "lateral.river.reservoir.demand",
        "wflow_v1": "reservoir_water_demand__required_downstream_volume_flow_rate",
        "hydromt_name": "reservoir_demand",
    },
    "reservoir_max_release": {
        "wflow_v0": "lateral.river.reservoir.maxrelease",
        "wflow_v1": "reservoir_water_release_below_spillway__max_volume_flow_rate",
        "hydromt_name": "reservoir_max_release",
    },
    "reservoir_max_volume": {
        "wflow_v0": "lateral.river.reservoir.maxvolume",
        "wflow_v1": "reservoir_water__max_volume",
        "hydromt_name": "reservoir_max_volume",
    },
    "reservoir_target_full_fraction": {
        "wflow_v0": "lateral.river.reservoir.targetfullfrac",
        "wflow_v1": "reservoir_water__target_full_volume_fraction",
        "hydromt_name": "reservoir_target_full_fraction",
    },
    "reservoir_target_min_fraction": {
        "wflow_v0": "lateral.river.reservoir.targetminfrac",
        "wflow_v1": "reservoir_water__target_min_volume_fraction",
        "hydromt_name": "reservoir_target_min_fraction",
    },
    # gwf
    "altitude": {
        "wflow_v0": "altitude",
        "wflow_v1": "land_surface__elevation",
    },
    "groundwater_constant_head": {
        "wflow_v0": "lateral.subsurface.constant_head",
        "wflow_v1": "model_constant_boundary_condition__hydraulic_head",
    },
    "groundwater_ksat_horizontal": {
        "wflow_v0": "lateral.subsurface.conductivity",
        "wflow_v1": "subsurface_surface_water__horizontal_saturated_hydraulic_conductivity",  # noqa: E501
    },
    "river_bed_elevation": {
        "wflow_v0": "lateral.subsurface.river_bottom",
        "wflow_v1": "river_bottom__elevation",
    },
    "river_bed_exfiltration_conductance": {
        "wflow_v0": "lateral.subsurface.infiltration_conductance",
        "wflow_v1": "river_water__infiltration_conductance",
    },
    "river_bed_infiltration_conductance": {
        "wflow_v0": "lateral.subsurface.exfiltration_conductance",
        "wflow_v1": "river_water__exfiltration_conductance",
    },
    "groundwater_specific_yield": {
        "wflow_v0": "lateral.subsurface.specific_yield",
        "wflow_v1": "subsurface_water__specific_yield",
    },
    "groundwater_f": {
        "wflow_v0": "lateral.subsurface.gwf_f",
        "wflow_v1": "subsurface__horizontal_saturated_hydraulic_conductivity_scale_parameter",  # noqa: E501
    },
    "drain": {
        "wflow_v0": "lateral.subsurface.drain",
        "wflow_v1": "land_drain_location__mask",
    },
    "drain_conductance": {
        "wflow_v0": "lateral.subsurface.drain_conductance",
        "wflow_v1": "land_drain__conductance",
    },
    "drain_elevation": {
        "wflow_v0": "lateral.subsurface.drain_elevation",
        "wflow_v1": "land_drain__elevation",
    },
}

WFLOW_STATES_NAMES: dict[str, dict[str, str | None]] = {
    "vegetation_water_depth": {
        "wflow_v0": "vertical.canopystorage",
        "wflow_v1": "vegetation_canopy_water__depth",
    },
    "soil_saturated_depth": {
        "wflow_v0": "vertical.satwaterdepth",
        "wflow_v1": "soil_water_saturated_zone__depth",
    },
    "soil_unsaturated_depth": {
        "wflow_v0": "vertical.ustorelayerdepth",
        "wflow_v1": "soil_layer_water_unsaturated_zone__depth",
    },
    "soil_temp": {
        "wflow_v0": "vertical.tsoil",
        "wflow_v1": "soil_surface__temperature",
    },
    "snow_leq_depth": {
        "wflow_v0": "vertical.snow",
        "wflow_v1": "snowpack_dry_snow__leq_depth",
    },
    "snow_water_depth": {
        "wflow_v0": "vertical.snowwater",
        "wflow_v1": "snowpack_liquid_water__depth",
    },
    "glacier_leq_depth": {
        "wflow_v0": "vertical.glacierstore",
        "wflow_v1": "glacier_ice__leq_depth",
    },
    "land_instantaneous_q": {
        "wflow_v0": "lateral.land.q",
        "wflow_v1": "land_surface_water__instantaneous_volume_flow_rate",
    },
    "land_instantaneous_qx": {
        "wflow_v0": "lateral.land.qx",
        "wflow_v1": "land_surface_water__x_component_of_instantaneous_volume_flow_rate",
    },
    "land_instantaneous_qy": {
        "wflow_v0": "lateral.land.qy",
        "wflow_v1": "land_surface_water__y_component_of_instantaneous_volume_flow_rate",
    },
    "land_h": {
        "wflow_v0": "lateral.land.h",
        "wflow_v1": "land_surface_water__depth",
    },
    "h_av_land": {
        "wflow_v0": "lateral.land.h_av",
        "wflow_v1": None,
    },
    "groundwater_head": {
        "wflow_v0": "lateral.subsurface.flow.aquifer.head",
        "wflow_v1": "subsurface_water__hydraulic_head",
    },
    "subsurface_q": {
        "wflow_v0": "lateral.subsurface.ssf",
        "wflow_v1": "subsurface_water__volume_flow_rate",
    },
    "river_instantaneous_q": {
        "wflow_v0": "lateral.river.q",
        "wflow_v1": "river_water__instantaneous_volume_flow_rate",
    },
    "river_h": {
        "wflow_v0": "lateral.river.h",
        "wflow_v1": "river_water__depth",
    },
    "h_av_river": {
        "wflow_v0": "lateral.river.h_av",
        "wflow_v1": None,
    },
    "floodplain_instantaneous_q": {
        "wflow_v0": "lateral.river.floodplain.q",
        "wflow_v1": "floodplain_water__instantaneous_volume_flow_rate",
    },
    "floodplain_h": {
        "wflow_v0": "lateral.river.floodplain.h",
        "wflow_v1": "floodplain_water__depth",
    },
    "lake_water_level": {
        "wflow_v0": "lateral.river.lake.waterlevel",
        "wflow_v1": "reservoir_water_surface__elevation",
    },
    "reservoir_volume": {
        "wflow_v0": "lateral.river.reservoir.volume",
        "wflow_v1": None,
    },
    "demand_paddy_h": {
        "wflow_v0": "vertical.paddy.h",
        "wflow_v1": "paddy_surface_water__depth",
    },
}

WFLOW_SEDIMENT_NAMES: dict[str, dict[str, str | None]] = {
    # general input
    "subcatchment": {
        "wflow_v0": "subcatchment",
        "wflow_v1": "subbasin_location__count",
        "hydromt_name": "basins",
    },
    "local_drain_direction": {
        "wflow_v0": "ldd",
        "wflow_v1": "basin__local_drain_direction",
        "hydromt_name": "flwdir",
    },
    "lake_area_id": {
        "wflow_v0": "vertical.lakeareas",
        "wflow_v1": None,
        "hydromt_name": "lake_area_id",
    },
    "lake_outlet_id": {
        "wflow_v0": "lateral.river.lakelocs",
        "wflow_v1": None,
        "hydromt_name": "lake_outlet_id",
    },
    "reservoir_area_id": {
        "wflow_v0": "vertical.resareas",
        "wflow_v1": "reservoir_area__count",
        "hydromt_name": "reservoir_area_id",
    },
    "reservoir_outlet_id": {
        "wflow_v0": "lateral.river.reslocs",
        "wflow_v1": "reservoir_location__count",
        "hydromt_name": "reservoir_outlet_id",
    },
    "river_mask": {
        "wflow_v0": "river_location",
        "wflow_v1": "river_location__mask",
        "hydromt_name": "rivmsk",
    },
    # forcing
    "precip": {
        "wflow_v0": "vertical.precipitation",
        "wflow_v1": "atmosphere_water__precipitation_volume_flux",
        "hydromt_name": "precip",
    },
    "vegetation_interception": {
        "wflow_v0": "vertical.interception",
        "wflow_v1": "vegetation_canopy_water__interception_volume_flux",
        "hydromt_name": "interception",
    },
    "land_h": {
        "wflow_v0": "vertical.h_land",
        "wflow_v1": "land_surface_water__depth",
    },
    "land_q": {
        "wflow_v0": "vertical.q_land",
        "wflow_v1": "land_surface_water__volume_flow_rate",
    },
    "river_h": {
        "wflow_v0": "lateral.river.h_riv",
        "wflow_v1": "river_water__depth",
    },
    "river_q": {
        "wflow_v0": "lateral.river.q_riv",
        "wflow_v1": "river_water__volume_flow_rate",
    },
    # land properties
    "land_slope": {
        "wflow_v0": "lateral.land.slope",
        "wflow_v1": "land_surface__slope",
        "hydromt_name": "lndslp",
    },
    # river properties
    "river_length": {
        "wflow_v0": "lateral.river.length",
        "wflow_v1": "river__length",
        "hydromt_name": "rivlen",
    },
    "river_slope": {
        "wflow_v0": "lateral.river.slope",
        "wflow_v1": "river__slope",
        "hydromt_name": "rivslp",
    },
    "river_width": {
        "wflow_v0": "lateral.river.width",
        "wflow_v1": "river__width",
        "hydromt_name": "rivwth",
    },
    # reservoirs
    "reservoir_area": {
        "wflow_v0": "lateral.river.resarea",
        "wflow_v1": "reservoir_surface__area",
        "hydromt_name": "reservoir_area",
    },
    "lake_area": {
        "wflow_v0": "lateral.river.lakearea",
        "wflow_v1": None,
        "hydromt_name": "lake_area",
    },
    "reservoir_trapping_efficiency": {
        "wflow_v0": "lateral.river.restrapeff",
        "wflow_v1": "reservoir_water_sediment__bedload_trapping_efficiency",
        "hydromt_name": "reservoir_trapping_efficiency",
    },
    # soil erosion
    "vegetation_gap_fraction": {
        "wflow_v0": "vertical.canopygapfraction",
        "wflow_v1": "vegetation_canopy__gap_fraction",
    },
    "vegetation_height": {
        "wflow_v0": "vertical.canopyheight",
        "wflow_v1": "vegetation_canopy__height",
    },
    "vegetation_kext": {
        "wflow_v0": "vertical.kext",
        "wflow_v1": None,
    },
    "leaf_storage": {
        "wflow_v0": "vertical.specific_leaf",
        "wflow_v1": None,
        "hydromt_name": "leaf_storage",
    },
    "vegetation_wood_storage": {
        "wflow_v0": "vertical.storage_wood",
        "wflow_v1": None,
        "hydromt_name": "wood_storage",
    },
    "soil_compacted_fraction": {
        "wflow_v0": "vertical.pathfrac",
        "wflow_v1": "compacted_soil__area_fraction",
    },
    "land_water_fraction": {
        "wflow_v0": None,
        "wflow_v1": "land_water_covered__area_fraction",
    },
    "erosion_soil_detachability": {
        "wflow_v0": "vertical.erosk",
        "wflow_v1": "soil_erosion__rainfall_soil_detachability_factor",
    },
    "erosion_eurosem_exp": {
        "wflow_v0": "vertical.erosspl",
        "wflow_v1": "soil_erosion__eurosem_exponent",
    },
    "erosion_usle_k": {
        "wflow_v0": "vertical.usleK",
        "wflow_v1": "soil_erosion__usle_k_factor",
        "hydromt_name": "usle_k",
    },
    "erosion_usle_c": {
        "wflow_v0": "vertical.usleC",
        "wflow_v1": "soil_erosion__usle_c_factor",
        "hydromt_name": "usle_c",
    },
    "erosion_answers_sheet_factor": {
        "wflow_v0": "vertical.erosov",
        "wflow_v1": "soil_erosion__answers_overland_flow_factor",
    },
    "erosion_answers_rain_factor": {
        "wflow_v0": None,
        "wflow_v1": "soil_erosion__answers_rainfall_factor",
    },
    # soil particles
    "soil_clay_fraction": {
        "wflow_v0": None,
        "wflow_v1": "soil_clay__mass_fraction",
    },
    "soil_silt_fraction": {
        "wflow_v0": None,
        "wflow_v1": "soil_silt__mass_fraction",
    },
    "soil_sand_fraction": {
        "wflow_v0": None,
        "wflow_v1": "soil_sand__mass_fraction",
    },
    "soil_sagg_fraction": {
        "wflow_v0": None,
        "wflow_v1": "soil_small_aggregates__mass_fraction",
    },
    "soil_lagg_fraction": {
        "wflow_v0": None,
        "wflow_v1": "soil_large_aggregates__mass_fraction",
    },
    # land transport
    "land_govers_c": {
        "wflow_v0": None,
        "wflow_v1": "land_surface_water_sediment__govers_transport_capacity_coefficient",  # noqa: E501
    },
    "land_govers_n": {
        "wflow_v0": None,
        "wflow_v1": "land_surface_water_sediment__govers_transport_capacity_exponent",
    },
    "soil_sediment_d50": {
        "wflow_v0": None,
        "wflow_v1": "land_surface_sediment__median_diameter",
    },
    # river transport
    "sediment_density": {
        "wflow_v0": "lateral.river.rhos",
        "wflow_v1": "sediment__particle_density",
    },
    "river_sediment_d50": {
        "wflow_v0": "lateral.river.d50engelund",
        "wflow_v1": None,
    },
    "river_bagnold_c": {
        "wflow_v0": "lateral.river.cbagnold",
        "wflow_v1": "river_water_sediment__bagnold_transport_capacity_coefficient",
    },
    "river_bagnold_exp": {
        "wflow_v0": "lateral.river.ebagnold",
        "wflow_v1": "river_water_sediment__bagnold_transport_capacity_exponent",
    },
    "river_kodatie_a": {
        "wflow_v0": None,
        "wflow_v1": "river_water_sediment__kodatie_transport_capacity_a_coefficient",
    },
    "river_kodatie_b": {
        "wflow_v0": None,
        "wflow_v1": "river_water_sediment__kodatie_transport_capacity_b_coefficient",
    },
    "river_kodatie_c": {
        "wflow_v0": None,
        "wflow_v1": "river_water_sediment__kodatie_transport_capacity_c_coefficient",
    },
    "river_kodatie_d": {
        "wflow_v0": None,
        "wflow_v1": "river_water_sediment__kodatie_transport_capacity_d_coefficient",
    },
    "river_bed_sediment_d50": {
        "wflow_v0": "lateral.river.d50",
        "wflow_v1": "river_bottom_and_bank_sediment__median_diameter",
    },
    "river_bed_clay_fraction": {
        "wflow_v0": "lateral.river.fclayriv",
        "wflow_v1": "river_bottom_and_bank_clay__mass_fraction",
    },
    "river_bed_silt_fraction": {
        "wflow_v0": "lateral.river.fsiltriv",
        "wflow_v1": "river_bottom_and_bank_silt__mass_fraction",
    },
    "river_bed_sand_fraction": {
        "wflow_v0": "lateral.river.fsandriv",
        "wflow_v1": "river_bottom_and_bank_sand__mass_fraction",
    },
    "river_bed_gravel_fraction": {
        "wflow_v0": "lateral.river.fgravelriv",
        "wflow_v1": "river_bottom_and_bank_gravel__mass_fraction",
    },
    "sediment_clay_dm": {
        "wflow_v0": None,
        "wflow_v1": "clay__mean_diameter",
    },
    "sediment_silt_dm": {
        "wflow_v0": None,
        "wflow_v1": "silt__mean_diameter",
    },
    "sediment_sand_dm": {
        "wflow_v0": None,
        "wflow_v1": "sand__mean_diameter",
    },
    "sediment_sagg_dm": {
        "wflow_v0": None,
        "wflow_v1": "sediment_small_aggregates__mean_diameter",
    },
    "sediment_lagg_dm": {
        "wflow_v0": None,
        "wflow_v1": "sediment_large_aggregates__mean_diameter",
    },
    "sediment_gravel_dm": {
        "wflow_v0": None,
        "wflow_v1": "gravel__mean_diameter",
    },
}

WFLOW_SEDIMENT_STATES_NAMES: dict[str, dict[str, str | None]] = {
    "river_clay_load": {
        "wflow_v0": "lateral.river.clayload",
        "wflow_v1": "river_water_clay__mass",
    },
    "river_bed_clay_store": {
        "wflow_v0": "lateral.river.claystore",
        "wflow_v1": "river_bed_clay__mass",
    },
    "river_clay_flux": {
        "wflow_v0": "lateral.river.outclay",
        "wflow_v1": "river_water_clay__mass_flow_rate",
    },
    "river_gravel_load": {
        "wflow_v0": "lateral.river.gravload",
        "wflow_v1": "river_water_gravel__mass",
    },
    "river_bed_gravel_store": {
        "wflow_v0": "lateral.river.gravstore",
        "wflow_v1": "river_bed_gravel__mass",
    },
    "river_gravel_flux": {
        "wflow_v0": "lateral.river.outgrav",
        "wflow_v1": "river_water_gravel__mass_flow_rate",
    },
    "river_lagg_load": {
        "wflow_v0": "lateral.river.laggload",
        "wflow_v1": "river_water_large_aggregates__mass",
    },
    "river_bed_lagg_store": {
        "wflow_v0": "lateral.river.laggstore",
        "wflow_v1": "river_bed_large_aggregates__mass",
    },
    "river_lagg_flux": {
        "wflow_v0": "lateral.river.outlagg",
        "wflow_v1": "river_water_large_aggregates__mass_flow_rate",
    },
    "river_sagg_load": {
        "wflow_v0": "lateral.river.saggload",
        "wflow_v1": "river_water_small_aggregates__mass",
    },
    "river_bed_sagg_store": {
        "wflow_v0": "lateral.river.saggstore",
        "wflow_v1": "river_bed_small_aggregates__mass",
    },
    "river_sagg_flux": {
        "wflow_v0": "lateral.river.outsagg",
        "wflow_v1": "river_water_small_aggregates__mass_flow_rate",
    },
    "river_sand_load": {
        "wflow_v0": "lateral.river.sandload",
        "wflow_v1": "river_water_sand__mass",
    },
    "river_bed_sand_store": {
        "wflow_v0": "lateral.river.sandstore",
        "wflow_v1": "river_bed_sand__mass",
    },
    "river_sand_flux": {
        "wflow_v0": "lateral.river.outsand",
        "wflow_v1": "river_water_sand__mass_flow_rate",
    },
    "river_silt_load": {
        "wflow_v0": "lateral.river.siltload",
        "wflow_v1": "river_water_silt__mass",
    },
    "river_bed_silt_store": {
        "wflow_v0": "lateral.river.siltstore",
        "wflow_v1": "river_bed_silt__mass",
    },
    "river_silt_flux": {
        "wflow_v0": "lateral.river.outsilt",
        "wflow_v1": "river_water_silt__mass_flow_rate",
    },
}


def _create_hydromt_wflow_mapping(
    hydromt_dict: dict,
    model_dict: dict,
    config_dict: dict,
) -> tuple[dict, dict]:
    """
    Create dictionaries to convert from hydromt/Wflow names to staticmaps names.

    The first dictionary will be used to rename from hydromt convention
    name into name in staticmaps/forcing.
    The second dictionary will be used to link the right name
    in staticmaps/forcing files to the right Wflow.jl variables in the toml file.

    Parameters
    ----------
    hydromt_dict : dict
        Dictionary of default hydromt names and name in staticmaps for the variables
        that are not needed by Wflow but needed for model building.
    model_dict : dict
        Dictionary of default name in staticmaps and a potential "hydromt_name" key if
        some of the variables have a convention on name in hydromt. Also contains
        the wflow variable names to be able to update the default names based on the
        config.
    config_dict : dict
        Dictionary of the current model config to update from the default name in
        staticmaps to the actual name in staticmaps.

    Returns
    -------
    mapping_hydromt : dict
        Dictionary of the mapping from hydromt names to staticmaps names.
    mapping_wflow : dict
        Dictionary of the mapping from staticmaps names to Wflow variable names.
    """
    wflow_version = "wflow_v1"  # variable version to use for Wflow.jl

    # First dictionary
    # name_in_staticmaps : name_in_hydromt
    # Instantiate the mapping with default names (ie non wflow variables)
    mapping_inv = {v: k for k, v in hydromt_dict.items()}

    # Second dictionary
    wflow_names = dict()  # wflow_variable: name_in_staticmaps

    # Go through the wflow variables and add them if hydromt name is not None
    for staticmap_name, staticmap_mapping in model_dict.items():
        if isinstance(staticmap_mapping, dict):
            # use hydromt_name if it is available, use default staticmap name otherwise
            mapping_inv[staticmap_name] = staticmap_mapping.get(
                "hydromt_name", staticmap_name
            )
            if wflow_version in staticmap_mapping:
                wflow_names[staticmap_mapping.get(wflow_version)] = staticmap_name

    # Update with the TOML
    # Check if wflow v0 config then do not update
    # (wflow_v0 only support upgrade function so this step is not needed)
    is_wflow_v0 = _check_wflow_version(config_dict)

    if "input" in config_dict and not is_wflow_v0:
        variable_types = ["", "forcing", "cyclic", "static"]
        for var_type in variable_types:
            if var_type != "" and var_type not in config_dict["input"]:
                continue
            # Variables in the general input section
            if var_type == "":
                vars_dict = {}
                for input_var in WFLOW_VARS_IN_INPUT:
                    if input_var in config_dict["input"]:
                        vars_dict[input_var] = config_dict["input"][input_var]
            # Others (static / cyclic / forcing)
            else:
                vars_dict = config_dict["input"][var_type]
            # Go through the variables that can be renamed
            for var_name, new_name in vars_dict.items():
                # Get the old name
                old_name = wflow_names.get(var_name)
                # If they are different update the mapping
                if isinstance(new_name, dict):
                    if "value" in new_name:
                        # Do not update
                        new_name = None
                    elif "netcdf_variable_name" in new_name:
                        new_name = new_name["netcdf_variable_name"]
                if new_name is not None and old_name != new_name:
                    # Update the mapping with the new name
                    mapping_inv[new_name] = mapping_inv.get(old_name, old_name)
                    # Remove the old name from the mapping
                    mapping_inv.pop(old_name, None)
                    # Update wflow_names with the new name
                    wflow_names[var_name] = new_name

    # Invert mapping to get hydromt_name: staticmap_name
    mapping_hydromt = {
        hydromt_name: staticmap_name
        for (staticmap_name, hydromt_name) in mapping_inv.items()
    }

    # # Get a mapping of staticmap_name: wflow_variable
    mapping_wflow = {
        staticmap_name: wflow_var for wflow_var, staticmap_name in wflow_names.items()
    }

    return mapping_hydromt, mapping_wflow


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


def _check_wflow_version(config_dict: dict) -> bool:
    """
    Check if the config is for Wflow v0 or v1.

    Parameters
    ----------
    config_dict : dict
        Dictionary of the current model config.

    Returns
    -------
    bool
        True if the config is for Wflow v0, False otherwise.
    """
    _is_wflow_v0 = False
    if "starttime" in config_dict:  # first check
        _is_wflow_v0 = True
    # second check for safety as starttime is not present for fews_run
    if "input" in config_dict and "vertical" in config_dict["input"]:
        _is_wflow_v0 = True

    return _is_wflow_v0
