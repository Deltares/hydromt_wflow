"""Mapping dictionaries from hydromt to hydromt_wflow to Wflow.jl names."""

# These names cannot be read from TOML
# Decide if users should be able to update these or not
# Eg landuse may be handy to have several maps in one model for scenarios
HYDROMT_NAMES_DEFAULT = {
    # additional hydromt outputs
    "elevtn": "wflow_dem",  # meta_dem
    "subelv": "dem_subgrid",  # meta_dem_subgrid
    "uparea": "wflow_uparea",  # meta_upstream_area
    "strord": "wflow_streamorder",  # meta_streamorder
    "landuse": "wflow_landuse",  # meta_landuse
    "soil_texture": "wflow_soil",  # meta_soil_texture
}
HYDROMT_NAMES_DEFAULT_SEDIMENT = {
    # additional hydromt outputs
    "elevtn": "wflow_dem",  # meta_dem
    "uparea": "wflow_uparea",  # meta_upstream_area
    "strord": "wflow_streamorder",  # meta_streamorder
    "landuse": "wflow_landuse",  # meta_landuse
    "soil": "wflow_soil",  # meta_soil_texture
}

# names in comment should become the new default names in hydromt_wflow version 1
# TODO in another PR or after the rest is well tested
WFLOW_NAMES = {
    # general input
    "wflow_subcatch": {  # subcatchment
        "wflow_v0": "subcatchment",
        "wflow_v1": "subcatchment_location__count",
        "hydromt_name": "basins",
    },
    "wflow_ldd": {  # local_drain_direction
        "wflow_v0": "ldd",
        "wflow_v1": "local_drain_direction",
        "hydromt_name": "flwdir",
    },
    "wflow_lakeareas": {  # lake_areas
        "wflow_v0": "lateral.river.lake.areas",
        "wflow_v1": "lake_area__count",
        "hydromt_name": "lakeareas",  # lake_areas
    },
    "wflow_lakelocs": {  # lake_locations
        "wflow_v0": "lateral.river.lake.locs",
        "wflow_v1": "lake_location__count",
        "hydromt_name": "lakelocs",  # lake_locations
    },
    "wflow_reservoirareas": {  # reservoir_areas
        "wflow_v0": "lateral.river.reservoir.areas",
        "wflow_v1": "reservoir_area__count",
        "hydromt_name": "resareas",  # reservoir_areas
    },
    "wflow_reservoirlocs": {  # reservoir_locations
        "wflow_v0": "lateral.river.reservoir.locs",
        "wflow_v1": "reservoir_location__count",
        "hydromt_name": "reslocs",  # reservoir_locations
    },
    "wflow_river": {  # river_mask
        "wflow_v0": "river_location",
        "wflow_v1": "river_location__mask",
        "hydromt_name": "rivmsk",
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
    "Cfmax": {  # cfmax
        "wflow_v0": "vertical.cfmax",
        "wflow_v1": "snowpack~dry__degree-day_coefficient",
    },
    "TT": {  # tt
        "wflow_v0": "vertical.tt",
        "wflow_v1": "atmosphere_air__snowfall_temperature_threshold",
    },
    "TTI": {
        "wflow_v0": "vertical.tti",
        "wflow_v1": "atmosphere_air__snowfall_temperature_interval",
    },
    "TTM": {
        "wflow_v0": "vertical.ttm",
        "wflow_v1": "snowpack__melting_temperature_threshold",
    },
    "WHC": {
        "wflow_v0": "vertical.water_holding_capacity",
        "wflow_v1": "snowpack__liquid_water_holding_capacity",
    },
    # glacier
    "wflow_glacierfrac": {  # glacier_fraction
        "wflow_v0": "vertical.glacierfrac",
        "wflow_v1": "glacier_surface__area_fraction",
        "hydromt_name": "glacfracs",
    },
    "wflow_glacierstore": {  # glacier_storage
        "wflow_v0": "vertical.glacierstore",
        "wflow_v1": "glacier_ice__leq-volume",
        "hydromt_name": "glacstore",
    },
    "G_TTM": {  # glacier_ttm
        "wflow_v0": "vertical.g_ttm",
        "wflow_v1": "glacier_ice__melting_temperature_threshold",
    },
    "G_Cfmax": {  # glacier_cfmax
        "wflow_v0": "vertical.g_cfmax",
        "wflow_v1": "glacier_ice__degree-day_coefficient",
    },
    "G_SIfrac": {  # glacier_sifrac
        "wflow_v0": "vertical.g_sifrac",
        "wflow_v1": "glacier_firn_accumulation__snowpack~dry_leq-depth_fraction",
    },
    # vegetation
    "EoverR": {
        "wflow_v0": "vertical.e_r",
        "wflow_v1": "vegetation_canopy_water__mean_evaporation-to-mean_precipitation_ratio",  # noqa: E501
    },
    "Kext": {  # kext
        "wflow_v0": "vertical.kext",
        "wflow_v1": "vegetation_canopy__light-extinction_coefficient",
    },
    "LAI": {  # lai
        "wflow_v0": "vertical.leaf_area_index",
        "wflow_v1": "vegetation__leaf-area_index",
        "hydromt_name": "LAI",  # lai
    },
    "RootingDepth": {  # root_depth
        "wflow_v0": "vertical.rootingdepth",
        "wflow_v1": "vegetation_root__depth",
    },
    "Sl": {  # leaf_storage
        "wflow_v0": "vertical.specific_leaf",
        "wflow_v1": "vegetation__specific-leaf_storage",
        "hydromt_name": "Sl",  # leaf_storage
    },
    "Swood": {  # wood_storage
        "wflow_v0": "vertical.storage_wood",
        "wflow_v1": "vegetation_wood_water__storage_capacity",
        "hydromt_name": "Swood",  # wood_storage
    },
    "kc": {  # crop_factor
        "wflow_v0": "vertical.kc",
        "wflow_v1": "vegetation__crop_factor",
    },
    "alpha_h1": {
        "wflow_v0": "vertical.alpha_h1",
        "wflow_v1": "vegetation_root__feddes_critial_pressure_head_h~1_reduction_coefficient",  # noqa: E501
    },
    "h1": {
        "wflow_v0": "vertical.h1",
        "wflow_v1": "vegetation_root__feddes_critial_pressure_head_h~1",
    },
    "h2": {
        "wflow_v0": "vertical.h2",
        "wflow_v1": "vegetation_root__feddes_critial_pressure_head_h~2",
    },
    "h3_high": {
        "wflow_v0": "vertical.h3_high",
        "wflow_v1": "vegetation_root__feddes_critial_pressure_head_h~3~high",
    },
    "h3_low": {
        "wflow_v0": "vertical.h3_low",
        "wflow_v1": "vegetation_root__feddes_critial_pressure_head_h~3~low",
    },
    "h4": {
        "wflow_v0": "vertical.h4",
        "wflow_v1": "vegetation_root__feddes_critial_pressure_head_h~4",
    },
    # soil
    "c": {
        "wflow_v0": "vertical.c",
        "wflow_v1": "soil_layer_water__brooks-corey_epsilon_parameter",
        "hydromt_name": "c",
    },
    "cf_soil": {
        "wflow_v0": "vertical.cf_soil",
        "wflow_v1": "soil_surface_water__infiltration_reduction_parameter",
    },
    "KsatVer": {  # ksat_vertical
        "wflow_v0": "vertical.kv_0",
        "wflow_v1": "soil_surface_water__vertical_saturated_hydraulic_conductivity",
        "hydromt_name": "ksat_vertical",
    },
    "kvfrac": {  # ksat_vertical_fraction
        "wflow_v0": "vertical.kvfrac",
        "wflow_v1": "soil_water__vertical_saturated_hydraulic_conductivity_factor",
    },
    "ksathorfrac": {  # ksat_horizontal_fraction
        "wflow_v0": "lateral.subsurface.ksathorfrac",
        "wflow_v1": "subsurface_water__horizontal-to-vertical_saturated_hydraulic_conductivity_ratio",  # noqa: E501
    },
    "f": {
        "wflow_v0": "vertical.f",
        "wflow_v1": "soil_water__vertical_saturated_hydraulic_conductivity_scale_parameter",  # noqa: E501
        "hydromt_name": "f",
    },
    "InfiltCapPath": {  # infiltcap_path
        "wflow_v0": "vertical.infiltcappath",
        "wflow_v1": "soil~compacted_surface_water__infiltration_capacity",
    },
    "InfiltCapSoil": {  # infiltcap_soil
        "wflow_v0": "vertical.infiltcapsoil",
        "wflow_v1": "soil~non-compacted_surface_water__infiltration_capacity",
    },
    "thetaR": {  # theta_r
        "wflow_v0": "vertical.theta_r",
        "wflow_v1": "soil_water__residual_volume_fraction",
        "hydromt_name": "thetaR",  # theta_r
    },
    "thetaS": {  # theta_s
        "wflow_v0": "vertical.theta_s",
        "wflow_v1": "soil_water__saturated_volume_fraction",
        "hydromt_name": "thetaS",  # theta_s
    },
    "MaxLeakage": {
        "wflow_v0": "vertical.maxleakage",
        "wflow_v1": "soil_water_sat-zone_bottom__max_leakage_volume_flux",
    },
    "PathFrac": {  # compacted_fraction
        "wflow_v0": "vertical.pathfrac",
        "wflow_v1": "soil~compacted__area_fraction",
    },
    "rootdistpar": {
        "wflow_v0": "vertical.rootdistpar",
        "wflow_v1": "soil_root~wet__sigmoid_function_shape_parameter",
    },
    "SoilThickness": {  # soil_thickness
        "wflow_v0": "vertical.soilthickness",
        "wflow_v1": "soil__thickness",
    },
    # land
    "WaterFrac": {  # water_fraction
        "wflow_v0": "vertical.waterfrac",
        "wflow_v1": "land~water-covered__area_fraction",
    },
    "allocation_areas": {
        "wflow_v0": "vertical.allocation.areas",
        "wflow_v1": "land_water_allocation_area__number",
        "hydromt_name": "allocation_areas",
    },
    "frac_sw_used": {
        "wflow_v0": "vertical.allocation.frac_sw_used",
        "wflow_v1": "land_surface_water__withdrawal_fraction",
    },
    "domestic_gross": {
        "wflow_v0": "vertical.domestic.demand_gross",
        "wflow_v1": "land~domestic__gross_water_demand_volume_flux",
        "hydromt_name": "dom_gross",
    },
    "domestic_net": {
        "wflow_v0": "vertical.domestic.demand_net",
        "wflow_v1": "land~domestic__net_water_demand_volume_flux",
        "hydromt_name": "dom_net",
    },
    "industry_gross": {
        "wflow_v0": "vertical.industry.demand_gross",
        "wflow_v1": "land~industry__gross_water_demand_volume_flux",
        "hydromt_name": "ind_gross",
    },
    "industry_net": {
        "wflow_v0": "vertical.industry.demand_net",
        "wflow_v1": "land~industry__net_water_demand_volume_flux",
        "hydromt_name": "ind_net",
    },
    "livestock_gross": {
        "wflow_v0": "vertical.livestock.demand_gross",
        "wflow_v1": "land~livestock__gross_water_demand_volume_flux",
        "hydromt_name": "lsk_gross",
    },
    "livestock_net": {
        "wflow_v0": "vertical.livestock.demand_net",
        "wflow_v1": "land~livestock__net_water_demand_volume_flux",
        "hydromt_name": "lsk_net",
    },
    "h_min": {
        "wflow_v0": "vertical.paddy.h_min",
        "wflow_v1": "land~irrigated-paddy__min_depth",
    },
    "h_opt": {
        "wflow_v0": "vertical.paddy.h_opt",
        "wflow_v1": "land~irrigated-paddy__optimal_depth",
    },
    "h_max": {
        "wflow_v0": "vertical.paddy.h_max",
        "wflow_v1": "land~irrigated-paddy__max_depth",
    },
    "paddy_irrigation_areas": {
        "wflow_v0": "vertical.paddy.irrigation_areas",
        "wflow_v1": "land~irrigated-paddy_area__number",
    },
    "paddy_irrigation_trigger": {
        "wflow_v0": "vertical.paddy.irrigation_trigger",
        "wflow_v1": "land~irrigated-paddy__irrigation_trigger_flag",
    },
    "nonpaddy_irrigation_areas": {
        "wflow_v0": "vertical.nonpaddy.irrigation_areas",
        "wflow_v1": "land~irrigated-non-paddy_area__number",
    },
    "nonpaddy_irrigation_trigger": {
        "wflow_v0": "vertical.nonpaddy.irrigation_trigger",
        "wflow_v1": "land~irrigated-non-paddy__irrigation_trigger_flag",
    },
    # land surface water flow
    "N": {  # land_n
        "wflow_v0": "lateral.land.n",
        "wflow_v1": "land_surface_water_flow__manning_n_parameter",
    },
    "lndelv": {
        "wflow_v0": "lateral.land.elevation",
        "wflow_v1": "land_surface_water_flow__ground_elevation",
        "hydromt_name": "lndelv",
    },
    "Slope": {  # slope
        "wflow_v0": "lateral.land.slope",
        "wflow_v1": "land_surface__slope",
        "hydromt_name": "lndslp",
    },
    # river
    "floodplain_volume": {
        "wflow_v0": "lateral.river.floodplain.volume",
        "wflow_v1": "floodplain_water__sum_of_volume-per-depth",
        "hydromt_name": "floodplain_volume",
    },
    "hydrodem": {
        "wflow_v0": "lateral.river.bankfull_elevation",
        "wflow_v1": "river_bank_water__elevation",
        "hydromt_name": "hydrodem",
    },
    "inflow": {
        "wflow_v0": "lateral.river.inflow",
        "wflow_v1": "river_water_inflow~external__volume_flow_rate",
        "hydromt_name": "inflow",
    },
    "RiverDepth": {  # river_depth
        "wflow_v0": "lateral.river.bankfull_depth",
        "wflow_v1": "river_bank_water__depth",
        "hydromt_name": "rivdph",
    },
    "wflow_riverlength": {  # river_length
        "wflow_v0": "lateral.river.length",
        "wflow_v1": "river__length",
        "hydromt_name": "rivlen",
    },
    "N_River": {  # river_n
        "wflow_v0": "lateral.river.n",
        "wflow_v1": "river_water_flow__manning_n_parameter",
    },
    "RiverSlope": {  # river_slope
        "wflow_v0": "lateral.river.slope",
        "wflow_v1": "river__slope",
        "hydromt_name": "rivslp",
    },
    "wflow_riverwidth": {  # river_width
        "wflow_v0": "lateral.river.width",
        "wflow_v1": "river__width",
        "hydromt_name": "rivwth",
    },
    # lakes
    "LakeArea": {  # lake_area
        "wflow_v0": "lateral.river.lake.area",
        "wflow_v1": "lake_surface__area",
    },
    "LakeAvgLevel": {  # lake_waterlevel
        "wflow_v0": "lateral.river.lake.waterlevel",
        "wflow_v1": "lake_water_level__initial_elevation",
    },
    "LakeThreshold": {  # lake_threshold
        "wflow_v0": "lateral.river.lake.threshold",
        "wflow_v1": "lake_water_flow_threshold-level__elevation",
    },
    "Lake_b": {  # lake_b
        "wflow_v0": "lateral.river.lake.b",
        "wflow_v1": "lake_water__rating_curve_coefficient",
    },
    "Lake_e": {  # lake_e
        "wflow_v0": "lateral.river.lake.e",
        "wflow_v1": "lake_water__rating_curve_exponent",
    },
    "LakeOutflowFunc": {  # lake_rating_curve
        "wflow_v0": "lateral.river.lake.outflowfunc",
        "wflow_v1": "lake_water__rating_curve_type_count",
    },
    "LakeStorFunc": {  # lake_storage_curve
        "wflow_v0": "lateral.river.lake.storfunc",
        "wflow_v1": "lake_water__storage_curve_type_count",
    },
    "LinkedLakeLocs": {  # lake_lower_locations
        "wflow_v0": "lateral.river.lake.linkedlakelocs",
        "wflow_v1": "lake~lower_location__count",
    },
    # reservoirs
    "ResSimpleArea": {  # reservoir_area
        "wflow_v0": "lateral.river.reservoir.area",
        "wflow_v1": "reservoir_surface__area",
    },
    "ResDemand": {  # reservoir_demand
        "wflow_v0": "lateral.river.reservoir.demand",
        "wflow_v1": "reservoir_water_demand~required~downstream__volume_flow_rate",
    },
    "ResMaxRelease": {  # reservoir_max_release
        "wflow_v0": "lateral.river.reservoir.maxrelease",
        "wflow_v1": "reservoir_water_release-below-spillway__max_volume_flow_rate",
    },
    "ResMaxVolume": {  # reservoir_max_volume
        "wflow_v0": "lateral.river.reservoir.maxvolume",
        "wflow_v1": "reservoir_water__max_volume",
    },
    "ResTargetFullFrac": {  # reservoir_target_full_fraction
        "wflow_v0": "lateral.river.reservoir.targetfullfrac",
        "wflow_v1": "reservoir_water~full-target__volume_fraction",
    },
    "ResTargetMinFrac": {  # reservoir_target_min_fraction
        "wflow_v0": "lateral.river.reservoir.targetminfrac",
        "wflow_v1": "reservoir_water~min-target__volume_fraction",
    },
}

WFLOW_STATES_NAMES = {
    "canopystorage": {
        "wflow_v0": "vertical.canopystorage",
        "wflow_v1": "vegetation_canopy_water__depth",
    },
    "satwaterdepth": {
        "wflow_v0": "vertical.satwaterdepth",
        "wflow_v1": "soil_water_sat-zone__depth",
    },
    "ustorelayerdepth": {
        "wflow_v0": "vertical.ustorelayerdepth",
        "wflow_v1": "soil_layer_water_unsat-zone__depth",
    },
    "tsoil": {
        "wflow_v0": "vertical.tsoil",
        "wflow_v1": "soil_surface__temperature",
    },
    "snow": {
        "wflow_v0": "vertical.snow",
        "wflow_v1": "snowpack~dry__leq-depth",
    },
    "snowwater": {
        "wflow_v0": "vertical.snowwater",
        "wflow_v1": "snowpack~liquid__depth",
    },
    "glacierstore": {
        "wflow_v0": "vertical.glacierstore",
        "wflow_v1": "glacier_ice__leq-volume",
    },
    "q_land": {
        "wflow_v0": "lateral.land.q",
        "wflow_v1": "land_surface_water__instantaneous_volume_flow_rate",
    },
    "qx_land": {
        "wflow_v0": "lateral.land.qx",
        "wflow_v1": "land_surface_water__x_component_of_instantaneous_volume_flow_rate",
    },
    "qy_land": {
        "wflow_v0": "lateral.land.qy",
        "wflow_v1": "land_surface_water__y_component_of_instantaneous_volume_flow_rate",
    },
    "h_land": {
        "wflow_v0": "lateral.land.h",
        "wflow_v1": "land_surface_water__instantaneous_depth",
    },
    "h_av_land": {
        "wflow_v0": "lateral.land.h_av",
        "wflow_v1": None,
    },
    "ssf": {
        "wflow_v0": "lateral.subsurface.ssf",
        "wflow_v1": "subsurface_water__volume_flow_rate",
    },
    "q_river": {
        "wflow_v0": "lateral.river.q",
        "wflow_v1": "river_water__instantaneous_volume_flow_rate",
    },
    "h_river": {
        "wflow_v0": "lateral.river.h",
        "wflow_v1": "river_water__instantaneous_depth",
    },
    "h_av_river": {
        "wflow_v0": "lateral.river.h_av",
        "wflow_v1": None,
    },
    "q_floodplain": {
        "wflow_v0": "lateral.river.floodplain.q",
        "wflow_v1": "floodplain_water__instantaneous_volume_flow_rate",
    },
    "h_floodplain": {
        "wflow_v0": "lateral.river.floodplain.h",
        "wflow_v1": "floodplain_water__instantaneous_depth",
    },
    "waterlevel_lake": {
        "wflow_v0": "lateral.river.lake.waterlevel",
        "wflow_v1": "lake_water_level__initial_elevation",
    },
    "volume_reservoir": {
        "wflow_v0": "lateral.river.reservoir.volume",
        "wflow_v1": "reservoir_water__instantaneous_volume",
    },
    "h_paddy": {
        "wflow_v0": "vertical.paddy.h",
        "wflow_v1": "land_surface_water~paddy__depth",
    },
}

WFLOW_SEDIMENT_NAMES = {
    "landuse": None,
    "PathFrac": {
        "wflow_v0": "vertical.pathfrac",
        "wflow_v1": "soil~compacted__area_fraction",
    },
    "USLE_C": {
        "wflow_v0": "vertical.usleC",
        "wflow_v1": "soil_erosion__usle_c_factor",
        "hydromt_name": "USLE_C",  # usle_c_factor
    },
}


def _create_hydromt_mapping(hydromt_dict: dict, model_dict: dict) -> dict:
    """
    Create a dictionnary to convert from hydromt names to default wflow input names.

    These names will be used to rename from hydromt convention
    name into name in staticmaps/forcing.
    """
    mapping = {}
    # Instantiate the mapping with default names (ie non wflow variables)
    for k, v in hydromt_dict.items():
        mapping[k] = v
    # Go trhough the wflow variables and add them if hydromt name is not None
    for k, v in model_dict.items():
        if isinstance(v, dict) and "hydromt_name" in v:
            mapping[v.get("hydromt_name")] = k
        # else use the default staticmap name as the name
        else:
            mapping[k] = k
    # TODO: read and update using toml for the others
    return mapping


def _create_hydromt_mapping_wflow() -> dict:
    """
    Create a dictionnary to convert from hydromt names to wflow sbm input names.

    These names will be used in staticmaps/forcing and linked to the right wflow
    internal variables in the toml.
    """
    mapping = _create_hydromt_mapping(HYDROMT_NAMES_DEFAULT, WFLOW_NAMES)
    return mapping


def _create_hydromt_mapping_sediment() -> dict:
    """
    Create a dictionnary to convert from hydromt names to wflow sediment input names.

    These names will be used in staticmaps/forcing and linked to the right wflow
    internal variables in the toml.
    """
    mapping = _create_hydromt_mapping(
        HYDROMT_NAMES_DEFAULT_SEDIMENT, WFLOW_SEDIMENT_NAMES
    )
    return mapping


def _create_variable_mapping(model_dict: dict, wflow_version="1.0.0") -> dict:
    """
    Create a dictionnary to convert from hydromt names to wflow internal variable names.

    These names will be used in the toml file to link the right wflow internal
    variables to the right sbm variables.
    """
    # wflow variable names
    mapping = dict()
    _is_v1 = wflow_version.startswith("1.")
    variable_version = "wflow_v1" if _is_v1 else "wflow_v0"
    for k, v in model_dict.items():
        if isinstance(v, dict) and variable_version in v:
            mapping[k] = v.get(variable_version)
        # else not a wflow but a hydromt variable so will not be added to toml

    return mapping


def _create_variable_mapping_wflow(wflow_version="1.0.0") -> dict:
    """
    Create a dictionnary to convert from hydromt to wflow sbm internal variable names.

    These names will be used in the toml file to link the right wflow internal
    variables to the right sbm variables.
    """
    mapping = _create_variable_mapping(WFLOW_NAMES, wflow_version)
    return mapping


def _create_variable_mapping_sediment(wflow_version="1.0.0") -> dict:
    """
    Create a dictionnary to convert from hydromt to wflow sediment internal names.

    These names will be used in the toml file to link the right wflow internal
    variables to the right sbm variables.
    """
    mapping = _create_variable_mapping(WFLOW_SEDIMENT_NAMES, wflow_version)
    return mapping
