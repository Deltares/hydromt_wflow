"""Mapping dictionaries from hydromt to hydromt_wflow to Wflow.jl names."""

from typing import Tuple

# These names cannot be read from TOML
# Decide if users should be able to update these or not
# Eg landuse may be handy to have several maps in one model for scenarios
HYDROMT_NAMES_DEFAULT = {
    # additional hydromt outputs
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
        "wflow_v1": "snowpack__degree-day_coefficient",
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
        "wflow_v1": "glacier_ice__initial_leq-depth",
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
        "wflow_v1": "soil_layer_water__brooks-corey_exponent",
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
        "wflow_v1": None,
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
    "wflow_dem": {
        "wflow_v0": "lateral.land.elevation",
        "wflow_v1": "land_surface_water_flow__ground_elevation",
        "hydromt_name": "elevtn",
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
        "hydromt_name": "LakeArea",
    },
    "LakeAvgLevel": {  # lake_waterlevel
        "wflow_v0": "lateral.river.lake.waterlevel",
        "wflow_v1": "lake_water_surface__initial_elevation",
        "hydromt_name": "LakeAvgLevel",
    },
    "LakeThreshold": {  # lake_threshold
        "wflow_v0": "lateral.river.lake.threshold",
        "wflow_v1": "lake_water_flow_threshold-level__elevation",
        "hydromt_name": "LakeThreshold",
    },
    "Lake_b": {  # lake_b
        "wflow_v0": "lateral.river.lake.b",
        "wflow_v1": "lake_water__rating_curve_coefficient",
        "hydromt_name": "Lake_b",
    },
    "Lake_e": {  # lake_e
        "wflow_v0": "lateral.river.lake.e",
        "wflow_v1": "lake_water__rating_curve_exponent",
        "hydromt_name": "Lake_e",
    },
    "LakeOutflowFunc": {  # lake_rating_curve
        "wflow_v0": "lateral.river.lake.outflowfunc",
        "wflow_v1": "lake_water__rating_curve_type_count",
        "hydromt_name": "LakeOutflowFunc",
    },
    "LakeStorFunc": {  # lake_storage_curve
        "wflow_v0": "lateral.river.lake.storfunc",
        "wflow_v1": "lake_water__storage_curve_type_count",
        "hydromt_name": "LakeStorFunc",
    },
    "LinkedLakeLocs": {  # lake_lower_locations
        "wflow_v0": "lateral.river.lake.linkedlakelocs",
        "wflow_v1": "lake~lower_location__count",
        "hydromt_name": "LinkedLakeLocs",
    },
    # reservoirs
    "ResSimpleArea": {  # reservoir_area
        "wflow_v0": "lateral.river.reservoir.area",
        "wflow_v1": "reservoir_surface__area",
        "hydromt_name": "ResSimpleArea",
    },
    "ResDemand": {  # reservoir_demand
        "wflow_v0": "lateral.river.reservoir.demand",
        "wflow_v1": "reservoir_water_demand~required~downstream__volume_flow_rate",
        "hydromt_name": "ResDemand",
    },
    "ResMaxRelease": {  # reservoir_max_release
        "wflow_v0": "lateral.river.reservoir.maxrelease",
        "wflow_v1": "reservoir_water_release-below-spillway__max_volume_flow_rate",
        "hydromt_name": "ResMaxRelease",
    },
    "ResMaxVolume": {  # reservoir_max_volume
        "wflow_v0": "lateral.river.reservoir.maxvolume",
        "wflow_v1": "reservoir_water__max_volume",
        "hydromt_name": "ResMaxVolume",
    },
    "ResTargetFullFrac": {  # reservoir_target_full_fraction
        "wflow_v0": "lateral.river.reservoir.targetfullfrac",
        "wflow_v1": "reservoir_water~full-target__volume_fraction",
        "hydromt_name": "ResTargetFullFrac",
    },
    "ResTargetMinFrac": {  # reservoir_target_min_fraction
        "wflow_v0": "lateral.river.reservoir.targetminfrac",
        "wflow_v1": "reservoir_water~min-target__volume_fraction",
        "hydromt_name": "ResTargetMinFrac",
    },
    # gwf
    "constant_head": {
        "wflow_v0": "lateral.subsurface.constant_head",
        "wflow_v1": "model_boundary_condition~constant_hydraulic_head",
    },
    "conductivity_profile": {
        "wflow_v0": "lateral.subsurface.conductivity_profile",
        "wflow_v1": "conductivity_profile",
    },
    "kh_surface": {
        "wflow_v0": "lateral.subsurface.conductivity",
        "wflow_v1": "subsurface_surface_water__horizontal_saturated_hydraulic_conductivity",  # noqa: E501
    },
    "zb_river": {
        "wflow_v0": "lateral.subsurface.river_bottom",
        "wflow_v1": "river_bottom__elevation",
    },
    "riverbed_cond_infilt": {
        "wflow_v0": "lateral.subsurface.infiltration_conductance",
        "wflow_v1": "river_water__infiltration_conductance",
    },
    "riverbed_cond_exfil": {
        "wflow_v0": "lateral.subsurface.exfiltration_conductance",
        "wflow_v1": "river_water__exfiltration_conductance",
    },
    "specific_yield": {
        "wflow_v0": "lateral.subsurface.specific_yield",
        "wflow_v1": "subsurface_water__specific_yield",
    },
    "gwf_f": {
        "wflow_v0": "lateral.subsurface.gwf_f",
        "wflow_v1": "subsurface__horizontal_saturated_hydraulic_conductivity_scale_parameter",  # noqa: E501
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
        "wflow_v1": "glacier_ice__leq-depth",
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
    "head": {
        "wflow_v0": "lateral.subsurface.flow.aquifer.head",
        "wflow_v1": "subsurface_water__hydraulic_head",
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
        "wflow_v1": "lake_water_surface__instantaneous_elevation",
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
        "wflow_v0": "vertical.lakeareas",
        "wflow_v1": "lake_area__count",
        "hydromt_name": "lakeareas",  # lake_areas
    },
    "wflow_lakelocs": {  # lake_locations
        "wflow_v0": "lateral.river.lakelocs",
        "wflow_v1": "lake_location__count",
        "hydromt_name": "lakelocs",  # lake_locations
    },
    "wflow_reservoirareas": {  # reservoir_areas
        "wflow_v0": "vertical.resareas",
        "wflow_v1": "reservoir_area__count",
        "hydromt_name": "resareas",  # reservoir_areas
    },
    "wflow_reservoirlocs": {  # reservoir_locations
        "wflow_v0": "lateral.river.reslocs",
        "wflow_v1": "reservoir_location__count",
        "hydromt_name": "reslocs",  # reservoir_locations
    },
    "wflow_river": {  # river_mask
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
    "interception": {
        "wflow_v0": "vertical.interception",
        "wflow_v1": "vegetation_canopy_water__interception_volume_flux",
        "hydromt_name": "interception",
    },
    "h_land": {
        "wflow_v0": "vertical.h_land",
        "wflow_v1": "land_surface_water__depth",
    },
    "q_land": {
        "wflow_v0": "vertical.q_land",
        "wflow_v1": "land_surface_water__volume_flow_rate",
    },
    "h_river": {
        "wflow_v0": "lateral.river.h_riv",
        "wflow_v1": "river_water__depth",
    },
    "q_river": {
        "wflow_v0": "lateral.river.q_riv",
        "wflow_v1": "river_water__volume_flow_rate",
    },
    # land properties
    "Slope": {
        "wflow_v0": "lateral.land.slope",
        "wflow_v1": "land_surface__slope",
        "hydromt_name": "lndslp",
    },
    # river properties
    "wflow_riverlength": {
        "wflow_v0": "lateral.river.length",
        "wflow_v1": "river__length",
        "hydromt_name": "rivlen",
    },
    "RiverSlope": {
        "wflow_v0": "lateral.river.slope",
        "wflow_v1": "river__slope",
        "hydromt_name": "rivslp",
    },
    "wflow_riverwidth": {
        "wflow_v0": "lateral.river.width",
        "wflow_v1": "river__width",
        "hydromt_name": "rivwth",
    },
    # waterbodies
    "ResSimpleArea": {
        "wflow_v0": "lateral.river.resarea",
        "wflow_v1": "reservoir_surface__area",
        "hydromt_name": "ResSimpleArea",
    },
    "LakeArea": {
        "wflow_v0": "lateral.river.lakearea",
        "wflow_v1": "lake_surface__area",
        "hydromt_name": "LakeArea",
    },
    "ResTrapEff": {
        "wflow_v0": "lateral.river.restrapeff",
        "wflow_v1": "reservoir_sediment~bedload__trapping_efficiency_coefficient",
        "hydromt_name": "ResTrapEff",
    },
    # soil erosion
    "CanopyGapFraction": {
        "wflow_v0": "vertical.canopygapfraction",
        "wflow_v1": "vegetation_canopy__gap_fraction",
    },
    "CanopyHeight": {
        "wflow_v0": "vertical.canopyheight",
        "wflow_v1": "vegetation_canopy__height",
    },
    "Kext": {  # kext
        "wflow_v0": "vertical.kext",
        "wflow_v1": None,
    },
    "Sl": {  # leaf_storage
        "wflow_v0": "vertical.specific_leaf",
        "wflow_v1": None,
    },
    "Swood": {  # wood_storage
        "wflow_v0": "vertical.storage_wood",
        "wflow_v1": None,
    },
    "PathFrac": {
        "wflow_v0": "vertical.pathfrac",
        "wflow_v1": "soil~compacted__area_fraction",
    },
    "WaterFrac": {
        "wflow_v0": None,
        "wflow_v1": "land~water-covered__area_fraction",
    },
    "soil_detachability": {
        "wflow_v0": "vertical.erosk",
        "wflow_v1": "soil_erosion__rainfall_soil_detachability_factor",
    },
    "eros_spl_EUROSEM": {
        "wflow_v0": "vertical.erosspl",
        "wflow_v1": "soil_erosion__eurosem_exponent",
    },
    "usle_k": {
        "wflow_v0": "vertical.usleK",
        "wflow_v1": "soil_erosion__usle_k_factor",
        "hydromt_name": "usle_k",  # usle_k
    },
    "USLE_C": {
        "wflow_v0": "vertical.usleC",
        "wflow_v1": "soil_erosion__usle_c_factor",
        "hydromt_name": "USLE_C",  # usle_c
    },
    "eros_ov": {
        "wflow_v0": "vertical.erosov",
        "wflow_v1": "soil_erosion__answers_overland_flow_factor",
    },
    # soil particles
    "fclay_soil": {
        "wflow_v0": None,
        "wflow_v1": "soil_clay__mass_fraction",
    },
    "fsilt_soil": {
        "wflow_v0": None,
        "wflow_v1": "soil_silt__mass_fraction",
    },
    "fsand_soil": {
        "wflow_v0": None,
        "wflow_v1": "soil_sand__mass_fraction",
    },
    "fsagg_soil": {
        "wflow_v0": None,
        "wflow_v1": "soil_aggregates~small__mass_fraction",
    },
    "flagg_soil": {
        "wflow_v0": None,
        "wflow_v1": "soil_aggregates~large__mass_fraction",
    },
    # land transport
    "sediment_density_land": {
        "wflow_v0": "vertical.rhos",
        "wflow_v1": "land_surface_sediment__particle_density",
    },
    "c_govers": {
        "wflow_v0": None,
        "wflow_v1": "land_surface_water_sediment__govers_transport_capacity_coefficient",  # noqa: E501
    },
    "n_govers": {
        "wflow_v0": None,
        "wflow_v1": "land_surface_water_sediment__govers_transport_capacity_exponent",
    },
    "d50_soil": {
        "wflow_v0": None,
        "wflow_v1": "land_surface_sediment__d50_diameter",
    },
    # river transport
    "sediment_density_river": {
        "wflow_v0": "lateral.river.rhos",
        "wflow_v1": "river_water_sediment__particle_density",
    },
    "D50_Engelund": {
        "wflow_v0": "lateral.river.d50engelund",
        "wflow_v1": "river_sediment__d50_diameter",
    },
    "c_Bagnold": {
        "wflow_v0": "lateral.river.cbagnold",
        "wflow_v1": "river_water_sediment__bagnold_transport_capacity_coefficient",
    },
    "exp_Bagnold": {
        "wflow_v0": "lateral.river.ebagnold",
        "wflow_v1": "river_water_sediment__bagnold_transport_capacity_exponent",
    },
    "a_kodatie": {
        "wflow_v0": None,
        "wflow_v1": "river_water_sediment__kodatie_transport_capacity_a-coefficient",
    },
    "b_kodatie": {
        "wflow_v0": None,
        "wflow_v1": "river_water_sediment__kodatie_transport_capacity_b-coefficient",
    },
    "c_kodatie": {
        "wflow_v0": None,
        "wflow_v1": "river_water_sediment__kodatie_transport_capacity_c-coefficient",
    },
    "d_kodatie": {
        "wflow_v0": None,
        "wflow_v1": "river_water_sediment__kodatie_transport_capacity_d-coefficient",
    },
    "D50_River": {
        "wflow_v0": "lateral.river.d50",
        "wflow_v1": "river_bottom-and-bank_sediment__d50_diameter",
    },
    "ClayF_River": {
        "wflow_v0": "lateral.river.fclayriv",
        "wflow_v1": "river_bottom-and-bank_clay__mass_fraction",
    },
    "SiltF_River": {
        "wflow_v0": "lateral.river.fsiltriv",
        "wflow_v1": "river_bottom-and-bank_silt__mass_fraction",
    },
    "SandF_River": {
        "wflow_v0": "lateral.river.fsandriv",
        "wflow_v1": "river_bottom-and-bank_sand__mass_fraction",
    },
    "GravelF_River": {
        "wflow_v0": "lateral.river.fgravelriv",
        "wflow_v1": "river_bottom-and-bank_gravel__mass_fraction",
    },
    "dm_clay": {
        "wflow_v0": None,
        "wflow_v1": "clay__d50_diameter",
    },
    "dm_silt": {
        "wflow_v0": None,
        "wflow_v1": "silt__d50_diameter",
    },
    "dm_sand": {
        "wflow_v0": None,
        "wflow_v1": "sand__d50_diameter",
    },
    "dm_sagg": {
        "wflow_v0": None,
        "wflow_v1": "sediment_aggregates~small__d50_diameter",
    },
    "dm_lagg": {
        "wflow_v0": None,
        "wflow_v1": "sediment_aggregates~large__d50_diameter",
    },
    "dm_gravel": {
        "wflow_v0": None,
        "wflow_v1": "gravel__d50_diameter",
    },
}

WFLOW_SEDIMENT_STATES_NAMES = {
    "clayload": {
        "wflow_v0": "lateral.river.clayload",
        "wflow_v1": "river_water_clay__mass",
    },
    "claystore": {
        "wflow_v0": "lateral.river.claystore",
        "wflow_v1": "river_bed_clay__mass",
    },
    "outclay": {
        "wflow_v0": "lateral.river.outclay",
        "wflow_v1": "river_water_clay__mass_flow_rate",
    },
    "gravload": {
        "wflow_v0": "lateral.river.gravload",
        "wflow_v1": "river_water_gravel__mass",
    },
    "gravstore": {
        "wflow_v0": "lateral.river.gravstore",
        "wflow_v1": "river_bed_gravel__mass",
    },
    "outgrav": {
        "wflow_v0": "lateral.river.outgrav",
        "wflow_v1": "river_water_gravel__mass_flow_rate",
    },
    "laggload": {
        "wflow_v0": "lateral.river.laggload",
        "wflow_v1": "river_water_aggregates~large__mass",
    },
    "laggstore": {
        "wflow_v0": "lateral.river.laggstore",
        "wflow_v1": "river_bed_aggregates~large__mass",
    },
    "outlagg": {
        "wflow_v0": "lateral.river.outlagg",
        "wflow_v1": "river_water_aggregates~large__mass_flow_rate",
    },
    "saggload": {
        "wflow_v0": "lateral.river.saggload",
        "wflow_v1": "river_water_aggregates~small__mass",
    },
    "saggstore": {
        "wflow_v0": "lateral.river.saggstore",
        "wflow_v1": "river_bed_aggregates~small__mass",
    },
    "outsagg": {
        "wflow_v0": "lateral.river.outsagg",
        "wflow_v1": "river_water_aggregates~small__mass_flow_rate",
    },
    "sandload": {
        "wflow_v0": "lateral.river.sandload",
        "wflow_v1": "river_water_sand__mass",
    },
    "sandstore": {
        "wflow_v0": "lateral.river.sandstore",
        "wflow_v1": "river_bed_sand__mass",
    },
    "outsand": {
        "wflow_v0": "lateral.river.outsand",
        "wflow_v1": "river_water_sand__mass_flow_rate",
    },
    "siltload": {
        "wflow_v0": "lateral.river.siltload",
        "wflow_v1": "river_water_silt__mass",
    },
    "siltstore": {
        "wflow_v0": "lateral.river.siltstore",
        "wflow_v1": "river_bed_silt__mass",
    },
    "outsilt": {
        "wflow_v0": "lateral.river.outsilt",
        "wflow_v1": "river_water_silt__mass_flow_rate",
    },
}


def _create_hydromt_wflow_mapping(
    hydromt_dict: dict,
    model_dict: dict,
    config_dict: dict,
    wflow_version="wflow_v1",
) -> Tuple[dict, dict]:
    """
    Create dictionnaries to convert from hydromt/Wflow names to staticmaps names.

    The first dictionnary will be used to rename from hydromt convention
    name into name in staticmaps/forcing.
    The second dictionnary will be used to link the right name
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
        Dictionnary of the current model config to update from the default name in
        staticmaps to the actual name in staticmaps.
    wflow_version : str
        Version of Wflow to use for the mapping (as defined in ``model_dict``). Default
        is "wflow_v1".

    Returns
    -------
    mapping : dict
        Dictionnary of the mapping from hydromt names to staticmaps names.
    """
    # First dictionnary
    mapping_inv = {}  # name_in_staticmaps : name_in_hydromt
    # Second dictionnary
    wflow_names = dict()  # wflow_variable: name_in_staticmaps
    # Instantiate the mapping with default names (ie non wflow variables)
    for k, v in hydromt_dict.items():
        mapping_inv[v] = k

    # Go through the wflow variables and add them if hydromt name is not None
    for k, v in model_dict.items():
        if isinstance(v, dict):
            if "hydromt_name" in v:
                mapping_inv[k] = v.get("hydromt_name")
            # else use the default staticmap name as the name
            else:
                mapping_inv[k] = k
            if wflow_version in v:
                wflow_names[v.get(wflow_version)] = k

    # Update with the TOML
    # Check if wflow v0 config then do not update
    wflow_v0 = False
    if "starttime" in config_dict:  # first check
        wflow_v0 = True
    # second check for safety
    if "input" in config_dict and "vertical" in config_dict["input"]:
        wflow_v0 = True

    if "input" in config_dict and not wflow_v0:
        variable_types = ["forcing", "cyclic", "static"]
        for var_type in variable_types:
            if var_type not in config_dict["input"]:
                continue
            for var_name, new_name in config_dict["input"][var_type].items():
                # Get the old name
                old_name = wflow_names.get(var_name)
                # If they are different update the mapping
                if isinstance(new_name, dict):
                    if "value" in new_name:
                        # Do not update
                        new_name = None
                    elif "netcdf" in new_name:
                        new_name = new_name["netcdf"]["variable"]["name"]
                if new_name is not None and old_name != new_name:
                    # Update the mapping with the new name
                    mapping_inv[new_name] = mapping_inv.get(old_name, old_name)
                    # Remove the old name from the mapping
                    mapping_inv.pop(old_name, None)
                    # Update wflow_names with the new name
                    wflow_names[var_name] = new_name

    # Invert mapping to get hydromt_name: staticmap_name
    mapping_hydromt = dict()
    for k, v in mapping_inv.items():
        mapping_hydromt[v] = k
    # Get a mapping of staticmap_name: wflow_variable
    mapping_wflow = dict()
    for k, v in wflow_names.items():
        mapping_wflow[v] = k

    return mapping_hydromt, mapping_wflow


def _create_hydromt_wflow_mapping_sbm(config: dict) -> Tuple[dict, dict]:
    """
    Create a dictionnary to convert from hydromt names to wflow sbm input names.

    These names will be used in staticmaps/forcing and linked to the right wflow
    internal variables in the toml.
    """
    return _create_hydromt_wflow_mapping(HYDROMT_NAMES_DEFAULT, WFLOW_NAMES, config)


def _create_hydromt_wflow_mapping_sediment(config: dict) -> Tuple[dict, dict]:
    """
    Create a dictionnary to convert from hydromt names to wflow sediment input names.

    These names will be used in staticmaps/forcing and linked to the right wflow
    internal variables in the toml.
    """
    return _create_hydromt_wflow_mapping(
        HYDROMT_NAMES_DEFAULT_SEDIMENT, WFLOW_SEDIMENT_NAMES, config
    )
