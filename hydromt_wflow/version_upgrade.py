"""Some utilities to upgrade Wflow model versions."""

from __future__ import annotations

import copy
import logging
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Generator

import xarray as xr
from hydromt.readers import read_toml
from hydromt.writers import write_toml
from packaging.version import Version

from hydromt_wflow.components.tables import WflowTablesComponent
from hydromt_wflow.utils import DATA_DIR, get_config, set_config
from hydromt_wflow.workflows.reservoirs import (
    RESERVOIR_COMMON_PARAMETERS,
    RESERVOIR_CONTROL_PARAMETERS,
    RESERVOIR_LAYERS_SEDIMENT,
    RESERVOIR_UNCONTROL_PARAMETERS,
    merge_reservoirs,
    merge_reservoirs_sediment,
    set_rating_curve_layer_data_type,
)

if TYPE_CHECKING:
    # requires the __future__ import at the top of the file.
    from hydromt_wflow.wflow_sbm import WflowSbmModel
    from hydromt_wflow.wflow_sediment import WflowSedimentModel

logger = logging.getLogger(f"hydromt.{__name__}")

# Version upgrades in order. Each entry is a tuple of (from_version, to_version).
_UPGRADES: list[tuple[Version, Version]] = [
    (Version("0.8"), Version("1.0")),
    (Version("1.0"), Version("1.1")),
]
# The latest Wflow.jl version supported by this hydromt_wflow release
WFLOW_LATEST_VERSION = _UPGRADES[-1][-1]

__all__ = ["upgrade_model"]

# Static version-upgrade mappings

# v0 variable to v1 variable for SBM input variables
# (None means the variable was removed in v1)
_V0_TO_V1_INPUT_SBM: dict[str, str | None] = {
    # general input
    "subcatchment": "subbasin_location__count",
    "ldd": "basin__local_drain_direction",
    "lateral.river.lake.areas": None,
    "lateral.river.lake.locs": None,
    "lateral.river.reservoir.areas": "reservoir_area__count",
    "lateral.river.reservoir.locs": "reservoir_location__count",
    "river_location": "river_location__mask",
    "pits": "basin_pit_location__mask ",
    # atmosphere / forcing
    "vertical.precipitation": "atmosphere_water__precipitation_volume_flux",
    "vertical.temperature": "atmosphere_air__temperature",
    "vertical.potential_evaporation": "land_surface_water__potential_evaporation_volume_flux",  # noqa: E501
    # snow
    "vertical.cfmax": "snowpack__degree_day_coefficient",
    "vertical.tt": "atmosphere_air__snowfall_temperature_threshold",
    "vertical.tti": "atmosphere_air__snowfall_temperature_interval",
    "vertical.ttm": "snowpack__melting_temperature_threshold",
    "vertical.water_holding_capacity": "snowpack__liquid_water_holding_capacity",
    # glacier
    "vertical.glacierfrac": "glacier_surface__area_fraction",
    "vertical.glacierstore": "glacier_ice__initial_leq_depth",
    "vertical.g_ttm": "glacier_ice__melting_temperature_threshold",
    "vertical.g_tt": "glacier_ice__melting_temperature_threshold",
    "vertical.g_cfmax": "glacier_ice__degree_day_coefficient",
    "vertical.g_sifrac": "glacier_firn_accumulation__snowpack_dry_snow_leq_depth_fraction",  # noqa: E501
    # vegetation
    "vertical.e_r": "vegetation_canopy_water__mean_evaporation_to_mean_precipitation_ratio",  # noqa: E501
    "vertical.kext": "vegetation_canopy__light_extinction_coefficient",
    "vertical.leaf_area_index": "vegetation__leaf_area_index",
    "vertical.rootingdepth": "vegetation_root__depth",
    "vertical.specific_leaf": "vegetation__specific_leaf_storage",
    "vertical.storage_wood": "vegetation_wood_water__storage_capacity",
    "vertical.kc": "vegetation__crop_factor",
    "vertical.etreftopot": "vegetation__crop_factor",
    "vertical.alpha_h1": "vegetation_root__feddes_critical_pressure_head_h1_reduction_coefficient",  # noqa: E501
    "vertical.h1": "vegetation_root__feddes_critical_pressure_head_h1",
    "vertical.h2": "vegetation_root__feddes_critical_pressure_head_h2",
    "vertical.h3_high": "vegetation_root__feddes_critical_pressure_head_h3_high",
    "vertical.h3_low": "vegetation_root__feddes_critical_pressure_head_h3_low",
    "vertical.h4": "vegetation_root__feddes_critical_pressure_head_h4",
    # soil
    "vertical.c": "soil_layer_water__brooks_corey_exponent",
    "vertical.cf_soil": "soil_surface_water__infiltration_reduction_parameter",
    "vertical.kv_0": "soil_surface_water__vertical_saturated_hydraulic_conductivity",
    "vertical.kv\u2080": "soil_surface_water__vertical_saturated_hydraulic_conductivity",  # noqa: E501
    "vertical.kvfrac": "soil_layer_water__vertical_saturated_hydraulic_conductivity_factor",  # noqa: E501
    "lateral.subsurface.ksathorfrac": "subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio",  # noqa: E501
    "vertical.f": "soil_water__vertical_saturated_hydraulic_conductivity_scale_parameter",  # noqa: E501
    "vertical.infiltcappath": "compacted_soil_surface_water__infiltration_capacity",
    "vertical.infiltcapsoil": None,
    "vertical.theta_r": "soil_water__residual_volume_fraction",
    "vertical.\u03b8\u1d63": "soil_water__residual_volume_fraction",
    "vertical.theta_s": "soil_water__saturated_volume_fraction",
    "vertical.\u03b8\u209b": "soil_water__saturated_volume_fraction",
    "vertical.maxleakage": "soil_water_saturated_zone_bottom__max_leakage_volume_flux",
    "vertical.pathfrac": "compacted_soil__area_fraction",
    "vertical.rootdistpar": "soil_wet_root__sigmoid_function_shape_parameter",
    "vertical.soilthickness": "soil__thickness",
    # land
    "vertical.waterfrac": "land_water_covered__area_fraction",
    "vertical.allocation.areas": "land_water_allocation_area__count",
    "vertical.allocation.frac_sw_used": "land_surface_water__withdrawal_fraction",
    "vertical.domestic.demand_gross": "domestic__gross_water_demand_volume_flux",
    "vertical.domestic.demand_net": "domestic__net_water_demand_volume_flux",
    "vertical.industry.demand_gross": "industry__gross_water_demand_volume_flux",
    "vertical.industry.demand_net": "industry__net_water_demand_volume_flux",
    "vertical.livestock.demand_gross": "livestock__gross_water_demand_volume_flux",
    "vertical.livestock.demand_net": "livestock__net_water_demand_volume_flux",
    "vertical.paddy.h_min": "irrigated_paddy__min_depth",
    "vertical.paddy.h_opt": "irrigated_paddy__optimal_depth",
    "vertical.paddy.h_max": "irrigated_paddy__max_depth",
    "vertical.paddy.irrigation_areas": "irrigated_paddy_area__count",
    "vertical.paddy.irrigation_trigger": "irrigated_paddy__irrigation_trigger_flag",
    "vertical.nonpaddy.irrigation_areas": "irrigated_non_paddy_area__count",
    "vertical.nonpaddy.irrigation_trigger": "irrigated_non_paddy__irrigation_trigger_flag",  # noqa: E501
    # land surface water flow
    "lateral.land.n": "land_surface_water_flow__manning_n_parameter",
    "lateral.land.elevation": "land_surface_water_flow__ground_elevation",
    "vertical.altitude": "land_surface__elevation",
    "lateral.land.slope": "land_surface__slope",
    # river
    "lateral.river.floodplain.volume": "floodplain_water__sum_of_volume_per_depth",
    "lateral.river.floodplain.n": "floodplain_water_flow__manning_n_parameter",
    "lateral.river.bankfull_elevation": "river_bank_water__elevation",
    "lateral.river.inflow": "river_water__external_inflow_volume_flow_rate",
    "lateral.river.bankfull_depth": "river_bank_water__depth",
    "lateral.river.length": "river__length",
    "lateral.river.n": "river_water_flow__manning_n_parameter",
    "lateral.river.slope": "river__slope",
    "lateral.river.width": "river__width",
    # lakes
    "lateral.river.lake.area": None,
    "lateral.river.lake.waterlevel": "reservoir_water_surface__initial_elevation",
    "lateral.river.lake.threshold": "reservoir_water_flow_threshold_level__elevation",
    "lateral.river.lake.b": "reservoir_water__rating_curve_coefficient",
    "lateral.river.lake.e": "reservoir_water__rating_curve_exponent",
    "lateral.river.lake.outflowfunc": "reservoir_water__rating_curve_type_count",
    "lateral.river.lake.storfunc": "reservoir_water__storage_curve_type_count",
    "lateral.river.lake.linkedlakelocs": "reservoir_lower_location__count",
    # reservoirs
    "lateral.river.reservoir.area": "reservoir_surface__area",
    "lateral.river.reservoir.demand": "reservoir_water_demand__required_downstream_volume_flow_rate",  # noqa: E501
    "lateral.river.reservoir.maxrelease": "reservoir_water_release_below_spillway__max_volume_flow_rate",  # noqa: E501
    "lateral.river.reservoir.maxvolume": "reservoir_water__max_volume",
    "lateral.river.reservoir.targetfullfrac": "reservoir_water__target_full_volume_fraction",  # noqa: E501
    "lateral.river.reservoir.targetminfrac": "reservoir_water__target_min_volume_fraction",  # noqa: E501
    # gwf
    "lateral.subsurface.constant_head": "model_constant_boundary_condition__hydraulic_head",  # noqa: E501
    "lateral.subsurface.conductivity": "subsurface_surface_water__horizontal_saturated_hydraulic_conductivity",  # noqa: E501
    "lateral.subsurface.river_bottom": "river_bottom__elevation",
    "lateral.subsurface.infiltration_conductance": "river_water__infiltration_conductance",  # noqa: E501
    "lateral.subsurface.exfiltration_conductance": "river_water__exfiltration_conductance",  # noqa: E501
    "lateral.subsurface.specific_yield": "subsurface_water__specific_yield",
    "lateral.subsurface.gwf_f": "subsurface__horizontal_saturated_hydraulic_conductivity_scale_parameter",  # noqa: E501
    "lateral.subsurface.drain": "land_drain_location__mask",
    "lateral.subsurface.drain_conductance": "land_drain__conductance",
    "lateral.subsurface.drain_elevation": "land_drain__elevation",
}

# v0 variable to v1 variable for SBM state variables
_V0_TO_V1_STATES_SBM: dict[str, str | None] = {
    "vertical.canopystorage": "vegetation_canopy_water__depth",
    "vertical.satwaterdepth": "soil_water_saturated_zone__depth",
    "vertical.ustorelayerdepth": "soil_layer_water_unsaturated_zone__depth",
    "vertical.tsoil": "soil_surface__temperature",
    "vertical.snow": "snowpack_dry_snow__leq_depth",
    "vertical.snowwater": "snowpack_liquid_water__depth",
    "vertical.glacierstore": "glacier_ice__leq_depth",
    "lateral.land.q": "land_surface_water__instantaneous_volume_flow_rate",
    "lateral.land.qx": "land_surface_water__x_component_of_instantaneous_volume_flow_rate",  # noqa: E501
    "lateral.land.qy": "land_surface_water__y_component_of_instantaneous_volume_flow_rate",  # noqa: E501
    "lateral.land.h": "land_surface_water__depth",
    "lateral.land.h_av": None,
    "lateral.subsurface.flow.aquifer.head": "subsurface_water__hydraulic_head",
    "lateral.subsurface.ssf": "subsurface_water__volume_flow_rate",
    "lateral.river.q": "river_water__instantaneous_volume_flow_rate",
    "lateral.river.h": "river_water__depth",
    "lateral.river.h_av": None,
    "lateral.river.floodplain.q": "floodplain_water__instantaneous_volume_flow_rate",
    "lateral.river.floodplain.h": "floodplain_water__depth",
    "lateral.river.lake.waterlevel": "reservoir_water_surface__elevation",
    "lateral.river.reservoir.volume": None,
    "vertical.paddy.h": "paddy_surface_water__depth",
}

# v0 variable to v1 variable for Sediment input variables
_V0_TO_V1_INPUT_SEDIMENT: dict[str, str | None] = {
    # general input
    "subcatchment": "subbasin_location__count",
    "ldd": "basin__local_drain_direction",
    "vertical.lakeareas": None,
    "lateral.river.lakelocs": None,
    "vertical.resareas": "reservoir_area__count",
    "lateral.river.reslocs": "reservoir_location__count",
    "river_location": "river_location__mask",
    # forcing
    "vertical.precipitation": "atmosphere_water__precipitation_volume_flux",
    "vertical.interception": "vegetation_canopy_water__interception_volume_flux",
    "vertical.h_land": "land_surface_water__depth",
    "vertical.q_land": "land_surface_water__volume_flow_rate",
    "lateral.river.h_riv": "river_water__depth",
    "lateral.river.q_riv": "river_water__volume_flow_rate",
    # land properties
    "vertical.altitude": "land_surface__elevation",
    "lateral.land.slope": "land_surface__slope",
    # river properties
    "lateral.river.length": "river__length",
    "lateral.river.slope": "river__slope",
    "lateral.river.width": "river__width",
    # reservoirs
    "lateral.river.resarea": "reservoir_surface__area",
    "lateral.river.lakearea": None,
    "lateral.river.restrapeff": "reservoir_water_sediment__bedload_trapping_efficiency",
    # soil erosion
    "vertical.canopygapfraction": "vegetation_canopy__gap_fraction",
    "vertical.canopyheight": "vegetation_canopy__height",
    "vertical.kext": None,
    "vertical.specific_leaf": None,
    "vertical.storage_wood": None,
    "vertical.pathfrac": "compacted_soil__area_fraction",
    "vertical.erosk": "soil_erosion__rainfall_soil_detachability_factor",
    "vertical.erosspl": "soil_erosion__eurosem_exponent",
    "vertical.usleK": "soil_erosion__usle_k_factor",
    "vertical.usleC": "soil_erosion__usle_c_factor",
    "vertical.erosov": "soil_erosion__answers_overland_flow_factor",
    # river transport
    "lateral.river.rhos": "sediment__particle_density",
    "lateral.river.d50engelund": None,
    "lateral.river.cbagnold": "river_water_sediment__bagnold_transport_capacity_coefficient",  # noqa: E501
    "lateral.river.ebagnold": "river_water_sediment__bagnold_transport_capacity_exponent",  # noqa: E501
    "lateral.river.d50": "river_bottom_and_bank_sediment__median_diameter",
    "lateral.river.fclayriv": "river_bottom_and_bank_clay__mass_fraction",
    "lateral.river.fsiltriv": "river_bottom_and_bank_silt__mass_fraction",
    "lateral.river.fsandriv": "river_bottom_and_bank_sand__mass_fraction",
    "lateral.river.fgravelriv": "river_bottom_and_bank_gravel__mass_fraction",
}

# v0 variable to v1 variable for Sediment state variables
_V0_TO_V1_STATES_SEDIMENT: dict[str, str | None] = {
    "lateral.river.clayload": "river_water_clay__mass",
    "lateral.river.claystore": "river_bed_clay__mass",
    "lateral.river.outclay": "river_water_clay__mass_flow_rate",
    "lateral.river.gravload": "river_water_gravel__mass",
    "lateral.river.gravstore": "river_bed_gravel__mass",
    "lateral.river.outgrav": "river_water_gravel__mass_flow_rate",
    "lateral.river.laggload": "river_water_large_aggregates__mass",
    "lateral.river.laggstore": "river_bed_large_aggregates__mass",
    "lateral.river.outlagg": "river_water_large_aggregates__mass_flow_rate",
    "lateral.river.saggload": "river_water_small_aggregates__mass",
    "lateral.river.saggstore": "river_bed_small_aggregates__mass",
    "lateral.river.outsagg": "river_water_small_aggregates__mass_flow_rate",
    "lateral.river.sandload": "river_water_sand__mass",
    "lateral.river.sandstore": "river_bed_sand__mass",
    "lateral.river.outsand": "river_water_sand__mass_flow_rate",
    "lateral.river.siltload": "river_water_silt__mass",
    "lateral.river.siltstore": "river_bed_silt__mass",
    "lateral.river.outsilt": "river_water_silt__mass_flow_rate",
}

# v1 to v1.1 renames (only the variables that actually changed)
_V1_TO_V1_1_SBM: dict[str, str] = {
    "subsurface_water__volume_flow_rate": "subsurface_water__instantaneous_volume_flow_rate",  # noqa: E501
}

# staticmap_name to v0 config variable (for reservoir conversion in SBM)
_STATICMAP_TO_V0_VAR_SBM: dict[str, str] = {
    "reservoir_area_id": "lateral.river.reservoir.areas",
    "reservoir_outlet_id": "lateral.river.reservoir.locs",
    "reservoir_area": "lateral.river.reservoir.area",
    "reservoir_initial_depth": "lateral.river.lake.waterlevel",
    "reservoir_outflow_threshold": "lateral.river.lake.threshold",
    "reservoir_b": "lateral.river.lake.b",
    "reservoir_e": "lateral.river.lake.e",
    "reservoir_rating_curve": "lateral.river.lake.outflowfunc",
    "reservoir_storage_curve": "lateral.river.lake.storfunc",
    "reservoir_lower_id": "lateral.river.lake.linkedlakelocs",
    "reservoir_max_volume": "lateral.river.reservoir.maxvolume",
    "reservoir_target_min_fraction": "lateral.river.reservoir.targetminfrac",
    "reservoir_target_full_fraction": "lateral.river.reservoir.targetfullfrac",
    "reservoir_demand": "lateral.river.reservoir.demand",
    "reservoir_max_release": "lateral.river.reservoir.maxrelease",
    "lake_area_id": "lateral.river.lake.areas",
    "lake_outlet_id": "lateral.river.lake.locs",
    "lake_area": "lateral.river.lake.area",
}

# staticmap_name to v1 config variable (for reservoir conversion in SBM)
_STATICMAP_TO_V1_VAR_SBM: dict[str, str | None] = {
    "reservoir_area_id": "reservoir_area__count",
    "reservoir_outlet_id": "reservoir_location__count",
    "reservoir_area": "reservoir_surface__area",
    "reservoir_initial_depth": "reservoir_water_surface__initial_elevation",
    "reservoir_outflow_threshold": "reservoir_water_flow_threshold_level__elevation",
    "reservoir_b": "reservoir_water__rating_curve_coefficient",
    "reservoir_e": "reservoir_water__rating_curve_exponent",
    "reservoir_rating_curve": "reservoir_water__rating_curve_type_count",
    "reservoir_storage_curve": "reservoir_water__storage_curve_type_count",
    "reservoir_lower_id": "reservoir_lower_location__count",
    "reservoir_max_volume": "reservoir_water__max_volume",
    "reservoir_target_min_fraction": "reservoir_water__target_min_volume_fraction",
    "reservoir_target_full_fraction": "reservoir_water__target_full_volume_fraction",
    "reservoir_demand": "reservoir_water_demand__required_downstream_volume_flow_rate",
    "reservoir_max_release": "reservoir_water_release_below_spillway__max_volume_flow_rate",  # noqa: E501
    "lake_area_id": None,
    "lake_outlet_id": None,
    "lake_area": None,
}

# staticmap_name to v0 config variable (for reservoir conversion in Sediment)
_STATICMAP_TO_V0_VAR_SEDIMENT: dict[str, str] = {
    "reservoir_area_id": "vertical.resareas",
    "reservoir_outlet_id": "lateral.river.reslocs",
    "reservoir_area": "lateral.river.resarea",
    "reservoir_trapping_efficiency": "lateral.river.restrapeff",
    "lake_area_id": "vertical.lakeareas",
    "lake_outlet_id": "lateral.river.lakelocs",
    "lake_area": "lateral.river.lakearea",
}

# staticmap_name to v1 config variable (for reservoir conversion in Sediment)
_STATICMAP_TO_V1_VAR_SEDIMENT: dict[str, str | None] = {
    "reservoir_area_id": "reservoir_area__count",
    "reservoir_outlet_id": "reservoir_location__count",
    "reservoir_area": "reservoir_surface__area",
    "reservoir_trapping_efficiency": "reservoir_water_sediment__bedload_trapping_efficiency",  # noqa: E501
    "lake_area_id": None,
    "lake_outlet_id": None,
    "lake_area": None,
}

# v0 output variable to v1 output variable (used for output section conversion in SBM)
_V0_TO_V1_OUTPUT_SBM: dict[str, str] = {
    "vertical.interception": "vegetation_canopy_water__interception_volume_flux",
    "vertical.actevap": "land_surface__evapotranspiration_volume_flux",
    "vertical.actinfilt": "soil_water__infiltration_volume_flux",
    "vertical.excesswatersoil": "compacted_soil_surface_water__excess_volume_flux",
    "vertical.excesswaterpath": "non_compacted_soil_surface_water__excess_volume_flux",
    "vertical.exfiltustore": "soil_surface_water_unsaturated_zone__exfiltration_volume_flux",  # noqa: E501
    "vertical.exfiltsatwater": "land.soil.variables.exfiltsatwater",
    "vertical.recharge": "soil_water_saturated_zone_top__net_recharge_volume_flux",
    "vertical.vwc_percroot": "soil_water_root_zone__volume_percentage",
    "lateral.land.q_av": "land_surface_water__volume_flow_rate",
    "lateral.land.h_av": "land_surface_water__depth",
    "lateral.land.to_river": "land_surface_water__to_river_volume_flow_rate",
    "lateral.subsurface.to_river": "subsurface_water__to_river_volume_flow_rate",
    "lateral.subsurface.drain.flux": "land_drain_water__to_subsurface_volume_flow_rate",
    "lateral.subsurface.flow.aquifer.head": "subsurface_water__hydraulic_head",
    "lateral.subsurface.river.flux": "river_water__to_subsurface_volume_flow_rate",
    "lateral.subsurface.recharge.rate": "subsurface_water_saturated_zone_top__net_recharge_volume_flow_rate",  # noqa: E501
    "lateral.river.q_av": "river_water__volume_flow_rate",
    "lateral.river.h_av": "river_water__depth",
    "lateral.river.volume": "river_water__volume",
    "lateral.river.inwater": "river_water__lateral_inflow_volume_flow_rate",
    "lateral.river.floodplain.volume": "floodplain_water__volume",
    "lateral.river.reservoir.volume": "reservoir_water__volume",
    "lateral.river.reservoir.totaloutflow": "reservoir_water__outgoing_volume_flow_rate",  # noqa: E501
    "lateral.river.reservoir.outflow": "reservoir_water__outgoing_volume_flow_rate",
    "lateral.river.reservoir.inflow": "reservoir_water__incoming_volume_flow_rate",
    "lateral.river.reservoir.precipitation": "reservoir_water__precipitation_volume_flux",  # noqa: E501
    "lateral.river.reservoir.evaporation": "reservoir_water__potential_evaporation_volume_flux",  # noqa: E501
    "lateral.river.reservoir.actevap": "reservoir_water__evaporation_volume_flux",
    "lateral.river.lake.storage": "reservoir_water__volume",
    "lateral.river.lake.totaloutflow": "reservoir_water__outgoing_volume_flow_rate",
    "lateral.river.lake.outflow": "reservoir_water__outgoing_volume_flow_rate",
    "lateral.river.lake.inflow": "reservoir_water__incoming_volume_flow_rate",
    "lateral.river.lake.waterlevel": "reservoir_water_surface__elevation",
    "lateral.river.lake.precipitation": "reservoir_water__precipitation_volume_flux",
    "lateral.river.lake.evaporation": "reservoir_water__potential_evaporation_volume_flux",  # noqa: E501
    "lateral.river.lake.actevap": "reservoir_water__evaporation_volume_flux",
}

# v0 model option to v1 model option (SBM)
_V0_TO_V1_MODEL_OPTIONS_SBM: dict[str, str | list[str] | None] = {
    "reinit": "cold_start__flag",
    "sizeinmetres": "cell_length_in_meter__flag",
    "reservoirs": "reservoir__flag",
    "lakes": "reservoir__flag",
    "snow": "snow__flag",
    "glacier": "glacier__flag",
    "pits": "pit__flag",
    "masswasting": "snow_gravitational_transport__flag",
    "thicknesslayers": "soil_layer__thickness",
    "min_streamorder_land": "land_streamorder__min_count",
    "min_streamorder_river": "river_streamorder__min_count",
    "drains": "drain__flag",
    "kin_wave_iteration": "kinematic_wave__adaptive_time_step_flag",
    "kw_land_tstep": "land_kinematic_wave__time_step",
    "kw_river_tstep": "river_kinematic_wave__time_step",
    "inertial_flow_alpha": [
        "river_local_inertial_flow__alpha_coefficient",
        "land_local_inertial_flow__alpha_coefficient",
    ],
    "h_thresh": [
        "river_water_flow_threshold__depth",
        "land_surface_water_flow_threshold__depth",
    ],
    "froude_limit": [
        "river_water_flow__froude_limit_flag",
        "land_surface_water_flow__froude_limit_flag",
    ],
    "floodplain_1d": "floodplain_1d__flag",
    "inertial_flow_theta": "land_local_inertial_flow__theta_coefficient",
    "soilinfreduction": "soil_infiltration_reduction__flag",
    "transfermethod": "topog_sbm_transfer__flag",
    "water_demand.domestic": "water_demand.domestic__flag",
    "water_demand.industry": "water_demand.industry__flag",
    "water_demand.livestock": "water_demand.livestock__flag",
    "water_demand.paddy": "water_demand.paddy__flag",
    "water_demand.nonpaddy": "water_demand.nonpaddy__flag",
    "constanthead": "constanthead__flag",
    "riverlength_bc": None,  # moved to cross_options
}

# v0 input top-level key to v1 input key (SBM)
_V0_TO_V1_INPUT_OPTIONS_SBM: dict[str, str] = {
    "ldd": "basin__local_drain_direction",
    "river_location": "river_location__mask",
    "subcatchment": "subbasin_location__count",
}

# v0 variables that belong in input (not input.static) in SBM
_V0_INPUT_SECTION_VARS_SBM: list[str] = [
    "lateral.river.lake.areas",
    "lateral.river.lake.locs",
    "lateral.river.lake.linkedlakelocs",
    "lateral.river.reservoir.areas",
    "lateral.river.reservoir.locs",
]

# v0 options that cross main config sections in SBM (old path to new path)
_V0_TO_V1_CROSS_OPTIONS_SBM: dict[str, str] = {
    "input.lateral.subsurface.conductivity_profile": "model.conductivity_profile",
    "model.riverlength_bc": "input.static.model_boundary_condition_river__length",
}

# v0 output variable to v1 output variable (Sediment)
_V0_TO_V1_OUTPUT_SEDIMENT: dict[str, str] = {
    "vertical.soilloss": "soil_erosion__mass_flow_rate",
    "lateral.river.SSconc": "river_water_sediment__suspended_mass_concentration",
    "lateral.river.Bedconc": "river_water_sediment__bedload_mass_concentration",
    "lateral.river.outsed": "land_surface_water_sediment__mass_flow_rate",
    "lateral.land.inlandclay": "land_surface_water_clay__to_river_mass_flow_rate",
    "lateral.land.inlandsilt": "land_surface_water_silt__to_river_mass_flow_rate",
    "lateral.land.inlandsand": "land_surface_water_sand__to_river_mass_flow_rate",
    "lateral.land.inlandsagg": "land_surface_water_small_aggregates__to_river_mass_flow_rate",  # noqa: E501
    "lateral.land.inlandlagg": "land_surface_water_large_aggregates__to_river_mass_flow_rate",  # noqa: E501
}

# v0 model option to v1 model option (Sediment)
_V0_TO_V1_MODEL_OPTIONS_SEDIMENT: dict[str, str] = {
    "reinit": "cold_start__flag",
    "sizeinmetres": "cell_length_in_meter__flag",
    "runrivermodel": "run_river_model__flag",
    "doreservoir": "reservoir__flag",
    "dolake": "reservoir__flag",
    "rainerosmethod": "rainfall_erosion",
    "landtransportmethod": "land_transport",
    "rivtransportmethod": "river_transport",
}

# v0 input top-level key to v1 input key (Sediment)
_V0_TO_V1_INPUT_OPTIONS_SEDIMENT: dict[str, str] = {
    "ldd": "basin__local_drain_direction",
    "river_location": "river_location__mask",
    "subcatchment": "subbasin_location__count",
}

# v0 variables that belong in input (not input.static) in Sediment
_V0_INPUT_SECTION_VARS_SEDIMENT: list[str] = [
    "vertical.lakeareas",
    "lateral.river.lakelocs",
    "vertical.resareas",
    "lateral.river.reslocs",
]


def _solve_var_name(
    var: str | dict, path: str, add: list
) -> Generator[tuple[str, str], None, None]:
    """Solve the config file into individual entries.

    Every entry is the entire path ("river.lateral.< something >") plus its value.

    Parameters
    ----------
    var : str | dict,
        Either the direct settings entry or a dictionary containing nested settings.
    path : str,
        Prepend the entries with the value (e.g. "lateral" or "lateral.river")
    add : list
        Usually an empty list in which the temporary headers are stored.

    Yields
    ------
    tuple[str, str]
        The path and value of each entry in the config.
    """
    if not isinstance(var, dict):
        sep = "." if path else ""
        add_str = ".".join(add) if add else ""
        yield (var, path + sep + add_str)
        return
    for key, item in var.items():
        yield from _solve_var_name(item, path, add + [key])


def _set_input_vars(
    wflow_v0_var: str,
    wflow_v1_var: str | None,
    config_in: dict,
    config_out: dict,
    input_options: dict,
    input_variables: dict,
    forcing_variables: list[str],
    cyclic_variables: list[str],
) -> dict:
    name = get_config(key=f"input.{wflow_v0_var}", config=config_in, fallback=None)
    if name is not None and wflow_v1_var is not None:
        if isinstance(name, dict) and "netcdf" in name.keys():
            name["netcdf_variable_name"] = name["netcdf"]["variable"]["name"]
            del name["netcdf"]
        if wflow_v0_var in input_options.keys():
            return config_out
        elif wflow_v0_var in input_variables:
            config_out["input"][wflow_v1_var] = name
        elif wflow_v0_var in forcing_variables:
            config_out["input"]["forcing"][wflow_v1_var] = name
        elif wflow_v0_var in cyclic_variables:
            config_out["input"]["cyclic"][wflow_v1_var] = name
        else:
            config_out["input"]["static"][wflow_v1_var] = name
    return config_out


# All `_convert_**` functions upgrade data in memory.
# They do not write to disk.
# They do not modify the input config (they return a new config).
def _convert_to_wflow_v1(
    config: dict,
    v0_to_v1_input: dict[str, str | None],
    v0_to_v1_states: dict[str, str | None],
    model_options: dict = {},
    cross_options: dict = {},
    input_options: dict = {},
    input_variables: list = [],
    additional_variables: dict = {},
) -> dict:
    """Convert the config to Wflow v1 format.

    Parameters
    ----------
    config: dict
        The config to convert.
    v0_to_v1_input: dict
        Flat mapping of v0 input variable names to v1 variable names (or None).
    v0_to_v1_states: dict
        Flat mapping of v0 state variable names to v1 variable names (or None).
    model_options: dict, optional
        Options in the [model] section of the TOML that were updated in Wflow v1.
    input_options: dict, optional
        Options in the [input] section of the TOML that were updated in Wflow v1.
    input_variables: list, optional
        Variables that were moved to input rather than input.static.

    Returns
    -------
    config_out: dict
        The converted config.
    """
    if config["model"].get("reinit") == False:
        logger.warning(
            "Converting states is not supported by this conversion code, therefore the "
            "reinit option (new name cold_start__flag) is adjusted to True. If model "
            "states are required, please produce new states using a Wflow simulation or"
            " use the setup_cold_states function."
        )
        config["model"]["reinit"] = True
    # Build the merged WFLOW_CONVERSION (states override inputs on conflict)
    WFLOW_CONVERSION = dict(v0_to_v1_input)
    WFLOW_CONVERSION.update(v0_to_v1_states)
    # Add a few extra output variables that are supported by the conversion
    WFLOW_CONVERSION.update(additional_variables)

    # Logging function for the other non supported variables
    def _warn_str(wflow_var, output_type):
        logger.warning(
            f"Output variable {wflow_var} not supported for the conversion. "
            f"Skipping from {output_type} output."
        )

    # Update function for the output.netcdf_grid
    def _update_output_netcdf_grid(wflow_var, var_name):
        if wflow_var in WFLOW_CONVERSION.keys():
            config_out["output"]["netcdf_grid"]["variables"][
                WFLOW_CONVERSION[wflow_var]
            ] = var_name
        else:
            _warn_str(var_name, "netcdf_grid")

    # Initialize the output config
    logger.info("Converting config to Wflow v1 format")
    logger.info("Converting config general, time and model sections")
    config_out = {}

    # Start with the general section - split into general, time and logging in v1
    input_section = {
        "general": ["dir_input", "dir_output", "fews_run"],
        "time": [
            "calendar",
            "starttime",
            "endtime",
            "time_units",
            "timestepsecs",
        ],
        "logging": ["loglevel", "path_log", "silent"],
    }
    for section, options in input_section.items():
        for key in options:
            value = get_config(key=key, config=config, fallback=None)
            if value is not None:
                if section == "general":
                    # only fews_run was renamed in general options
                    if key == "fews_run":
                        key = "fews_run__flag"
                    config_out[key] = value
                else:
                    config_out[section] = config_out.get(section, {})
                    config_out[section][key] = value

    # Model section
    config_out["model"] = {}
    for value, config_var in _solve_var_name(config["model"], "", []):
        if config_var not in model_options:
            # Model option that did not change in v1
            set_config(config_out, f"model.{config_var}", value)
        else:
            new_config_var = model_options[config_var]
            if isinstance(new_config_var, (list, tuple)):
                for elem in new_config_var:
                    set_config(config_out, f"model.{elem}", value)
                continue
            elif new_config_var is None:
                # This option was moved to cross_options
                continue
            set_config(config_out, f"model.{new_config_var}", value)

    # Routing options were renamed
    routing_rename = {
        "kinematic-wave": "kinematic_wave",
        "local-inertial": "local_inertial",
    }
    for key in ["river_routing", "land_routing"]:
        routing_option = get_config(key=f"model.{key}", config=config, fallback=None)
        if routing_option is not None:
            set_config(config_out, f"model.{key}", routing_rename.get(routing_option))

    # State
    logger.info("Converting config state section")
    config_out["state"] = {
        "path_input": get_config(
            key="state.path_input", config=config, fallback="instate/instates.nc"
        ),
        "path_output": get_config(
            key="state.path_output", config=config, fallback="outstate/outstates.nc"
        ),
    }
    # Go through the states variables
    config_out["state"]["variables"] = {}
    for v0_var, v1_var in v0_to_v1_states.items():
        if v1_var is None:
            continue
        name = get_config(key=f"state.{v0_var}", config=config, fallback=None)
        if name is not None:
            config_out["state"]["variables"][v1_var] = name

    # Input section
    logger.info("Converting config input section")
    cyclic_variables = []
    forcing_variables = []

    reservoir_locs = []
    if get_config(config, "input.lateral.river.reservoir.locs") is not None:
        reservoir_locs.append(get_config(config, "input.lateral.river.reservoir.locs"))
    if get_config(config, "input.lateral.river.lake.locs") is not None:
        reservoir_locs.append(get_config(config, "input.lateral.river.lake.locs"))
    reservoir_locs_map = []

    config_out["input"] = {}
    for key, name in config["input"].items():
        if key in input_options.keys():
            config_out["input"][input_options[key]] = name
        elif key == "forcing":
            forcing_variables = name
        elif key == "cyclic":
            cyclic_variables = name
        elif key not in ["vertical", "lateral"]:  # variables are done separately
            # For reservoir / lake locations map, skip
            if name in reservoir_locs:
                reservoir_locs_map.append(key)
            else:
                config_out["input"][key] = name

    # Go through the input variables
    config_out["input"]["forcing"] = {}
    config_out["input"]["cyclic"] = {}
    config_out["input"]["static"] = {}
    for wflow_v0_var, wflow_v1_var in v0_to_v1_input.items():
        if wflow_v0_var is None:
            continue
        config_out = _set_input_vars(
            wflow_v0_var=wflow_v0_var,
            wflow_v1_var=wflow_v1_var,
            config_in=config,
            config_out=config_out,
            input_options=input_options,
            input_variables=input_variables,
            forcing_variables=forcing_variables,
            cyclic_variables=cyclic_variables,
        )

    # Cross options
    for opt_old, opt_new in cross_options.items():
        value = get_config(key=opt_old, config=config)
        if value is None:
            continue
        # Ensure that it is set either as a value or as a map
        elif isinstance(value, str):
            set_config(config_out, opt_new, value)
        else:
            set_config(config_out, f"{opt_new}.value", value)

    # Output netcdf_grid section
    logger.info("Converting config output sections")
    if get_config(key="output", config=config, fallback=None) is not None:
        config_out["output"] = {}
        config_out["output"]["netcdf_grid"] = {
            "path": get_config(key="output.path", config=config, fallback="output.nc"),
            "compressionlevel": get_config(
                key="output.compressionlevel", config=config, fallback=1
            ),
        }
        config_out["output"]["netcdf_grid"]["variables"] = {}
        for key, value in config["output"].items():
            if key in ["path", "compressionlevel"]:
                continue

            for var_name, wflow_var in _solve_var_name(value, key, []):
                _update_output_netcdf_grid(wflow_var, var_name)

    # Output netcdf_scalar section
    map_variable_conversions = {
        "lateral.river.lake.locs": "reservoir_location__count",
        "lateral.river.reservoir.locs": "reservoir_location__count",
    }
    # Add the reservoir maps that were potentially in the input section
    for var in reservoir_locs_map:
        map_variable_conversions[var] = "reservoir_location__count"

    if get_config(key="netcdf", config=config, fallback=None) is not None:
        if "output" not in config_out:
            config_out["output"] = {}
        config_out["output"]["netcdf_scalar"] = {
            "path": get_config(
                key="netcdf.path", config=config, fallback="output_scalar.nc"
            ),
        }
        config_out["output"]["netcdf_scalar"]["variable"] = []
        nc_scalar_vars = get_config(key="netcdf.variable", config=config, fallback=[])
        for nc_scalar in nc_scalar_vars:
            if nc_scalar["parameter"] in WFLOW_CONVERSION:
                nc_scalar["parameter"] = WFLOW_CONVERSION[nc_scalar["parameter"]]
                if map_var := nc_scalar.get("map"):
                    if map_var in input_options:
                        nc_scalar["map"] = input_options[map_var]
                    elif map_var in map_variable_conversions:
                        nc_scalar["map"] = map_variable_conversions[map_var]
                config_out["output"]["netcdf_scalar"]["variable"].append(nc_scalar)
            else:
                _warn_str(nc_scalar["parameter"], "netcdf_scalar")

    # Output csv section
    if get_config(key="csv", config=config, fallback=None) is not None:
        if "output" not in config_out:
            config_out["output"] = {}
        config_out["output"]["csv"] = {}

        config_out["output"]["csv"]["path"] = get_config(
            key="csv.path", config=config, fallback="output.csv"
        )
        config_out["output"]["csv"]["column"] = []
        csv_vars = get_config(key="csv.column", config=config, fallback=[])
        for csv_var in csv_vars:
            if csv_var["parameter"] in WFLOW_CONVERSION:
                csv_var["parameter"] = WFLOW_CONVERSION[csv_var["parameter"]]
                if map_var := csv_var.get("map"):
                    if map_var in input_options:
                        csv_var["map"] = input_options[map_var]
                    elif map_var in map_variable_conversions:
                        csv_var["map"] = map_variable_conversions[map_var]
                config_out["output"]["csv"]["column"].append(csv_var)
            else:
                _warn_str(csv_var["parameter"], "csv")

    config_out["wflow_version"] = "1.0"
    return config_out


def _convert_sbm_config_v0_to_v1(config: dict) -> dict:
    """Convert the config to Wflow v1 format for SBM.

    Parameters
    ----------
    config: dict
        The config to convert.

    Returns
    -------
    config_out: dict
        The converted config.
    """
    return _convert_to_wflow_v1(
        config=config,
        v0_to_v1_input=_V0_TO_V1_INPUT_SBM,
        v0_to_v1_states=_V0_TO_V1_STATES_SBM,
        model_options=_V0_TO_V1_MODEL_OPTIONS_SBM,
        cross_options=_V0_TO_V1_CROSS_OPTIONS_SBM,
        input_options=_V0_TO_V1_INPUT_OPTIONS_SBM,
        input_variables=_V0_INPUT_SECTION_VARS_SBM,
        additional_variables=_V0_TO_V1_OUTPUT_SBM,
    )


def _convert_sediment_config_v0_to_v1(config: dict) -> dict:
    """Convert the config to Wflow v1 format for sediment.

    Parameters
    ----------
    config: dict
        The config to convert.

    Returns
    -------
    config_out: dict
        The converted config.
    """
    return _convert_to_wflow_v1(
        config=config,
        v0_to_v1_input=_V0_TO_V1_INPUT_SEDIMENT,
        v0_to_v1_states=_V0_TO_V1_STATES_SEDIMENT,
        model_options=_V0_TO_V1_MODEL_OPTIONS_SEDIMENT,
        input_options=_V0_TO_V1_INPUT_OPTIONS_SEDIMENT,
        input_variables=_V0_INPUT_SECTION_VARS_SEDIMENT,
        additional_variables=_V0_TO_V1_OUTPUT_SEDIMENT,
    )


def _convert_sbm_reservoirs_v0_to_v1(
    grid: xr.Dataset,
    config: dict,
) -> tuple[xr.Dataset | None, list[str], dict[str, str]]:
    """
    Merge reservoirs and lakes layers from a v0.x model config.

    Returns None if no reservoirs or lakes are present in the config.
    If reservoirs are present, also add rating_curve, storage_curve and initial_depth.
    The output variables will use standard names for the variables (e.g.
    "reservoir_area_id").

    Parameters
    ----------
    grid : xr.Dataset
        The grid dataset containing the reservoirs and lakes layers.
    config : dict
        The model configuration dictionary.

    Returns
    -------
    xr.Dataset | None
        The merged dataset with reservoirs and lakes layers, or None if not applicable.
    list[str]
        List of variables to remove from the grid (will be replaced by the merged ones).
    dict[str, str]
        Dictionary of wflow v1 config options to update with the new variable names.
    """
    variables_to_remove = []
    config_options = {}
    ds_res = xr.Dataset()

    has_reservoirs = get_config(key="model.reservoirs", config=config, fallback=False)
    has_lakes = get_config(key="model.lakes", config=config, fallback=False)

    if not has_reservoirs and not has_lakes:
        logger.info("No reservoirs or lakes found in the config. Skipping conversion.")
        return None, variables_to_remove, config_options
    else:
        logger.info(
            "Reservoirs/lakes found in the grid, converting to Wflow v1 format."
        )
        # config options will need to be updated with the standard names
        config_options["model.reservoir__flag"] = True
        config_options["state.variables.reservoir_water_surface__elevation"] = (
            "reservoir_water_level"
        )

    # Start with the reservoir layers
    if has_reservoirs:
        reservoir_layers = (
            RESERVOIR_COMMON_PARAMETERS
            + RESERVOIR_CONTROL_PARAMETERS
            + [
                "reservoir_area_id",
                "reservoir_outlet_id",
            ]
        )
        for layer in reservoir_layers:
            if layer in [
                "reservoir_initial_depth",
                "reservoir_rating_curve",
                "reservoir_storage_curve",
            ]:
                # These layers are not in the config, so we skip them
                continue
            wflow_var = _STATICMAP_TO_V0_VAR_SBM[layer]
            # get the map name from config
            map_name = get_config(key=f"input.{wflow_var}", config=config)
            # add to the variables to remove
            variables_to_remove.append(map_name)
            if map_name in grid:
                ds_res[layer] = grid[map_name].copy()
            else:
                logger.warning(f"Reservoir layer {map_name} not found in the grid.")

        # Add the additional layers that are not in the config
        res_mask_inv = (
            ds_res["reservoir_outlet_id"] == ds_res["reservoir_outlet_id"].raster.nodata
        )
        ds_res["reservoir_rating_curve"] = ds_res["reservoir_outlet_id"].where(
            res_mask_inv,
            4,
        )
        ds_res["reservoir_storage_curve"] = ds_res["reservoir_outlet_id"].where(
            res_mask_inv,
            1,
        )
        ds_res["reservoir_initial_depth"] = (
            ds_res["reservoir_target_full_fraction"]
            * ds_res["reservoir_max_volume"]
            / ds_res["reservoir_area"]
        )

        # Update the config options
        for layer in reservoir_layers:
            # layers that are in "main" input section
            if layer in [
                "reservoir_area_id",
                "reservoir_outlet_id",
                "reservoir_lower_id",
            ]:
                wflow_var_v1 = f"input.{_STATICMAP_TO_V1_VAR_SBM[layer]}"
            else:
                # Find if in cyclic / forcing / static
                v0_var = _STATICMAP_TO_V0_VAR_SBM[layer]
                if v0_var in get_config(config=config, key="input.cyclic", fallback=[]):
                    wflow_var_v1 = f"input.cyclic.{_STATICMAP_TO_V1_VAR_SBM[layer]}"
                elif v0_var in get_config(
                    config=config, key="input.forcing", fallback=[]
                ):
                    wflow_var_v1 = f"input.forcing.{_STATICMAP_TO_V1_VAR_SBM[layer]}"
                else:
                    wflow_var_v1 = f"input.static.{_STATICMAP_TO_V1_VAR_SBM[layer]}"
            config_options[wflow_var_v1] = layer

    # Move to the lake layers
    if has_lakes:
        lake_layers = RESERVOIR_UNCONTROL_PARAMETERS + [
            "lake_area_id",
            "lake_outlet_id",
            "lake_area",
            "reservoir_initial_depth",
            "reservoir_rating_curve",
            "reservoir_storage_curve",
        ]
        ds_lakes = xr.Dataset()

        for layer in lake_layers:
            wflow_var = _STATICMAP_TO_V0_VAR_SBM[layer]
            # get the map name from config
            map_name = get_config(key=f"input.{wflow_var}", config=config)
            # add to the variables to remove
            variables_to_remove.append(map_name)
            if map_name in grid:
                # Replace lake by reservoir in the output layer name
                layer_out = layer.replace("lake", "reservoir")
                ds_lakes[layer_out] = grid[map_name].copy()

        # Merge lakes into the reservoirs dataset if needed
        if not has_reservoirs:
            ds_res = ds_lakes
        else:
            ds_merge = merge_reservoirs(
                ds_lakes,
                ds_res,
                duplicate_id="skip",
            )
            if ds_merge is None:
                logger.warning(
                    "Merging lakes into reservoirs failed because of duplicated IDs. "
                    "Only reservoirs will be converted and you may need to update the "
                    "models to add the missing lakes manually."
                )
            else:
                ds_res = ds_merge

        # Update the config options
        for layer in lake_layers:
            layer_out = layer.replace("lake", "reservoir")
            if layer in ["lake_area_id", "lake_outlet_id", "reservoir_lower_id"]:
                wflow_var_v1 = f"input.{_STATICMAP_TO_V1_VAR_SBM[layer_out]}"
            else:
                wflow_var_v1 = f"input.static.{_STATICMAP_TO_V1_VAR_SBM[layer_out]}"
            config_options[wflow_var_v1] = layer_out

    # Ensure correct data types for rating curve
    ds_res = set_rating_curve_layer_data_type(ds_res)
    return ds_res, variables_to_remove, config_options


def _convert_sediment_reservoirs_v0_to_v1(
    grid: xr.Dataset,
    config: dict,
) -> tuple[xr.Dataset | None, list[str], dict[str, str]]:
    """
    Merge reservoirs and lakes layers from a v0.x model config.

    Returns None if no reservoirs or lakes are present in the config.
    If lakes are present, also add trapping_efficiency of 0. The output variables will
    use standard names for the variables (e.g. "reservoir_area_id").

    Parameters
    ----------
    grid : xr.Dataset
        The grid dataset containing the reservoirs and lakes layers.
    config : dict
        The model configuration dictionary.

    Returns
    -------
    xr.Dataset | None
        The merged dataset with reservoirs and lakes layers, or None if not applicable.
    list[str]
        List of variables to remove from the grid (will be replaced by the merged ones).
    dict[str, str]
        Dictionary of wflow v1 config options to update with the new variable names.
    """
    variables_to_remove = []
    config_options = {}
    ds_res = xr.Dataset()

    has_reservoirs = get_config(key="model.doreservoir", config=config, fallback=False)
    has_lakes = get_config(key="model.dolake", config=config, fallback=False)

    if not has_reservoirs and not has_lakes:
        logger.info("No reservoirs or lakes found in the config. Skipping conversion.")
        return None, variables_to_remove, config_options
    else:
        logger.info(
            "Reservoirs/lakes found in the grid, converting to Wflow v1 format."
        )
        # config options will need to be updated with the standard names
        config_options["model.reservoir__flag"] = True
        for layer in RESERVOIR_LAYERS_SEDIMENT:
            if layer in ["reservoir_area_id", "reservoir_outlet_id"]:
                wflow_var_v1 = f"input.{_STATICMAP_TO_V1_VAR_SEDIMENT[layer]}"
            else:
                wflow_var_v1 = f"input.static.{_STATICMAP_TO_V1_VAR_SEDIMENT[layer]}"
            config_options[wflow_var_v1] = layer

    # Start with the reservoir layers
    if has_reservoirs:
        for layer in RESERVOIR_LAYERS_SEDIMENT:
            wflow_var = _STATICMAP_TO_V0_VAR_SEDIMENT[layer]
            # get the map name from config
            map_name = get_config(key=f"input.{wflow_var}", config=config)
            # add to the variables to remove
            variables_to_remove.append(map_name)
            if map_name in grid:
                ds_res[layer] = grid[map_name].copy()
            else:
                logger.warning(f"Reservoir layer {map_name} not found in the grid.")

    # Move to the lake layers
    if has_lakes:
        ds_lakes = xr.Dataset()

        lake_layers = [
            "lake_area_id",
            "lake_outlet_id",
            "lake_area",
            "reservoir_trapping_efficiency",
        ]
        for layer in lake_layers:
            if layer == "reservoir_trapping_efficiency":
                # This layer is not in the config, so we skip it
                continue
            wflow_var = _STATICMAP_TO_V0_VAR_SEDIMENT[layer]
            # get the map name from config
            map_name = get_config(key=f"input.{wflow_var}", config=config)
            # add to the variables to remove
            variables_to_remove.append(map_name)
            if map_name in grid:
                # Replace lake by reservoir in the output layer name
                layer_out = layer.replace("lake", "reservoir")
                ds_lakes[layer_out] = grid[map_name].copy()

        # Add trapping efficiency
        ds_lakes["reservoir_trapping_efficiency"] = ds_lakes[
            "reservoir_outlet_id"
        ].where(
            ds_lakes["reservoir_outlet_id"]
            == ds_lakes["reservoir_outlet_id"].raster.nodata,
            0.0,
        )

        # Merge lakes into the reservoirs dataset if needed
        if not has_reservoirs:
            ds_res = ds_lakes
        else:
            ds_merge = merge_reservoirs_sediment(
                ds_lakes,
                ds_res,
                duplicate_id="skip",
            )
            if ds_merge is None:
                logger.warning(
                    "Merging lakes into reservoirs failed because of duplicated IDs. "
                    "Only reservoirs will be converted and you may need to update the "
                    "models to add the missing lakes manually."
                )
            else:
                ds_res = ds_merge

    return ds_res, variables_to_remove, config_options


def _convert_tables_v0_to_v1(tables: WflowTablesComponent) -> None:
    logger.info("Renaming lake_*.csv files to reservoir_*.csv files.")
    for key in [t for t in tables.data.keys() if t.startswith("lake_")]:
        data_table = tables.data.pop(key)
        new_name = key.replace("lake_", "reservoir_")
        tables.set(data_table, name=new_name)
        logger.debug(f"Renamed table {key} to {new_name}.")


def _convert_sbm_config_v1_to_v1_1(config: dict) -> dict:
    """Convert the config of a Wflow v1.0 model into a Wflow v1.1 format for SBM."""
    config_out = copy.deepcopy(config)

    for v1, v1_1 in _V1_TO_V1_1_SBM.items():
        # input.static
        static = config_out.get("input", {}).get("static", {})
        if v1 in static:
            static[v1_1] = static.pop(v1)
        # state.variables
        state_vars = config_out.get("state", {}).get("variables", {})
        if v1 in state_vars:
            state_vars[v1_1] = state_vars.pop(v1)

    config_out["wflow_version"] = "1.1"
    return config_out


# Helper functions for version detection and validation for the upgrade functions.
def _is_v1_schema(config: dict) -> bool:
    """Detect the Wflow v1.0 config schema."""
    # v1 split sections created by converter
    if "logging" in config:
        logger.debug("Found logging section, indicating v1 schema.")
        return True

    # v1 input structure
    input_cfg = config.get("input", {})

    if isinstance(input_cfg.get("static"), dict):
        logger.debug("Found input.static section, indicating v1 schema.")
        return True

    # In v0 forcing/cyclic are lists. In v1 they are tables.
    if isinstance(input_cfg.get("forcing"), dict):
        logger.debug("Found input.forcing section, indicating v1 schema.")
        return True

    if isinstance(input_cfg.get("cyclic"), dict):
        logger.debug("Found input.cyclic section, indicating v1 schema.")
        return True

    # v1 model names
    model_cfg = config.get("model", {})
    v1_model_keys = {
        "cold_start__flag",
        "reservoir__flag",
        "snow__flag",
        "cell_length_in_meter__flag",
        "river_streamorder__min_count",
        "land_streamorder__min_count",
    }

    if v1_model_keys.intersection(model_cfg):
        logger.debug("Found v1 model options.")
        return True

    # v1 output structure
    output_cfg = config.get("output", {})

    if "netcdf_grid" in output_cfg or "netcdf_scalar" in output_cfg:
        logger.debug("Found v1 output structure.")
        return True

    return False


def _detect_version_from_config(config: dict) -> Version:
    """Detect the Wflow.jl version from a raw config dictionary.

    Parameters
    ----------
    config : dict
        The raw TOML config dictionary.

    Returns
    -------
    Version
        The detected Wflow.jl version.
    """
    version = get_config(config=config, key="wflow_version", fallback=None)
    if version is not None:
        logger.debug(f"Detected Wflow version {version} from config.")
        return Version(str(version))

    if _is_v1_schema(config):
        logger.debug(
            "No 'wflow_version' found in config but v1.0 schema detected. "
            "Assuming v1.0 model."
        )
        return Version("1.0")

    logger.warning(
        "No 'wflow_version' found in config and no v1.0-only keys detected. "
        "Assuming pre-v1.0 model with v0.8 schema. Please check the config and update "
        "the version if needed."
    )
    return Version("0.8")


def _validate_options(
    options: dict | None,
) -> dict[tuple[Version, Version], dict[str, dict]]:
    """Validate the options dictionary for the upgrade functions."""
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise TypeError(
            f"Expected 'options' to be a dict but got {type(options).__name__}."
        )

    validated = {}
    for version_string, opts in options.items():
        if not isinstance(opts, dict):
            raise TypeError(
                f"Expected 'options[{version_string}]' to be a dict "
                f"but got {type(opts).__name__}."
            )
        if not isinstance(version_string, str) or "_" not in version_string:
            raise TypeError(
                f"Expected 'options' keys to be strings of the format 'from_to' "
                f"but got {version_string}."
            )
        _from, _to = version_string.split("_")
        version_tuple = (Version(_from), Version(_to))
        if version_tuple not in _UPGRADES:
            raise ValueError(
                f"Unknown upgrade versions '{version_tuple}'. "
                f"Available versions: {_UPGRADES}."
            )
        validated[version_tuple] = opts
    return validated


def _get_model_class(model_type: str):
    """Import and return the model class for the given model type."""
    if model_type == "wflow_sbm":
        from hydromt_wflow.wflow_sbm import WflowSbmModel

        return WflowSbmModel
    elif model_type == "wflow_sediment":
        from hydromt_wflow.wflow_sediment import WflowSedimentModel

        return WflowSedimentModel
    raise ValueError(
        f"Unknown model type: {model_type!r}. Expected 'wflow_sbm' or 'wflow_sediment'."
    )


# All `_upgrade_**` functions use the `_convert_**` functions to perform the actual
# conversion, but they also handle reading/writing configs and accessing model
# components as needed.
def _upgrade_config_v0_to_v1(
    model_root: Path, config_v0: dict, model_type: str, config_filename: str
) -> None:
    """Convert a v0 config to v1 format and write the result to disk.

    This performs the config-only part of the v0tov1 upgrade without needing
    an initialized Model object.
    """
    if model_type == "wflow_sbm":
        config_out = _convert_sbm_config_v0_to_v1(config_v0)
    elif model_type == "wflow_sediment":
        config_out = _convert_sediment_config_v0_to_v1(config_v0)
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    # Load the v1 config template and apply converted values
    with open(DATA_DIR / "default_config_headers.toml", "rb") as file:
        new_config = tomllib.load(file)

    for key, value in config_out.items():
        set_config(new_config, key, value)

    # Write the upgraded config to disk
    config_path = model_root / config_filename
    write_toml(config_path, new_config)
    logger.info(f"Wrote upgraded v1.0 config to {config_path}.")


def _upgrade_config_v1_to_v1_1(
    model_root: Path, model_type: str, config_filename: str
) -> bool:
    """Upgrade a v1.0 config to v1.1 format and write the result to disk.

    Returns
    -------
    bool
        True when the staticmaps does not contain a known elevation layer and
        a ``setup_basemaps`` call is required to create one.
    """

    def _get_land_surface_elevation_name(config: dict, root: Path) -> str | None:
        configured_name = get_config(
            key="input.static.land_surface__elevation",
            config=config,
            fallback=None,
        )
        if configured_name is not None:
            return configured_name

        path_static = get_config(
            key="input.path_static",
            config=config,
            root=root,
            abs_path=True,
            fallback="staticmaps.nc",
        )
        if not path_static.exists():
            logger.warning(
                f"Could not find staticmaps at {path_static}; "
                "cannot infer 'input.static.land_surface__elevation'."
            )
            return None

        with xr.open_dataset(path_static) as staticmaps:
            for candidate in (
                "land_surface__elevation",
                "land_elevation",
                "wflow_dem",
            ):
                if candidate in staticmaps.data_vars:
                    return candidate
        return None

    config_path = model_root / config_filename
    config = read_toml(config_path)
    requires_setup_basemaps = False

    if model_type == "wflow_sbm":
        config_out = _convert_sbm_config_v1_to_v1_1(config)
    elif model_type == "wflow_sediment":
        config_out = copy.deepcopy(config)
        config_out["wflow_version"] = "1.1"
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    elevation_name = _get_land_surface_elevation_name(config_out, model_root)
    if elevation_name is not None:
        set_config(
            config=config_out,
            key="input.static.land_surface__elevation",
            value=elevation_name,
        )
    else:
        requires_setup_basemaps = True
        logger.warning(
            "No elevation layer found in staticmaps for "
            "'input.static.land_surface__elevation'. "
            "Run setup_basemaps during upgrade to create the layer."
        )

    write_toml(config_path, config_out)
    logger.info(f"Wrote upgraded v1.1 config to {config_path}.")
    return requires_setup_basemaps


def _upgrade_components_v0_to_v1_sbm(
    model: WflowSbmModel, config_v0: dict, **kwargs
) -> None:
    """Perform grid and table operations for the SBM v0tov1 upgrade.

    This requires an initialized Model (with valid v1 config) to access
    staticmaps and tables.
    """
    ds_res, vars_to_remove, config_opt = _convert_sbm_reservoirs_v0_to_v1(
        model.staticmaps.data, config_v0
    )
    if ds_res is not None:
        model.staticmaps.drop_vars(vars_to_remove)
        model.staticmaps.set(ds_res)
        for option in config_opt:
            model.config.set(option, config_opt[option])

    _convert_tables_v0_to_v1(model.tables)


def _upgrade_components_v0_to_v1_sediment(
    model: WflowSedimentModel,
    config_v0: dict,
    *,
    soil_fn: str = "soilgrids",
    usle_k_method: str = "renard",
    strord_name: str = "wflow_streamorder",
    **kwargs,
) -> None:
    """Perform grid and table operations for the sediment v0tov1 upgrade.

    This requires an initialized Model (with valid v1 config) to access
    staticmaps, tables, and run setup workflows.
    """
    # Rerun setup workflows that compute new parameters
    model.setup_soilmaps(
        soil_fn=soil_fn,
        usle_k_method=usle_k_method,
        add_aggregates=True,
    )
    model.setup_riverbedsed(bedsed_mapping_fn=None, strord_name=strord_name)

    # Merge lakes and reservoirs
    ds_res, vars_to_remove, config_opt = _convert_sediment_reservoirs_v0_to_v1(
        model.staticmaps.data, config_v0
    )
    if ds_res is not None:
        model.staticmaps.drop_vars(vars_to_remove)
        model.staticmaps.set(ds_res)
        for option in config_opt:
            model.config.set(option, config_opt[option])


# Main entry point
def upgrade_model(
    model_root: str | Path,
    model_type: str,
    config_filename: str | None = None,
    data_libs: list[str] | str | None = None,
    options: dict | None = None,
) -> None:
    """Upgrade a wflow model on disk to the latest Wflow.jl version.

    This function:

    - Reads the config file from disk to detect the version
    - Applies all necessary upgrade steps to the config file and writes it back to disk
    - Initializes a Model object with the upgraded config
    - Applies all necessary upgrade steps to the model components (staticmaps, tables)

    Parameters
    ----------
    model_root : str or Path
        Path to the model root directory.
    model_type : str
        Type of model: ``"wflow_sbm"`` or ``"wflow_sediment"``.
    config_filename : str, optional
        Config filename relative to model_root.
        Defaults to ``"{model_type}.toml"``.
    data_libs : list[str] or str, optional
        Data catalog configuration files. Required for sediment v0tov1 upgrades
        that need to re-run ``setup_soilmaps`` and ``setup_riverbedsed``.
    options : dict, optional
        Options passed to upgrade functions. Keys should be strings like
        ``"0.8_1.0"`` and values should be dicts of keyword arguments.
    """
    model_root = Path(model_root)
    if config_filename is None:
        config_filename = f"{model_type}.toml"

    config_path = model_root / config_filename
    config = read_toml(config_path)

    version = _detect_version_from_config(config)
    if version >= WFLOW_LATEST_VERSION:
        logger.info("Model is already at the latest version, no upgrade needed.")
        return

    validated_options = _validate_options(options)

    # Chain upgrade config on disk without a Model object (v0 to latest)
    config_v0 = None
    if version < Version("1.0"):
        config_v0 = config.copy()
        _upgrade_config_v0_to_v1(model_root, config, model_type, config_filename)
        logger.info("Upgrading config from v0.x to v1.0 format.")
        version = Version("1.0")
    if version < Version("1.1"):
        needs_setup_basemaps = _upgrade_config_v1_to_v1_1(
            model_root,
            model_type,
            config_filename,
        )
        if needs_setup_basemaps:
            opts_v1_to_v1_1 = validated_options.get(
                (Version("1.0"), Version("1.1")), {}
            )
            setup_basemaps_kwargs = opts_v1_to_v1_1.get("setup_basemaps")
            if setup_basemaps_kwargs is None:
                raise ValueError(
                    "Missing required elevation layer in staticmaps and no "
                    "'setup_basemaps' options were provided. "
                    "Pass options={'1.0_1.1': {'setup_basemaps': {...}}} "
                    "to run setup_basemaps during upgrade."
                )
            ModelClass = _get_model_class(model_type)
            model = ModelClass(
                str(model_root),
                config_filename=config_filename,
                mode="r+",
                data_libs=data_libs,
            )
            model.setup_basemaps(**setup_basemaps_kwargs)
            model.write()
        version = Version("1.1")

    # Upgrade component data using previous and updated config (needs initialized Model)
    if config_v0 is not None:
        ModelClass = _get_model_class(model_type)
        model = ModelClass(
            str(model_root),
            config_filename=config_filename,
            mode="r+",
            data_libs=data_libs,
        )
        v0_opts = validated_options.get((Version("0.8"), Version("1.0")), {})
        if model_type == "wflow_sbm":
            _upgrade_components_v0_to_v1_sbm(model, config_v0, **v0_opts)
        elif model_type == "wflow_sediment":
            _upgrade_components_v0_to_v1_sediment(model, config_v0, **v0_opts)
        model.write()

    logger.info(f"Model upgraded to Wflow.jl v{version}. {model_root=}")
