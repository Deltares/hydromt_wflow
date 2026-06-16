"""Some utilities to upgrade Wflow model versions."""

from __future__ import annotations

import copy
import logging
import tomllib
from typing import TYPE_CHECKING, Protocol

import xarray as xr
from packaging.version import Version

from hydromt_wflow.components.tables import WflowTablesComponent
from hydromt_wflow.naming import (
    WFLOW_NAMES,
    WFLOW_SEDIMENT_NAMES,
    WFLOW_SEDIMENT_STATES_NAMES,
    WFLOW_STATES_NAMES,
)
from hydromt_wflow.utils import DATADIR, get_config, set_config
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

# The latest Wflow.jl version supported by this hydromt_wflow release
WFLOW_LATEST_VERSION = Version("1.1")

__all__ = [
    "convert_to_wflow_v1_sbm",
    "convert_to_wflow_v1_sediment",
]


def _solve_var_name(var: str | dict, path: str, add: list):
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
    """
    if not isinstance(var, dict):
        sep = "." if path else ""
        add_str = ".".join(add) if add else ""
        yield (var, path + sep + add_str)
        return
    for key, item in var.items():
        yield from _solve_var_name(item, path, add + [key])


def _create_v0_to_v1_var_mapping(wflow_vars: dict) -> dict:
    output_dict = {}
    for value in wflow_vars.values():
        if isinstance(value["wflow_v0"], list):
            for var in value["wflow_v0"]:
                output_dict[var] = value["wflow_v1"]
        else:
            output_dict[value["wflow_v0"]] = value["wflow_v1"]
    return output_dict


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


def _convert_to_wflow_v1(
    config: dict,
    wflow_vars: dict,
    states_vars: dict,
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
    wflow_vars: dict
        The Wflow variables dict to use for the conversion between versions.
        Either WFLOW_NAMES or WFLOW_SEDIMENT_NAMES.
    states_vars: dict
        The Wflow states variables dict to use for the conversion between versions.
        Either WFLOW_STATES_NAMES or WFLOW_SEDIMENT_STATES_NAMES.
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
    WFLOW_CONVERSION = _create_v0_to_v1_var_mapping(wflow_vars)
    for k, v in states_vars.items():
        WFLOW_CONVERSION[v["wflow_v0"]] = v["wflow_v1"]
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
    for key, variables in states_vars.items():
        name = get_config(
            key=f"state.{variables['wflow_v0']}", config=config, fallback=None
        )
        if name is not None and variables["wflow_v1"] is not None:
            config_out["state"]["variables"][variables["wflow_v1"]] = name

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
    for key, variables in wflow_vars.items():
        if isinstance(variables["wflow_v0"], list):
            for wflow_v0_var in variables["wflow_v0"]:
                config_out = _set_input_vars(
                    wflow_v0_var=wflow_v0_var,
                    wflow_v1_var=variables["wflow_v1"],
                    config_in=config,
                    config_out=config_out,
                    input_options=input_options,
                    input_variables=input_variables,
                    forcing_variables=forcing_variables,
                    cyclic_variables=cyclic_variables,
                )
        else:
            config_out = _set_input_vars(
                wflow_v0_var=variables["wflow_v0"],
                wflow_v1_var=variables["wflow_v1"],
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


def convert_to_wflow_v1_sbm(config: dict) -> dict:
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
    additional_variables = {
        "vertical.interception": "vegetation_canopy_water__interception_volume_flux",
        "vertical.actevap": "land_surface__evapotranspiration_volume_flux",
        "vertical.actinfilt": "soil_water__infiltration_volume_flux",
        "vertical.excesswatersoil": "compacted_soil_surface_water__excess_volume_flux",
        "vertical.excesswaterpath": "non_compacted_soil_surface_water__excess_volume_flux",  # noqa : E501
        "vertical.exfiltustore": "soil_surface_water_unsaturated_zone__exfiltration_volume_flux",  # noqa : E501
        "vertical.exfiltsatwater": "land.soil.variables.exfiltsatwater",
        "vertical.recharge": "soil_water_saturated_zone_top__net_recharge_volume_flux",
        "vertical.vwc_percroot": "soil_water_root_zone__volume_percentage",
        "lateral.land.q_av": "land_surface_water__volume_flow_rate",
        "lateral.land.h_av": "land_surface_water__depth",
        "lateral.land.to_river": "land_surface_water__to_river_volume_flow_rate",
        "lateral.subsurface.to_river": "subsurface_water__to_river_volume_flow_rate",
        "lateral.subsurface.drain.flux": "land_drain_water__to_subsurface_volume_flow_rate",  # noqa : E501
        "lateral.subsurface.flow.aquifer.head": "subsurface_water__hydraulic_head",
        "lateral.subsurface.river.flux": "river_water__to_subsurface_volume_flow_rate",
        "lateral.subsurface.recharge.rate": "subsurface_water_saturated_zone_top__net_recharge_volume_flow_rate",  # noqa : E501
        "lateral.river.q_av": "river_water__volume_flow_rate",
        "lateral.river.h_av": "river_water__depth",
        "lateral.river.volume": "river_water__volume",
        "lateral.river.inwater": "river_water__lateral_inflow_volume_flow_rate",
        "lateral.river.floodplain.volume": "floodplain_water__volume",
        "lateral.river.reservoir.volume": "reservoir_water__volume",
        "lateral.river.reservoir.totaloutflow": "reservoir_water__outgoing_volume_flow_rate",  # noqa : E501
        "lateral.river.reservoir.outflow": "reservoir_water__outgoing_volume_flow_rate",
        "lateral.river.reservoir.inflow": "reservoir_water__incoming_volume_flow_rate",
        "lateral.river.reservoir.precipitation": "reservoir_water__precipitation_volume_flux",  # noqa : E501
        "lateral.river.reservoir.evaporation": "reservoir_water__potential_evaporation_volume_flux",  # noqa : E501
        "lateral.river.reservoir.actevap": "reservoir_water__evaporation_volume_flux",
        "lateral.river.lake.storage": "reservoir_water__volume",
        "lateral.river.lake.totaloutflow": "reservoir_water__outgoing_volume_flow_rate",
        "lateral.river.lake.outflow": "reservoir_water__outgoing_volume_flow_rate",
        "lateral.river.lake.inflow": "reservoir_water__incoming_volume_flow_rate",
        "lateral.river.lake.waterlevel": "reservoir_water_surface__elevation",
        "lateral.river.lake.precipitation": "reservoir_water__precipitation_volume_flux",  # noqa : E501
        "lateral.river.lake.evaporation": "reservoir_water__potential_evaporation_volume_flux",  # noqa : E501
        "lateral.river.lake.actevap": "reservoir_water__evaporation_volume_flux",
    }

    # Options in model section that were renamed
    model_options = {
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

    # Options in input section that were renamed
    input_options = {
        "ldd": "basin__local_drain_direction",
        "river_location": "river_location__mask",
        "subcatchment": "subbasin_location__count",
    }

    # variables that were moved to input rather than input.static
    input_variables = [
        "lateral.river.lake.areas",
        "lateral.river.lake.locs",
        "lateral.river.lake.linkedlakelocs",
        "lateral.river.reservoir.areas",
        "lateral.river.reservoir.locs",
    ]

    # Wflow entries that cross main headers (i.e. [input, state, model, output])
    cross_options = {
        "input.lateral.subsurface.conductivity_profile": "model.conductivity_profile",
        "model.riverlength_bc": "input.static.model_boundary_condition_river__length",
    }

    config_out = _convert_to_wflow_v1(
        config=config,
        wflow_vars=WFLOW_NAMES,
        states_vars=WFLOW_STATES_NAMES,
        model_options=model_options,
        cross_options=cross_options,
        input_options=input_options,
        input_variables=input_variables,
        additional_variables=additional_variables,
    )

    return config_out


def convert_to_wflow_v1_sediment(config: dict) -> dict:
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
    additional_variables = {
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

    # Options in model section that were renamed
    model_options = {
        "reinit": "cold_start__flag",
        "sizeinmetres": "cell_length_in_meter__flag",
        "runrivermodel": "run_river_model__flag",
        "doreservoir": "reservoir__flag",
        "dolake": "reservoir__flag",
        "rainerosmethod": "rainfall_erosion",
        "landtransportmethod": "land_transport",
        "rivtransportmethod": "river_transport",
    }

    # Options in input section that were renamed
    input_options = {
        "ldd": "basin__local_drain_direction",
        "river_location": "river_location__mask",
        "subcatchment": "subbasin_location__count",
    }

    # variables that were moved to input rather than input.static
    input_variables = [
        "vertical.lakeareas",
        "lateral.river.lakelocs",
        "vertical.resareas",
        "lateral.river.reslocs",
    ]

    config_out = _convert_to_wflow_v1(
        config=config,
        wflow_vars=WFLOW_SEDIMENT_NAMES,
        states_vars=WFLOW_SEDIMENT_STATES_NAMES,
        model_options=model_options,
        input_options=input_options,
        input_variables=input_variables,
        additional_variables=additional_variables,
    )

    return config_out


def convert_reservoirs_to_wflow_v1_sbm(
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
            wflow_var = WFLOW_NAMES[layer]["wflow_v0"]
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
                wflow_var_v1 = f"input.{WFLOW_NAMES[layer]['wflow_v1']}"
            else:
                # Find if in cyclic / forcing / static
                v0_var = WFLOW_NAMES[layer]["wflow_v0"]
                if v0_var in get_config(config=config, key="input.cyclic", fallback=[]):
                    wflow_var_v1 = f"input.cyclic.{WFLOW_NAMES[layer]['wflow_v1']}"
                elif v0_var in get_config(
                    config=config, key="input.forcing", fallback=[]
                ):
                    wflow_var_v1 = f"input.forcing.{WFLOW_NAMES[layer]['wflow_v1']}"
                else:
                    wflow_var_v1 = f"input.static.{WFLOW_NAMES[layer]['wflow_v1']}"
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
            wflow_var = WFLOW_NAMES[layer]["wflow_v0"]
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
                wflow_var_v1 = f"input.{WFLOW_NAMES[layer_out]['wflow_v1']}"
            else:
                wflow_var_v1 = f"input.static.{WFLOW_NAMES[layer_out]['wflow_v1']}"
            config_options[wflow_var_v1] = layer_out

    # Ensure correct data types for rating curve
    ds_res = set_rating_curve_layer_data_type(ds_res)
    return ds_res, variables_to_remove, config_options


def convert_reservoirs_to_wflow_v1_sediment(
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
                wflow_var_v1 = f"input.{WFLOW_SEDIMENT_NAMES[layer]['wflow_v1']}"
            else:
                wflow_var_v1 = f"input.static.{WFLOW_SEDIMENT_NAMES[layer]['wflow_v1']}"
            config_options[wflow_var_v1] = layer

    # Start with the reservoir layers
    if has_reservoirs:
        for layer in RESERVOIR_LAYERS_SEDIMENT:
            wflow_var = WFLOW_SEDIMENT_NAMES[layer]["wflow_v0"]
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
            wflow_var = WFLOW_SEDIMENT_NAMES[layer]["wflow_v0"]
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


def upgrade_lake_tables_to_reservoir_tables_v1(tables: WflowTablesComponent) -> None:
    logger.info("Renaming lake_*.csv files to reservoir_*.csv files.")
    for key in [t for t in tables.data.keys() if t.startswith("lake_")]:
        data_table = tables.data.pop(key)
        new_name = key.replace("lake_", "reservoir_")
        tables.set(data_table, name=new_name)
        logger.debug(f"Renamed table {key} to {new_name}.")


def convert_to_wflow_v1_1_sbm(config: dict) -> dict:
    """Convert the config of a Wflow v1.0 model into a Wflow v1.1 format for SBM."""
    config_out = copy.deepcopy(config)

    all_vars = {**WFLOW_NAMES, **WFLOW_STATES_NAMES}
    for variables in all_vars.values():
        v1 = variables.get("wflow_v1")
        v1_1 = variables.get("wflow_v1.1")
        if v1 is None or v1_1 is None:
            continue
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


def _upgrade_sbm_v0_to_v1(model: WflowSbmModel, **kwargs) -> None:
    """
    Upgrade the model from v0 to wflow v1.0 format.

    The function reads a TOML from wflow v0x and converts it to wflow v1.0 format.
    The other components stay the same.

    Lakes and reservoirs have also been merged into one structure and parameters in
    the resulted staticmaps will be combined.

    This function should be followed by write_config() to write the upgraded file.
    """
    if _detect_wflow_version(model) >= Version("1.0"):
        logger.info("Config is already at v1.0 or later, no upgrade needed.")
        return

    logger.info("Upgrading config from v0.x to v1.0 format.")
    config_v0 = model.config.data.copy()
    config_out = convert_to_wflow_v1_sbm(model.config.data)

    # Update the config
    with open(model._DATADIR / "default_config_headers.toml", "rb") as file:
        model.config._data = tomllib.load(file)
    for option in config_out:
        model.config.set(option, config_out[option])

    # Merge lakes and reservoirs layers
    ds_res, vars_to_remove, config_opt = convert_reservoirs_to_wflow_v1_sbm(
        model.staticmaps.data, config_v0
    )
    if ds_res is not None:
        # Remove older maps from grid
        model.staticmaps.drop_vars(vars_to_remove)
        # Add new reservoir maps to grid
        model.staticmaps.set(ds_res)
        # Update the config with the new names
        for option in config_opt:
            model.config.set(option, config_opt[option])
    # also update tables
    upgrade_lake_tables_to_reservoir_tables_v1(model.tables)


def _upgrade_sbm_v1_to_v1_1(model: WflowSbmModel, **kwargs) -> None:
    """Upgrade the model config from Wflow v1.0 to v1.1 format.

    Currently, only the TOML config is updated.
    All other model components remain unchanged.

    This function should be followed by write_config() to write the upgraded file.
    """
    version = _detect_wflow_version(model)

    if version < Version("1.0"):
        raise ValueError(
            f"Expected a v1.0 model but got v{version}. Run the v0 to v1 upgrade first."
        )
    elif version >= Version("1.1"):
        logger.info(f"Config is already at v{version}, no upgrade needed.")
        return

    logger.info("Upgrading config from v1.0 to v1.1 format.")
    model.config._data = convert_to_wflow_v1_1_sbm(model.config.data)


def _upgrade_sediment_v0_to_v1(model: WflowSedimentModel, **kwargs):
    """
    Upgrade the model from v0x to wflow v1.0 format.

    The function reads a TOML from wflow v0x and converts it to wflow v1x format.
    The other components stay the same.

    A few variables that used to be computed within Wflow.jl are now moved to
    HydroMT to allow more flexibility for the users to update if they do get local
    data or calibrate some of the parameters specifically. For this, the
    ``setup_soilmaps`` and ``setup_riverbedsed`` functions are called again.

    Lakes and reservoirs have also been merged into one structure and parameters in
    the resulted staticmaps will be combined.

    This function should be followed by ``write_config`` to write the upgraded TOML
    file and by ``write_grid`` to write the upgraded static netcdf input file.

    Parameters
    ----------
    model : WflowSedimentModel
        Model instance to upgrade.
    **kwargs
        Optional arguments passed to the setup workflows.

        soil_fn : str, default "soilgrids"
            Value passed to ``setup_soilmaps``.
        usle_k_method : str, default "renard"
            Value passed to ``setup_soilmaps``.
        strord_name : str, default "wflow_streamorder"
            Value passed to ``setup_riverbedsed``.
    """
    config_v0 = model.config.data.copy()
    config_out = convert_to_wflow_v1_sediment(model.config.data)

    soil_fn: str = kwargs.get("soil_fn", "soilgrids")
    usle_k_method: str = kwargs.get("usle_k_method", "renard")
    strord_name: str = kwargs.get("strord_name", "wflow_streamorder")

    # Update the config
    with open(DATADIR / "default_config_headers.toml", "rb") as file:
        model.config._data = tomllib.load(file)
    for option in config_out:
        model.config.set(option, config_out[option])

    # Rerun setup_soilmaps
    model.setup_soilmaps(
        soil_fn=soil_fn,
        usle_k_method=usle_k_method,
        add_aggregates=True,
    )

    # Rerun setup_riverbedsed
    model.setup_riverbedsed(bedsed_mapping_fn=None, strord_name=strord_name)

    # Merge lakes and reservoirs layers
    ds_res, vars_to_remove, config_opt = convert_reservoirs_to_wflow_v1_sediment(
        model.staticmaps.data, config_v0
    )
    if ds_res is not None:
        # Remove older maps from grid
        model.staticmaps.drop_vars(vars_to_remove)
        # Add new reservoir maps to grid
        model.staticmaps.set(ds_res)
        # Update the config with the new names
        for option in config_opt:
            model.config.set(option, config_opt[option])


def _upgrade_sediment_v1_to_v1_1(model: WflowSedimentModel, **kwargs):
    """Upgrade the model from wflow v1.0 to v1.1 format."""
    model.config.set("wflow_version", "1.1")


class UpgradeFunction(Protocol):
    def __call__(
        self, model: "WflowSedimentModel | WflowSbmModel", **kwargs
    ) -> None: ...


_UPGRADES: dict[str, dict[tuple[Version, Version], UpgradeFunction]] = {
    "wflow_sbm": {
        (Version("0.8"), Version("1.0")): _upgrade_sbm_v0_to_v1,
        (Version("1.0"), Version("1.1")): _upgrade_sbm_v1_to_v1_1,
    },
    "wflow_sediment": {
        (Version("0.8"), Version("1.0")): _upgrade_sediment_v0_to_v1,
        (Version("1.0"), Version("1.1")): _upgrade_sediment_v1_to_v1_1,
    },
}


def _is_v1_schema(config: dict) -> bool:
    """Detect the Wflow v1.0 config schema."""
    # v1.1+
    if "wflow_version" in config:
        logger.debug("Found wflow_version, indicating v1 schema.")
        return True

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


def _detect_wflow_version(model: WflowSbmModel | WflowSedimentModel) -> Version:
    """Detect the Wflow.jl version of the model."""
    version = model.config.get_value("wflow_version")
    if version is not None:
        # models built from v1.1 an onwards will always have this version in the config.
        logger.debug(f"Detected Wflow version {version} from config.")
        return Version(str(version))

    # v1.0-only config key
    if _is_v1_schema(model.config.data):
        logger.debug(
            "No 'wflow_version' found in config but v1.0 schema detected. "
            "Assuming v1.0 model."
        )
        return Version("1.0")

    logger.warning(
        "No 'wflow_version' found in config and no v1.0-only keys detected. "
        "Assuming pre-v1.0 model."
    )
    return Version("0.0")


def _validate_options(options: dict | None, model_name: str) -> dict:
    """Validate the options dictionary for the upgrade functions."""
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise TypeError(
            f"Expected 'options' to be a dict but got {type(options).__name__}."
        )
    for version_tuple, opts in options.items():
        if not isinstance(opts, dict):
            raise TypeError(
                f"Expected 'options[{version_tuple}]' to be a dict "
                "but got {type(opts).__name__}."
            )
        if version_tuple not in _UPGRADES[model_name]:
            raise ValueError(
                f"Unknown upgrade versions '{version_tuple}' for model '{model_name}'. "
                f"Available versions: {list(_UPGRADES[model_name].keys())}."
            )
    return options


def upgrade_to_latest(
    model: WflowSbmModel | WflowSedimentModel, options: dict | None = None
) -> None:
    """Upgrade the model to the latest Wflow.jl version.

    Applies all necessary upgrade steps in order based on the ``wflow_version``
    key in the config. If absent, the model is assumed to be pre-v1.0 and all
    upgrade steps are applied.

    This function should be followed by write() to write all upgraded components
    to disk.

    Parameters
    ----------
    model: WflowSbmModel | WflowSedimentModel
        The model to upgrade.
    options: dict, optional
        A dictionary of options to pass to the upgrade functions. The keys should be
        tuples of two version numbers associated with this upgrade (e.g. ("0.x", "1.0")
        or ("1.0", "1.1")) and the values should be dictionaries of options for that
        version.
    """
    version = _detect_wflow_version(model)

    if version >= WFLOW_LATEST_VERSION:
        logger.info("Model is already at the latest version, no upgrade needed.")
        return

    options = _validate_options(options, model.name)
    for (from_version, to_version), upgrade_func in _UPGRADES[model.name].items():
        if version < to_version:
            logger.debug(f"Upgrading model from v{from_version} to v{to_version}.")
            upgrade_func(model, **options.get((from_version, to_version), {}))
            version = Version(model.config.get_value("wflow_version"))

    logger.info(f"Model upgraded to Wflow.jl v{version}.")
