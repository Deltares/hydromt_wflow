"""Implement Wflow model class."""

# Implement model class following model API
import logging
import os
import tomllib
from os.path import isfile, join
from pathlib import Path
from typing import Any

# Implement model class following model API
import geopandas as gpd
import hydromt
import numpy as np
import pandas as pd
import pyflwdir
import xarray as xr
from hydromt import hydromt_step
from hydromt.error import NoDataStrategy
from hydromt.gis import flw

import hydromt_wflow.utils as utils
from hydromt_wflow import workflows
from hydromt_wflow.naming import _create_hydromt_wflow_mapping_sbm
from hydromt_wflow.version_upgrade import (
    convert_reservoirs_to_wflow_v1_sbm,
    convert_to_wflow_v1_sbm,
    upgrade_lake_tables_to_reservoir_tables_v1,
)
from hydromt_wflow.wflow_base import WflowBaseModel

__all__ = ["WflowSbmModel"]
__hydromt_eps__ = ["WflowSbmModel"]  # core entrypoints
logger = logging.getLogger(f"hydromt.{__name__}")


class WflowSbmModel(WflowBaseModel):
    """Read or Write a wflow model.

    Parameters
    ----------
    root : str, optional
        Model root, by default None (current working directory)
    config_filename : str, optional
        A path relative to the root where the configuration file will
        be read and written if user does not provide a path themselves.
        By default "wflow_sbm.toml"
    mode : {'r','r+','w'}, optional
        read/append/write mode, by default "w"
    data_libs : list[str] | str, optional
        List of data catalog configuration files, by default None
    **catalog_keys:
        Additional keyword arguments to be passed down to the DataCatalog.
    """

    name: str = "wflow_sbm"

    def __init__(
        self,
        root: str | None = None,
        config_filename: str | None = None,
        mode: str = "w",
        data_libs: list[str] | str | None = None,
        **catalog_keys,
    ):
        super().__init__(
            root,
            config_filename=config_filename,
            mode=mode,
            data_libs=data_libs,
            **catalog_keys,
        )

        # hydromt mapping and wflow variable names
        self._MAPS, self._WFLOW_NAMES = _create_hydromt_wflow_mapping_sbm(
            self.config.data
        )

    ## SETUP METHODS
    @hydromt_step
    def setup_rivers(
        self,
        hydrography_fn: str | xr.Dataset,
        river_geom_fn: str | gpd.GeoDataFrame | None = None,
        river_upa: float = 30,
        rivdph_method: str = "powlaw",
        slope_len: float = 2e3,
        min_rivlen_ratio: float = 0.0,
        min_rivdph: float = 1,
        min_rivwth: float = 30,
        smooth_len: float = 5e3,
        elevtn_map: str = "land_elevation",
        river_routing: str = "kinematic_wave",
        connectivity: int = 8,
        output_names: dict = {
            "river_location__mask": "river_mask",
            "river__length": "river_length",
            "river__width": "river_width",
            "river_bank_water__depth": "river_depth",
            "river__slope": "river_slope",
            "river_bank_water__elevation": "river_bank_elevation",
        },
    ):
        """
        Set full river parameter maps including river depth and bank elevation.

        The river mask is defined by all cells with a minimum upstream area threshold
        ``river_upa`` [km2].

        The river length is defined as the distance from the subgrid outlet pixel to
        the next upstream subgrid outlet pixel. The ``min_rivlen_ratio`` is the minimum
        global river length to avg. cell resolution ratio and is used as a threshold in
        window based smoothing of river length.

        The river slope is derived from the subgrid elevation difference between pixels
        at a half distance ``slope_len`` [m] up-
        and downstream from the subgrid outlet pixel.

        The river width is derived from the nearest river segment in ``river_geom_fn``.
        Data gaps are filled by the nearest valid upstream value and averaged along
        the flow directions over a length ``smooth_len`` [m]

        The river depth can be directly derived from ``river_geom_fn`` property or
        calculated using the ``rivdph_method``, by default powlaw:
        h = hc*Qbf**hp, which is based on qbankfull discharge from the nearest river
        segment in ``river_geom_fn`` and takes optional arguments for the hc
        (default = 0.27) and hp (default = 0.30) parameters. For other methods see
        :py:meth:`hydromt.workflows.river_depth`.

        If ``river_routing`` is set to "local_inertial", the bankfull elevation map
        can be conditioned based on the average cell elevation ("land_elevation")
        or subgrid outlet pixel elevation ("meta_subgrid_elevation").
        The subgrid elevation might provide a better representation
        of the river elevation profile, however in combination with
        local_inertial land routing (see :py:meth:`setup_floodplains`)
        the subgrid elevation will likely overestimate the floodplain storage capacity.
        Note that the same input elevation map should be used for river bankfull
        elevation and land elevation when using local_inertial land routing.

        Adds model layers:

        * **wflow_river** map: river mask [-]
        * **river_length** map: river length [m]
        * **river_width** map: river width [m]
        * **river_depth** map: bankfull river depth [m]
        * **river_slope** map: river slope [m/m]
        * **rivers** geom: river vector based on wflow_river mask
        * **river_bank_elevation** map: hydrologically conditioned elevation [m+REF]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        hydrography_fn : str, Path, xarray.Dataset
            Name of RasterDataset source for hydrography data. Must be same as
            setup_basemaps for consistent results.

            * **Required variables**: 'flwdir' [LLD or D8 or NEXTXY], 'uparea' [km2],
              'elevtn' [m+REF]

            * **Optional variables**: 'rivwth' [m], 'qbankfull' [m3/s]

        river_geom_fn : str, Path, geopandas.GeoDataFrame, optional
            Name of GeoDataFrame source for river data.

            * **Required variables**: 'rivwth' [m], 'rivdph' [m] or 'qbankfull' [m3/s]

        river_upa : float, optional
            Minimum upstream area threshold for the river map [km2]. By default 30.0
        slope_len : float, optional
            Length over which the river slope is calculated [km]. By default 2.0
        min_rivlen_ratio: float, optional
            Ratio of cell resolution used minimum length threshold in a moving
            window based smoothing of river length, by default 0.0
            The river length smoothing is skipped if `min_riverlen_ratio` = 0.
            For details about the river length smoothing,
            see :py:meth:`pyflwdir.FlwdirRaster.smooth_rivlen`
        rivdph_method : {'gvf', 'manning', 'powlaw'}
            see :py:meth:`hydromt.workflows.river_depth` for details, by default
            "powlaw"
        river_routing : {'kinematic_wave', 'local_inertial'}
            Routing methodology to be used, by default "kinematic_wave".
        smooth_len : float, optional
            Length [m] over which to smooth the output river width and depth,
            by default 5e3
        min_rivdph : float, optional
            Minimum river depth [m], by default 1.0
        min_rivwth : float, optional
            Minimum river width [m], by default 30.0
        elevtn_map : str, optional
            Name of the elevation map in the current WflowBaseModel.staticmaps.
            By default "land_elevation"
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.

        See Also
        --------
        workflows.river_bathymetry
        hydromt.workflows.river_depth
        pyflwdir.FlwdirRaster.river_depth
        setup_floodplains
        """
        super().setup_rivers(
            hydrography_fn=hydrography_fn,
            river_geom_fn=river_geom_fn,
            river_upa=river_upa,
            slope_len=slope_len,
            min_rivlen_ratio=min_rivlen_ratio,
            smooth_len=smooth_len,
            min_rivwth=min_rivwth,
            rivdph_method=rivdph_method,
            min_rivdph=min_rivdph,
            output_names=output_names,
        )

        routing_options = ["kinematic_wave", "local_inertial"]
        if river_routing not in routing_options:
            raise ValueError(
                f'river_routing="{river_routing}" unknown. '
                f"Select from {routing_options}."
            )

        logger.info(f'Update wflow config model.river_routing="{river_routing}"')
        self.set_config("model.river_routing", river_routing)

        if river_routing == "local_inertial":
            postfix = {
                "land_elevation": "_avg",
                "meta_subgrid_elevation": "_subgrid",
            }.get(elevtn_map, "")
            name = f"river_bank_elevation{postfix}"

            hydrodem_var = self._WFLOW_NAMES.get(self._MAPS["hydrodem"])
            if hydrodem_var in output_names:
                name = output_names[hydrodem_var]
            self._update_naming({hydrodem_var: name})

            ds_out = flw.dem_adjust(
                da_flwdir=self.staticmaps.data[self._MAPS["flwdir"]],
                da_elevtn=self.staticmaps.data[elevtn_map],
                da_rivmsk=self.staticmaps.data[self._MAPS["rivmsk"]],
                flwdir=self.flwdir,
                connectivity=connectivity,
                river_d8=True,
            ).rename(name)
            self.set_grid(ds_out)
            self._update_config_variable_name(name)

    @hydromt_step
    def setup_river_roughness(
        self,
        rivman_mapping_fn: str | Path | pd.DataFrame | None = None,
        strord_name: str = "meta_streamorder",
        output_name: str = "river_manning_n",
    ):
        """Set river Manning roughness coefficient for SBM.

        Adds model layers:
        - **river_manning_n** map: river Manning roughness coefficient [-]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`
        * :py:meth:`~WflowSbmModel.setup_rivers`

        Parameters
        ----------
        rivman_mapping_fn : str | Path | pd.DataFrame
            Path to the river Manning mapping file or a DataFrame with the mapping.
        strord_name : str
            Name of the stream order map.
        output_names : dict
            Mapping of output variable names.

        See Also
        --------
        WflowSbmModel.setup_basemaps
        WflowSbmModel.setup_rivers
        """
        logger.info("Preparing river Manning roughness.")

        # then add Manning roughness mapping
        if self._MAPS["strord"] not in self.staticmaps.data:
            if strord_name not in self.staticmaps.data:
                raise ValueError(
                    f"Streamorder map {strord_name} not found in grid. "
                    "Please run setup_basemaps or update the strord_name argument."
                )
            else:
                self._MAPS["strord"] = strord_name
        # update self._MAPS and self._WFLOW_NAMES with user defined output names
        self._update_naming({"river_water_flow__manning_n_parameter": output_name})

        # Make river_manning_n map from csv file with mapping with streamorder
        if rivman_mapping_fn is None:
            fn_map = "roughness_river_mapping_default"
        else:
            fn_map = rivman_mapping_fn
        df = self.data_catalog.get_dataframe(fn_map)

        strord = self.staticmaps.data[self._MAPS["strord"]].copy()
        # max streamorder value above which values get the same river_manning_n value
        max_str = df.index[-2]
        nodata = df.index[-1]
        # if streamorder value larger than max_str, assign last value
        strord = strord.where(strord <= max_str, max_str)
        # handle missing value (last row of csv is mapping of missing values)
        strord = strord.where(strord != strord.raster.nodata, nodata)
        strord.raster.set_nodata(nodata)
        ds_nriver = workflows.landuse(
            da=strord,
            ds_like=self.staticmaps.data,
            df=df,
        )
        rmdict = {k: self._MAPS.get(k, k) for k in ds_nriver.data_vars}
        self.set_grid(ds_nriver.rename(rmdict))
        # update config
        self._update_config_variable_name(ds_nriver.rename(rmdict).data_vars)

    @hydromt_step
    def setup_floodplains(
        self,
        hydrography_fn: str | xr.Dataset,
        floodplain_type: str,
        ### Options for 1D floodplains
        river_upa: float | None = None,
        flood_depths: list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        ### Options for 2D floodplains
        elevtn_map: str = "land_elevation",
        connectivity: int = 4,
        output_names: dict = {
            "floodplain_water__sum_of_volume_per_depth": "floodplain_volume",
            "river_bank_elevation": "river_bank_elevation_avg_D4",
        },
    ):
        """
        Add floodplain information to the model schematisation.

        The user can define what type of floodplains are required (1D or 2D),
        through the ``floodplain_type`` argument.

        If ``floodplain_type`` is set to "1d", a floodplain profile is derived for every
        river cell. It adds a map with floodplain volume per flood depth,
        which is used in the wflow 1D floodplain schematisation.

        Note, it is important to use the same river uparea value as used in the
        :py:meth:`setup_rivers` method.

        If ``floodplain_type`` is set to "2d", this component adds
        a hydrologically conditioned elevation (river_bank_elevation) map for
        land routing (local_inertial). For this options, landcells need to be
        conditioned to D4 flow directions otherwise pits may remain in the land cells.

        The conditioned elevation can be based on the average cell elevation
        ("land_elevation") or subgrid outlet pixel elevation ("meta_subgrid_elevation").
        Note that the subgrid elevation will likely overestimate
        the floodplain storage capacity.

        Additionally, note that the same input elevation map should be used for river
        bankfull elevation and land elevation when using local_inertial land routing.

        Requires :py:meth:`setup_rivers` to be executed beforehand
        (with ``river_routing`` set to "local_inertial").

        Adds model layers:

        * **floodplain_volume** map: map with floodplain volumes, has flood depth as \
            third dimension [m3] (for 1D floodplains)
        * **river_bank_elevation** map: hydrologically conditioned elevation [m+REF]
          (for 2D floodplains)

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_rivers`

        Parameters
        ----------
        floodplain_type: {"1d", "2d"}
            Option defining the type of floodplains, see below what arguments
            are related to the different floodplain types
        hydrography_fn : str, Path, xarray.Dataset
            Name of RasterDataset source for hydrography data. Must be same as
            setup_basemaps for consistent results.

            * Required variables: ['flwdir', 'uparea', 'elevtn']
        river_upa : float, optional
            (1D floodplains) minimum upstream area threshold for drain in the HAND.
            Optional value, as it is inferred from the grid metadata,
            to be consistent with setup_rivers.
        flood_depths : tuple of float, optional
            (1D floodplains) flood depths at which a volume is derived.
            By default [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

        elevtn_map: {"land_elevation", "meta_subgrid_elevation"}
            (2D floodplains) Name of staticmap to hydrologically condition.
            By default "land_elevation"
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.

        See Also
        --------
        workflows.river_floodplain_volume
        hydromt.flw.dem_adjust
        pyflwdir.FlwdirRaster.dem_adjust
        setup_rivers
        """
        if self.get_config("model.river_routing") != "local_inertial":
            raise ValueError(
                "Floodplains (1d or 2d) are currently only supported with \
local inertial river routing"
            )
        # update self._MAPS and self._WFLOW_NAMES with user defined output names
        var = "floodplain_water__sum_of_volume_per_depth"
        if var in output_names:
            self._update_naming({var: output_names[var]})

        r_list = ["1d", "2d"]
        if floodplain_type not in r_list:
            raise ValueError(
                f'river_routing="{floodplain_type}" unknown. Select from {r_list}.'
            )

        # Adjust settings based on floodplain_type selection
        if floodplain_type == "1d":
            floodplain_1d = True
            land_routing = "kinematic_wave"

            if not hasattr(pyflwdir.FlwdirRaster, "ucat_volume"):
                logger.warning("This method requires pyflwdir >= 0.5.6")
                return

            logger.info("Preparing 1D river floodplain_volume map.")

            # read data
            ds_hydro = self.data_catalog.get_rasterdataset(
                hydrography_fn, geom=self.region, buffer=10
            )
            ds_hydro.coords["mask"] = ds_hydro.raster.geometry_mask(self.region)

            # try to get river uparea from grid, throw error if not specified
            # or when found but different from specified value
            new_river_upa = self.staticmaps.data[self._MAPS["rivmsk"]].attrs.get(
                "river_upa", river_upa
            )
            if new_river_upa is None:
                raise ValueError(
                    "No value for `river_upa` specified, and the value cannot \
be inferred from the grid attributes"
                )
            elif new_river_upa != river_upa and river_upa is not None:
                raise ValueError(
                    f"Value specified for river_upa ({river_upa}) is different from \
the value found in the grid ({new_river_upa})"
                )
            logger.debug(f"Using river_upa value value of: {new_river_upa}")

            # get river floodplain volume
            inv_rename = {
                v: k for k, v in self._MAPS.items() if v in self.staticmaps.data
            }
            da_fldpln = workflows.river_floodplain_volume(
                ds=ds_hydro,
                ds_model=self.staticmaps.data.rename(inv_rename),
                river_upa=new_river_upa,
                flood_depths=flood_depths,
            )

            # check if the layer already exists, since overwriting with different
            # flood_depth values is not working properly if this is the case
            if self._MAPS["floodplain_volume"] in self.staticmaps.data:
                logger.warning(
                    "Layer `floodplain_volume` already in grid, removing layer \
and `flood_depth` dimension to ensure correctly \
setting new flood_depth dimensions"
                )
                self._grid = self._grid.drop_dims("flood_depth")

            da_fldpln.name = self._MAPS["floodplain_volume"]
            self.set_grid(da_fldpln)
            self._update_config_variable_name(da_fldpln.name)

        elif floodplain_type == "2d":
            floodplain_1d = False
            land_routing = "local_inertial"

            if elevtn_map not in self.staticmaps.data:
                raise ValueError(f'"{elevtn_map}" not found in grid')

            postfix = {
                "land_elevation": "_avg",
                "meta_subgrid_elevation": "_subgrid",
            }.get(elevtn_map, "")
            name = f"river_bank_elevation{postfix}_D{connectivity}"
            # Check if users wanted a specific name for the river_bank_elevation
            hydrodem_var = self._WFLOW_NAMES.get(self._MAPS["hydrodem"])
            lndelv_var = self._WFLOW_NAMES.get(self._MAPS["elevtn"])
            # river_bank_elevation is used for two wflow variables
            if hydrodem_var in output_names:
                name = output_names[hydrodem_var]
            self._update_naming(
                {
                    hydrodem_var: name,
                    lndelv_var: elevtn_map,
                }
            )
            logger.info(f"Preparing {name} map for land routing.")
            ds_out = flw.dem_adjust(
                da_flwdir=self.staticmaps.data[self._MAPS["flwdir"]],
                da_elevtn=self.staticmaps.data[elevtn_map],
                da_rivmsk=self.staticmaps.data[self._MAPS["rivmsk"]],
                flwdir=self.flwdir,
                connectivity=connectivity,
                river_d8=True,
            ).rename(name)
            self.set_grid(ds_out)
            # Update the bankfull elevation map
            self.set_config("input.static.river_bank_water__elevation", name)
            # In this case river_bank_elevation is also used for the ground elevation?
            self.set_config(
                "input.static.land_surface_water_flow__ground_elevation", elevtn_map
            )

        # Update config
        logger.info(
            f'Update wflow config model.floodplain_1d__flag="{floodplain_1d}"',
        )
        self.set_config("model.floodplain_1d__flag", floodplain_1d)
        logger.info(f'Update wflow config model.land_routing="{land_routing}"')
        self.set_config("model.land_routing", land_routing)

        if floodplain_type == "1d":
            # Add states
            self.set_config(
                "state.variables.floodplain_water__instantaneous_volume_flow_rate",
                "floodplain_instantaneous_q",
            )
            self.set_config(
                "state.variables.floodplain_water__depth",
                "floodplain_h",
            )
            self.set_config(
                "state.variables.land_surface_water__instantaneous_volume_flow_rate",
                "land_instantaneous_q",
            )
            # Remove local_inertial land states
            self.config.remove(
                "state.variables.land_surface_water__x_component_of_instantaneous_volume_flow_rate",
                errors="ignore",
            )
            self.config.remove(
                "state.variables.land_surface_water__y_component_of_instantaneous_volume_flow_rate",
                errors="ignore",
            )

            # Remove from output.netcdf_grid section
            self.config.remove(
                "output.netcdf_grid.variables.land_surface_water__x_component_of_instantaneous_volume_flow_rate",
                errors="ignore",
            )
            self.config.remove(
                "output.netcdf_grid.variables.land_surface_water__y_component_of_instantaneous_volume_flow_rate",
                errors="ignore",
            )
        else:
            # Add local_inertial land routing states
            self.set_config(
                "state.variables.land_surface_water__x_component_of_instantaneous_volume_flow_rate",
                "land_instantaneous_qx",
            )
            self.set_config(
                "state.variables.land_surface_water__y_component_of_instantaneous_volume_flow_rate",
                "land_instantaneous_qy",
            )
            # Remove kinematic_wave and 1d floodplain states
            self.config.remove(
                "state.variables.land_surface_water__instantaneous_volume_flow_rate",
                errors="ignore",
            )
            self.config.remove(
                "state.variables.floodplain_water__instantaneous_volume_flow_rate",
                errors="ignore",
            )
            self.config.remove(
                "state.variables.floodplain_water__depth",
                errors="ignore",
            )
            # Remove from output.netcdf_grid section
            self.config.remove(
                "output.netcdf_grid.variables.land_surface_water__instantaneous_volume_flow_rate",
                errors="ignore",
            )

    @hydromt_step
    def setup_reservoirs_no_control(
        self,
        reservoirs_fn: str | Path | gpd.GeoDataFrame,
        rating_curve_fns: list[str | Path | pd.DataFrame] | None = None,
        overwrite_existing: bool = False,
        duplicate_id: str = "error",
        min_area: float = 10.0,
        output_names: dict = {
            "reservoir_area__count": "reservoir_area_id",
            "reservoir_location__count": "reservoir_outlet_id",
            "reservoir_surface__area": "reservoir_area",
            "reservoir_water_surface__initial_elevation": "reservoir_initial_depth",
            "reservoir_water_flow_threshold_level__elevation": "reservoir_outflow_threshold",  # noqa: E501
            "reservoir_water__rating_curve_coefficient": "reservoir_b",
            "reservoir_water__rating_curve_exponent": "reservoir_e",
            "reservoir_water__rating_curve_type_count": "reservoir_rating_curve",
            "reservoir_water__storage_curve_type_count": "reservoir_storage_curve",
            "reservoir_lower_location__count": "reservoir_lower_id",
        },
        geom_name: str = "meta_reservoirs_no_control",
        **kwargs,
    ):
        """Generate maps of reservoir areas, outlets and parameters.

        This function adds (uncontrolled) reservoirs such as natural lakes or weirs to
        the model. It prepares rating and storage curves parameters for the reservoirs
        modelled with the following rating curve types (see
        `Wflow reservoir concepts <https://deltares.github.io/Wflow.jl/stable/model_docs/lateral/kinwave/>`__ ):

        * 1 for Q = f(H) from reservoir data and interpolation
        * 2 for Q = b(H - H0)^e (general power law)
        * 3 for Q = b(H - H0)^2 (Modified Puls Approach)

        Created reservoirs can be added to already existing ones in the model
        `overwrite_existing=False` (default) or overwrite them
        `overwrite_existing=True`.

        Reservoir data is generated from features with ``min_area`` [km2] (default 1
        km2) from a database with reservoir geometry, IDs and metadata. Parameters can
        be directly provided in the GeoDataFrame or derived using common properties such
        as average depth, area and discharge.

        If rating curve data is available for storage and discharge they can be prepared
        via ``rating_curve_fns`` (see below for syntax and requirements).
        Else the parameters 'reservoir_b' and 'reservoir_e' will be used for discharge,
        and a rectangular profile will be used to compute storage. This corresponds to
        the following storage curve types in Wflow:

        * 1 for S = A * H
        * 2 for S = f(H) from reservoir data and interpolation

        Adds model layers:

        * **reservoir_area_id** map: reservoir IDs [-]
        * **reservoir_outlet_id** map: reservoir IDs at outlet locations [-]
        * **reservoir_area** map: reservoir area [m2]
        * **reservoir_initial_depth** map: reservoir average water level [m]
        * **reservoir_outflow_threshold** map: reservoir outflow threshold water
            level [m]
        * **meta_reservoir_mean_outflow** map: reservoir average discharge [m3/s]
        * **reservoir_b** map: reservoir rating curve coefficient [-]
        * **reservoir_e** map: reservoir rating curve exponent [-]
        * **reservoir_rating_curve** map: option to compute rating curve [-]
        * **reservoir_storage_curve** map: option to compute storage curve [-]
        * **reservoir_lower_id** map: optional, lower linked reservoir locations [-]
        * **meta_reservoirs_no_control** geom: polygon with reservoirs (e.g. lakes or
          weirs) and wflow parameters.
        * **reservoirs** geom: polygon with all reservoirs as in the model

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_rivers`

        Parameters
        ----------
        reservoirs_fn : str, Path, gpd.GeoDataFrame
            Name of GeoDataFrame source for uncontrolled reservoir parameters.

                * Required variables for direct use:
                    - 'waterbody_id' [-],
                    - 'Area_avg' [m2],
                    - 'Depth_avg' [m],
                    - 'Dis_avg' [m3/s],
                    - 'reservoir_b' [-],
                    - 'reservoir_e' [-],
                    - 'reservoir_rating_curve' [-],
                    - 'reservoir_storage_curve' [-],
                    - 'reservoir_outflow_threshold' [m],
                    - 'reservoir_lower_id' [-]
                * Required variables for parameter estimation:
                    - 'waterbody_id' [-],
                    - 'Area_avg' [m2],
                    - 'Vol_avg' [m3],
                    - 'Depth_avg' [m],
                    - 'Dis_avg'[m3/s]

        rating_curve_fns: str, Path, pandas.DataFrame, List, optional
            Data catalog entry/entries, path(s) or pandas.DataFrame containing rating
            curve values for reservoirs. If None then will be derived from properties of
            `reservoirs_fn`.
            Assumes one file per reservoir (with all variables) and that the reservoir
            ID is either in the filename or data catalog entry name (eg using
            placeholder). The ID should be placed at the end separated by an underscore
            (eg 'rating_curve_12.csv' or 'rating_curve_12')

                * Required variables for storage curve:
                    - 'elevtn' [m+REF],
                    - 'volume' [m3]
                * Required variables for rating curve:
                    - 'elevtn' [m+REF],
                    - 'discharge' [m3/s]

        overwrite_existing : bool, optional
            If False (default), update existing reservoirs in the model with the new
            reservoirs_fn data.
        duplicate_id: str, optional {"error", "skip"}
            Action to take if duplicate reservoir IDs are found when merging with
            existing reservoirs. Options are "error" to raise an error (default); "skip"
            to skip adding new reservoirs.
        min_area : float, optional
            Minimum reservoir area threshold [km2], by default 10.0 km2.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        geom_name : str, optional
            Name of the reservoir geometry in the staticgeoms folder, by default
            'meta_reservoirs_no_control' for meta_reservoirs_no_control.geojson.
        kwargs: optional
            Keyword arguments passed to the method
            hydromt.DataCatalog.get_rasterdataset()
        """  # noqa: E501
        # retrieve data for basin
        logger.info("Preparing reservoir maps.")
        kwargs.setdefault("predicate", "contains")
        gdf_org = self.data_catalog.get_geodataframe(
            reservoirs_fn,
            geom=self.basins_highres,
            handle_nodata=NoDataStrategy.IGNORE,
            **kwargs,
        )
        if gdf_org is None:
            logger.info("Skipping method, as no data has been found")
            return

        # Derive reservoir area and outlet maps
        ds_reservoirs, gdf_org = workflows.reservoir_id_maps(
            gdf=gdf_org,
            ds_like=self.staticmaps.data,
            min_area=min_area,
            uparea_name=self._MAPS["uparea"],
        )
        if ds_reservoirs is None:
            # No reservoirs of sufficient size found
            return

        self._update_naming(output_names)

        # If rating_curve_fn prepare rating curve dict
        rating_dict = dict()
        if rating_curve_fns is not None:
            rating_curve_fns = np.atleast_1d(rating_curve_fns)
            # Find ids in rating_curve_fns
            fns_ids = []
            for fn in rating_curve_fns:
                try:
                    fns_ids.append(int(fn.split("_")[-1].split(".")[0]))
                except Exception:
                    logger.warning(
                        "Could not parse integer reservoir index from "
                        f"rating curve fn {fn}. Skipping."
                    )
            # assume reservoir index will be in the path
            # Assume one rating curve per reservoir index
            for _id in gdf_org["waterbody_id"].values:
                _id = int(_id)
                # Find if _id is is one of the paths in rating_curve_fns
                if _id in fns_ids:
                    # Update path based on current waterbody_id
                    i = fns_ids.index(_id)
                    rating_fn = rating_curve_fns[i]
                    # Read data
                    if isfile(rating_fn) or rating_fn in self.data_catalog.sources:
                        logger.info(
                            f"Preparing reservoir rating curve data from {rating_fn}"
                        )
                        df_rate = self.data_catalog.get_dataframe(rating_fn)
                        # Add to dict
                        rating_dict[_id] = df_rate
                else:
                    logger.warning(
                        f"Rating curve file not found for reservoir with id {_id}. "
                        "Using default storage/outflow function parameters."
                    )
        else:
            logger.info(
                "No rating curve data provided. "
                "Using default storage/outflow function parameters."
            )

        # add reservoir parameters
        ds_reservoirs, gdf_reservoirs, rating_curves = (
            workflows.reservoirs.reservoir_parameters(
                ds_reservoirs,
                gdf_org,
                rating_dict,
            )
        )

        # merge with existing reservoirs
        if (
            not overwrite_existing
            and self._MAPS["reservoir_area"] in self.staticmaps.data
        ):
            inv_rename = {
                v: k
                for k, v in self._MAPS.items()
                if v in self.staticmaps.data.data_vars
            }
            ds_reservoirs = workflows.reservoirs.merge_reservoirs(
                ds_reservoirs,
                self.staticmaps.data.rename(inv_rename),
                duplicate_id=duplicate_id,
            )
            # Check if ds_res is None ie duplicate IDs
            if ds_reservoirs is None:
                logger.warning(
                    "Duplicate reservoir IDs found. Skipping adding new reservoirs."
                )
                return
        else:
            # remove all reservoir layers from the grid as some control parameters
            # like demand will not be in ds_reservoirs and won't be overwritten
            reservoir_maps = [
                self._MAPS.get(k, k) for k in workflows.reservoirs.RESERVOIR_LAYERS
            ]
            self.staticmaps.drop_vars(reservoir_maps, errors="ignore")

        # add to grid
        rmdict = {k: self._MAPS.get(k, k) for k in ds_reservoirs.data_vars}
        self.set_grid(ds_reservoirs.rename(rmdict))
        # write reservoirs with attr tables to static geoms.
        self.set_geoms(gdf_reservoirs, name=geom_name)
        # Prepare a combined geoms of all reservoirs
        gdf_res_all = workflows.reservoirs.create_reservoirs_geoms(
            ds_reservoirs.rename(rmdict),
        )
        self.set_geoms(gdf_res_all, name="reservoirs")
        # add the tables
        for k, v in rating_curves.items():
            self.set_tables(v, name=k)

        # Reservoir settings in the toml to update
        self.set_config("model.reservoir__flag", True)
        self.set_config(
            "state.variables.reservoir_water_surface__elevation",
            "reservoir_water_level",
        )

        for dvar in ds_reservoirs.data_vars:
            if dvar in [
                "reservoir_area_id",
                "reservoir_outlet_id",
                "reservoir_lower_id",
            ]:
                self._update_config_variable_name(self._MAPS[dvar], data_type=None)
            elif dvar in self._WFLOW_NAMES:
                self._update_config_variable_name(self._MAPS[dvar])

    @hydromt_step
    def setup_reservoirs_simple_control(
        self,
        reservoirs_fn: str | gpd.GeoDataFrame,
        timeseries_fn: str | None = None,
        overwrite_existing: bool = False,
        duplicate_id: str = "error",
        min_area: float = 1.0,
        output_names: dict = {
            "reservoir_area__count": "reservoir_area_id",
            "reservoir_location__count": "reservoir_outlet_id",
            "reservoir_surface__area": "reservoir_area",
            "reservoir_water_surface__initial_elevation": "reservoir_initial_depth",
            "reservoir_water__rating_curve_type_count": "reservoir_rating_curve",
            "reservoir_water__storage_curve_type_count": "reservoir_storage_curve",
            "reservoir_water__max_volume": "reservoir_max_volume",
            "reservoir_water__target_min_volume_fraction": "reservoir_target_min_fraction",  # noqa: E501
            "reservoir_water__target_full_volume_fraction": "reservoir_target_full_fraction",  # noqa: E501
            "reservoir_water_demand__required_downstream_volume_flow_rate": "reservoir_demand",  # noqa: E501
            "reservoir_water_release_below_spillway__max_volume_flow_rate": "reservoir_max_release",  # noqa: E501
        },
        geom_name: str = "meta_reservoirs_simple_control",
        **kwargs,
    ):
        """Generate maps of controlled reservoir areas, outlets and parameters.

        Also generates parameters with average reservoir area, demand,
        min and max target storage capacities and discharge capacity values.

        This function adds reservoirs with simple control operations to the model. It
        prepares rating and storage curves parameters for the reservoirs modelled with
        the following rating curve types (see
        `Wflow reservoir concepts <https://deltares.github.io/Wflow.jl/stable/model_docs/lateral/waterbodies/>`__
        ):

        * 4 simple reservoir operational parameters

        Created reservoirs can be added to already existing ones in the model
        `overwrite_existing=False` (default) or overwrite them
        `overwrite_existing=True`.

        The data is generated from features with ``min_area`` [km2] (default is 1 km2)
        from a database with reservoir geometry, IDs and metadata. Parameters can
        be directly provided in the GeoDataFrame or derived using common properties such
        as average depth, area and discharge.

        Data requirements for direct use (i.e. wflow parameters are data already present
        in reservoirs_fn) are reservoir ID 'waterbody_id', area 'reservoir_area' [m2],
        initial depth 'reservoir_initial_depth' [m], rating curve type
        'reservoir_rating_curve' [-], storage curve type 'reservoir_storage_curve' [-],
        maximum volume 'reservoir_max_volume' [m3], the targeted minimum and maximum
        fraction of water volume in the reservoir 'reservoir_target_min_fraction' and
        'reservoir_target_full_fraction' [-], the average water demand
        'reservoir_demand' [m3/s] and the maximum release of the reservoir before
        spilling 'reservoir_max_release' [m3/s].

        In case the wflow parameters are not directly available they can be computed by
        HydroMT based on time series of reservoir surface water area.
        These time series can be retrieved from either the hydroengine or the gwwapi,
        based on the Hylak_id the reservoir, found in the GrandD database.

        The required variables for computation of the parameters with time series data
        are reservoir ID 'waterbody_id', reservoir ID in the HydroLAKES database
        'Hylak_id', average volume 'Vol_avg' [m3], average depth 'Depth_avg' [m],
        average discharge 'Dis_avg' [m3/s] and dam height 'Dam_height' [m].
        To compute parameters without using time series data, the required variables in
        reservoirs_fn are reservoir ID 'waterbody_id', average area 'Area_avg' [m2],
        average volume 'Vol_avg' [m3], average depth 'Depth_avg' [m], average discharge
        'Dis_avg' [m3/s] and dam height 'Dam_height' [m]
        and minimum / normal / maximum storage capacity of the dam 'Capacity_min',
        'Capacity_norm', 'Capacity_max' [m3].

        Adds model layers:

            * **reservoir_area_id** map: reservoir IDs [-]
            * **reservoir_outlet_id** map: reservoir IDs at outlet locations [-]
            * **reservoir_area** map: reservoir area [m2]
            * **reservoir_initial_depth** map: reservoir initial water level [m]
            * **reservoir_rating_curve** map: option to compute rating curve [-]
            * **reservoir_storage_curve** map: option to compute storage curve [-]
            * **reservoir_max_volume** map: reservoir max volume [m3]
            * **reservoir_target_min_fraction** map: reservoir target min frac [m3/m3]
            * **reservoir_target_full_fraction** map: reservoir target full frac [m3/m3]
            * **reservoir_demand** map: reservoir demand flow [m3/s]
            * **reservoir_max_release** map: reservoir max release flow [m3/s]
            * **meta_reservoirs_simple_control** geom: polygon with
                reservoirs and parameters
            * **reservoirs** geom: polygon with all reservoirs as in the model

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_rivers`

        Parameters
        ----------
        reservoirs_fn : str
            Name of data source for reservoir parameters, see data/data_sources.yml.

            * Required variables for direct use:
              'waterbody_id' [-], 'reservoir_area' [m2], 'reservoir_max_volume' [m3],
              'reservoir_initial_depth' [m], 'reservoir_rating_curve' [-],
              'reservoir_storage_curve' [-], 'reservoir_target_min_fraction' [m3/m3],
              'reservoir_target_full_fraction' [m3/m3], 'reservoir_demand' [m3/s],
              'reservoir_max_release' [m3/s]
            * Required variables for computation with timeseries_fn:
              'waterbody_id' [-], 'Hylak_id' [-], 'Vol_avg' [m3], 'Depth_avg' [m],
              'Dis_avg' [m3/s], 'Dam_height' [m]
            * Required variables for computation without timeseries_fn:
              'waterbody_id' [-], 'Area_avg' [m2], 'Vol_avg' [m3], 'Depth_avg' [m],
              'Dis_avg' [m3/s], 'Capacity_max' [m3], 'Capacity_norm' [m3],
              'Capacity_min' [m3], 'Dam_height' [m]

        timeseries_fn : {'gww', 'hydroengine', None}, optional
            Download and use time series of reservoir surface water area to calculate
            and overwrite the reservoir volume/areas of the data source. Timeseries are
            either downloaded from Global Water Watch 'gww' (using gwwapi package) or
            JRC 'jrc' (using hydroengine package). By default None.
        overwrite_existing : bool, optional
            If False (default), update existing reservoirs in the model with the new
            reservoirs_fn data.
        duplicate_id: str, optional {"error", "skip"}
            Action to take if duplicate reservoir IDs are found when merging with
            existing reservoirs. Options are "error" to raise an error (default); "skip"
            to skip adding new reservoirs.
        min_area : float, optional
            Minimum reservoir area threshold [km2], by default 1.0 km2.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        geom_name : str, optional
            Name of the reservoirs geometry in the staticgeoms folder, by default
            "meta_reservoirs_simple_control" for meta_reservoirs_simple_control.geojson.
        kwargs: optional
            Keyword arguments passed to the method
            hydromt.DataCatalog.get_rasterdataset()
        """  # noqa: E501
        # retrieve data for basin
        logger.info("Preparing reservoir with simple control maps.")
        kwargs.setdefault("predicate", "contains")
        gdf_org = self.data_catalog.get_geodataframe(
            reservoirs_fn,
            geom=self.basins_highres,
            handle_nodata=NoDataStrategy.IGNORE,
            **kwargs,
        )
        # Skip method if no data is returned
        if gdf_org is None:
            logger.info("Skipping method, as no data has been found")
            return

        # Derive reservoir area and outlet maps
        ds_res, gdf_org = workflows.reservoir_id_maps(
            gdf=gdf_org,
            ds_like=self.staticmaps.data,
            min_area=min_area,
            uparea_name=self._MAPS["uparea"],
        )
        if ds_res is None:
            # No reservoir of sufficient size found
            return
        self._update_naming(output_names)

        # add parameters
        ds_res, gdf_res = workflows.reservoir_simple_control_parameters(
            gdf=gdf_org,
            ds_reservoirs=ds_res,
            timeseries_fn=timeseries_fn,
            output_folder=self.root.path,
        )

        # merge with existing reservoirs
        if (
            not overwrite_existing
            and self._MAPS["reservoir_area"] in self.staticmaps.data
        ):
            inv_rename = {
                v: k
                for k, v in self._MAPS.items()
                if v in self.staticmaps.data.data_vars
            }
            ds_res = workflows.reservoirs.merge_reservoirs(
                ds_res,
                self.staticmaps.data.rename(inv_rename),
                duplicate_id=duplicate_id,
            )
            # Check if ds_res is None ie duplicate IDs
            if ds_res is None:
                logger.warning(
                    "Duplicate reservoir IDs found. Skipping adding new reservoirs."
                )
                return
        else:
            # remove all reservoir layers from the grid as some parameters
            # like b or e will not be in ds_res and won't be overwritten
            reservoir_maps = [
                self._MAPS.get(k, k) for k in workflows.reservoirs.RESERVOIR_LAYERS
            ]
            self.staticmaps.drop_vars(reservoir_maps, errors="ignore")

        # add to grid
        rmdict = {k: self._MAPS.get(k, k) for k in ds_res.data_vars}
        self.set_grid(ds_res.rename(rmdict))

        # write reservoirs with param values to geoms
        self.set_geoms(gdf_res, name=geom_name)
        # Prepare a combined geoms of all reservoirs
        gdf_res_all = workflows.reservoirs.create_reservoirs_geoms(
            ds_res.rename(rmdict),
        )
        self.set_geoms(gdf_res_all, name="reservoirs")

        # update toml
        self.set_config("model.reservoir__flag", True)
        self.set_config(
            "state.variables.reservoir_water_surface__elevation",
            "reservoir_water_level",
        )
        for dvar in ds_res.data_vars:
            if dvar in ["reservoir_area_id", "reservoir_outlet_id"]:
                self._update_config_variable_name(self._MAPS[dvar], data_type=None)
            elif dvar in self._WFLOW_NAMES:
                self._update_config_variable_name(self._MAPS[dvar], data_type="static")

    @hydromt_step
    def setup_glaciers(
        self,
        glaciers_fn: str | Path | gpd.GeoDataFrame,
        min_area: float = 1.0,
        output_names: dict = {
            "glacier_surface__area_fraction": "glacier_fraction",
            "glacier_ice__initial_leq_depth": "glacier_initial_leq_depth",
        },
        geom_name: str = "glaciers",
    ):
        """
        Generate maps of glacier areas, area fraction and volume fraction.

        The data is generated from features with ``min_area`` [km2] (default is 1 km2)
        from a database with glacier geometry, IDs and metadata.

        The required variables from glaciers_fn dataset are glacier ID 'simple_id'.
        Optionally glacier area 'AREA' [km2] can be present to filter the glaciers
        by size. If not present it will be computed on the fly.

        Adds model layers:

        * **meta_glacier_area_id** map: glacier IDs [-]
        * **glacier_fraction** map: area fraction of glacier per cell [-]
        * **glacier_initial_leq_depth** map: storage (volume) of glacier per cell [mm]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        glaciers_fn :
            Name of data source for glaciers, see data/data_sources.yml.

            * Required variables: ['simple_id']
        min_area : float, optional
            Minimum glacier area threshold [km2], by default 0 (all included)
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        geom_name : str, optional
            Name of the geometry to be used in the model, by default "glaciers" for
            glaciers.geojson.
        """
        self._update_naming(output_names)
        # retrieve data for basin
        logger.info("Preparing glacier maps.")
        gdf_org = self.data_catalog.get_geodataframe(
            glaciers_fn,
            geom=self.basins_highres,
            predicate="intersects",
            handle_nodata=NoDataStrategy.IGNORE,
        )
        # Check if there are glaciers found
        if gdf_org is None:
            logger.info("Skipping method, as no data has been found")
            return

        # skip small size glacier
        if "AREA" in gdf_org.columns and gdf_org.geometry.size > 0:
            gdf_org = gdf_org[gdf_org["AREA"] >= min_area]
        # get glacier maps and parameters
        nb_glac = gdf_org.geometry.size
        if nb_glac == 0:
            logger.warning(
                "No glaciers of sufficient size found within region!"
                "Skipping glacier procedures!"
            )
            return

        logger.info(f"{nb_glac} glaciers of sufficient size found within region.")
        # add glacier maps
        ds_glac = workflows.glaciermaps(
            gdf=gdf_org,
            ds_like=self.staticmaps.data,
            id_column="simple_id",
            elevtn_name=self._MAPS["elevtn"],
        )

        rmdict = {k: self._MAPS.get(k, k) for k in ds_glac.data_vars}
        self.set_grid(ds_glac.rename(rmdict))
        # update config
        self._update_config_variable_name(ds_glac.rename(rmdict).data_vars)
        self.set_config("model.glacier__flag", True)
        self.set_config("state.variables.glacier_ice__leq_depth", "glacier_leq_depth")
        # update geoms
        self.set_geoms(gdf_org, name=geom_name)

    @hydromt_step
    def setup_lulcmaps_with_paddy(
        self,
        lulc_fn: str | Path | xr.DataArray,
        paddy_class: int,
        output_paddy_class: int | None = None,
        lulc_mapping_fn: str | Path | pd.DataFrame | None = None,
        paddy_fn: str | Path | xr.DataArray | None = None,
        paddy_mapping_fn: str | Path | pd.DataFrame | None = None,
        soil_fn: str | Path | xr.DataArray = "soilgrids",
        wflow_thicknesslayers: list[int] = [50, 100, 50, 200, 800],
        target_conductivity: list[None | int | float] = [
            None,
            None,
            5,
            None,
            None,
        ],
        lulc_vars: dict = {
            "landuse": None,
            "vegetation_kext": "vegetation_canopy__light_extinction_coefficient",
            "land_manning_n": "land_surface_water_flow__manning_n_parameter",
            "soil_compacted_fraction": "compacted_soil__area_fraction",
            "vegetation_root_depth": "vegetation_root__depth",
            "vegetation_leaf_storage": "vegetation__specific_leaf_storage",
            "vegetation_wood_storage": "vegetation_wood_water__storage_capacity",
            "land_water_fraction": "land_water_covered__area_fraction",
            "vegetation_crop_factor": "vegetation__crop_factor",
            "vegetation_feddes_alpha_h1": "vegetation_root__feddes_critical_pressure_head_h1_reduction_coefficient",  # noqa: E501
            "vegetation_feddes_h1": "vegetation_root__feddes_critical_pressure_head_h1",
            "vegetation_feddes_h2": "vegetation_root__feddes_critical_pressure_head_h2",
            "vegetation_feddes_h3_high": "vegetation_root__feddes_critical_pressure_head_h3_high",  # noqa: E501
            "vegetation_feddes_h3_low": "vegetation_root__feddes_critical_pressure_head_h3_low",  # noqa: E501
            "vegetation_feddes_h4": "vegetation_root__feddes_critical_pressure_head_h4",
        },
        paddy_waterlevels: dict = {
            "demand_paddy_h_min": 20,
            "demand_paddy_h_opt": 50,
            "demand_paddy_h_max": 80,
        },
        save_high_resolution_lulc: bool = False,
        output_names_suffix: str | None = None,
    ):
        """Set up landuse maps and parameters including for paddy fields.

        Lookup table `lulc_mapping_fn` columns are converted to lulc classes model
        parameters based on literature. The data is remapped at its original resolution
        and then resampled to the model resolution using the average value, unless noted
        differently.

        If paddies are present either directly as a class in the landuse_fn or in a
        separate paddy_fn, the paddy class is used to derive the paddy parameters.

        To allow for water to pool on the surface (for paddy/rice fields), the layers in
        the model can be updated to new depths, such that we can allow a thin layer with
        limited vertical conductivity. These updated layers means that the
        ``soil_brooks_corey_c`` parameter needs to be calculated again. Next, the
        soil_ksat_vertical_factor layer corrects the vertical conductivity
        (by multiplying) such that the bottom of the layer corresponds to the
        ``target_conductivity`` for that layer. This currently assumes the wflow models
        to have an exponential declining vertical conductivity (using the ``f``
        parameter). If no target_conductivity is specified for a layer (``None``),
        the soil_ksat_vertical_factor value is set to 1.

        The different values for the minimum/optimal/maximum water levels for paddy
        fields will be added as constant values in the toml file, through the
        ``irrigated_paddy__min_depth.value = 20`` interface.

        Adds model layers:

        * **landuse** map:
            Landuse class [-]
        * **vegetation_kext** map:
            Extinction coefficient in the canopy gap fraction equation [-]
        * **vegetation_leaf_storage** map:
            Specific leaf storage [mm]
        * **vegetation_wood_storage** map:
            Fraction of wood in the vegetation/plant [-]
        * **vegetation_root_depth** map:
            Length of vegetation roots [mm]
        * **soil_compacted_fraction** map:
            The fraction of compacted or urban area per grid cell [-]
        * **land_water_fraction** map:
            The fraction of open water per grid cell [-]
        * **land_manning_n** map:
            Manning Roughness [-]
        * **vegetation_crop_factor** map:
            Crop coefficient [-]
        * **vegetation_feddes_alpha_h1** map:
            Root water uptake reduction at soil water pressure head
            h1 (0 or 1) [-]
        * **vegetation_feddes_h1** map:
            Soil water pressure head h1 at which root water
            uptake is reduced (Feddes) [cm]
        * **vegetation_feddes_h2** map:
            Soil water pressure head h2 at which root water
            uptake is reduced (Feddes) [cm]
        * **vegetation_feddes_h3_high** map:
            Soil water pressure head h3 (high) at which root water uptake is
            reduced (Feddes) [cm]
        * **vegetation_feddes_h3_low** map:
            Soil water pressure head h3 (low) at which root water uptake is
            reduced (Feddes) [cm]
        * **vegetation_feddes_h4** map:
            Soil water pressure head h4 at which root water
            uptake is reduced (Feddes) [cm]
        * **demand_paddy_h_min** value:
            Minimum required water depth for paddy fields [mm]
        * **demand_paddy_h_opt** value:
            Optimal water depth for paddy fields [mm]
        * **demand_paddy_h_max** value:
            Maximum water depth for paddy fields [mm]
        * **soil_ksat_vertical_factor**:
            Map with a multiplication factor for the vertical conductivity [-]

        Updates model layers:

        * **soil_brooks_corey_c**:
            Brooks Corey coefficients [-] based on pore size
            distribution, a map for each of the wflow_sbm soil layers (updated based
            on the newly specified layers)


        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_soilmaps`

        Parameters
        ----------
        lulc_fn : str, Path, xr.DataArray
            RasterDataset or name in data catalog / path to landuse map.
        paddy_class : int
            Landuse class value for paddy fields either in landuse_fn or paddy_fn if
            provided.
        output_paddy_class : int, optional
            Landuse class value for paddy fields in the output landuse map. If None,
            the ``paddy_class`` is used, by default None. This can be useful when
            merging paddy location from ``paddy_fn`` into ``landuse_fn``.
        lulc_mapping_fn : str, Path, pd.DataFrame, optional
            Path to a mapping csv file from landuse in source name to parameter values
            in lulc_vars. If lulc_fn is one of {"globcover", "vito", "corine",
            "esa_worldcover", "glmnco"}, a default mapping is used and this argument
            becomes optional.
        paddy_fn : str, Path, xr.DataArray, optional
            RasterDataset or name in data catalog / path to paddy map.
        paddy_mapping_fn : str, Path, pd.DataFrame, optional
            Path to a mapping csv file from paddy in source name to parameter values
            in lulc_vars. A default mapping table for rice parameters is used if not
            provided.
        soil_fn : str, Path, xr.DataArray, optional
            Soil data to be used to recalculate the Brooks-Corey coefficients
            (`soil_brooks_corey_c` parameter), based on the provided
            ``wflow_thicknesslayers``, by default "soilgrids", but should ideally
            be equal to the data used in :py:meth:`setup_soilmaps`

            * Required variables: 'bd_sl*' [g/cm3], 'clyppt_sl*' [%], 'sltppt_sl*' [%],
              'ph_sl*' [-].

        wflow_thicknesslayers: list
            List of soil thickness per layer [mm], by default [50, 100, 50, 200, 800, ]
        target_conductivity: list
            List of target vertical conductivities [mm/day] for each layer in
            ``wflow_thicknesslayers``. Set value to `None` if no specific value is
            required, by default [None, None, 5, None, None].
        lulc_vars : dict
            Dictionnary of landuse parameters to prepare. The names are the
            the columns of the mapping file and the values are the corresponding
            Wflow.jl variables.
        paddy_waterlevels : dict
            Dictionary with the minimum, optimal and maximum water levels for paddy
            fields [mm]. By default {"demand_paddy_h_min": 20, "demand_paddy_h_opt": 50,
            "demand_paddy_h_max": 80}
        save_high_resolution_lulc : bool
            Save the high resolution landuse map merged with the paddies to the static
            folder. By default False.
        output_names_suffix : str, optional
            Suffix to be added to the output names to avoid having to rename all the
            columns of the mapping tables. For example if the suffix is "vito", all
            variables in lulc_vars will be renamed to "landuse_vito", "Kext_vito", etc.
            Note that the suffix will also be used to rename the paddy parameter
            soil_ksat_vertical_factor but not the soil_brooks_corey_c parameter.
        """
        logger.info("Preparing LULC parameter maps including paddies.")
        if output_names_suffix is not None:
            # rename lulc_vars with the suffix
            output_names = {
                v: f"{k}_{output_names_suffix}" for k, v in lulc_vars.items()
            }
            # Add soil_ksat_vertical_factor
            output_names[self._WFLOW_NAMES[self._MAPS["soil_ksat_vertical_factor"]]] = (
                f"soil_ksat_vertical_factor_{output_names_suffix}"
            )

        else:
            output_names = {v: k for k, v in lulc_vars.items()}
        # update self._MAPS and self._WFLOW_NAMES with user defined output names
        self._update_naming(output_names)

        # As landuse is not a wflow variable, we update the name manually in self._MAPS
        rmdict = {"landuse": "meta_landuse"} if "landuse" in lulc_vars else {}
        if output_names_suffix is not None:
            self._MAPS["landuse"] = f"meta_landuse_{output_names_suffix}"
            # rename dict for the staticmaps (hydromt names are not used in that case)
            rmdict = {k: f"{k}_{output_names_suffix}" for k in lulc_vars.keys()}
            if "landuse" in lulc_vars:
                rmdict["landuse"] = f"meta_landuse_{output_names_suffix}"

        # Check if soil data is available
        if self._MAPS["ksat_vertical"] not in self.staticmaps.data.data_vars:
            raise ValueError(
                "ksat_vertical and f are required to update the soil parameters with "
                "paddies. Please run setup_soilmaps first."
            )

        if lulc_mapping_fn is None:
            lulc_mapping_fn = f"{lulc_fn}_mapping_default"
        # read landuse map and mapping table
        landuse = self.data_catalog.get_rasterdataset(
            lulc_fn, geom=self.region, buffer=2, variables=["landuse"]
        )
        df_mapping = self.data_catalog.get_dataframe(
            lulc_mapping_fn,
            driver_kwargs={"index_col": 0},  # only used if fn_map is a file path
        )
        output_paddy_class = (
            paddy_class if output_paddy_class is None else output_paddy_class
        )

        # if needed, add paddies to landuse
        if paddy_fn is not None:
            # Read paddy map and mapping table
            paddy = self.data_catalog.get_rasterdataset(
                paddy_fn, geom=self.region, buffer=2, variables=["paddy"]
            )
            if paddy_mapping_fn is None:
                paddy_mapping_fn = "paddy_mapping_default"
            df_paddy_mapping = self.data_catalog.get_dataframe(
                paddy_mapping_fn,
                driver_kwargs={"index_col": 0},
            )

            landuse, df_mapping = workflows.add_paddy_to_landuse(
                landuse,
                paddy,
                paddy_class,
                output_paddy_class=output_paddy_class,
                df_mapping=df_mapping,
                df_paddy_mapping=df_paddy_mapping,
            )

            if save_high_resolution_lulc:
                output_dir = join(self.root.path, "maps")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                landuse.raster.to_raster(join(output_dir, "landuse_with_paddy.tif"))
                df_mapping.to_csv(join(output_dir, "landuse_with_paddy_mapping.csv"))

        # Prepare landuse parameters
        landuse_maps = workflows.landuse(
            da=landuse,
            ds_like=self.staticmaps.data,
            df=df_mapping,
            params=list(lulc_vars.keys()),
        )
        self.set_grid(landuse_maps.rename(rmdict))
        # update config
        self._update_config_variable_name(landuse_maps.rename(rmdict).data_vars)

        # Update soil parameters if there are paddies in the domain
        # Get paddy pixels at model resolution
        wflow_paddy = landuse_maps["landuse"] == output_paddy_class
        if wflow_paddy.any():
            if self.get_config("model.soil_layer__thickness") == len(
                wflow_thicknesslayers
            ):
                logger.info(
                    "same thickness already present, skipping updating"
                    " `soil_brooks_corey_c` parameter"
                )
                update_c = False
            else:
                logger.info(
                    "Different thicknesslayers requested, updating "
                    "`soil_brooks_corey_c` parameter"
                )
                update_c = True
            # Read soil data
            soil = self.data_catalog.get_rasterdataset(
                soil_fn, geom=self.region, buffer=2
            )
            # update soil parameters soil_brooks_corey_c and soil_ksat_vertical_factor
            inv_rename = {
                v: k
                for k, v in self._MAPS.items()
                if v in self.staticmaps.data.data_vars
            }
            soil_maps = workflows.update_soil_with_paddy(
                ds=soil,
                ds_like=self.staticmaps.data.rename(inv_rename),
                paddy_mask=wflow_paddy,
                soil_fn=soil_fn,
                update_c=update_c,
                wflow_layers=wflow_thicknesslayers,
                target_conductivity=target_conductivity,
            )
            self.set_grid(
                soil_maps["soil_ksat_vertical_factor"],
                name=self._MAPS["soil_ksat_vertical_factor"],
            )
            self._update_config_variable_name(self._MAPS["soil_ksat_vertical_factor"])
            if "soil_brooks_corey_c" in soil_maps:
                self.set_grid(
                    soil_maps["soil_brooks_corey_c"],
                    name=self._MAPS["soil_brooks_corey_c"],
                )
                self._update_config_variable_name(self._MAPS["soil_brooks_corey_c"])
                self.set_config("model.soil_layer__thickness", wflow_thicknesslayers)
            # Add paddy water levels to the config
            for key, value in paddy_waterlevels.items():
                self.set_config(f"input.static.{self._WFLOW_NAMES[key]}.value", value)
            # Update the states
            self.set_config(
                "state.variables.paddy_surface_water__depth", "demand_paddy_h"
            )
        else:
            logger.info("No paddy fields found, skipping updating soil parameters")

    @hydromt_step
    def setup_laimaps(
        self,
        lai_fn: str | xr.DataArray,
        lulc_fn: str | xr.DataArray | None = None,
        lulc_sampling_method: str = "any",
        lulc_zero_classes: list[int] = [],
        buffer: int = 2,
        output_name: str = "vegetation_leaf_area_index",
    ):
        """
        Set leaf area index (LAI) climatology maps per month [1,2,3,...,12].

        The values are resampled to the model resolution using the average value.
        Currently only directly cyclic LAI data is supported.

        If `lulc_fn` is provided, mapping tables from landuse classes to LAI values
        will be derived from the LULC data. These tables can then be re-used later if
        you would like to add new LAI maps derived from this mapping table and new
        landuse scenarios. We advise to use a larger `buffer` to ensure that LAI values
        can be assigned for all landuse classes and based on a large enough sample of
        the LULC data.

        Adds model layers:

        * **vegetation_leaf_area_index** map: Leaf Area Index climatology [-]
            Resampled from source data using average. Assuming that missing values
            correspond to bare soil, these are set to zero before resampling.

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        lai_fn : str, xarray.DataArray
            Name of RasterDataset source for LAI parameters, see data/data_sources.yml.

            * Required variables: 'LAI' [-]

            * Required dimensions: 'time' = [1,2,3,...,12] (months)
        lulc_fn : str, xarray.DataArray, optional
            Name of RasterDataset source for landuse-landcover data.
            If provided, the LAI values are mapped to landuse classes and will be saved
            to a csv file.
        lulc_sampling_method : str, optional
            Resampling method for the LULC data to the LAI resolution. Two methods are
            supported:

            * 'any' (default): if any cell of the desired landuse class is present in
              the resampling window (even just one), it will be used to derive LAI
              values. This method is less exact but will provide LAI values for all
              landuse classes for the high resolution landuse map.
            * 'mode': the most frequent value in the resampling window is
              used. This method is less precise as for cells with a lot of different
              landuse classes, the most frequent value might still be only a small
              fraction of the cell. More landuse classes should however be covered and
              it can always be used with the landuse map of the wflow model instead of
              the original high resolution one.
            * 'q3': only cells with the most frequent value (mode) and that cover 75%
              (q3) of the resampling window will be used. This method is more exact but
              for small basins, you may have less or no samples to derive LAI values
              for some classes.
        lulc_zero_classes : list, optional
            List of landuse classes that should have zero for leaf area index values
            for example waterbodies, open ocean etc. For very high resolution landuse
            maps, urban surfaces and bare areas can be included here as well.
            By default empty.
        buffer : int, optional
            Buffer in pixels around the region to read the data, by default 2.
        output_name : str
            Name of the output vegetation__leaf_area_index map.
            By default "vegetation_leaf_area_index".
        """
        # retrieve data for region
        logger.info("Preparing LAI maps.")
        wflow_var = self._WFLOW_NAMES[self._MAPS["LAI"]]
        self._update_naming({wflow_var: output_name})
        da = self.data_catalog.get_rasterdataset(
            lai_fn, geom=self.region, buffer=buffer
        )
        if lulc_fn is not None:
            logger.info("Preparing LULC-LAI mapping table.")
            da_lulc = self.data_catalog.get_rasterdataset(
                lulc_fn, geom=self.region, buffer=buffer
            )
            # derive mapping
            df_lai_mapping = workflows.create_lulc_lai_mapping_table(
                da_lulc=da_lulc,
                da_lai=da.copy(),
                sampling_method=lulc_sampling_method,
                lulc_zero_classes=lulc_zero_classes,
            )
            # Save to csv
            if isinstance(lulc_fn, str) and not isfile(lulc_fn):
                df_fn = f"lai_per_lulc_{lulc_fn}.csv"
            else:
                df_fn = "lai_per_lulc.csv"
            df_lai_mapping.to_csv(join(self.root.path, df_fn))

        # Resample LAI data to wflow model resolution
        da_lai = workflows.lai(
            da=da,
            ds_like=self.staticmaps.data,
        )
        # Rename the first dimension to time
        rmdict = {da_lai.dims[0]: "time"}
        self.set_grid(da_lai.rename(rmdict), name=self._MAPS["LAI"])
        self._update_config_variable_name(self._MAPS["LAI"], data_type="cyclic")

    @hydromt_step
    def setup_laimaps_from_lulc_mapping(
        self,
        lulc_fn: str | xr.DataArray,
        lai_mapping_fn: str | pd.DataFrame,
        output_name: str = "vegetation_leaf_area_index",
    ):
        """
        Derive cyclic LAI maps from a LULC data source and a LULC-LAI mapping table.

        Adds model layers:

        * **vegetation_leaf_area_index** map: Leaf Area Index climatology [-]
            Resampled from source data using average. Assuming that missing values
            correspond to bare soil, these are set to zero before resampling.

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        lulc_fn : str, xarray.DataArray
            Name of RasterDataset source for landuse-landcover data.
        lai_mapping_fn : str, pd.DataFrame
            Path to a mapping csv file from landuse in source name to
            LAI values. The csv file should contain rows with landuse classes
            and LAI values for each month. The columns should be named as the
            months (1,2,3,...,12).
            This table can be created using the :py:meth:`setup_laimaps` method.
        output_name : str
            Name of the output vegetation__leaf_area_index map.
            By default "vegetation_leaf_area_index".
        """
        logger.info("Preparing LAI maps from LULC data using LULC-LAI mapping table.")
        # update self._MAPS and self._WFLOW_NAMES with user defined output names
        wflow_var = self._WFLOW_NAMES[self._MAPS["LAI"]]
        self._update_naming({wflow_var: output_name})

        # read landuse map to DataArray
        da = self.data_catalog.get_rasterdataset(
            lulc_fn, geom=self.region, buffer=2, variables=["landuse"]
        )
        df_lai_mapping = self.data_catalog.get_dataframe(
            lai_mapping_fn,
            driver_kwargs={"index_col": 0},  # only used if fn_map is a file path
        )
        # process landuse with LULC-LAI mapping table
        da_lai = workflows.lai_from_lulc_mapping(
            da=da,
            ds_like=self.staticmaps.data,
            df=df_lai_mapping,
        )
        # Add to grid
        self.set_grid(da_lai, name=self._MAPS["LAI"])
        # Add to config
        self._update_config_variable_name(self._MAPS["LAI"], data_type="cyclic")

    @hydromt_step
    def setup_rootzoneclim(
        self,
        run_fn: str | Path | xr.Dataset,
        forcing_obs_fn: str | Path | xr.Dataset,
        forcing_cc_hist_fn: str | Path | xr.Dataset | None = None,
        forcing_cc_fut_fn: str | Path | xr.Dataset | None = None,
        chunksize: int | None = 100,
        return_period: list[int] = [2, 3, 5, 10, 15, 20, 25, 50, 60, 100],
        Imax: float = 2.0,
        start_hydro_year: str = "Sep",
        start_field_capacity: str = "Apr",
        LAI: bool = False,
        rootzone_storage: bool = False,
        correct_cc_deficit: bool = False,
        time_range: tuple | None = None,
        time_range_fut: tuple | None = None,
        missing_days_threshold: int | None = 330,
        output_name_rootingdepth: str = "vegetation_root_depth_obs_20",
    ) -> None:
        """
        Set the vegetation_root_depth.

        Done by estimating the catchment-scale root-zone storage capacity from observed
        hydroclimatic data (and optionally also for climate change historical and
        future periods).

        This presents an alternative approach to determine the vegetation_root_depth
        based on hydroclimatic data instead of through a look-up table relating
        land use to rooting depth (as usually done for the wflow_sbm model).
        The method is based on the estimation of maximum annual storage deficits
        based on precipitation and estimated actual evaporation time series,
        which in turn are estimated from observed streamflow data and
        long-term precipitation and potential evap. data, as explained in
        Bouaziz et al. (2022).

        The main assumption is that vegetation adapts its rootzone storage capacity
        to overcome dry spells with a certain return period (typically 20 years for
        forest ecosystems). In response to a changing climtate,
        it is likely that vegetation also adapts its rootzone storage capacity,
        thereby changing model parameters for future conditions.
        This method also allows to estimate the change in rootzone storage capacity
        in response to a changing climate.

        As the method requires precipitation and potential evaporation timeseries,
        it may be useful to run this method as an update step in the setting-up of
        the hydrological model, once the forcing files have already been derived.
        In addition the setup_soilmaps method is also required to calculate
        the vegetation_root_depth (rootzone_storage / (theta_s-theta_r)).
        The setup_laimaps method is also required if LAI is set to True
        (interception capacity estimated from LAI maps, instead of providing
        a default maximum interception capacity).

        References
        ----------
        Bouaziz, L. J. E., Aalbers, E. E., Weerts, A. H., Hegnauer, M., Buiteveld,
        H., Lammersen, R., Stam, J., Sprokkereef, E., Savenije, H. H. G. and
        Hrachowitz, M. (2022). Ecosystem adaptation to climate change: the
        sensitivity of hydrological predictions to time-dynamic model parameters,
        Hydrology and Earth System Sciences, 26(5), 1295-1318. DOI:
        10.5194/hess-26-1295-2022.

        Adds model layer:

        * **vegetation_root_depth_{forcing}_{RP}** map: rooting depth [mm of the soil \
column] estimated from hydroclimatic data {forcing: obs, cc_hist or cc_fut} for \
different return periods RP. The translation to vegetation_root_depth is done by \
dividing the rootzone_storage by (theta_s - theta_r).
        * **meta_rootzone_storage_{forcing}_{RP}** geom: polygons of rootzone \
storage capacity [mm of water] for each catchment estimated before filling \
the missing with data from downstream catchments.
        * **meta_rootzone_storage_{forcing}_{RP}** map: rootzone storage capacity \
[mm of water] estimated from hydroclimatic data {forcing: obs, cc_hist or cc_fut} for \
different return periods RP. Only if rootzone_storage is set to True!

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_soilmaps`
        * :py:meth:`~WflowSbmModel.setup_laimaps` or equivalent


        Parameters
        ----------
        run_fn : str, Path, xr.Dataset
            Geodataset with streamflow timeseries (m3/s) per x,y location.
            The geodataset expects the coordinate names "index" (for each station id)
            and the variable name "discharge".
        forcing_obs_fn : str, Path, xr.Dataset
            Gridded timeseries with the observed forcing [mm/timestep].
            Expects to have variables "precip" and "pet".
        forcing_cc_hist_fn : str, Path, xr.Dataset, optional
            Gridded timeseries with the simulated historical forcing [mm/timestep],
            based on a climate model. Expects to have variables "precip" and "pet".
            The default is None.
        forcing_cc_fut_fn : str, optional
            Gridded timeseries with the simulated climate forcing [mm/timestep],
            based on a climate model. Expects to have variables "precip" and "pet".
            The default is None.
        chunksize : int, optional
            Chunksize on time dimension for processing data (not for saving to
            disk!). The default is 100.
        return_period : list, optional
            List with one or more values indicating the return period(s) (in
            years) for which the rootzone storage depth should be calculated. The
            default is [2,3,5,10,15,20,25,50,60,100] years.
        Imax : float, optional
            The maximum interception storage capacity [mm]. The default is 2.0 mm.
        start_hydro_year : str, optional
            The start month (abbreviated to the first three letters of the month,
            starting with a capital letter) of the hydrological year. The
            default is 'Sep'.
        start_field_capacity : str, optional
            The end of the wet season / commencement of dry season. This is the
            moment when the soil is at field capacity, i.e. there is no storage
            deficit yet. The default is 'Apr'.
        LAI : bool, optional
            Determine whether the leaf area index will be used to
            determine Imax. The default is False.
            If set to True, requires to have run setup_laimaps.
        rootzone_storage : bool, optional
            Determines whether the rootzone storage maps
            should be stored in the grid or not. The default is False.
        correct_cc_deficit : bool, optional
            Determines whether a bias-correction of the future deficit should be
            applied using the cc_hist deficit. Only works if the time periods of
            cc_hist and cc_fut are the same. If the climate change scenario and
            hist period are bias-corrected, this should probably set to False.
            The default is False.
        time_range: tuple, optional
            Select which time period to read from all the forcing files.
            There should be some overlap between the time period available in the
            forcing files for the historical period and in the observed streamflow data.
        missing_days_threshold: int, optional
            Minimum number of days within a year for that year to be counted in
            the long-term Budyko analysis.
        output_name_rootingdepth: str, optional
            Update the wflow_sbm model config of the vegetation_root_depth variable with
            the estimated vegetation_root_depth.
            The default is vegetation_root_depth_obs_20,
            which requires to have RP 20 in the list provided for \
the return_period argument.
        """
        logger.info("Preparing climate based root zone storage parameter maps.")
        # Open the data sets
        ds_obs = self.data_catalog.get_rasterdataset(
            forcing_obs_fn,
            geom=self.region,
            buffer=2,
            variables=["pet", "precip"],
            time_range=time_range,
        )
        ds_cc_hist = None
        if forcing_cc_hist_fn is not None:
            ds_cc_hist = self.data_catalog.get_rasterdataset(
                forcing_cc_hist_fn,
                geom=self.region,
                buffer=2,
                variables=["pet", "precip"],
                time_range=time_range,
            )
        ds_cc_fut = None
        if forcing_cc_fut_fn is not None:
            ds_cc_fut = self.data_catalog.get_rasterdataset(
                forcing_cc_fut_fn,
                geom=self.region,
                buffer=2,
                variables=["pet", "precip"],
                time_range=time_range_fut,
            )
        # observed streamflow data
        dsrun = self.data_catalog.get_geodataset(
            run_fn, single_var_as_array=False, time_range=time_range
        )

        # make sure dsrun overlaps with ds_obs, otherwise give error
        if dsrun.time[0] < ds_obs.time[0]:
            dsrun = dsrun.sel(time=slice(ds_obs.time[0], None))
        if dsrun.time[-1] > ds_obs.time[-1]:
            dsrun = dsrun.sel(time=slice(None, ds_obs.time[-1]))
        if len(dsrun.time) == 0:
            logger.error(
                "No overlapping period between the meteo and observed streamflow data"
            )

        # check if setup_soilmaps and setup_laimaps were run when:
        # if LAI == True and rooting_depth == True
        if (LAI == True) and (self._MAPS["LAI"] not in self.staticmaps.data):
            logger.error(
                "LAI variable not found in grid. \
Set LAI to False or run setup_laimaps first"
            )

        if (self._MAPS["theta_r"] not in self.staticmaps.data) or (
            self._MAPS["theta_s"] not in self.staticmaps.data
        ):
            logger.error(
                "theta_s or theta_r variables not found in grid. \
Run setup_soilmaps first"
            )

        # Run the rootzone clim workflow
        inv_rename = {
            v: k for k, v in self._MAPS.items() if v in self.staticmaps.data.data_vars
        }
        dsout, gdf = workflows.rootzoneclim(
            dsrun=dsrun,
            ds_obs=ds_obs,
            ds_like=self.staticmaps.data.rename(inv_rename),
            flwdir=self.flwdir,
            ds_cc_hist=ds_cc_hist,
            ds_cc_fut=ds_cc_fut,
            return_period=return_period,
            Imax=Imax,
            start_hydro_year=start_hydro_year,
            start_field_capacity=start_field_capacity,
            LAI=LAI,
            rootzone_storage=rootzone_storage,
            correct_cc_deficit=correct_cc_deficit,
            chunksize=chunksize,
            missing_days_threshold=missing_days_threshold,
        )

        # set nodata value outside basin
        dsout = dsout.where(self.staticmaps.data[self._MAPS["basins"]] > 0, -999)
        for var in dsout.data_vars:
            dsout[var].raster.set_nodata(-999)
        self.set_grid(dsout)
        self.set_geoms(gdf, name="rootzone_storage")

        # update config
        self.set_config("input.static.vegetation_root__depth", output_name_rootingdepth)

    @hydromt_step
    def setup_soilmaps(
        self,
        soil_fn: str = "soilgrids",
        ptf_ksatver: str = "brakensiek",
        wflow_thicknesslayers: list[int] = [100, 300, 800],
        output_names: dict = {
            "soil_water__saturated_volume_fraction": "soil_theta_s",
            "soil_water__residual_volume_fraction": "soil_theta_r",
            "soil_surface_water__vertical_saturated_hydraulic_conductivity": "soil_ksat_vertical",  # noqa: E501
            "soil__thickness": "soil_thickness",
            "soil_water__vertical_saturated_hydraulic_conductivity_scale_parameter": "soil_f",  # noqa: E501
            "soil_layer_water__brooks_corey_exponent": "soil_brooks_corey_c",
        },
    ):
        """
        Derive several (layered) soil parameters.

        Based on a database with physical soil properties using available point-scale
        (pedo)transfer functions (PTFs) from literature with upscaling rules to
        ensure flux matching across scales.

        Currently, supported ``soil_fn`` is "soilgrids" and "soilgrids_2020".
        ``ptf_ksatver`` (PTF for the vertical hydraulic conductivity) options are
        "brakensiek" and "cosby". "soilgrids" provides data at 7 specific depths,
        while "soilgrids_2020" provides data averaged over 6 depth intervals.
        This leads to small changes in the workflow:
        (1) M parameter uses midpoint depths in soilgrids_2020 versus \
specific depths in soilgrids,
        (2) weighted average of soil properties over soil thickness is done with \
the trapezoidal rule in soilgrids versus simple block weighted average in \
soilgrids_2020,
        (3) the soil_brooks_corey_c parameter is computed as weighted average over \
wflow_sbm soil layers defined in ``wflow_thicknesslayers``.

        The required data from soilgrids are soil bulk density 'bd_sl*' [g/cm3], \
clay content 'clyppt_sl*' [%], silt content 'sltppt_sl*' [%], organic carbon content \
'oc_sl*' [%], pH 'ph_sl*' [-], sand content 'sndppt_sl*' [%] and soil thickness \
'soilthickness' [cm].

        A ``soil_mapping_fn`` can optionnally be provided to derive parameters based
        on soil texture classes. A default table *soil_mapping_default* is available
        to derive the infiltration capacity of the soil.

        The following maps are added to grid:

        * **soil_theta_s** map:
            average saturated soil water content [m3/m3]
        * **soil_theta_r** map:
            average residual water content [m3/m3]
        * **soil_ksat_vertical** map:
            vertical saturated hydraulic conductivity at soil surface [mm/day]
        * **soil_thickness** map:
            soil thickness [mm]
        * **soil_f** map: scaling parameter controlling the decline of ksat_vertical \
[mm-1] (fitted with curve_fit (scipy.optimize)), bounds are checked
        * **soil_f_** map:
            scaling parameter controlling the decline of soil_ksat_vertical \
[mm-1] (fitted with numpy linalg regression), bounds are checked
        * **soil_brooks_corey_c_n** map:
            Brooks Corey coefficients [-] based on pore size distribution, \
a map for each of the wflow_sbm soil layers (n in total)
        * **meta_{soil_fn}_ksat_vertical_[z]cm** map: vertical hydraulic conductivity
            [mm/day] at soil depths [z] of ``soil_fn`` data
            [0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0]
        * **meta_soil_texture** map: soil texture based on USDA soil texture triangle \
(mapping: [1:Clay, 2:Silty Clay, 3:Silty Clay-Loam, 4:Sandy Clay, 5:Sandy Clay-Loam, \
6:Clay-Loam, 7:Silt, 8:Silt-Loam, 9:Loam, 10:Sand, 11: Loamy Sand, 12:Sandy Loam])


        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        soil_fn : {'soilgrids', 'soilgrids_2020'}
            Name of RasterDataset source for soil parameter maps, see
            data/data_sources.yml.
            Should contain info for the 7 soil depths of soilgrids
            (or 6 depths intervals for soilgrids_2020).
            * Required variables: \
'bd_sl*' [g/cm3], 'clyppt_sl*' [%], 'sltppt_sl*' [%], 'oc_sl*' [%], 'ph_sl*' [-], \
'sndppt_sl*' [%], 'soilthickness' [cm]
        ptf_ksatver : {'brakensiek', 'cosby'}
            Pedotransfer function (PTF) to use for calculation of ksat vertical
            (vertical saturated hydraulic conductivity [mm/day]).
            By default 'brakensiek'.
        wflow_thicknesslayers : list of int, optional
            Thickness of soil layers [mm] for wflow_sbm soil model.
            By default [100, 300, 800] for layers at depths 100, 400, 1200 and >1200 mm.
            Used only for Brooks Corey coefficients.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        """
        logger.info("Preparing soil parameter maps.")
        self._update_naming(output_names)
        # TODO add variables list with required variable names
        dsin = self.data_catalog.get_rasterdataset(soil_fn, geom=self.region, buffer=2)

        dsout = workflows.soilgrids(
            ds=dsin,
            ds_like=self.staticmaps.data,
            ptfKsatVer=ptf_ksatver,
            soil_fn=soil_fn,
            wflow_layers=wflow_thicknesslayers,
        ).reset_coords(drop=True)
        rmdict = {k: self._MAPS.get(k, k) for k in dsout.data_vars}
        self.set_grid(dsout.rename(rmdict))

        # Update the toml file
        self.set_config("model.soil_layer__thickness", wflow_thicknesslayers)
        self._update_config_variable_name(dsout.rename(rmdict).data_vars)

    @hydromt_step
    def setup_ksathorfrac(
        self,
        ksat_fn: str | xr.DataArray,
        variable: str | None = None,
        resampling_method: str = "average",
        output_name: str | None = None,
    ):
        """Set KsatHorFrac parameter values from a predetermined map.

        This predetermined map contains (preferably) 'calibrated' values of \
the KsatHorFrac parameter. This map is either selected from the wflow Deltares data \
or created by a third party/ individual.

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        ksat_fn : str, xr.DataArray
            The identifier of the KsatHorFrac dataset in the data catalog.
        variable : str, optional
            The variable name for the subsurface_ksat_horizontal_ratio map to
            use in ``ksat_fn`` in case ``ksat_fn`` contains several variables.
            By default None.
        resampling_method : str, optional
            The resampling method when up- or downscaled, by default "average"
        output_name : str, optional
            The name of the output map. If None (default), the name will be set
            to the name of the ksat_fn DataArray.
        """
        logger.info("Preparing KsatHorFrac parameter map.")
        wflow_var = "subsurface_water__horizontal_to_vertical_saturated_hydraulic_conductivity_ratio"  # noqa: E501
        dain = self.data_catalog.get_rasterdataset(
            ksat_fn,
            geom=self.region,
            buffer=2,
            variables=variable,
            single_var_as_array=True,
        )

        # Ensure its a DataArray
        if isinstance(dain, xr.Dataset):
            raise ValueError(
                "The ksat_fn data contains several variables. \
Select the variable to use for subsurface_ksat_horizontal_ratio \
using 'variable' argument."
            )

        # Create scaled subsurface_ksat_horizontal_ratio map
        daout = workflows.ksat_horizontal_ratio(
            dain,
            ds_like=self.staticmaps.data,
            resampling_method=resampling_method,
        )
        if output_name is not None:
            daout.name = output_name
        self._update_naming({wflow_var: daout.name})
        # Set the grid
        self.set_grid(daout)
        self._update_config_variable_name(daout.name)

    @hydromt_step
    def setup_ksatver_vegetation(
        self,
        soil_fn: str = "soilgrids",
        alfa: float = 4.5,
        beta: float = 5,
        output_name: str = "soil_ksat_vertical_vegetation",
    ):
        """Correct vertical saturated hydraulic conductivity with vegetation properties.

        This allows to account for biologically-promoted soil structure and \
        heterogeneities in natural landscapes based on the work of \
        Bonetti et al. (2021) https://www.nature.com/articles/s43247-021-00180-0.

        This method requires to have run setup_soilgrids and setup_lai first.

        The following map is added to grid:

        * **KsatVer_vegetation** map: saturated hydraulic conductivity considering \
        vegetation characteristics [mm/d]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_soilmaps`
        * :py:meth:`~WflowSbmModel.setup_laimaps` or equivalent

        Parameters
        ----------
        soil_fn : {'soilgrids', 'soilgrids_2020'}
            Name of RasterDataset source for soil parameter maps, see
            data/data_sources.yml.
            Should contain info for the sand percentage of the upper layer
            * Required variable: 'sndppt_sl1' [%]
        alfa : float, optional
            Shape parameter. The default is 4.5 when using LAI.
        beta : float, optional
            Shape parameter. The default is 5 when using LAI.
        output_name : dict, optional
            Name of the output map. By default 'KsatVer_vegetation'.
        """
        logger.info("Modifying ksat_vertical based on vegetation characteristics")
        wflow_var = self._WFLOW_NAMES[self._MAPS["ksat_vertical"]]

        # open soil dataset to get sand percentage
        sndppt = self.data_catalog.get_rasterdataset(
            soil_fn, geom=self.region, buffer=2, variables=["sndppt_sl1"]
        )

        # in ksatver_vegetation, ksat_vertical should be provided in mm/d
        inv_rename = {
            v: k for k, v in self._MAPS.items() if v in self.staticmaps.data.data_vars
        }
        ksatver_vegetation = workflows.ksatver_vegetation(
            ds_like=self.staticmaps.data.rename(inv_rename),
            sndppt=sndppt,
            alfa=alfa,
            beta=beta,
        )
        self._update_naming({wflow_var: output_name})
        # add to grid
        self.set_grid(ksatver_vegetation, output_name)
        # update config file
        self._update_config_variable_name(output_name)

    @hydromt_step
    def setup_allocation_areas(
        self,
        waterareas_fn: str | gpd.GeoDataFrame,
        priority_basins: bool = True,
        minimum_area: float = 50.0,
        output_name: str = "demand_allocation_area_id",
    ):
        """Create water demand allocation areas.

        The areas are based on the wflow model basins (at model resolution), the
        wflow model rivers and water areas or regions for allocation.

        Water regions are generally defined by sub-river-basins within a Country. In
        order to mimic reality, it is advisable to avoid cross-Country-border
        abstractions. Whenever information is available, it is strongly recommended to
        align the water regions with the actual areas managed by water management
        authorities, such as regional water boards.

        The allocation area will be an intersection of the wflow model basins and the
        water areas. For areas that do not contain river cells after intersection with
        the water areas, the priority_basins flag can be used to decide if these basins
        should be merged with the closest downstream basin or with any large enough
        basin in the same water area.

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_rivers`

        Parameters
        ----------
        waterareas_fn : str | gpd.GeoDataFrame
            Administrative boundaries GeoDataFrame data, this could be
            e.g. water management areas by water boards or the administrative
            boundaries of countries.
        priority_basins : bool, optional
            If True, merge the basins with the closest downstream basin, else merge
            with any large enough basin in the same water area, by default True.
        minimum_area : float
            Minimum area of the subbasins to keep in km2. Default is 50 km2.
        output_name : str, optional
            Name of the allocation areas map to be saved in the wflow model staticmaps
            and staticgeoms. Default is 'demand_allocation_area_id'.

        """
        logger.info("Preparing water demand allocation map.")
        self._update_naming({"land_water_allocation_area__count": output_name})
        # Read the data
        waterareas = self.data_catalog.get_geodataframe(
            waterareas_fn,
            geom=self.region,
        )

        # Create the allocation grid
        inv_rename = {v: k for k, v in self._MAPS.items() if v in self.staticmaps.data}
        da_alloc, gdf_alloc = workflows.demand.allocation_areas(
            ds_like=self.staticmaps.data.rename(inv_rename),
            waterareas=waterareas,
            basins=self.basins,
            priority_basins=priority_basins,
            minimum_area=minimum_area,
        )
        self.set_grid(da_alloc, name=output_name)
        # Update the config
        self.set_config("input.static.land_water_allocation_area__count", output_name)
        # Add alloc to geoms
        self.set_geoms(gdf_alloc, name=output_name)

    @hydromt_step
    def setup_allocation_surfacewaterfrac(
        self,
        gwfrac_fn: str | xr.DataArray,
        waterareas_fn: str | xr.DataArray | None = None,
        gwbodies_fn: str | xr.DataArray | None = None,
        ncfrac_fn: str | xr.DataArray | None = None,
        interpolate_nodata: bool = False,
        mask_and_scale_gwfrac: bool = True,
        output_name: str = "demand_surface_water_ratio",
    ):
        """Create the fraction of water allocated from surface water.

        This fraction entails the division of the water demand between surface water,
        ground water (aquifers) and non conventional sources (e.g. desalination plants).

        The surface water fraction is based on the raw groundwater fraction, if
        groundwater bodies are present (these are absent in e.g. mountainous regions),
        a fraction of water consumed that is obtained by non-conventional means and the
        water source areas.

        Non-conventional water could e.g. be water acquired by desalination of ocean or
        other brackish water.

        Adds model layer:

        * **demand_surface_water_ratio**: fraction of water allocated from surface water
          [0-1]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        gwfrac_fn : str | xr.DataArray
            The raw groundwater fraction per grid cell. The values of these cells need
            to be between 0 and 1.
        waterareas_fn : str| xr.DataArray
            The areas over which the water has to be distributed. This may either be
            a global (or more local map). If not provided, the source areas created by
            the `setup_allocation_areas` will be used.
        gwbodies_fn : str | xr.DataArray | None
            The presence of groundwater bodies per grid cell. The values are ought to
            be binary (either 0 or 1). If they are not provided, we assume groundwater
            bodies are present where gwfrac is more than 0.
        ncfrac_fn : str | xr.DataArray | None
            The non-conventional fraction. Same types of values apply as for
            `gwfrac_fn`. If not provided, we assume no non-conventional sources are
            used.
        interpolate_nodata : bool, optional
            If True, nodata values in the resulting demand_surface_water_ratio map will
            be linearly
            interpolated. Else a default value of 1 will be used for nodata values
            (default).
        mask_and_scale_gwfrac : bool, optional
            If True, gwfrac will be masked for areas with no groundwater bodies. To keep
            the average gwfrac used over waterareas similar after the masking, gwfrac
            for areas with groundwater bodies can increase. If False, gwfrac will be
            used as is. By default True.
        output_name : str, optional
            Name of the fraction of surface water used map to be saved in the wflow
            model staticmaps file. Default is 'demand_surface_water_ratio'.
        """
        logger.info("Preparing surface water fraction map.")
        # Load the data
        gwfrac_raw = self.data_catalog.get_rasterdataset(
            gwfrac_fn,
            geom=self.region,
            buffer=2,
        )
        if gwbodies_fn is not None:
            gwbodies = self.data_catalog.get_rasterdataset(
                gwbodies_fn,
                geom=self.region,
                buffer=2,
            )
        else:
            gwbodies = None
        if ncfrac_fn is not None:
            ncfrac = self.data_catalog.get_rasterdataset(
                ncfrac_fn,
                geom=self.region,
                buffer=2,
            )
        else:
            ncfrac = None

        # check whether to use the models own allocation areas
        if waterareas_fn is None:
            logger.info("Using wflow model allocation areas.")
            if self._MAPS["allocation_areas"] not in self.staticmaps.data:
                logger.error(
                    "No allocation areas found. Run setup_allocation_areas first "
                    "or provide a waterareas_fn."
                )
                return
            waterareas = self.staticmaps.data[self._MAPS["allocation_areas"]]
        else:
            waterareas = self.data_catalog.get_rasterdataset(
                waterareas_fn,
                geom=self.region,
                buffer=2,
            )

        # Call the workflow
        w_frac = workflows.demand.surfacewaterfrac_used(
            gwfrac_raw=gwfrac_raw,
            da_like=self.staticmaps.data[self._MAPS["elevtn"]],
            waterareas=waterareas,
            gwbodies=gwbodies,
            ncfrac=ncfrac,
            interpolate=interpolate_nodata,
            mask_and_scale_gwfrac=mask_and_scale_gwfrac,
        )

        # Update the settings toml
        wflow_var = "land_surface_water__withdrawal_fraction"
        self._update_naming({wflow_var: output_name})
        self.set_config(f"input.static.{wflow_var}", output_name)

        # Set the dataarray to the wflow grid
        self.set_grid(w_frac, name=output_name)

    @hydromt_step
    def setup_domestic_demand(
        self,
        domestic_fn: str | xr.Dataset,
        population_fn: str | xr.Dataset | None = None,
        domestic_fn_original_res: float | None = None,
        output_names: dict = {
            "domestic__gross_water_demand_volume_flux": "demand_domestic_gross",
            "domestic__net_water_demand_volume_flux": "demand_domestic_net",
        },
    ):
        """
        Prepare domestic water demand maps from a raster dataset.

        Both gross and netto domestic demand should be provided in `domestic_fn`. They
        can either be cyclic or non-cyclic.

        To improve accuracy, the domestic demand can be downsampled based on a provided
        population dataset. If the data you are using was already downscaled using a
        different source for population data, you may decide to first resample to the
        original resolution of `domestic_fn` before downsampling with `population_fn`.
        For example, the pcr_globwb dataset is at a resolution of 0.0083333333 degrees,
        while the original data has a resolution of 0.5 degrees. Use the
        `domestic_fn_original_res` parameter to specify the original resolution.

        Adds model layer:

        * **demand_domestic_gross**: gross domestic water demand [mm/day]
        * **demand_domestic_net**: net domestic water demand [mm/day]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        domestic_fn : str | xr.Dataset
            The domestic dataset. This can either be the dataset directly (xr.Dataset),
            a string referring to an entry in the data catalog or a dictionary
            containing the name of the dataset (keyword: `source`) and any optional
            keyword arguments (e.g. `version`). The data can be cyclic
            (with a `time` dimension) or non-cyclic. Allowed cyclic data can be monthly
            (12) or dayofyear (365 or 366).

            * Required variables: 'domestic_gross' [mm/day], 'domestic_net' [mm/day]
        population_fn : str | xr.Dataset
            The population dataset in capita. Either provided as a dataset directly or
            as a string referring to an entry in the data catalog.
        domestic_fn_original_res : Optional[float], optional
            The original resolution of the domestic dataset, by default None to skip
            upscaling before downsampling with population.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        """
        logger.info("Preparing domestic demand maps.")
        self._update_naming(output_names)
        # Set flag for cyclic data
        _cyclic = False

        # Read data
        domestic_raw = self.data_catalog.get_rasterdataset(
            domestic_fn,
            geom=self.region,
            buffer=2,
            variables=["domestic_gross", "domestic_net"],
        )
        # Increase the buffer if original resolution is provided
        if domestic_fn_original_res is not None:
            buffer = np.ceil(domestic_fn_original_res / abs(domestic_raw.raster.res[0]))
            domestic_raw = self.data_catalog.get_rasterdataset(
                domestic_fn,
                geom=self.region,
                buffer=buffer,
                variables=["domestic_gross", "domestic_net"],
            )
        # Check if data is time dependent
        if "time" in domestic_raw.coords:
            # Check that this is indeed cyclic data
            if len(domestic_raw.time) in [12, 365, 366]:
                _cyclic = True
                domestic_raw["time"] = domestic_raw.time.astype("int32")
            else:
                raise ValueError(
                    "The provided domestic demand data is cyclic but the time "
                    "dimension does not match the expected length of 12, 365 or 366."
                )

        # Get population data
        pop_raw = None
        if population_fn is not None:
            pop_raw = self.data_catalog.get_rasterdataset(
                population_fn,
                bbox=domestic_raw.raster.bounds,
                buffer=2,
            )

        # Compute domestic demand
        domestic, pop = workflows.demand.domestic(
            domestic_raw,
            ds_like=self.staticmaps.data,
            popu=pop_raw,
            original_res=domestic_fn_original_res,
        )
        # Add to grid
        rmdict = {k: self._MAPS.get(k, k) for k in domestic.data_vars}
        self.set_grid(domestic.rename(rmdict))
        if population_fn is not None:
            self.set_grid(pop, name="meta_population")

        # Update toml
        self.set_config("model.water_demand.domestic__flag", True)
        data_type = "cyclic" if _cyclic else "static"
        self._update_config_variable_name(domestic.rename(rmdict).data_vars, data_type)

    @hydromt_step
    def setup_domestic_demand_from_population(
        self,
        population_fn: str | xr.Dataset,
        domestic_gross_per_capita: float | list[float],
        domestic_net_per_capita: float | list[float] | None = None,
        output_names: dict = {
            "domestic__gross_water_demand_volume_flux": "demand_domestic_gross",
            "domestic__net_water_demand_volume_flux": "demand_domestic_net",
        },
    ):
        """
        Prepare domestic water demand maps from statistics per capita.

        Gross and net demands per capita can be provide as cyclic (list) or non-cyclic
        (constant). The statistics are then multiplied by the population dataset to
        derive the gross and net domestic demand.

        Adds model layer:

        * **demand_domestic_gross**: gross domestic water demand [mm/day]
        * **demand_domestic_net**: net domestic water demand [mm/day]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        population_fn : str | xr.Dataset
            The (gridded) population dataset in capita. Either provided as a dataset
            directly or as a string referring to an entry in the data catalog.
        domestic_gross_per_capita : float | list[float]
            The gross domestic water demand per capita [m3/day]. If cyclic, provide a
            list with 12 values for monthly data or 365/366 values for daily data.
        domestic_net_per_capita : float | list[float] | None
            The net domestic water demand per capita [m3/day]. If cyclic, provide a
            list with 12 values for monthly data or 365/366 values for daily data. If
            not provided, the gross demand will be used as net demand.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        """
        logger.info("Preparing domestic demand maps based on population.")

        # Set flag for cyclic data
        _cyclic = False
        self._update_naming(output_names)

        # Check if data is time dependent
        time_length = len(np.atleast_1d(domestic_gross_per_capita))
        if time_length in [12, 365, 366]:
            _cyclic = True
        elif time_length > 1:
            raise ValueError(
                "The provided domestic demand data is cyclic but the length "
                f"({time_length})does not match the expected length of 12, 365 or 366."
            )
        if domestic_net_per_capita is None:
            domestic_net_per_capita = domestic_gross_per_capita
            logger.info("Net domestic demand not provided, using gross demand.")

        # Get population data
        popu = self.data_catalog.get_rasterdataset(
            population_fn,
            bbox=self.staticmaps.data.raster.bounds,
            buffer=1000,
        )

        # Compute domestic demand
        domestic, popu_scaled = workflows.demand.domestic_from_population(
            popu,
            ds_like=self.staticmaps.data,
            gross_per_capita=domestic_gross_per_capita,
            net_per_capita=domestic_net_per_capita,
        )

        # Add to grid
        rmdict = {k: self._MAPS.get(k, k) for k in domestic.data_vars}
        self.set_grid(domestic.rename(rmdict))
        if population_fn is not None:
            self.set_grid(popu_scaled, name="meta_population")

        # Update toml
        self.set_config("model.water_demand.domestic__flag", True)
        data_type = "cyclic" if _cyclic else "static"
        self._update_config_variable_name(domestic.rename(rmdict).data_vars, data_type)

    @hydromt_step
    def setup_other_demand(
        self,
        demand_fn: str | dict[str, dict[str, Any]] | xr.Dataset,
        variables: list = [
            "industry_gross",
            "industry_net",
            "livestock_gross",
            "livestock_net",
        ],
        resampling_method: str = "average",
        output_names: dict = {
            "industry__gross_water_demand_volume_flux": "demand_industry_gross",
            "industry__net_water_demand_volume_flux": "demand_industry_net",
            "livestock__gross_water_demand_volume_flux": "demand_livestock_gross",
            "livestock__net_water_demand_volume_flux": "demand_livestock_net",
        },
    ):
        """Create water demand maps from other sources (e.g. industry, livestock).

        These maps are created from a supplied dataset that either contains one
        or all of the following variables:
        - `Industrial` water demand
        - `Livestock` water demand
        - `Domestic` water demand (without population downsampling)

        For each of these datasets/ variables a gross and a netto water demand
        should be provided. They can either be provided cyclic or non-cyclic. The
        maps are then resampled to the model resolution using the provided
        `resampling_method`.

        Adds model layer:

        * **{var}_gross**: gross water demand [mm/day]
        * **{var}_net**: net water demand [mm/day]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        demand_fn : str | dict[str, dict[str, Any]], xr.Dataset]
            The water demand dataset. This can either be the dataset directly
            (xr.Dataset), a string referring to an entry in the data catalog or
            a dictionary containing the name of the dataset (keyword: `source`) and
            any optional keyword arguments (e.g. `version`). The data can be cyclic
            (with a `time` dimension) or non-cyclic. Allowed cyclic data can be monthly
            (12) or dayofyear (365 or 366).

            * Required variables: variables listed in `variables` in [mm/day].
        variables : list, optional
            The variables to be processed. Supported variables are ['domestic_gross',
            'domestic_net', 'industry_gross', 'industry_net', 'livestock_gross',
            'livestock_net']. By default gross and net demand for industry and livestock
            are processed.
        resampling_method : str, optional
            Resampling method for the demand maps, by default "average"
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        """
        logger.info(f"Preparing water demand maps for {variables}.")
        # Set flag for cyclic data
        _cyclic = False
        self._update_naming(output_names)

        # Selecting data
        demand_raw = self.data_catalog.get_rasterdataset(
            demand_fn,
            geom=self.region,
            buffer=2,
            variables=variables,
        )
        if "time" in demand_raw.coords:
            # Check that this is indeed cyclic data
            if len(demand_raw.time) in [12, 365, 366]:
                _cyclic = True
                demand_raw["time"] = demand_raw.time.astype("int32")
            else:
                raise ValueError(
                    "The provided demand data is cyclic but the time dimension does "
                    "not match the expected length of 12, 365 or 366."
                )

        # Create static water demand rasters
        demand = workflows.demand.other_demand(
            demand_raw,
            ds_like=self.staticmaps.data,
            ds_method=resampling_method,
        )
        rmdict = {k: self._MAPS.get(k, k) for k in demand.data_vars}
        self.set_grid(demand.rename(rmdict))

        # Update the settings toml
        if "domestic_gross" in demand.data_vars:
            self.set_config("model.water_demand.domestic__flag", True)
        if "industry_gross" in demand.data_vars:
            self.set_config("model.water_demand.industry__flag", True)
        if "livestock_gross" in demand.data_vars:
            self.set_config("model.water_demand.livestock__flag", True)
        data_type = "cyclic" if _cyclic else "static"
        self._update_config_variable_name(demand.rename(rmdict).data_vars, data_type)

    @hydromt_step
    def setup_irrigation(
        self,
        irrigated_area_fn: str | Path | xr.DataArray,
        irrigation_value: list[int],
        cropland_class: list[int],
        paddy_class: list[int] = [],
        area_threshold: float = 0.6,
        lai_threshold: float = 0.2,
        lulcmap_name: str = "meta_landuse",
        output_names: dict = {
            "irrigated_paddy_area__count": "demand_paddy_irrigated_mask",
            "land~irrigated-non-paddy_area__count": "demand_nonpaddy_irrigated_mask",
            "irrigated_paddy__irrigation_trigger_flag": "demand_paddy_irrigation_trigger",  # noqa: E501
            "irrigated_non_paddy__irrigation_trigger_flag": "demand_nonpaddy_irrigation_trigger",  # noqa: E501
        },
    ):
        """
        Add required information to simulate irrigation water demand from grid.

        THIS FUNCTION SHOULD BE RUN AFTER LANDUSE AND LAI MAPS ARE CREATED.

        The function requires data that contains information about the location of the
        irrigated areas (``irrigated_area_fn``). This, combined with the wflow landuse
        map that contains classes for cropland (``cropland_class``) and optionally for
        paddy (rice) (``paddy_class``), determines which locations are considered to be
        paddy irrigation, and which locations are considered to be non-paddy irrigation.

        Next, the irrigated area map is reprojected to the model resolution, where a
        threshold (``area_threshold``) determines when pixels are considered to be
        classified as irrigation or rainfed cells (both paddy and non-paddy). It adds
        the resulting maps to the input data.

        To determine when irrigation is allowed to occur, an irrigation trigger map is
        defined. This is a cyclic map, that defines (with a mask) when irrigation is
        expected to occur. This is done based on the Leaf Area Index (LAI), that is
        already present in the wflow model configuration. We follow the procedure
        described by Peano et al. (2019). They describe a threshold value based on the
        LAI variability to determine the growing season. This threshold is defined as
        20% (default value) of the LAI variability, but can be adjusted via the
        ``lai_threshold`` argument.

        Adds model layers:

        * **demand_nonpaddy_irrigated_mask**: Irrigated (non-paddy) mask [-]
        * **demand_paddy_irrigated_mask**: Irrigated (paddy) mask [-]
        * **demand_paddy_irrigation_trigger**: Map with monthly values, indicating
          whether irrigation is allowed (1) or not (0) [-] for paddy areas
        * **demand_paddy_irrigation_trigger**: Map with monthly values, indicating
          whether irrigation is allowed (1) or not (0) [-] for non-paddy areas

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_lulcmaps` or equivalent
        * :py:meth:`~WflowSbmModel.setup_laimaps` or equivalent

        Parameters
        ----------
        irrigated_area_fn: str, Path, xarray.DataArray
            Name of the (gridded) dataset that contains the location of irrigated areas.
        irrigation_value: list
            List of values that are considered to be irrigated areas in
            ``irrigated_area_fn``.
        cropland_class: list
            List of values that are considered to be cropland in the wflow landuse data.
        paddy_class: int
            Class in the wflow landuse data that is considered as paddy or rice. Leave
            empty if not present (default).
        area_threshold: float
            Fractional area of a (wflow) pixel before it gets classified as an irrigated
            pixel, by default 0.6
        lai_threshold: float
            Value of LAI variability to be used to determine the irrigation trigger. By
            default 0.2.
        lulcmap_name: str
            Name of the landuse map layer in the wflow model staticmaps. By default
            'meta_landuse'. Please update if your landuse map has a different name
            (eg 'landuse_globcover').
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.

        See Also
        --------
        workflows.demand.irrigation

        References
        ----------
        Peano, D., Materia, S., Collalti, A., Alessandri, A., Anav, A., Bombelli, A., &
        Gualdi, S. (2019). Global variability of simulated and observed vegetation
        growing season. Journal of Geophysical Research: Biogeosciences, 124, 3569–3587.
        https://doi.org/10.1029/2018JG004881
        """
        logger.info("Preparing irrigation maps.")
        if lulcmap_name in self.staticmaps.data:
            # update the internal mapping
            self._MAPS["landuse"] = lulcmap_name
        else:
            raise ValueError(
                f"Landuse map {lulcmap_name} not found in the model grid. Please "
                "provide a valid landuse map name or run setup_lulcmaps."
            )

        # Extract irrigated area dataset
        irrigated_area = self.data_catalog.get_rasterdataset(
            irrigated_area_fn, bbox=self.staticmaps.data.raster.bounds, buffer=3
        )

        # Get irrigation areas for paddy, non paddy and irrigation trigger
        inv_rename = {v: k for k, v in self._MAPS.items() if v in self.staticmaps.data}
        ds_irrigation = workflows.demand.irrigation(
            da_irrigation=irrigated_area,
            ds_like=self.staticmaps.data.rename(inv_rename),
            irrigation_value=irrigation_value,
            cropland_class=cropland_class,
            paddy_class=paddy_class,
            area_threshold=area_threshold,
            lai_threshold=lai_threshold,
        )

        # Check if paddy and non paddy are present
        cyclic_lai = len(self.staticmaps.data[self._MAPS["LAI"]].dims) > 2
        if (
            "demand_paddy_irrigated_mask" in ds_irrigation.data_vars
            and ds_irrigation["demand_paddy_irrigated_mask"]
            .raster.mask_nodata()
            .sum()
            .values
            != 0
        ):
            # Select the paddy variables in output_names
            paddy_names = {
                k: v for k, v in output_names.items() if "irrigated-paddy" in k
            }
            self._update_naming(paddy_names)
            ds_paddy = ds_irrigation[
                ["demand_paddy_irrigated_mask", "demand_paddy_irrigation_trigger"]
            ]
            rmdict = {k: self._MAPS.get(k, k) for k in ds_paddy.data_vars}
            self.set_grid(ds_paddy.rename(rmdict))
            self.set_config("model.water_demand.paddy__flag", True)
            self._update_config_variable_name(
                self._MAPS.get(
                    "demand_paddy_irrigated_mask", "demand_paddy_irrigated_mask"
                ),
                "static",
            )
            data_type = "cyclic" if cyclic_lai else "static"
            self._update_config_variable_name(
                self._MAPS.get(
                    "demand_paddy_irrigation_trigger", "demand_paddy_irrigation_trigger"
                ),
                data_type,
            )
        else:
            self.set_config("model.water_demand.paddy__flag", False)

        if (
            ds_irrigation["demand_nonpaddy_irrigated_mask"]
            .raster.mask_nodata()
            .sum()
            .values
            != 0
        ):
            nonpaddy_names = {
                k: v for k, v in output_names.items() if "irrigated-non-paddy" in k
            }
            self._update_naming(nonpaddy_names)
            ds_nonpaddy = ds_irrigation[
                ["demand_nonpaddy_irrigated_mask", "demand_nonpaddy_irrigation_trigger"]
            ]
            rmdict = {k: self._MAPS.get(k, k) for k in ds_nonpaddy.data_vars}
            self.set_grid(ds_nonpaddy.rename(rmdict))
            # Update the config
            self.set_config("model.water_demand.nonpaddy__flag", True)
            self._update_config_variable_name(
                self._MAPS.get(
                    "demand_nonpaddy_irrigated_mask", "demand_nonpaddy_irrigated_mask"
                ),
                "static",
            )
            data_type = "cyclic" if cyclic_lai else "static"
            self._update_config_variable_name(
                self._MAPS.get(
                    "demand_nonpaddy_irrigation_trigger",
                    "demand_nonpaddy_irrigation_trigger",
                ),
                data_type,
            )
        else:
            self.set_config("model.water_demand.nonpaddy__flag", False)

    @hydromt_step
    def setup_irrigation_from_vector(
        self,
        irrigated_area_fn: str | Path | gpd.GeoDataFrame,
        cropland_class: list[int],
        paddy_class: list[int] = [],
        area_threshold: float = 0.6,
        lai_threshold: float = 0.2,
        output_names: dict = {
            "irrigated_paddy_area__count": "demand_paddy_irrigated_mask",
            "land~irrigated-non-paddy_area__count": "demand_nonpaddy_irrigated_mask",
            "irrigated_paddy__irrigation_trigger_flag": "demand_paddy_irrigation_trigger",  # noqa: E501
            "irrigated_non_paddy__irrigation_trigger_flag": "demand_nonpaddy_irrigation_trigger",  # noqa: E501
        },
    ):
        """
        Add required information to simulate irrigation water demand from vector.

        THIS FUNCTION SHOULD BE RUN AFTER LANDUSE AND LAI MAPS ARE CREATED.

        The function requires data that contains information about the location of the
        irrigated areas (``irrigated_area_fn``). This, combined with the wflow landuse
        map that contains classes for cropland (``cropland_class``) and optionally for
        paddy (rice) (``paddy_class``), determines which locations are considered to be
        paddy irrigation, and which locations are considered to be non-paddy irrigation.

        Next, the irrigated area geometries are rasterized, where a threshold
        (``area_threshold``) determines when pixels are considered to be
        classified as irrigation or rainfed cells (both paddy and non-paddy). It adds
        the resulting maps to the input data.

        To determine when irrigation is allowed to occur, an irrigation trigger map is
        defined. This is a cyclic map, that defines (with a mask) when irrigation is
        expected to occur. This is done based on the Leaf Area Index (LAI), that is
        already present in the wflow model configuration. We follow the procedure
        described by Peano et al. (2019). They describe a threshold value based on the
        LAI variability to determine the growing season. This threshold is defined as
        20% (default value) of the LAI variability, but can be adjusted via the
        ``lai_threshold`` argument.

        Adds model layers:

        * **demand_paddy_irrigated_mask**: Irrigated (paddy) mask [-]
        * **demand_nonpaddy_irrigated_mask**: Irrigated (non-paddy) mask [-]
        * **demand_paddy_irrigation_trigger**: Map with monthly values, indicating
          whether irrigation is allowed (1) or not (0) [-] for paddy areas
        * **demand_nonpaddy_irrigation_trigger**: Map with monthly values, indicating
          whether irrigation is allowed (1) or not (0) [-] for non-paddy areas

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_lulcmaps` or equivalent
        * :py:meth:`~WflowSbmModel.setup_laimaps` or equivalent

        Parameters
        ----------
        irrigated_area_fn: str, Path, geopandas.GeoDataFrame
            Name of the (vector) dataset that contains the location of irrigated areas.
        cropland_class: list
            List of values that are considered to be cropland in the wflow landuse data.
        paddy_class: int
            Class in the wflow landuse data that is considered as paddy or rice. Leave
            empty if not present (default).
        area_threshold: float
            Fractional area of a (wflow) pixel before it gets classified as an irrigated
            pixel, by default 0.6
        lai_threshold: float
            Value of LAI variability to be used to determine the irrigation trigger. By
            default 0.2.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.

        See Also
        --------
        workflows.demand.irrigation

        References
        ----------
        Peano, D., Materia, S., Collalti, A., Alessandri, A., Anav, A., Bombelli, A., &
        Gualdi, S. (2019). Global variability of simulated and observed vegetation
        growing season. Journal of Geophysical Research: Biogeosciences, 124, 3569–3587.
        https://doi.org/10.1029/2018JG004881
        """
        logger.info("Preparing irrigation maps.")

        # Extract irrigated area dataset
        irrigated_area = self.data_catalog.get_geodataframe(
            irrigated_area_fn,
            bbox=self.staticmaps.data.raster.bounds,
            buffer=1000,
            predicate="intersects",
            handle_nodata=NoDataStrategy.IGNORE,
        ).copy()  # Ensure we have a copy to resolve SettingWithCopyWarning

        # Check if the geodataframe is empty
        if irrigated_area is None or irrigated_area.empty:
            logger.info("No irrigated areas found in the provided geodataframe.")
            return

        # Get irrigation areas for paddy, non paddy and irrigation trigger
        inv_rename = {v: k for k, v in self._MAPS.items() if v in self.staticmaps.data}
        ds_irrigation = workflows.demand.irrigation_from_vector(
            gdf_irrigation=irrigated_area,
            ds_like=self.staticmaps.data.rename(inv_rename),
            cropland_class=cropland_class,
            paddy_class=paddy_class,
            area_threshold=area_threshold,
            lai_threshold=lai_threshold,
        )

        # Check if paddy and non paddy are present
        cyclic_lai = len(self.staticmaps.data[self._MAPS["LAI"]].dims) > 2
        if (
            "demand_paddy_irrigated_mask" in ds_irrigation.data_vars
            and ds_irrigation["demand_paddy_irrigated_mask"]
            .raster.mask_nodata()
            .sum()
            .values
            != 0
        ):
            paddy_names = {
                k: v for k, v in output_names.items() if "irrigated-paddy" in k
            }
            self._update_naming(paddy_names)
            ds_paddy = ds_irrigation[
                ["demand_paddy_irrigated_mask", "demand_paddy_irrigation_trigger"]
            ]
            rmdict = {k: self._MAPS.get(k, k) for k in ds_paddy.data_vars}
            self.set_grid(ds_paddy.rename(rmdict))
            # Update the config
            self.set_config("model.water_demand.paddy__flag", True)
            self._update_config_variable_name(
                self._MAPS.get(
                    "demand_paddy_irrigated_mask", "demand_paddy_irrigated_mask"
                ),
                "static",
            )
            data_type = "cyclic" if cyclic_lai else "static"
            self._update_config_variable_name(
                self._MAPS.get(
                    "demand_paddy_irrigation_trigger", "demand_paddy_irrigation_trigger"
                ),
                data_type,
            )
        else:
            self.set_config("model.water_demand.paddy__flag", False)

        if (
            ds_irrigation["demand_nonpaddy_irrigated_mask"]
            .raster.mask_nodata()
            .sum()
            .values
            != 0
        ):
            nonpaddy_names = {
                k: v for k, v in output_names.items() if "irrigated-non-paddy" in k
            }
            self._update_naming(nonpaddy_names)
            ds_nonpaddy = ds_irrigation[
                ["demand_nonpaddy_irrigated_mask", "demand_nonpaddy_irrigation_trigger"]
            ]
            rmdict = {k: self._MAPS.get(k, k) for k in ds_nonpaddy.data_vars}
            self.set_grid(ds_nonpaddy.rename(rmdict))
            # Update the config
            self.set_config("model.water_demand.nonpaddy__flag", True)
            self._update_config_variable_name(
                self._MAPS.get(
                    "demand_nonpaddy_irrigated_mask", "demand_nonpaddy_irrigated_mask"
                ),
                "static",
            )
            data_type = "cyclic" if cyclic_lai else "static"
            self._update_config_variable_name(
                self._MAPS.get(
                    "demand_nonpaddy_irrigation_trigger",
                    "demand_nonpaddy_irrigation_trigger",
                ),
                data_type,
            )
        else:
            self.set_config("model.water_demand.nonpaddy__flag", False)

    @hydromt_step
    def setup_1dmodel_connection(
        self,
        river1d_fn: str | Path | gpd.GeoDataFrame,
        connection_method: str = "subbasin_area",
        area_max: float = 30.0,
        basin_buffer_cells: int = 0,
        geom_snapping_tolerance: float = 0.001,
        add_tributaries: bool = True,
        include_river_boundaries: bool = True,
        mapname: str = "1dmodel",
        update_toml: bool = True,
        toml_output: str = "netcdf_scalar",
        **kwargs,
    ):
        """
        Connect wflow to a 1D model by deriving linked subcatch (and tributaries).

        There are two methods to connect models:

        - `subbasin_area`:
            creates subcatchments linked to the 1d river based
            on an area threshold (area_max) for the subbasin size. With this method,
            if a tributary is larger than the `area_max`, it will be connected to
            the 1d river directly.
        - `nodes`:
            subcatchments are derived based on the 1driver nodes (used as
            gauges locations). With this method, large tributaries can also be derived
            separately using the `add_tributaries` option and adding a `area_max`
            threshold for the tributaries.

        If `add_tributary` option is on, you can decide to include or exclude the
        upstream boundary of the 1d river as an additional tributary using the
        `include_river_boundaries` option.

        River edges or river nodes are snapped to the closest downstream wflow river
        cell using the :py:meth:`hydromt.flw.gauge_map` method.

        Optionally, the toml file can also be updated to save lateral.river.inwater to
        save all river inflows for the subcatchments and lateral.river.q_av for the
        tributaries using
        :py:meth:`hydromt_wflow.wflow_base.setup_config_output_timeseries`.

        Note that if the method is `subbasin_area`, only connecting one continuous
        river1d is supported. To connect several riv1d networks to the same wflow model,
        the method can be run several times.

        Adds model layer:

        * **subcatchment_{mapname}** map/geom:  connection subbasins between
          wflow and the 1D model.
        * **subcatchment_river_{mapname}** map/geom:  connection subbasins between
          wflow and the 1D model for river cells only.
        * **gauges_{mapname}** map/geom, optional: outlets of the tributaries
          flowing into the 1D model.

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_rivers`

        Parameters
        ----------
        river1d_fn : str, Path, gpd.GeoDataFrame
            GeodataFrame with the 1D model river network and nodes where to derive
            subbasins for connection_method **nodes**.
        connection_method : str, default subbasin_area
            Method to connect wflow to the 1D model. Available methods are {
                'subbasin_area', 'nodes'}.
        area_max : float, default 10.0
            Maximum area [km2] of the subbasins to connect to the 1D model in km2 with
            connection_method **subbasin_area** or **nodes** with add_tributaries
            set to True.
        basin_buffer_cells : int, default 0
            Number of cells to use when clipping the river1d geometry to the basin
            extent. This can be used to not include river geometries near the basin
            border.
        geom_snapping_tolerance : float, default 0.1
            Distance used to determine whether to snap parts of the river1d geometry
            that are close to each other. This can be useful if some of the tributaries
            of the 1D river are not perfectly connected to the main river.
        add_tributaries : bool, default True
            If True, derive tributaries for the subbasins larger than area_max. Always
            True for **subbasin_area** method.
        include_river_boundaries : bool, default True
            If True, include the upstream boundary(ies) of the 1d river as an
            additional tributary(ies).
        mapname : str, default 1dmodel
            Name of the map to save the subcatchments and tributaries in the wflow model
            staticmaps and geoms (subcatchment_{mapname}).
        update_toml : bool, default True
            If True, updates the wflow configuration file to save the required outputs
            for the 1D model.
        toml_output : str, optional
            One of ['csv', 'netcdf_scalar', None] to update [output.csv] or
            [output.netcdf_scalar] section of wflow toml file or do nothing. By
            default, 'netcdf_scalar'.
        **kwargs
            Additional keyword arguments passed to the snapping method
            hydromt.flw.gauge_map. See its documentation for more information.

        See Also
        --------
        hydromt.flw.gauge_map
        hydromt_wflow.workflows.wflow_1dmodel_connection
        """
        # Check connection method values
        if connection_method not in ["subbasin_area", "nodes"]:
            raise ValueError(
                f"Unknown connection method {connection_method},"
                "select from ['subbasin_area', 'nodes']"
            )
        # read 1d model river network
        gdf_riv = self.data_catalog.get_geodataframe(
            river1d_fn,
            geom=self.region,
            buffer=2,
        )

        # derive subcatchments and tributaries
        inv_rename = {v: k for k, v in self._MAPS.items() if v in self.staticmaps.data}
        ds_out = workflows.wflow_1dmodel_connection(
            gdf_riv,
            ds_model=self.staticmaps.data.rename(inv_rename),
            connection_method=connection_method,
            area_max=area_max,
            basin_buffer_cells=basin_buffer_cells,
            geom_snapping_tolerance=geom_snapping_tolerance,
            add_tributaries=add_tributaries,
            include_river_boundaries=include_river_boundaries,
            **kwargs,
        )

        # Derive tributary gauge map
        if "gauges" in ds_out.data_vars:
            self.set_grid(ds_out["gauges"], name=f"gauges_{mapname}")
            # Derive the gauges staticgeoms
            gdf_tributary = ds_out["gauges"].raster.vectorize()
            centroid = utils.planar_operation_in_utm(
                gdf_tributary["geometry"], lambda geom: geom.centroid
            )
            gdf_tributary["geometry"] = centroid
            gdf_tributary["value"] = gdf_tributary["value"].astype(
                ds_out["gauges"].dtype
            )
            self.set_geoms(gdf_tributary, name=f"gauges_{mapname}")

            # Add a check that all gauges are on the river
            if (
                self.staticmaps.data[self._MAPS["rivmsk"]].raster.sample(gdf_tributary)
                == self.staticmaps.data[self._MAPS["rivmsk"]].raster.nodata
            ).any():
                river_upa = self.staticmaps.data[self._MAPS["rivmsk"]].attrs.get(
                    "river_upa", ""
                )
                logger.warning(
                    "Not all tributary gauges are on the river network and river "
                    "discharge cannot be saved. You should use a higher threshold "
                    f"for the subbasin area than {area_max} to match better the "
                    f"wflow river in your model {river_upa}."
                )
                all_gauges_on_river = False
            else:
                all_gauges_on_river = True

            # Update toml
            if update_toml and all_gauges_on_river:
                self.setup_config_output_timeseries(
                    mapname=f"gauges_{mapname}",
                    toml_output=toml_output,
                    header=["Q"],
                    param=["river_water__volume_flow_rate"],
                    reducer=None,
                )

        # Derive subcatchment map
        self.set_grid(ds_out["subcatch"], name=f"subcatchment_{mapname}")
        gdf_subcatch = ds_out["subcatch"].raster.vectorize()
        gdf_subcatch["value"] = gdf_subcatch["value"].astype(ds_out["subcatch"].dtype)
        self.set_geoms(gdf_subcatch, name=f"subcatchment_{mapname}")

        # Subcatchment map for river cells only (to be able to save river outputs
        # in wflow)
        self.set_grid(ds_out["subcatch_riv"], name=f"subcatchment_riv_{mapname}")
        gdf_subcatch_riv = ds_out["subcatch_riv"].raster.vectorize()
        gdf_subcatch_riv["value"] = gdf_subcatch_riv["value"].astype(
            ds_out["subcatch"].dtype
        )
        self.set_geoms(gdf_subcatch_riv, name=f"subcatchment_riv_{mapname}")

        # Update toml
        if update_toml:
            self.setup_config_output_timeseries(
                mapname=f"subcatchment_river_{mapname}",
                toml_output=toml_output,
                header=["Qlat"],
                param=["river_water__lateral_inflow_volume_flow_rate"],
                reducer=["sum"],
            )

    @hydromt_step
    def setup_precip_forcing(
        self,
        precip_fn: str | xr.DataArray,
        precip_clim_fn: str | xr.DataArray | None = None,
        chunksize: int | None = None,
        **kwargs,
    ) -> None:
        """Generate gridded precipitation forcing at model resolution.

        Adds model layer:

        * **precip**: precipitation [mm]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        precip_fn : str, xarray.DataArray
            Precipitation RasterDataset source.

            * Required variable: 'precip' [mm]

            * Required dimension: 'time'  [timestamp]
        precip_clim_fn : str, xarray.DataArray, optional
            High resolution climatology precipitation RasterDataset source to correct
            precipitation.

            * Required variable: 'precip' [mm]

            * Required dimension: 'time'  [cyclic month]
        chunksize: int, optional
            Chunksize on time dimension for processing data (not for saving to disk!).
            If None the data chunksize is used, this can however be optimized for
            large/small catchments. By default None.
        **kwargs : dict, optional
            Additional arguments passed to the forcing function.
            See hydromt.model.processes.meteo.precip for more details.
        """
        starttime = self.get_config("time.starttime")
        endtime = self.get_config("time.endtime")
        freq = pd.to_timedelta(self.get_config("time.timestepsecs"), unit="s")
        mask = self.staticmaps.data[self._MAPS["basins"]].values > 0

        precip = self.data_catalog.get_rasterdataset(
            precip_fn,
            geom=self.region,
            buffer=2,
            time_range=(starttime, endtime),
            variables=["precip"],
        )
        precip = precip.astype("float32")

        if chunksize is not None:
            precip = precip.chunk({"time": chunksize})

        clim = None
        if precip_clim_fn is not None:
            clim = self.data_catalog.get_rasterdataset(
                precip_clim_fn,
                geom=precip.raster.box,
                buffer=2,
                variables=["precip"],
            )
            clim = clim.astype("float32")

        precip_out = hydromt.model.processes.meteo.precip(
            precip=precip,
            da_like=self.staticmaps.data[self._MAPS["elevtn"]],
            clim=clim,
            freq=freq,
            resample_kwargs=dict(label="right", closed="right"),
        )

        # Update meta attributes (used for default output filename later)
        precip_out.attrs.update({"precip_fn": precip_fn})
        if precip_clim_fn is not None:
            precip_out.attrs.update({"precip_clim_fn": precip_clim_fn})
        self.set_forcing(precip_out.where(mask), name="precip")
        self._update_config_variable_name(self._MAPS["precip"], data_type="forcing")

    @hydromt_step
    def setup_precip_from_point_timeseries(
        self,
        precip_fn: str | pd.DataFrame | xr.Dataset,
        interp_type: str = "nearest",
        precip_stations_fn: str | gpd.GeoDataFrame | None = None,
        index_col: str | None = None,
        buffer: float = 1e5,
        **kwargs,
    ) -> None:
        """
        Generate gridded precipitation from point timeseries (requires wradlib).

        Adds model layer:

        * **precip**: precipitation [mm]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Supported interpolation methods:

        * uniform: Applies spatially uniform precipitation to the model. \
        Only works when `precip_fn` contains a single timeseries.
        * nearest: Nearest-neighbour interpolation, also works with a single station.
        * idw: Inverse-distance weighting using 1 / distance ** p.
        * linear: Linear interpolation using scipy.interpolate.LinearNDInterpolator, \
        may result in missing values when station coverage is limited.
        * ordinarykriging: Interpolate using Ordinary Kriging, see wradlib \
        documentation for a full explanation: `wradlib.ipol.OrdinaryKriging <https://docs.wradlib.org/en/latest/generated/wradlib.ipol.OrdinaryKriging.html>`.
        * externaldriftkriging: Kriging interpolation including an external drift, \
        see wradlib documentation for a full explanation: \
        `wradlib.ipol.ExternalDriftKriging <https://docs.wradlib.org/en/latest/generated/wradlib.ipol.ExternalDriftKriging.html>`.


        Parameters
        ----------
        precip_fn : str, pd.DataFrame, xr.Dataset
            Precipitation source as DataFrame or GeoDataset. \
            - DataFrame: the index column should contain time and the other \
            columns should correspond to the name or ID values of the stations \
            in `precip_stations_fn`.
            - GeoDataset: the dataset should contain the variable 'precip' and \
            the dimensions 'time' and 'index'.

            * Required variable: 'time', 'precip' [mm]
        interp_type : str
            Interpolation method. Options: "nearest", "idw", "linear", \
            "ordinarykriging", "externaldriftkriging".
        precip_stations_fn : str, gpd.GeoDataFrame, optional
            Source for the locations of the stations as points: (x, y) or (lat, lon). \
            Only required if precip_fn is of type DataFrame.
        index_col : str, optional
            Column in precip_stations_fn to use for station ID values, by default None.
        buffer: float, optional
            Buffer around the basins in metres to determine which
            stations to include. Set to 100 km (1e5 metres) by default.
        **kwargs
            Additional keyword arguments passed to the interpolation function. \
            Supported arguments depend on the interpolation type:
            - nnearest: Maximum number of neighbors for interpolation (default: 4).
            - p: Power parameter for IDW interpolation (default: 2).
            - remove_missing: Mask NaN values in the input data (default: False).
            - cov: Covariance model for Kriging (default: '1.0 Exp(10000.)').
            - src_drift: External drift values at source points (stations).
            - trg_drift: External drift values at target points (grid).

        See Also
        --------
        hydromt_wflow.workflows.forcing.spatial_interpolation
        `wradlib.ipol.interpolate <https://docs.wradlib.org/en/latest/ipol.html#wradlib.ipol.interpolate>`
        """
        starttime = self.get_config("time.starttime")
        endtime = self.get_config("time.endtime")
        timestep = self.get_config("time.timestepsecs")
        freq = pd.to_timedelta(timestep, unit="s")
        mask = self.staticmaps.data[self._MAPS["basins"]].values > 0

        # Check data type of precip_fn if it is provided through the data catalog
        if isinstance(precip_fn, str) and precip_fn in self.data_catalog:
            _data_type = self.data_catalog[precip_fn].data_type
        else:
            _data_type = None

        # Read the precipitation timeseries
        if isinstance(precip_fn, xr.Dataset) or _data_type == "GeoDataset":
            da_precip = self.data_catalog.get_geodataset(
                precip_fn,
                geom=self.region,
                buffer=buffer,
                variables=["precip"],
                time_range=(starttime, endtime),
                single_var_as_array=True,
            )
        else:
            # Read timeseries
            df_precip = self.data_catalog.get_dataframe(
                precip_fn,
                time_range=(starttime, endtime),
            )
            # Get locs
            if interp_type == "uniform":
                # Use basin centroid as 'station' for uniform case
                gdf_stations = gpd.GeoDataFrame(
                    data=None,
                    geometry=[self.basins.union_all().centroid],
                    index=df_precip.columns,
                    crs=self.crs,
                )
                interp_type = "nearest"
                if df_precip.shape[1] != 1:
                    raise ValueError(
                        f"""
                        Data source ({precip_fn}) should contain
                        a single timeseries, not {df_precip.shape[1]}."""
                    )
                logger.info("Uniform interpolation is applied using method 'nearest'.")
            elif precip_stations_fn is None:
                raise ValueError(
                    "Using a DataFrame as precipitation source requires that station "
                    "locations are provided separately through precip_station_fn."
                )
            else:
                # Load the stations and their coordinates
                gdf_stations = self.data_catalog.get_geodataframe(
                    precip_stations_fn,
                    geom=self.basins,
                    buffer=buffer,
                    assert_gtype="Point",
                    handle_nodata=NoDataStrategy.IGNORE,
                )
                # Use station ids from gdf_stations when reading the DataFrame
                if index_col is not None:
                    gdf_stations = gdf_stations.set_index(index_col)

            # Index is required to contruct GeoDataArray
            if gdf_stations.index.name is None:
                gdf_stations.index.name = "stations"

            # Convert to geodataset
            da_precip = hydromt.vector.GeoDataArray.from_gdf(
                gdf=gdf_stations,
                data=df_precip,
                name="precip",
                index_dim=None,
                dims=["time", gdf_stations.index.name],
                keep_cols=False,
                merge_index="gdf",
            )

        # Calling interpolation workflow
        precip = workflows.forcing.spatial_interpolation(
            forcing=da_precip,
            interp_type=interp_type,
            ds_like=self.staticmaps.data,
            mask_name=self._MAPS["basins"],
            **kwargs,
        )

        # Use precip workflow to create the forcing file
        precip_out = hydromt.model.processes.meteo.precip(
            precip=precip,
            da_like=self.staticmaps.data[self._MAPS["elevtn"]],
            clim=None,
            freq=freq,
            resample_kwargs=dict(label="right", closed="right"),
        )

        # Turn precip_fn into a string if needed
        if isinstance(precip_fn, pd.DataFrame):
            precip_fn_str = getattr(precip_fn, "name", None) or "unnamed_DataFrame"
        elif isinstance(precip_fn, xr.Dataset):
            precip_fn_str = precip_fn.attrs.get("name", "unnamed_Dataset")
        else:
            precip_fn_str = precip_fn

        # Update meta attributes (used for default output filename later)
        precip_out.attrs.update({"precip_fn": precip_fn_str})
        self.set_forcing(precip_out.where(mask), name="precip")
        self._update_config_variable_name(self._MAPS["precip"], data_type="forcing")

        # Add to geoms
        gdf_stations = da_precip.vector.to_gdf().to_crs(self.crs)
        self.set_geoms(gdf_stations, name="stations_precipitation")

    @hydromt_step
    def setup_temp_pet_forcing(
        self,
        temp_pet_fn: str | xr.Dataset,
        pet_method: str = "debruin",
        press_correction: bool = True,
        temp_correction: bool = True,
        wind_correction: bool = True,
        wind_altitude: int = 10,
        reproj_method: str = "nearest_index",
        fillna_method: str | None = None,
        dem_forcing_fn: str | xr.DataArray | None = None,
        skip_pet: bool = False,
        chunksize: int | None = None,
    ) -> None:
        """Generate gridded temperature and reference evapotranspiration forcing.

        If `temp_correction` is True, the temperature will be reprojected and then
        downscaled to model resolution using the elevation lapse rate. For better
        accuracy, you can provide the elevation grid of the climate data in
        `dem_forcing_fn`. If not present, the upscaled elevation grid of the wflow model
        is used ('land_elevation').

        To compute PET (`skip_pet` is False), several methods are available. Before
        computation, both the temperature and pressure can be downscaled. Wind speed
        should be given at 2m altitude and can be corrected if `wind_correction` is True
        and the wind data altitude is provided in `wind_altitude` [m].
        Several methods to compute pet are available: {'debruin', 'makkink',
        'penman-monteith_rh_simple', 'penman-monteith_tdew'}.

        Depending on the methods, `temp_pet_fn` should contain temperature 'temp' [°C],
        pressure 'press_msl' [hPa], incoming shortwave radiation 'kin' [W/m2], outgoing
        shortwave radiation 'kout' [W/m2], wind speed 'wind' [m/s], relative humidity
        'rh' [%], dew point temperature 'temp_dew' [°C], wind speed either total 'wind'
        or the U- 'wind10_u' [m/s] and V- 'wind10_v' components [m/s].

        Adds model layer:

        * **pet**: reference evapotranspiration [mm]
        * **temp**: temperature [°C]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        temp_pet_fn : str, xarray.Dataset
            Name or path of RasterDataset source with variables to calculate temperature
            and reference evapotranspiration.

            * Required variable for temperature: 'temp' [°C]

            * Required variables for De Bruin reference evapotranspiration: \
'temp' [°C], 'press_msl' [hPa], 'kin' [W/m2], 'kout' [W/m2]

            * Required variables for Makkink reference evapotranspiration: \
'temp' [°C], 'press_msl' [hPa], 'kin'[W/m2]

            * Required variables for daily Penman-Monteith \
reference evapotranspiration: \
either {'temp' [°C], 'temp_min' [°C], 'temp_max' [°C], 'wind' [m/s], 'rh' [%], 'kin' \
[W/m2]} for 'penman-monteith_rh_simple' or {'temp' [°C], 'temp_min' [°C], 'temp_max' \
[°C], 'temp_dew' [°C], 'wind' [m/s], 'kin' [W/m2], 'press_msl' [hPa], 'wind10_u' [m/s],\
"wind10_v" [m/s]} for 'penman-monteith_tdew' (these are the variables available in ERA5)
        pet_method : {'debruin', 'makkink', 'penman-monteith_rh_simple', \
'penman-monteith_tdew'}, optional
            Reference evapotranspiration method, by default 'debruin'.
            If penman-monteith is used, requires the installation of the pyet package.
        press_correction, temp_correction : bool, optional
            If True pressure, temperature are corrected using elevation lapse rate,
            by default False.
        dem_forcing_fn : str, default None
            Elevation data source with coverage of entire meteorological forcing domain.
            If temp_correction is True and dem_forcing_fn is provided this is used in
            combination with elevation at model resolution to correct the temperature.

            * Required variable: 'elevtn' [m+REF]
        wind_correction : bool, optional
            If True wind speed is corrected to wind at 2m altitude using
            ``wind_altitude``. By default True.
        wind_altitude : int, optional
            Altitude of wind speed [m] variable, by default 10. Only used if
            ``wind_correction`` is True.
        skip_pet : bool, optional
            If True calculate temp only.
        reproj_method : str, optional
            Reprojection method from rasterio.enums.Resampling. to reproject the climate
            data to the model resolution. By default 'nearest_index'.
        fillna_method: str, optional
            Method to fill NaN cells within the active model domain in the
            temperature data e.g. 'nearest'
            By default None for no interpolation.
        chunksize: int, optional
            Chunksize on time dimension for processing data (not for saving to disk!).
            If None the data chunksize is used, this can however be optimized for
            large/small catchments. By default None.
        """
        starttime = self.get_config("time.starttime")
        endtime = self.get_config("time.endtime")
        timestep = self.get_config("time.timestepsecs")
        freq = pd.to_timedelta(timestep, unit="s")
        mask = self.staticmaps.data[self._MAPS["basins"]].values > 0

        variables = ["temp"]
        if not skip_pet:
            if pet_method == "debruin":
                variables += ["press_msl", "kin", "kout"]
            elif pet_method == "makkink":
                variables += ["press_msl", "kin"]
            elif pet_method == "penman-monteith_rh_simple":
                variables += ["temp_min", "temp_max", "wind", "rh", "kin"]
            elif pet_method == "penman-monteith_tdew":
                variables += [
                    "temp_min",
                    "temp_max",
                    "wind10_u",
                    "wind10_v",
                    "temp_dew",
                    "kin",
                    "press_msl",
                ]
            else:
                methods = [
                    "debruin",
                    "makkink",
                    "penman-monteith_rh_simple",
                    "penman-monteith_tdew",
                ]
                raise ValueError(
                    f"Unknown pet method {pet_method}, select from {methods}"
                )

        ds = self.data_catalog.get_rasterdataset(
            temp_pet_fn,
            geom=self.region,
            buffer=1,
            time_range=(starttime, endtime),
            variables=variables,
            single_var_as_array=False,  # always return dataset
        )
        if chunksize is not None:
            ds = ds.chunk({"time": chunksize})
        for var in ds.data_vars:
            ds[var] = ds[var].astype("float32")

        dem_forcing = None
        if dem_forcing_fn is not None:
            dem_forcing = self.data_catalog.get_rasterdataset(
                dem_forcing_fn,
                geom=ds.raster.box,  # clip dem with forcing bbox for full coverage
                buffer=2,
                variables=["elevtn"],
            ).squeeze()
            dem_forcing = dem_forcing.astype("float32")

        temp_in = hydromt.model.processes.meteo.temp(
            ds["temp"],
            dem_model=self.staticmaps.data[self._MAPS["elevtn"]],
            dem_forcing=dem_forcing,
            lapse_correction=temp_correction,
            freq=None,  # resample time after pet workflow
        )

        if (
            "penman-monteith" in pet_method
        ):  # also downscaled temp_min and temp_max for Penman needed
            temp_max_in = hydromt.model.processes.meteo.temp(
                ds["temp_max"],
                dem_model=self.staticmaps.data[self._MAPS["elevtn"]],
                dem_forcing=dem_forcing,
                lapse_correction=temp_correction,
                freq=None,  # resample time after pet workflow
            )
            temp_max_in.name = "temp_max"

            temp_min_in = hydromt.model.processes.meteo.temp(
                ds["temp_min"],
                dem_model=self.staticmaps.data[self._MAPS["elevtn"]],
                dem_forcing=dem_forcing,
                lapse_correction=temp_correction,
                freq=None,  # resample time after pet workflow
            )
            temp_min_in.name = "temp_min"

            temp_in = xr.merge([temp_in, temp_max_in, temp_min_in], compat="override")

        # Turn temp_pet_fn into a string
        if isinstance(temp_pet_fn, pd.DataFrame):
            temp_pet_fn_str = getattr(temp_pet_fn, "name", None) or "unnamed_DataFrame"
        elif isinstance(temp_pet_fn, xr.Dataset):
            temp_pet_fn_str = temp_pet_fn.attrs.get("name", "unnamed_Dataset")
        else:
            temp_pet_fn_str = temp_pet_fn

        if not skip_pet:
            pet_out = hydromt.model.processes.meteo.pet(
                ds[variables[1:]],
                temp=temp_in,
                dem_model=self.staticmaps.data[self._MAPS["elevtn"]],
                method=pet_method,
                press_correction=press_correction,
                wind_correction=wind_correction,
                wind_altitude=wind_altitude,
                reproj_method=reproj_method,
                freq=freq,
                resample_kwargs=dict(label="right", closed="right"),
            )
            # Update meta attributes with setup opt
            opt_attr = {
                "pet_fn": temp_pet_fn_str,
                "pet_method": pet_method,
            }
            pet_out.attrs.update(opt_attr)
            self.set_forcing(pet_out.where(mask), name="pet")
            self._update_config_variable_name(self._MAPS["pet"], data_type="forcing")

        # make sure only temp is written to netcdf
        if "penman-monteith" in pet_method:
            temp_in = temp_in["temp"]
        # resample temp after pet workflow
        temp_out = hydromt.model.processes.meteo.resample_time(
            temp_in,
            freq,
            upsampling="bfill",  # we assume right labeled original data
            downsampling="mean",
            label="right",
            closed="right",
            conserve_mass=False,
        )

        # Update meta attributes with setup opt (used for default naming later)
        opt_attr = {
            "temp_fn": temp_pet_fn_str,
            "temp_correction": str(temp_correction),
        }
        temp_out.attrs.update(opt_attr)
        if fillna_method is not None:
            temp_out = temp_out.raster.interpolate_na(
                dim=temp_out.raster.x_dim,
                method=fillna_method,
                fill_value="extrapolate",
            )
        self.set_forcing(temp_out.where(mask), name="temp")
        self._update_config_variable_name(self._MAPS["temp"], data_type="forcing")

    @hydromt_step
    def setup_pet_forcing(
        self,
        pet_fn: str | xr.DataArray,
        chunksize: int | None = None,
    ):
        """
        Prepare PET forcing from existig PET data.

        Adds model layer:

        * **pet**: reference evapotranspiration [mm]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_basemaps`

        Parameters
        ----------
        pet_fn: str, xr.DataArray
            RasterDataset source or data for PET to be resampled.

            * Required variable: 'pet' [mm]

        chunksize: int, optional
            Chunksize on time dimension for processing data (not for saving to disk!).
            If None the data chunksize is used, this can however be optimized for
            large/small catchments. By default None.
        """
        logger.info("Preparing potential evapotranspiration forcing maps.")

        starttime = self.get_config("time.starttime")
        endtime = self.get_config("time.endtime")
        freq = pd.to_timedelta(self.get_config("time.timestepsecs"), unit="s")

        pet = self.data_catalog.get_rasterdataset(
            pet_fn,
            geom=self.region,
            buffer=2,
            variables=["pet"],
            time_range=(starttime, endtime),
        )
        pet = pet.astype("float32")

        pet_out = workflows.forcing.pet(
            pet=pet,
            ds_like=self.staticmaps.data,
            freq=freq,
            mask_name=self._MAPS["basins"],
            chunksize=chunksize,
        )

        # Update meta attributes (used for default output filename later)
        pet_out.attrs.update({"pet_fn": pet_fn})
        self.set_forcing(pet_out, name="pet")
        self._update_config_variable_name(self._MAPS["pet"], data_type="forcing")

    @hydromt_step
    def setup_cold_states(
        self,
        timestamp: str = None,
    ) -> None:
        """Prepare cold states for Wflow.

        To be run last as this requires some soil parameters or constant_pars to be
        computed already.

        To be run after setup_reservoirs methods and setup_glaciers to also create
        cold states for them if they are present in the basin.

        This function is mainly useful in case the wflow model is read into Delft-FEWS.

        Adds model layers:

        * **soil_saturated_depth**: saturated store [mm]
        * **snow_leq_depth**: snow storage [mm]
        * **soil_temp**: top soil temperature [°C]
        * **soil_unsaturated_depth**: amount of water in the unsaturated store, per
          layer [mm]
        * **snow_water_depth**: liquid water content in the snow pack [mm]
        * **vegetation_water_depth**: canopy storage [mm]
        * **river_instantaneous_q**: river discharge [m3/s]
        * **river_h**: river water level [m]
        * **subsurface_q**: subsurface flow [m3/d]
        * **land_h**: land water level [m]
        * **land_instantaneous_q** or **land_instantaneous_qx** +
          **land_instantaneous_qy**: overland flow for kinwave [m3/s] or
          overland flow in x/y directions for local_inertial [m3/s]

        If reservoirs, also adds:

        * **reservoir_water_level**: reservoir water level [m]

        If glaciers, also adds:

        * **glacier_leq_depth**: water within the glacier [mm]

        If paddy, also adds:

        * **demand_paddy_h**: water on the paddy fields [mm]

        Required setup methods:

        * :py:meth:`~WflowSbmModel.setup_soilmaps`
        * :py:meth:`~WflowSbmModel.setup_constant_pars`
        * :py:meth:`~WflowSbmModel.setup_lakes`
        * :py:meth:`~WflowSbmModel.setup_reservoirs`
        * :py:meth:`~WflowSbmModel.setup_glaciers`
        * :py:meth:`~WflowSbmModel.setup_irrigation` or equivalent

        Parameters
        ----------
        timestamp : str, optional
            Timestamp of the cold states. By default uses the (starttime - timestepsecs)
            from the config.
        """
        states, states_config = workflows.prepare_cold_states(
            self.staticmaps.data,
            config=self.config.data,
            timestamp=timestamp,
            mask_name_land=self._MAPS["basins"],
            mask_name_river=self._MAPS["rivmsk"],
        )

        self.set_states(states)

        # Update config to read the states
        self.set_config("model.cold_start__flag", False)
        # Update states variables names in config
        for option in states_config:
            self.set_config(option, states_config[option])

    @hydromt_step
    def clip(
        self,
        region: dict,
        inverse_clip: bool = False,
        clip_forcing: bool = True,
        clip_states: bool = True,
        crs: int = 4326,
        **kwargs,
    ):
        """Clip model to region.

        First the staticmaps are clipped to the region.
        Then the staticgeoms are re-generated to match the new grid for basins and
        rivers and clipped for the others.
        Finally the forcing and states are clipped to the new grid extent.

        Parameters
        ----------
        region : dict
            See :meth:`models.wflow_base.WflowBaseModel.setup_basemaps`
        inverse_clip: bool, optional
            Flag to perform "inverse clipping": removing an upstream part of the model
            instead of the subbasin itself, by default False
        clip_forcing: bool, optional
            Flag to clip the forcing to the new grid extent, by default True
        clip_states: bool, optional
            Flag to clip the states to the new grid extent, by default True
        crs: int, optional
            Default crs of the grid to clip.
        **kwargs: dict
            Additional keyword arguments passed to
            :py:meth:`~hydromt.raster.Raster.clip_geom`
        """
        # Reservoir maps that will be removed if no reservoirs after clipping
        # key: staticmaps name,  value: wflow intput variable name
        reservoir_maps = [
            self._MAPS["reservoir_area_id"],
            self._MAPS["reservoir_outlet_id"],
            self._MAPS["reservoir_lower_id"],
            self._MAPS["reservoir_storage_curve"],
            self._MAPS["reservoir_rating_curve"],
            self._MAPS["reservoir_area"],
            self._MAPS["reservoir_initial_depth"],
            self._MAPS["reservoir_demand"],
            self._MAPS["reservoir_target_full_fraction"],
            self._MAPS["reservoir_target_min_fraction"],
            self._MAPS["reservoir_max_release"],
            self._MAPS["reservoir_max_volume"],
            "meta_reservoir_mean_outflow",  # this is a hydromt meta map
            self._MAPS["reservoir_outflow_threshold"],
            self._MAPS["reservoir_b"],
            self._MAPS["reservoir_e"],
        ]
        reservoir_maps = {k: self._WFLOW_NAMES.get(k, None) for k in reservoir_maps}

        # Reservoir states that will be removed if no reservoirs after clipping
        # key: states name,  value: wflow state variable name
        reservoir_state = self.get_config(
            "state.variables.reservoir_water_surface__elevation",
            fallback="reservoir_water_level",
        )
        reservoir_states = {reservoir_state: "reservoir_water_surface__elevation"}

        super().clip(
            region,
            inverse_clip=inverse_clip,
            clip_forcing=clip_forcing,
            clip_states=clip_states,
            reservoir_maps=reservoir_maps,
            reservoir_states=reservoir_states,
            crs=crs,
            **kwargs,
        )

        # Update reservoirs tables
        logger.info("Updating reservoir tables to match clipped model.")
        if self._MAPS["reservoir_area_id"] not in self.staticmaps.data:
            # no more reservoirs in the model, tables can be cleared
            self.tables._data = {}
        else:
            old_tables = self.tables.data.copy()
            reservoir = self.staticmaps.data[self._MAPS["reservoir_area_id"]]
            ids = np.unique(reservoir.raster.mask_nodata())
            # remove nan from ids
            ids = ids[~np.isnan(ids)].astype(int)
            keys_to_keep = []
            for res_id in ids:
                res_tbl = [k for k in self.tables.data if str(res_id) in k]
                keys_to_keep.extend(res_tbl)

            # Clear and add
            self.tables._data = {}
            for k in keys_to_keep:
                self.set_tables(old_tables[k], name=k)

    @hydromt_step
    def upgrade_to_v1_wflow(self):
        """
        Upgrade the model to wflow v1 format.

        The function reads a TOML from wflow v0x and converts it to wflow v1x format.
        The other components stay the same.

        Lakes and reservoirs have also been merged into one structure and parameters in
        the resulted staticmaps will be combined.

        This function should be followed by write_config() to write the upgraded file.
        """
        config_v0 = self.config.data.copy()
        config_out = convert_to_wflow_v1_sbm(self.config.data)
        # Update the config
        with open(self._DATADIR / "default_config_headers.toml", "rb") as file:
            self.config._data = tomllib.load(file)
        for option in config_out:
            self.set_config(option, config_out[option])

        # Merge lakes and reservoirs layers
        ds_res, vars_to_remove, config_opt = convert_reservoirs_to_wflow_v1_sbm(
            self.staticmaps.data, config_v0
        )
        upgrade_lake_tables_to_reservoir_tables_v1(self.tables)
        if ds_res is not None:
            # Remove older maps from grid
            self.staticmaps.drop_vars(vars_to_remove)
            # Add new reservoir maps to grid
            self.set_grid(ds_res)
            # Update the config with the new names
            for option in config_opt:
                self.set_config(option, config_opt[option])
