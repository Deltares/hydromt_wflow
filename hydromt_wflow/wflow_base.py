"""Implement Wflow base model class."""

# Implement model class following model API
import logging
import os
from os.path import isfile, join
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import pyflwdir
import pyproj
import xarray as xr
from hydromt import hydromt_step
from hydromt.error import NoDataStrategy
from hydromt.gis import flw
from hydromt.model import Model
from hydromt.typing import ModeLike

import hydromt_wflow.utils as utils
from hydromt_wflow import workflows
from hydromt_wflow.components import (
    WflowConfigComponent,
    WflowForcingComponent,
    WflowGeomsComponent,
    WflowOutputCsvComponent,
    WflowOutputGridComponent,
    WflowOutputScalarComponent,
    WflowStatesComponent,
    WflowStaticmapsComponent,
    WflowTablesComponent,
)

__all__ = ["WflowBaseModel"]
logger = logging.getLogger(f"hydromt.{__name__}")


class WflowBaseModel(Model):
    """Base Class for all Wflow Model implementations.

    DO NOT USE THIS CLASS DIRECTLY. ONLY USE SUBCLASSES OF THIS CLASS.

    It provides common functionality (IO, components and workflows).
    Any specific implementation details for either `sediment` of `sbm` should be
    handled in the derived classes.

    Parameters
    ----------
    root : str, optional
        Model root, by default None (current working directory)
    config_filename : str, optional
        A path relative to the root where the configuration file will
        be read and written if user does not provide a path themselves.
        By default "wflow.toml"
    mode : {'r','r+','w'}, optional
        read/append/write mode, by default "w"
    data_libs : list[str] | str, optional
        List of data catalog configuration files, by default None
    **catalog_keys:
        Additional keyword arguments to be passed down to the DataCatalog.
    """

    name: str = "wflow"

    _MODEL_VERSION = None
    # TODO supported model version should be filled by the plugins
    # e.g. _MODEL_VERSION = ">=1.0, <1.1

    _DATADIR: Path = utils.DATADIR

    def __init__(
        self,
        root: str | None = None,
        config_filename: str | None = None,
        mode: str = "w",
        data_libs: list[str] | str | None = None,
        **catalog_keys,
    ):
        if type(self) is WflowBaseModel:
            raise TypeError(
                "``WflowBaseModel`` is an abstract class and cannot be instantiated "
                "directly. Please use one of its subclasses defined as hydromt-entry "
                "points: [``WflowSbmModel``, ``WflowSedimentModel``]"
            )

        default_filename = self._DATADIR / self.name / f"{self.name}.toml"
        if config_filename is None:
            config_filename = default_filename.name

        components = {
            "config": WflowConfigComponent(
                self,
                filename=str(config_filename),
                default_template_filename=default_filename.as_posix(),
            ),
            "forcing": WflowForcingComponent(self, region_component="staticmaps"),
            "geoms": WflowGeomsComponent(self, region_component="staticmaps"),
            "states": WflowStatesComponent(self, region_component="staticmaps"),
            "staticmaps": WflowStaticmapsComponent(self),
            "tables": WflowTablesComponent(self),
            "output_grid": WflowOutputGridComponent(
                self, region_component="staticmaps"
            ),
            "output_scalar": WflowOutputScalarComponent(self),
            "output_csv": WflowOutputCsvComponent(
                self, locations_component="staticmaps"
            ),
        }

        super().__init__(
            root,
            components=components,
            mode=mode,
            region_component="staticmaps",
            data_libs=data_libs,
            **catalog_keys,
        )
        self._MAPS: dict[str, str] = {}
        self._WFLOW_NAMES: dict[str, str] = {}

        # wflow specific
        self._flwdir = None
        self.data_catalog.from_yml(self._DATADIR / "parameters_data.yml")

        # Supported Wflow.jl version
        logger.info("Supported Wflow.jl version v1+")

    ## Properties
    # Components
    @property
    def config(self) -> WflowConfigComponent:
        """Return the config component."""
        return self.components["config"]

    @property
    def forcing(self) -> WflowForcingComponent:
        """Return the forcing component."""
        return self.components["forcing"]

    @property
    def geoms(self) -> WflowGeomsComponent:
        """Return the geoms component."""
        return self.components["geoms"]

    @property
    def states(self) -> WflowStatesComponent:
        """Return the states component."""
        return self.components["states"]

    @property
    def staticmaps(self) -> WflowStaticmapsComponent:
        """Return the staticmaps component."""
        return self.components["staticmaps"]

    @property
    def tables(self) -> WflowTablesComponent:
        """Return the WflowTablesComponent instance."""
        return self.components["tables"]

    @property
    def output_grid(self) -> WflowOutputGridComponent:
        """Return the WflowOutputGridComponent instance."""
        return self.components["output_grid"]

    @property
    def output_scalar(self) -> WflowOutputScalarComponent:
        """Return the WflowOutputScalarComponent instance."""
        return self.components["output_scalar"]

    @property
    def output_csv(self) -> WflowOutputCsvComponent:
        """Return the WflowOutputCsvComponent instance."""
        return self.components["output_csv"]

    ## SETUP METHODS
    @hydromt_step
    def setup_config(self, data: dict[str, Any]):
        """Set the config dictionary at key(s) with values.

        Parameters
        ----------
        data : dict[str, Any]
            A dictionary with the values to be set. keys can be dotted like in
            :py:meth:`~hydromt_wflow.components.config.WflowConfigComponent.set`

        Examples
        --------
        Setting data as a nested dictionary::


            >> self.setup_config({'a': 1, 'b': {'c': {'d': 2}}})
            >> self.config.data
            {'a': 1, 'b': {'c': {'d': 2}}}

        Setting data using dotted notation::

            >> self.setup_config({'a.d.f.g': 1, 'b': {'c': {'d': 2}}})
            >> self.config.data
            {'a': {'d':{'f':{'g': 1}}}, 'b': {'c': {'d': 2}}}

        """
        self.config.update(data)

    @hydromt_step
    def setup_config_output_timeseries(
        self,
        mapname: str,
        toml_output: str | None = "csv",
        header: list[str] | None = ["river_q"],
        param: list[str] | None = ["river_water__volume_flow_rate"],
        reducer: list[str] | None = None,
    ):
        """Set the default gauge map based on basin outlets.

        Adds model layers:

        * **csv.column** config: csv timeseries to save based on mapname locations
        * **netcdf.variable** config: netcdf timeseries to save based on mapname \
            locations

        Required setup methods:

        * :py:meth:`~WflowBaseModel.setup_basemaps`

        Parameters
        ----------
        mapname : str
            Name of the gauge map (in staticmaps.nc) to use for scalar output.
        toml_output : str, optional
            One of ['csv', 'netcdf_scalar', None] to update [output.csv] or
            [output.netcdf_scalar] section of wflow toml file or do nothing. By
            default, 'csv'.
        header : list, optional
            Save specific model parameters in csv section. This option defines
            the header of the csv file.
            By default saves river_q (for river_water__volume_flow_rate).
        param: list, optional
            Save specific model parameters in csv section. This option defines
            the wflow variable corresponding to the
            names in gauge_toml_header. By default saves river_water__volume_flow_rate
            (for river_q).
        reducer: list, optional
            If map is an area rather than a point location, provides the reducer
            for the parameters to save. By default None.
        """
        # # Add new outputcsv section in the config
        if toml_output == "csv" or toml_output == "netcdf_scalar":
            logger.info(f"Adding {param} to {toml_output} section of toml.")
            # Add map to the input section of config
            self.set_config(f"input.{mapname}", mapname)
            # Settings and add csv or netcdf sections if not already in config
            # csv
            if toml_output == "csv":
                header_name = "header"
                var_name = "column"
                current_config = self.get_config("output.csv")
                if current_config is None or len(current_config) == 0:
                    self.set_config("output.csv.path", "output.csv")
            # netcdf
            if toml_output == "netcdf_scalar":
                header_name = "name"
                var_name = "variable"
                current_config = self.get_config("output.netcdf_scalar")
                if current_config is None or len(current_config) == 0:
                    self.set_config("output.netcdf_scalar.path", "output_scalar.nc")
            # initialise column / variable section
            if self.get_config(f"output.{toml_output}.{var_name}") is None:
                self.set_config(f"output.{toml_output}.{var_name}", [])

            # Add new output column/variable to config
            for o in range(len(param)):
                gauge_toml_dict = {
                    header_name: header[o],
                    "map": mapname,
                    "parameter": param[o],
                }
                if reducer is not None:
                    gauge_toml_dict["reducer"] = reducer[o]
                # If the gauge column/variable already exists skip writing twice
                variables = self.get_config(f"output.{toml_output}.{var_name}")
                if gauge_toml_dict not in variables:
                    variables.append(gauge_toml_dict)
                    self.set_config(f"output.{toml_output}.{var_name}", variables)
        else:
            logger.info(
                f"toml_output set to {toml_output}, \
skipping adding gauge specific outputs to the toml."
            )

    @hydromt_step
    def setup_basemaps(
        self,
        region: dict,
        hydrography_fn: str | xr.Dataset,
        basin_index_fn: str | xr.Dataset | None = None,
        res: float | int = 1 / 120.0,
        upscale_method: str = "ihu",
        output_names: dict = {
            "basin__local_drain_direction": "local_drain_direction",
            "subbasin_location__count": "subcatchment",
            "land_surface__slope": "land_slope",
        },
    ):
        """
        Build the DEM and flow direction for a Wflow model.

        Setup basemaps sets the `region` of interest and `res`
        (resolution in degrees) of the model.
        All DEM and flow direction related maps are then build.

        We strongly recommend using the region types ``basin`` or ``subbasin`` to build
        a wflow model. If you know what you are doing, you can also use the ``bbox``
        region type (e.g for an island) with the bbox coordinates in EPSG 4326 or the
        ``geom`` region type (e.g. basin polygons have been pre-processed and match
        EXACTLY with ``hydrography_fn``).

        E.g. of `region` argument for a subbasin based on a point and snapped using
        upstream area threshold in `hydrography_fn`, where the maximum boundary box of
        the output subbasin is known:
        region = {'subbasin': [x,y], 'uparea': 10, 'bounds': [xmin, ymin, xmax, ymax]}

        (Sub)Basin delineation is done using hydromt.workflows.get_basin_geometry
        method. Because the delineation is computed from the flow direction data in
        memory, to avoid memory error when using large datasets in `hydrography_fn`, the
        user can either supply 'bounds' in `region` or a basin index dataset in
        `basin_index_fn` to limit the flow direction data to the region of interest.
        The basin index dataset is a GeoDataframe containing either basins polygons or
        bounding boxes of basin boundaries. To select the correct basins, basin ID
        'basins' in `hydrography_fn` and `basin_index_fn` should match.

        If the model resolution is larger than the source data resolution,
        the flow direction is upscaled using the `upscale_method`, by default the
        Iterative Hydrography Upscaling (IHU).

        The default `hydrography_fn` is "merit_hydro"
        (`MERIT hydro <http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/index.html>`_
        at 3 arcsec resolution).
        Alternative sources include "merit_hydro_1k" at 30 arcsec resolution.
        Users can also supply their own elevation and flow direction data
        in any CRS and not only EPSG:4326. The ArcGIS D8 convention is supported
        (see also `PyFlwDir documentation
        <https://deltares.github.io/pyflwdir/latest/_examples/flwdir.html>`).

        Note that in order to define the region, using points or bounding box,
        the coordinates of the points / bounding box
        should be in the same CRS than the hydrography data.
        The wflow model will then also be in the same CRS than the
        hydrography data in order to avoid assumptions and reprojection errors.
        If the user wishes to use a different CRS,
        we recommend first to reproject the hydrography data separately,
        before calling hydromt build.
        You can find examples on how to reproject or prepare hydrography data in the
        `prepare flow directions example notebook
        <https://deltares.github.io/hydromt_wflow/latest/_examples/prepare_ldd.html>`_.

        Adds model layers:

        * **local_drain_direction** map: flow direction in LDD format [-]
        * **subcatchment** map: basin ID map [-]
        * **meta_upstream_area** map: upstream area [km2]
        * **meta_streamorder** map: Strahler stream order [-]
        * **land_elevation** map: average elevation [m+REF]
        * **meta_subgrid_elevation** map: subgrid outlet elevation [m+REF]
        * **land_slope** map: average land surface slope [m/m]
        * **basins** geom: basins boundary vector
        * **region** geom: region boundary vector

        Parameters
        ----------
        region : dict
            Dictionary describing region of interest.
            See :py:meth:`hydromt.workflows.basin_mask.parse_region()` for all options
        hydrography_fn : str, xarray.Dataset
            Name of RasterDataset source for basemap parameters.

            * Required variables: 'flwdir' [LLD or D8 or NEXTXY], 'elevtn' [m+REF]

            * Required variables if used with `basin_index_fn`: 'basins' [-]

            * Required variables if used for snapping in `region`: 'uparea' [km2],
                'strord' [-]

            * Optional variables: 'lndslp' [m/m], 'mask' [bool]
        basin_index_fn : str, geopandas.GeoDataFrame, optional
            Name of GeoDataFrame source for basin_index data linked to hydrography_fn.

            * Required variables: 'basid' [-]
        res : float, optional
            Output model resolution
        upscale_method : {'ihu', 'eam', 'dmm'}, optional
            Upscaling method for flow direction data, by default 'ihu'.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.

        See Also
        --------
        hydromt.workflows.parse_region
        hydromt.workflows.get_basin_geometry
        workflows.hydrography
        workflows.topography
        """
        logger.info("Preparing base hydrography basemaps.")
        # update self._MAPS and self._WFLOW_NAMES with user defined output names
        self._update_naming(output_names)

        geometries, xy, ds_org = workflows.parse_region(
            data_catalog=self.data_catalog,
            region=region,
            hydrography_fn=hydrography_fn,
            resolution=res,
            basin_index_fn=basin_index_fn,
        )

        # setup hydrography maps and set staticmap attribute with renamed maps
        ds_base, _ = workflows.hydrography(
            ds=ds_org,
            res=res,
            xy=xy,
            upscale_method=upscale_method,
        )

        # Rename and add to grid
        # rename idx_out coords
        if "idx_out" in ds_base:
            ds_base = ds_base.rename({"idx_out": "meta_subgrid_outlet_idx"})
            # Set meta_subgrid as a data variable instead of coordinate
            da_outlet = ds_base["meta_subgrid_outlet_idx"]
            ds_base = ds_base.drop_vars("meta_subgrid_outlet_idx")
            ds_base["meta_subgrid_outlet_idx"] = da_outlet

        rmdict = {k: self._MAPS.get(k, k) for k in ds_base.data_vars}
        if "mask" in ds_base.coords:
            ds_base = ds_base.drop_vars("mask")

        self.set_grid(ds_base.rename(rmdict))

        # Add basin geometries after grid is set to avoid warning
        logger.info("Adding basin shape to staticgeoms.")
        for name, _geom in geometries.items():
            self.set_geoms(_geom, name=name)

        # update config
        # skip adding elevtn to config as it will only be used if floodplain 2d are on
        rmdict = {k: v for k, v in rmdict.items() if k != "elevtn"}
        self._update_config_variable_name(ds_base.rename(rmdict).data_vars, None)

        # Call basins once to set it
        self.basins

        # setup topography maps
        ds_topo = workflows.topography(
            ds=ds_org,
            ds_like=self.staticmaps.data,
            method="average",
        )
        rmdict = {k: self._MAPS.get(k, k) for k in ds_topo.data_vars}
        self.set_grid(ds_topo.rename(rmdict))

        # update config
        # skip adding elevtn to config as it will only be used if floodplain 2d are on
        rmdict = {k: v for k, v in rmdict.items() if k != "elevtn"}
        self._update_config_variable_name(ds_topo.rename(rmdict).data_vars)

        # update toml for degree/meters if needed
        if ds_base.raster.crs.is_projected:
            self.set_config("model.cell_length_in_meter__flag", True)

    @hydromt_step
    def setup_rivers(
        self,
        hydrography_fn: str | xr.Dataset,
        river_geom_fn: str | gpd.GeoDataFrame | None = None,
        river_upa: float = 30,
        slope_len: float = 2e3,
        min_rivlen_ratio: float = 0.0,
        smooth_len: float = 5e3,
        min_rivwth: float = 30,
        rivdph_method: str | None = "powlaw",
        min_rivdph: float = 1,
        output_names: dict = {
            "river_location__mask": "river_mask",
            "river__length": "river_length",
            "river__width": "river_width",
            "river__slope": "river_slope",
            "river__depth": "river_depth",
        },
    ):
        """
        Set all river parameter maps.

        The river mask is defined by all cells with a minimum upstream area threshold
        ``river_upa`` [km2].

        The river length is defined as the distance from the subgrid outlet pixel to
        the next upstream subgrid outlet pixel. The ``min_rivlen_ratio`` is the minimum
        global river length to avg. cell resolution ratio and is used as a threshold in
        window-based smoothing of river length.

        The river slope is derived from the subgrid elevation difference between pixels
        at a half distance ``slope_len`` [m] up- and downstream from the subgrid outlet
        pixel.

        The river width is derived from the nearest river segment in ``river_geom_fn``.
        Data gaps are filled by the nearest valid upstream value and averaged along
        the flow directions over a length ``smooth_len`` [m].

        Optionally, river depth is derived using ``rivdph_method`` if provided
        (default is "powlaw"). Data gaps are filled similarly to river width.

        Adds model layers:

        * **wflow_river** map: river mask [-]
        * **river_length** map: river length [m]
        * **river_width** map: river width [m]
        * **river_slope** map: river slope [m/m]
        * **river_depth** map: river depth [m] (if ``rivdph_method`` is not None)
        * **rivers** geom: river vector based on wflow_river mask

        Required setup methods:

        * :py:meth:`~WflowBaseModel.setup_basemaps`

        Parameters
        ----------
        hydrography_fn : str, Path, xarray.Dataset
            Name of RasterDataset source for hydrography data. Must be same as
            setup_basemaps for consistent results.

            * **Required variables**: 'flwdir' [LLD or D8 or NEXTXY], 'uparea' [km2],
              'elevtn' [m+REF]

            * **Optional variables**: 'rivwth' [m]

        river_geom_fn : str, Path, geopandas.GeoDataFrame, optional
            Name of GeoDataFrame source for river data.

            * **Required variables**: 'rivwth' [m]

        river_upa : float, optional
            Minimum upstream area threshold for the river map [km2]. By default 30.0
        slope_len : float, optional
            Length [km] over which the river slope is calculated. By default 2.0
        min_rivlen_ratio: float, optional
            Ratio of cell resolution used as a minimum length threshold in a moving
            window-based smoothing of river length, by default 0.0.
            The river length smoothing is skipped if `min_rivlen_ratio` = 0.
            For details about the river length smoothing,
            see :py:meth:`pyflwdir.FlwdirRaster.smooth_rivlen`
        smooth_len : float, optional
            Length [m] over which to smooth the output river width and depth,
            by default 5000.0
        min_rivwth : float, optional
            Minimum river width [m], by default 30.0
        rivdph_method : str, optional
            Method for estimating river depth. Default is "powlaw".
            Set to None to skip river depth estimation.
        min_rivdph : float, optional
            Minimum river depth [m], by default 1.0
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file. By default, includes river mask, length, width, slope,
            and depth.
        """
        logger.info("Preparing river maps.")
        self._update_naming(output_names)

        if river_upa > float(self.staticmaps.data[self._MAPS["uparea"]].max()):
            raise ValueError(
                f"river_upa threshold {river_upa} should be larger than the maximum "
                f"uparea {float(self.staticmaps.data[self._MAPS['uparea']].max())} "
                "to create river cells."
            )

        # read hydrography data
        ds_hydro = self.data_catalog.get_rasterdataset(
            hydrography_fn, geom=self.region, buffer=10
        )
        ds_hydro.coords["mask"] = ds_hydro.raster.geometry_mask(self.region)

        # derive rivmsk, rivlen, rivslp
        inv_rename = {v: k for k, v in self._MAPS.items() if v in self.staticmaps.data}
        ds_riv = workflows.river(
            ds=ds_hydro,
            ds_model=self.staticmaps.data.rename(inv_rename),
            river_upa=river_upa,
            slope_len=slope_len,
            channel_dir="up",
            min_rivlen_ratio=min_rivlen_ratio,
        )[0]

        ds_riv["rivmsk"] = ds_riv["rivmsk"].assign_attrs(
            river_upa=river_upa, slope_len=slope_len, min_rivlen_ratio=min_rivlen_ratio
        )
        dvars = ["rivmsk", "rivlen", "rivslp"]
        rmdict = {k: self._MAPS.get(k, k) for k in dvars}
        self.set_grid(ds_riv[dvars].rename(rmdict))
        for dvar in dvars:
            if dvar == "rivmsk":
                self._update_config_variable_name(self._MAPS[dvar], data_type=None)
            else:
                self._update_config_variable_name(self._MAPS[dvar])

        # optional width/depth
        if river_geom_fn is not None:
            gdf_riv = self.data_catalog.get_geodataframe(
                river_geom_fn, geom=self.region
            )
            inv_rename = {
                v: k for k, v in self._MAPS.items() if v in self.staticmaps.data
            }
            ds_riv1 = workflows.river_bathymetry(
                ds_model=self.staticmaps.data.rename(inv_rename),
                gdf_riv=gdf_riv,
                method=rivdph_method,  # None skips depth
                smooth_len=smooth_len,
                min_rivdph=min_rivdph,
                min_rivwth=min_rivwth,
            )
            rmdict = {k: self._MAPS.get(k, k) for k in ds_riv1.data_vars}
            self.set_grid(ds_riv1.rename(rmdict))
            self._update_config_variable_name(ds_riv1.rename(rmdict).data_vars)

        logger.debug("Adding rivers vector to geoms.")
        if "rivers" in self.geoms.data:
            self.geoms.pop("rivers")
        self.rivers

    @hydromt_step
    def setup_riverwidth(
        self,
        predictor: str = "discharge",
        fill: bool = False,
        fit: bool = False,
        min_wth: float = 1.0,
        precip_fn: str | xr.DataArray = "chelsa",
        climate_fn: str | xr.DataArray = "koppen_geiger",
        output_name: str = "river_width",
        **kwargs,
    ):
        """
        Set the river width parameter based on power-law relationship with a predictor.

        By default the riverwidth is estimated based on discharge as ``predictor``
        and used to set the riverwidth globally based on pre-defined power-law
        parameters per climate class. With ``fit`` set to True,
        the power-law relationships parameters are set on-the-fly.
        With ``fill`` set to True, the estimated river widths are only used
        to fill gaps in the observed data. Alternative ``predictor`` values
        are precip (accumulated precipitation) and uparea (upstream area).
        For these predictors values ``fit`` default to True.
        By default the predictor is based on discharge which is estimated through
        multiple linear regression with precipitation and upstream area
        per climate zone.

        * **river_width** map: river width [m]

        Parameters
        ----------
        predictor : {"discharge", "precip", "uparea"}
            Predictor used in the power-law equation: width = a * predictor ^ b.
            Discharge is based on multiple linear regression per climate zone.
            Precip is based on the 10x the daily average
            accumulated precipitation [m3/s].
            Uparea is based on the upstream area grid [km2].
            Other variables, e.g. bankfull discharge, can also be provided if present
            in the grid
        fill : bool, optional
            If True (default), use estimate to fill gaps, outliers and lake/res areas
            in observed width data (if present);
            if False, set all riverwidths based on predictor
            (automatic choice if no observations found)
        fit : bool, optional
            If True, the power-law parameters are fitted on the fly
            By default True for all but "discharge" predictor.
            A-priori derived parameters will be overwritten if True.
        a, b : float, optional kwarg
            Manual power-law parameters
        min_wth : float, optional
            Minimum river width, by default 1.0
        precip_fn : str, xarray.DataArray
            Source of long term precipitation grid if the predictor
            is set to 'discharge' or 'precip'. By default "chelsa".
        climate_fn: str, xarray.DataArray
            Source of long-term climate grid if the predictor is set to 'discharge'.
            By default "koppen_geiger".
        output_name : str
            The name of the output river__width map.
        """
        logger.warning(
            'The "setup_riverwidth" method has been deprecated \
and will soon be removed. '
            'You can now use the "setup_river" method for all river parameters.'
        )
        if self._MAPS["rivmsk"] not in self.staticmaps.data:
            raise ValueError(
                "The setup_riverwidth method requires to run setup_river method first."
            )

        # update self._MAPS and self._WFLOW_NAMES with user defined output names
        wflow_var = "river__width"
        self._update_naming({wflow_var: output_name})

        # derive river width
        data = {}
        if predictor in ["discharge", "precip"]:
            da_precip = self.data_catalog.get_rasterdataset(
                precip_fn, geom=self.region, buffer=2
            )
            da_precip.name = precip_fn
            data["da_precip"] = da_precip
        if predictor == "discharge":
            da_climate = self.data_catalog.get_rasterdataset(
                climate_fn, geom=self.region, buffer=2
            )
            da_climate.name = climate_fn
            data["da_climate"] = da_climate

        inv_rename = {v: k for k, v in self._MAPS.items() if v in self.staticmaps.data}
        da_rivwth = workflows.river_width(
            ds_like=self.staticmaps.data.rename(inv_rename),
            flwdir=self.flwdir,
            data=data,
            fill=fill,
            fill_outliers=kwargs.pop("fill_outliers", fill),
            min_wth=min_wth,
            mask_names=["lake_area_id", "reservoir_area_id"],
            predictor=predictor,
            a=kwargs.get("a", None),
            b=kwargs.get("b", None),
            fit=fit,
            **kwargs,
        )
        self.set_grid(da_rivwth, name=output_name)
        self._update_config_variable_name(output_name)

    @hydromt_step
    def setup_lulcmaps(
        self,
        lulc_fn: str | xr.DataArray,
        lulc_mapping_fn: str | Path | pd.DataFrame | None = None,
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
        output_names_suffix: str | None = None,
    ):
        """
        Derive several wflow maps based on landuse-landcover (LULC) data.

        Lookup table `lulc_mapping_fn` columns are converted to lulc classes model
        parameters based on literature. The data is remapped at its original resolution
        and then resampled to the model resolution using the average value, unless noted
        differently.

        Currently, if `lulc_fn` is set to the "vito", "globcover", "esa_worldcover"
        "corine" or "glmnco", default lookup tables are available and will be used if
        `lulc_mapping_fn` is not provided.

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

        Required setup methods:

        * :py:meth:`~WflowBaseModel.setup_basemaps`

        Parameters
        ----------
        lulc_fn : str, xarray.DataArray
            Name of RasterDataset source in data_sources.yml file.
        lulc_mapping_fn : str, Path, pd.DataFrame
            Path to a mapping csv file from landuse in source name to parameter values
            in lulc_vars. If lulc_fn is one of {"globcover", "vito", "corine",
            "esa_worldcover", "glmnco"}, a default mapping is used and this argument
            becomes optional.
        lulc_vars : dict
            Dictionnary of landuse parameters to prepare. The names are the
            the columns of the mapping file and the values are the corresponding
            Wflow.jl variables if any.
        output_names_suffix : str, optional
            Suffix to be added to the output names to avoid having to rename all the
            columns of the mapping tables. For example if the suffix is "vito", all
            variables in lulc_vars will be renamed to "landuse_vito", "Kext_vito", etc.
        """
        output_names = {
            v: f"{k}_{output_names_suffix}" if output_names_suffix else k
            for k, v in lulc_vars.items()
        }
        self._update_naming(output_names)

        # As landuse is not a wflow variable, we update the name manually
        rmdict = {"landuse": "meta_landuse"} if "landuse" in lulc_vars else {}
        if output_names_suffix is not None:
            self._MAPS["landuse"] = f"meta_landuse_{output_names_suffix}"
            # rename dict for the staticmaps (hydromt names are not used in that case)
            rmdict = {k: f"{k}_{output_names_suffix}" for k in lulc_vars.keys()}
            if "landuse" in lulc_vars:
                rmdict["landuse"] = f"meta_landuse_{output_names_suffix}"

        logger.info("Preparing LULC parameter maps.")
        if lulc_mapping_fn is None:
            lulc_mapping_fn = f"{lulc_fn}_mapping_default"

        # read landuse map to DataArray
        da = self.data_catalog.get_rasterdataset(
            lulc_fn, geom=self.region, buffer=2, variables=["landuse"]
        )
        df_map = self.data_catalog.get_dataframe(
            lulc_mapping_fn,
            driver_kwargs={"index_col": 0},  # only used if fn_map is a file path
        )
        # process landuse
        ds_lulc_maps = workflows.landuse(
            da=da,
            ds_like=self.staticmaps.data,
            df=df_map,
            params=list(lulc_vars.keys()),
        )
        self.set_grid(ds_lulc_maps.rename(rmdict))

        # Add entries to the config
        self._update_config_variable_name(ds_lulc_maps.rename(rmdict).data_vars)

    @hydromt_step
    def setup_lulcmaps_from_vector(
        self,
        lulc_fn: str | gpd.GeoDataFrame,
        lulc_mapping_fn: str | Path | pd.DataFrame | None = None,
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
        lulc_res: float | int | None = None,
        all_touched: bool = False,
        buffer: int = 1000,
        save_raster_lulc: bool = False,
        output_names_suffix: str | None = None,
    ):
        """
        Derive several wflow maps based on vector landuse-landcover (LULC) data.

        The vector lulc data is first rasterized to a raster map at the model resolution
        or at a higher resolution specified in ``lulc_res`` (recommended).

        Lookup table `lulc_mapping_fn` columns are converted to lulc classes model
        parameters based on literature. The data is remapped at its original resolution
        and then resampled to the model resolution using the average value, unless noted
        differently.

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
        * **land_manning_n** map: Manning Roughness [-]
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

        Required setup methods:

        * :py:meth:`~WflowBaseModel.setup_basemaps`

        Parameters
        ----------
        lulc_fn : str, gpd.GeoDataFrame
            GeoDataFrame or name in data catalog / path to (vector) landuse map.

            * Required columns: 'landuse' [-]
        lulc_mapping_fn : str, Path, pd.DataFrame
            Path to a mapping csv file from landuse in source name to parameter values
            in lulc_vars. If lulc_fn is one of {"globcover", "vito", "corine",
            "esa_worldcover", "glmnco"}, a default mapping is used and this argument
            becomes optional.
        lulc_vars : dict
            Dictionnary of landuse parameters to prepare. The names are the
            the columns of the mapping file and the values are the corresponding
            Wflow.jl variables.
        lulc_res : float, int, optional
            Resolution of the intermediate rasterized landuse map. The unit (meter or
            degree) depends on the CRS of lulc_fn (projected or not). By default None,
            which uses the model resolution.
        all_touched : bool, optional
            If True, all pixels touched by the vector will be burned in the raster,
            by default False.
        buffer : int, optional
            Buffer around the bounding box of the vector data to ensure that all
            landuse classes are included in the rasterized map, by default 1000.
        save_raster_lulc : bool, optional
            If True, the (high) resolution rasterized landuse map will be saved to
            maps/landuse_raster.tif, by default False.
        output_names_suffix : str, optional
            Suffix to be added to the output names to avoid having to rename all the
            columns of the mapping tables. For example if the suffix is "vito", all
            variables in lulc_vars will be renamed to "landuse_vito", "Kext_vito", etc.

        See Also
        --------
        workflows.landuse_from_vector
        """
        output_names = {
            v: f"{k}_{output_names_suffix}" if output_names_suffix else k
            for k, v in lulc_vars.items()
        }
        self._update_naming(output_names)

        # As landuse is not a wflow variable, we update the name manually
        rmdict = {"landuse": "meta_landuse"} if "landuse" in lulc_vars else {}
        if output_names_suffix is not None:
            self._MAPS["landuse"] = f"meta_landuse_{output_names_suffix}"
            # rename dict for the staticmaps (hydromt names are not used in that case)
            rmdict = {k: f"{k}_{output_names_suffix}" for k in lulc_vars.keys()}
            if "landuse" in lulc_vars:
                rmdict["landuse"] = f"meta_landuse_{output_names_suffix}"

        logger.info("Preparing LULC parameter maps.")
        # Read mapping table
        if lulc_mapping_fn is None:
            lulc_mapping_fn = f"{lulc_fn}_mapping_default"
        df_map = self.data_catalog.get_dataframe(
            lulc_mapping_fn,
            driver_kwargs={"index_col": 0},  # only used if fn_map is a file path
        )
        # read landuse map
        gdf = self.data_catalog.get_geodataframe(
            lulc_fn,
            bbox=self.staticmaps.data.raster.bounds,
            buffer=buffer,
            variables=["landuse"],
        )
        if save_raster_lulc:
            lulc_out = join(self.root.path, "maps", "landuse_raster.tif")
        else:
            lulc_out = None

        # process landuse
        ds_lulc_maps = workflows.landuse_from_vector(
            gdf=gdf,
            ds_like=self.staticmaps.data,
            df=df_map,
            params=list(lulc_vars.keys()),
            lulc_res=lulc_res,
            all_touched=all_touched,
            buffer=buffer,
            lulc_out=lulc_out,
        )
        self.set_grid(ds_lulc_maps.rename(rmdict))
        # update config variable names
        self._update_config_variable_name(ds_lulc_maps.rename(rmdict).data_vars)

    @hydromt_step
    def setup_outlets(
        self,
        river_only: bool = True,
        toml_output: str = "csv",
        gauge_toml_header: list[str] = ["river_q"],
        gauge_toml_param: list[str] = ["river_water__volume_flow_rate"],
    ):
        """Set the default gauge map based on basin outlets.

        If the subcatchment map is available, the catchment outlets IDs will be matching
        the subcatchment IDs. If not, then IDs from 1 to number of outlets are used.

        Can also add csv/netcdf_scalar output settings in the TOML.

        Adds model layers:

        * **outlets** map: IDs map from catchment outlets [-]
        * **outlets** geom: polygon of catchment outlets

        Required setup methods:

        * :py:meth:`~WflowBaseModel.setup_rivers`

        Parameters
        ----------
        river_only : bool, optional
            Only derive outlet locations if they are located on a river instead of
            locations for all catchments, by default True.
        toml_output : str, optional
            One of ['csv', 'netcdf_scalar', None] to update [output.csv] or
            [output.netcdf_scalar] section of wflow toml file or do nothing. By
            default, 'csv'.
        gauge_toml_header : list, optional
            Save specific model parameters in csv section. This option defines
            the header of the csv file.
            By default saves river_q (for river_water__volume_flow_rate).
        gauge_toml_param: list, optional
            Save specific model parameters in csv section. This option defines
            the wflow variable corresponding to the names in gauge_toml_header.
            By default saves river_water__volume_flow_rate (for river_q).
        """
        # read existing geoms; important to get the right basin when updating
        # fix in set_geoms / set_geoms method
        self.geoms

        logger.info("Gauges locations set based on river outlets.")
        idxs_out = self.flwdir.idxs_pit
        # Only keep river outlets for gauges
        if river_only:
            idxs_out = idxs_out[
                (self.staticmaps.data[self._MAPS["rivmsk"]] > 0).values.flat[idxs_out]
            ]
        # Use the subcatchment ids
        if self._MAPS["basins"] in self.staticmaps.data:
            ids = self.staticmaps.data[self._MAPS["basins"]].values.flat[idxs_out]
        else:
            ids = None
        da_out, idxs_out, ids_out = flw.gauge_map(
            self.staticmaps.data,
            idxs=idxs_out,
            ids=ids,
            flwdir=self.flwdir,
        )
        self.set_grid(da_out, name="outlets")
        points = gpd.points_from_xy(*self.staticmaps.data.raster.idx_to_xy(idxs_out))
        gdf = gpd.GeoDataFrame(
            index=ids_out.astype(np.int32), geometry=points, crs=self.crs
        )
        gdf["fid"] = ids_out.astype(np.int32)
        self.set_geoms(gdf, name="outlets")

        logger.info("Gauges map based on catchment river outlets added.")
        self.setup_config_output_timeseries(
            mapname="outlets",
            toml_output=toml_output,
            header=gauge_toml_header,
            param=gauge_toml_param,
        )

    @hydromt_step
    def setup_gauges(
        self,
        gauges_fn: str | Path | gpd.GeoDataFrame,
        index_col: str | None = None,
        snap_to_river: bool = True,
        mask: np.ndarray | None = None,
        snap_uparea: bool = False,
        max_dist: float = 10e3,
        wdw: int = 3,
        rel_error: float = 0.05,
        abs_error: float = 50.0,
        fillna: bool = False,
        derive_subcatch: bool = False,
        basename: str | None = None,
        toml_output: str = "csv",
        gauge_toml_header: list[str] = ["river_q", "precip"],
        gauge_toml_param: list[str] = [
            "river_water__volume_flow_rate",
            "atmosphere_water__precipitation_volume_flux",
        ],
        **kwargs,
    ):
        """Set a gauge map based on ``gauges_fn`` data.

        Supported gauge datasets include data catlog entries, direct GeoDataFrame
        or "<path_to_source>" for user supplied csv or geometry files
        with gauge locations. If a csv file is provided, a "x" or "lon" and
        "y" or "lat" column is required and the first column will be used as
        IDs in the map.

        There are four available methods to prepare the gauge map:

        * no snapping: ``mask=None``, ``snap_to_river=False``, ``snap_uparea=False``.
          The gauge locations are used as is.
        * snapping to mask: the gauge locations are snapped to a boolean mask map based
          on the closest dowsntream cell within the mask:
          either provide ``mask`` or set ``snap_to_river=True``
          to snap to the river cells (default).
          ``max_dist`` can be used to set the maximum distance to snap to the mask.
        * snapping based on upstream area matching: : ``snap_uparea=True``.
          The gauge locations are snapped to the closest matching upstream area value.
          Requires gauges_fn to have an ``uparea`` [km2] column. The closest value will
          be looked for in a cell window of size ``wdw`` and the absolute and relative
          differences between the gauge and the closest value should be smaller than
          ``abs_error`` and ``rel_error``.
        * snapping based on upstream area matching and mask: ``snap_uparea=True``,
          ``mask`` or ``snap_to_river=True``. The gauge locations are snapped to the
          closest matching upstream area value within the mask.

        If ``derive_subcatch`` is set to True, an additional subcatch map is derived
        from the gauge locations.

        Finally the output locations can be added to wflow TOML file sections
        [output.csv] or [output.netcdf_scalar] using the ``toml_output`` option. The
        ``gauge_toml_header`` and ``gauge_toml_param`` options can be used to define
        the header and corresponding wflow variable names in the TOML file.

        Adds model layers:

        * **gauges_source** map: gauge IDs map from source [-] (if gauges_fn)
        * **subcatchment_source** map: subcatchment based on gauge locations [-] \
(if derive_subcatch)
        * **gauges_source** geom: polygon of gauges from source
        * **subcatchment_source** geom: polygon of subcatchment based on \
gauge locations [-] (if derive_subcatch)

        Required setup methods:

        * :py:meth:`~WflowBaseModel.setup_rivers`

        Parameters
        ----------
        gauges_fn : str, Path, geopandas.GeoDataFrame
            Catalog source name, path to gauges file geometry file or
            geopandas.GeoDataFrame.

            * Required variables if snap_uparea is True: 'uparea' [km2]
        index_col : str, optional
            Column in gauges_fn to use for ID values, by default None
            (use the default index column)
        mask : np.boolean, optional
            If provided snaps to the mask, else snaps to the river (default).
        snap_to_river : bool, optional
            Snap point locations to the closest downstream river cell, by default True
        snap_uparea: bool, optional
            Snap gauges based on upstream area. Gauges_fn should have "uparea"
            in its attributes.
        max_dist : float, optional
            Maximum distance [m] between original and snapped point location.
            A warning is logged if exceeded. By default 10 000m.
        wdw: int, optional
            Window size in number of cells around the gauge locations
            to snap uparea to, only used if ``snap_uparea`` is True. By default 3.
        rel_error: float, optional
            Maximum relative error (default 0.05)
            between the gauge location upstream area and the upstream area of
            the best fit grid cell, only used if snap_uparea is True.
        abs_error: float, optional
            Maximum absolute error (default 50.0)
            between the gauge location upstream area and the upstream area of
            the best fit grid cell, only used if snap_uparea is True.
        fillna: bool, optional
            Fill missing values in the gauges uparea column with the values from wflow
            upstream area (ie no snapping). By default False and the gauges with NaN
            values are skipped.
        derive_subcatch : bool, optional
            Derive subcatch map for gauges, by default False
        basename : str, optional
            Map name in grid (gauges_basename)
            if None use the gauges_fn basename.
        toml_output : str, optional
            One of ['csv', 'netcdf_scalar', None] to update [output.csv] or
            [output.netcdf_scalar] section of wflow toml file or do nothing. By
            default, 'csv'.
        gauge_toml_header : list, optional
            Save specific model parameters in csv section.
            This option defines the header of the csv file.
            By default saves river_q (for river_water__volume_flow_rate) and
            precip (for "atmosphere_water__precipitation_volume_flux").
        gauge_toml_param: list, optional
            Save specific model parameters in csv section. This option defines
            the wflow variable corresponding to the names in gauge_toml_header.
            By default saves river_water__volume_flow_rate (for river_q) and
            "atmosphere_water__precipitation_volume_flux" (for precip).
        kwargs : dict, optional
            Additional keyword arguments to pass to the get_data method ie
            get_geodataframe or get_geodataset depending  on the data_type of gauges_fn.
        """
        # Read data
        if isinstance(gauges_fn, gpd.GeoDataFrame):
            gdf_gauges = gauges_fn
            if not np.all(np.isin(gdf_gauges.geometry.type, "Point")):
                raise ValueError(f"{gauges_fn} contains other geometries than Point")
        elif isfile(gauges_fn):
            # hydromt#1243
            # try to get epsg number directly, important when writing back data_catalog
            if hasattr(self.crs, "to_epsg"):
                code = self.crs.to_epsg()
            else:
                code = self.crs
            if "metadata" in kwargs:
                kwargs["metadata"].update(crs=code)
            else:
                kwargs.update({"metadata": {"crs": code}})
            gdf_gauges = self.data_catalog.get_geodataframe(
                gauges_fn,
                geom=self.basins,
                # assert_gtype="Point", hydromt#1243
                handle_nodata=NoDataStrategy.IGNORE,
                **kwargs,
            )
        elif self.data_catalog.contains_source(gauges_fn):
            if self.data_catalog.get_source(gauges_fn).data_type == "GeoDataFrame":
                gdf_gauges = self.data_catalog.get_geodataframe(
                    gauges_fn,
                    geom=self.basins,
                    # assert_gtype="Point", hydromt#1243
                    handle_nodata=NoDataStrategy.IGNORE,
                    **kwargs,
                )
            elif self.data_catalog.get_source(gauges_fn).data_type == "GeoDataset":
                da = self.data_catalog.get_geodataset(
                    gauges_fn,
                    geom=self.basins,
                    # assert_gtype="Point", hydromt#1243
                    handle_nodata=NoDataStrategy.IGNORE,
                    **kwargs,
                )
                gdf_gauges = da.vector.to_gdf()
                # Check for point geometry
                if not np.all(np.isin(gdf_gauges.geometry.type, "Point")):
                    raise ValueError(
                        f"{gauges_fn} contains other geometries than Point"
                    )
        else:
            raise ValueError(
                f"{gauges_fn} data source not found or incorrect data_type "
                f"({self.data_catalog.get_source(gauges_fn).data_type} instead of "
                "GeoDataFrame or GeoDataset)."
            )

        # Create basename
        if basename is None:
            basename = os.path.basename(gauges_fn).split(".")[0].replace("_", "-")

        # Check if there is data found
        if gdf_gauges is None:
            logger.info("Skipping method, as no data has been found")
            return

        # Create the gauges map
        logger.info(
            f"{gdf_gauges.index.size} {basename} gauge locations found within domain"
        )

        # read existing geoms; important to get the right basin when updating
        self.geoms
        # Reproject to model crs
        gdf_gauges = gdf_gauges.to_crs(self.crs).copy()

        # Get coords, index and ID
        xs, ys = np.vectorize(lambda p: (p.xy[0][0], p.xy[1][0]))(
            gdf_gauges["geometry"]
        )
        idxs = self.staticmaps.data.raster.xy_to_idx(xs, ys)
        if index_col is not None and index_col in gdf_gauges.columns:
            gdf_gauges = gdf_gauges.set_index(index_col)
        if np.any(gdf_gauges.index == 0):
            logger.warning("Gauge ID 0 is not allowed, setting to 1")
            gdf_gauges.index = gdf_gauges.index.values + 1
        ids = gdf_gauges.index.values

        # if snap_to_river use river map as the mask
        if snap_to_river and mask is None:
            mask = self._MAPS["rivmsk"]
        if mask is not None:
            mask = self.staticmaps.data[mask].values
        if snap_uparea and "uparea" in gdf_gauges.columns:
            # Derive gauge map based on upstream area snapping
            da, idxs, ids = workflows.gauge_map_uparea(
                self.staticmaps.data,
                gdf_gauges,
                uparea_name=self._MAPS["uparea"],
                mask=mask,
                wdw=wdw,
                rel_error=rel_error,
                abs_error=abs_error,
                fillna=fillna,
            )
        else:
            # Derive gauge map
            da, idxs, ids = flw.gauge_map(
                self.staticmaps.data,
                idxs=idxs,
                ids=ids,
                stream=mask,
                flwdir=self.flwdir,
                max_dist=max_dist,
            )
            # Filter gauges that could not be snapped to rivers
            if snap_to_river:
                ids_old = ids.copy()
                da = da.where(
                    self.staticmaps.data[self._MAPS["rivmsk"]] != 0, da.raster.nodata
                )
                ids_new = np.unique(da.values[da.values > 0])
                idxs = idxs[np.isin(ids_old, ids_new)]
                ids = da.values.flat[idxs]

        # Check if there are gauges left
        if ids.size == 0:
            logger.warning(
                "No gauges found within domain after snapping, skipping method."
            )
            return

        # Add to grid
        mapname = f"gauges_{basename}"
        self.set_grid(da, name=mapname)

        # geoms
        points = gpd.points_from_xy(*self.staticmaps.data.raster.idx_to_xy(idxs))
        # if csv contains additional columns, these are also written in the geoms
        gdf_snapped = gpd.GeoDataFrame(
            index=ids.astype(np.int32), geometry=points, crs=self.crs
        )
        # Set the index name of gdf snapped based on original gdf
        if gdf_gauges.index.name is not None:
            gdf_snapped.index.name = gdf_gauges.index.name
        else:
            gdf_snapped.index.name = "fid"
            gdf_gauges.index.name = "fid"
        # Add gdf attributes to gdf_snapped (filter on snapped index before merging)
        df_attrs = pd.DataFrame(gdf_gauges.drop(columns="geometry"))
        df_attrs = df_attrs[np.isin(df_attrs.index, gdf_snapped.index)]
        gdf_snapped = gdf_snapped.merge(df_attrs, how="inner", on=gdf_gauges.index.name)
        # Add gdf_snapped to geoms
        self.set_geoms(gdf_snapped, name=mapname)

        # Add output timeseries for gauges in the toml
        self.setup_config_output_timeseries(
            mapname=mapname,
            toml_output=toml_output,
            header=gauge_toml_header,
            param=gauge_toml_param,
        )

        # add subcatch
        if derive_subcatch:
            da_basins = flw.basin_map(
                self.staticmaps.data, self.flwdir, idxs=idxs, ids=ids
            )[0]
            mapname = self._MAPS["basins"] + "_" + basename
            self.set_grid(da_basins, name=mapname)
            gdf_basins = self.staticmaps.data[mapname].raster.vectorize()
            self.set_geoms(gdf_basins, name=mapname)

    @hydromt_step
    def setup_constant_pars(self, **kwargs):
        """Generate constant parameter maps for all active model cells.

        Adds model layer:

        * **param_name** map: constant parameter map.

        Parameters
        ----------
        dtype: str
            data type
        nodata: int or float
            nodata value
        kwargs
            "param_name: value" pairs for constant grid.
            Param_name should be the Wflow.jl variable name.
        """
        wflow_variables = [v for k, v in self._WFLOW_NAMES.items()]
        for wflow_var, value in kwargs.items():
            if wflow_var not in wflow_variables:
                raise ValueError(
                    f"Parameter {wflow_var} not recognised as a Wflow variable. "
                    f"Please check the name."
                )
            # check if param is already in toml and will be overwritten
            if self.get_config(wflow_var) is not None:
                logger.info(
                    f"Parameter {wflow_var} already in toml and will be overwritten."
                )
            # remove from config
            self.config.data.pop(wflow_var, None)
            # Add to config
            self.set_config(f"input.static.{wflow_var}.value", value)

    @hydromt_step
    def setup_grid_from_raster(
        self,
        raster_fn: str | xr.Dataset,
        reproject_method: str,
        variables: list[str] | None = None,
        wflow_variables: list[str] | None = None,
        fill_method: str | None = None,
    ) -> None:
        """
        Add data variable(s) from ``raster_fn`` to grid object.

        If raster is a dataset, all variables will be added unless ``variables``
        list is specified. The config toml can also be updated to include
        the new maps using ``wflow_variables``.

        Adds model layers:

        * **raster.name** or **variables** grid: data from raster_fn

        Parameters
        ----------
        raster_fn: str
            Source name of RasterDataset in data_catalog.
        reproject_method: str
            Reprojection method from rasterio.enums.Resampling.
            Available methods: ['nearest', 'bilinear', 'cubic', 'cubic_spline', \
'lanczos', 'average', 'mode', 'gauss', 'max', 'min', 'med', 'q1', 'q3', \
'sum', 'rms']
        variables: list, optional
            List of variables to add to grid from raster_fn. By default all.
        wflow_variables: list, optional
            List of corresponding wflow variables to update the config toml
            (e.g: ["vegetation_root__depth"]).
            Should match the variables list. variables list should be provided unless
            raster_fn contains a single variable (len 1).
        fill_method : str, optional
            If specified, fills nodata values using fill_nodata method.
            Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.
        """
        logger.info(f"Preparing grid data from raster source {raster_fn}")
        # Read raster data and select variables
        ds = self.data_catalog.get_rasterdataset(
            raster_fn,
            geom=self.region,
            buffer=2,
            variables=variables,
            single_var_as_array=False,
        )
        # Fill nodata
        if fill_method is not None:
            ds = ds.raster.interpolate_na(method=fill_method)
        # Reprojection
        ds_out = ds.raster.reproject_like(self.staticmaps.data, method=reproject_method)
        # Add to grid
        self.set_grid(ds_out)

        # Update config
        if wflow_variables is not None:
            logger.info(f"Updating the config for wflow_variables: {wflow_variables}")
            if variables is None:
                if len(ds_out.data_vars) == 1:
                    variables = list(ds_out.data_vars.keys())
                else:
                    raise ValueError(
                        "Cannot update the toml if raster_fn has more than \
one variable and variables list is not provided."
                    )

            # Check on len
            if len(wflow_variables) != len(variables):
                raise ValueError(
                    f"Length of variables {variables} do not match wflow_variables \
{wflow_variables}. Cannot update the toml."
                )
            else:
                for i in range(len(variables)):
                    self.set_config(f"input.static.{wflow_variables[i]}", variables[i])

    @hydromt_step
    def setup_areamap(
        self,
        area_fn: str | gpd.GeoDataFrame,
        col2raster: str,
        nodata: int | float = -1,
        output_names: dict = {},
    ):
        """Set area map from vector data to save wflow outputs for specific area.

        Adds model layer:

        * **col2raster** map:  output area data map

        Required setup methods:

        * :py:meth:`~WflowBaseModel.setup_basemaps`

        Parameters
        ----------
        area_fn : str, geopandas.GeoDataFrame
            Name of GeoDataFrame data corresponding to wflow output area.
        col2raster : str
            Name of the column from `area_fn` to rasterize.
        nodata : int/float, optional
            Nodata value to use when rasterizing. Should match the dtype of `col2raster`
            . By default -1.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file. If another name is provided than col2raster, this name
            will be used.
        """
        logger.info(f"Preparing '{col2raster}' map from '{area_fn}'.")
        self._update_naming(output_names)
        gdf_org = self.data_catalog.get_geodataframe(
            area_fn, geom=self.basins, dst_crs=self.crs
        )
        if gdf_org.empty:
            logger.warning(
                f"No shapes of {area_fn} found within region, skipping areamap."
            )
            return
        else:
            da_area = self.staticmaps.data.raster.rasterize(
                gdf=gdf_org,
                col_name=col2raster,
                nodata=nodata,
                all_touched=True,
            )
        if any(output_names):
            if len(output_names) > 1:
                raise ValueError(
                    "Only one output name is allowed for areamap, \
                    please provide a dictionary with one key."
                )
            col2raster_name = list(output_names.values())[0]
            self._update_config_variable_name(col2raster_name)
        else:
            col2raster_name = col2raster
        self.set_grid(da_area.rename(col2raster_name))

    ## WFLOW other step methods (clip and upgrade)
    @hydromt_step
    def clip(
        self,
        region: dict,
        inverse_clip: bool = False,
        clip_forcing: bool = True,
        clip_states: bool = True,
        reservoir_maps: dict[str, str | None] = {},
        reservoir_states: dict[str, str | None] = {},
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
        reservoir_maps: dict[str, str | None] = {},
            Dictionnary of staticmap names in the wflow model staticmaps and
            corresponding wflow input variables to be treated as
            reservoirs. These maps are removed if empty after clipping.
        reservoir_states: dict[str, str | None] = {},
            Dictionnary of state names in the wflow model states and corresponding
            wflow state variables to be treated as reservoirs.
            These states are removed if no reservoir was found after clipping.
        crs: int, optional
            Default crs of the model in case it cannot be read.
            By default 4326 (WGS84)
        **kwargs: dict
            Additional keyword arguments passed to
            :py:meth:`~hydromt.raster.Raster.clip_geom`
        """
        # 0. Read the model in case not done yet
        if self.root.mode.is_reading_mode():
            logger.info("Reading full model before clipping..")
            self.read()

        # 1. Clip staticmaps
        logger.info("Clipping staticmaps..")
        basins_name = self._MAPS["basins"]
        flwdir_name = self._MAPS["flwdir"]
        reservoir_name = self._MAPS["reservoir_area_id"]

        self.staticmaps.clip(
            region=region,
            inverse_clip=inverse_clip,
            crs=crs,
            basins_name=basins_name,
            flwdir_name=flwdir_name,
            reservoir_name=reservoir_name,
            reservoir_maps=list(reservoir_maps.keys()),
            **kwargs,
        )

        # Re-derive flwdir after clipping
        self._flwdir = None  # make sure old flwdir object is removed
        self.flwdir

        # 2. Reinitialize geoms, re-create basins/rivers/outlets and clip the rest
        logger.info("Re-generating/clipping staticgeoms..")
        old_geoms = self.geoms.data.copy()
        self.geoms.clear()
        self.basins
        self.rivers
        self.setup_outlets()
        exclude_geoms = ["basins", "basins_highres", "region", "rivers", "outlets"]
        for name, gdf in old_geoms.items():
            if name not in exclude_geoms:
                logger.debug(f"Clipping geometry {name}..")
                self.set_geoms(
                    geometry=gdf.clip(self.basins, keep_geom_type=True),
                    name=name,
                )

        # 3. Clip states
        if clip_states:
            self.states.clip(
                reservoir_name=reservoir_name,
                reservoir_states=list(reservoir_states.keys()),
            )
        else:
            self.states._data = xr.Dataset()  # clear states

        # 4. Clip forcing
        if clip_forcing:
            self.forcing.clip()
        else:
            self.forcing._data = xr.Dataset()  # clear forcing

        # 5. Update config
        if reservoir_name not in self.staticmaps.data:
            self.config.remove_reservoirs(
                input=list(reservoir_maps.values()),
                state=list(reservoir_states.values()),
            )

    # I/O
    @hydromt_step
    def write(
        self,
        config_filename: str | None = None,
        grid_filename: str | None = None,
        geoms_folder: str = "staticgeoms",
        forcing_filename: str | None = None,
        states_filename: str | None = None,
    ):
        """
        Write the complete model schematization and configuration to file.

        From this function, the output filenames/folder of the different components can
        be set. If not set, the default filenames/folder are used.

        To change more advanced settings, use the specific write methods directly.

        Parameters
        ----------
        config_filename : str, optional
            Name of the config file, relative to model root. By default None to use the
            default name.
        grid_filename : str, optional
            Name of the grid file, relative to model root/dir_input. By default None
            to use the name as defined in the model config file.
        geoms_folder : str, optional
            Name of the geoms folder relative to grid_filename (ie model
            root/dir_input). By default 'staticgeoms'.
        forcing_filename : str, optional
            Name of the forcing file relative to model root/dir_input. By default None
            to use the name as defined in the model config file.
        states_filename : str, optional
            Name of the states file relative to model root/dir_input. By default None
            to use the name as defined in the model config file.
        """
        logger.info(f"Write model data to {self.root.path}")
        # if in r, r+ mode, only write updated components
        if not self.root.is_writing_mode():
            logger.warning("Cannot write in read-only mode")
            return
        self.write_data_catalog()
        _ = self.config.data  # try to read default if not yet set
        if "staticmaps" in self.components:
            self.write_grid(filename=grid_filename)
            self.staticmaps.write_region(
                filename=str(Path(geoms_folder) / "region.geojson"), to_wgs84=True
            )
        if "geoms" in self.components:
            self.write_geoms(folder=geoms_folder)
        if "forcing" in self.components:
            self.write_forcing(filename=forcing_filename)
        if "tables" in self.components:
            self.write_tables()
        if "states" in self.components:
            self.write_states(filename=states_filename)

        # Write the config last as variables can get set in other write methods
        self.write_config(filename=config_filename)

    @hydromt_step
    def read(
        self,
        config_filename: str | None = None,
        geoms_folder: str = "staticgeoms",
    ):
        """Read components from disk.

        From this function, the input filenames/folder of the config and geoms
        components can be set. For the others, the filenames/folder as defined in the
        config file are used.

        To change more advanced settings, use the specific read methods directly.

        Parameters
        ----------
        config_filename : str | None, optional
            Name of the config file, relative to model root. By default None to use the
            default name.
        geoms_folder : str | None, optional
            Name of the geoms folder relative to grid_filename (ie model
            root/dir_input). By default 'staticgeoms'.
        """
        self.read_config(filename=config_filename)
        self.read_grid()
        self.read_geoms(folder=geoms_folder)
        self.read_forcing()
        self.read_states()
        self.read_tables()

    @hydromt_step
    def read_config(
        self,
        filename: str | None = None,
    ):
        """
        Read config from <root/filename>.

        Parameters
        ----------
        filename : str, optional
            Name of the config file. By default None to use the default name
            wflow.toml.
        """
        # Call the component
        self.config.read(filename)

    @hydromt_step
    def write_config(
        self,
        filename: str | None = None,
        config_root: Path | str | None = None,
    ):
        """
        Write the model ``config`` file to ``<config_root>/<config_fn>``.

        Parameters
        ----------
        filename : str, optional
            Name of the config file. Default is ``None``, which uses the
            default name ``wflow.toml``.
        config_root : str or Path, optional
            Root folder to write the config file. If ``None`` (default),
            the model root is used. Can be absolute or relative to the model root.
        """
        # Call the component
        self.config.write(filename, config_root)

    def read_grid(
        self,
        **kwargs,
    ):
        """Read grid model data.

        Checks the path of the file in the config toml using both ``input.path_static``
        and ``dir_input``. If not found uses the default path ``staticmaps.nc`` in the
        root folder.
        Key-word arguments are passed to :py:meth:`~hydromt._io.readers._read_nc`

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        # Call the component method
        self.staticmaps.read(**kwargs)

    @hydromt_step
    def write_grid(
        self,
        filename: str | None = None,
        **kwargs,
    ):
        """
        Write grid to wflow static data file.

        Checks the path of the file in the config toml using both ``input.path_static``
        and ``dir_input``. If not found uses the default path ``staticmaps.nc`` in the
        root folder.

        If filename is supplied, the config will be updated.

        Parameters
        ----------
        filename : Path, str, optional
            Name or path to the outgoing staticmaps file (including extension).
            This is the path/name relative to the root folder and if present the
            ``dir_input`` folder. By default None.
        **kwargs : dict
            Additional keyword arguments to be passed to the `write_nc` method.
        """
        # Call the component write method
        self.staticmaps.write(filename=filename, **kwargs)

    def set_grid(
        self,
        data: xr.DataArray | xr.Dataset,
        name: str | None = None,
    ):
        """Add data to grid.

        All layers of grid must have identical spatial coordinates. If basin data is
        available the grid will be masked to that upon setting.

        The first fix is when data with a time axis is being added. Since Wflow.jl
        v0.7.3, cyclic data at different lengths (12, 365, 366) is supported, as long as
        the dimension name starts with "time". In this function, a check is done if a
        time axis with that exact shape is already present in the grid object, and will
        use that dimension (and its name) to set the data. If a time dimension does not
        yet exist with that shape, it is created following the format
        "time_{length_data}".

        The other fix is that when the model is updated with a different number of
        layers, this is not automatically updated correctly. With this fix, the old
        layer dimension is removed (including all associated data), and the new data is
        added with the correct "layer" dimension.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to grid
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray and
            ignored if data is a Dataset
        """
        # Call the staticmaps set method
        self.staticmaps.set(data, name=name)

    @hydromt_step
    def read_geoms(
        self,
        folder: str = "staticgeoms",
    ):
        """
        Read static geometries and adds to ``geoms``.

        If ``dir_input`` is set in the config, the path where all static geometries are
        read, will be constructed as ``<model_root>/<dir_input>/<geoms_fn>``.
        Where <dir_input> is relative to the model root. Depending on the config value
        ``dir_input``, the path will be constructed differently.

        Parameters
        ----------
        folder : str, optional
            Folder name/path where the static geometries are stored relative to the
            model root and ``dir_input`` if any. By default "staticgeoms".
        """
        self.geoms.read(folder=folder)

    @hydromt_step
    def write_geoms(
        self,
        folder: str = "staticgeoms",
        precision: int | None = None,
        to_wgs84: bool = False,
        **kwargs: dict,
    ):
        """
        Write geoms in GeoJSON format.

        Checks the path of ``geoms_fn`` using both model root and
        ``dir_input``. If not found uses the default path ``staticgeoms`` in the root
        folder.

        Parameters
        ----------
        folder : str, optional
            Folder name/path where the static geometries are stored relative to the
            model root and ``dir_input`` if any. By default "staticgeoms".
        precision : int, optional
            Decimal precision to write the geometries. By default None to use 1 decimal
            for projected crs and 6 for non-projected crs.
        to_wgs84 : bool, optional
            If True, geometries are transformed to WGS84 before writing. By default
            False, which means geometries are written in their original CRS.
        """
        # Call the component write method
        self.geoms.write(
            folder=folder,
            to_wgs84=to_wgs84,
            precision=precision,
            **kwargs,
        )

    def set_geoms(self, geometry: gpd.GeoDataFrame | gpd.GeoSeries, name: str):
        """
        Set geometries to the model.

        This is an inherited method from HydroMT-core's GeomsModel.set_geoms.
        """
        self.geoms.set(
            geom=geometry,
            name=name,
        )

    @hydromt_step
    def read_forcing(
        self,
        **kwargs,
    ):
        """
        Read forcing.

        Checks the path of the file in the config toml using both ``input.path_forcing``
        and ``dir_input``. If not found uses the default path ``inmaps.nc`` in the
        root folder.

        If several files are used using '*' in ``input.path_forcing``, all corresponding
        files are read and merged into one xarray dataset before being split to one
        xarray DataArray per forcing variable in the hydromt ``forcing`` dictionary.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the `read_nc` method.
        """
        self.forcing.read(**kwargs)

    @hydromt_step
    def write_forcing(
        self,
        filename: str | None = None,
        output_frequency: str | None = None,
        time_chunk: int = 1,
        time_units="days since 1900-01-01T00:00:00",
        decimals: int = 2,
        overwrite: bool = False,
        **kwargs,
    ):
        """
        Write forcing at ``<root>/dir_input/<filename>`` in model-ready format.

        If no ``filename`` path is provided and ``path_forcing`` from the
        ``wflow.toml`` exists, the following default filenames are used:

        * With downscaling:
          ``inmaps_sourcePd_sourceTd_methodPET_freq_startyear_endyear.nc``

        * Without downscaling:
          ``inmaps_sourceP_sourceT_methodPET_freq_startyear_endyear.nc``

        Parameters
        ----------
        filename : Path or str, optional
            Path to save output NetCDF file. If ``None``, the name is read
            from the ``wflow.toml`` file.
        output_frequency : str, optional
            Write several files for the forcing according to frequency. For example
            ``'Y'`` for one file per year or ``'M'`` for one file per month.
            By default writes a single file.
            For more options see:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        time_chunk : int, optional
            Chunk size on the time dimension when saving to disk. Default is ``1``.
        time_units : str, optional
            Common time units when writing several NetCDF forcing files.
            Default is ``"days since 1900-01-01T00:00:00"``.
        decimals : int, optional
            Number of decimals to use when writing the forcing data. Default is ``2``.
        overwrite : bool, optional
            Whether to overwrite existing files. Default is ``False`` unless the model
            is in w+ mode (FORCED_WRITE).
        **kwargs : dict
            Additional keyword arguments passed to ``write_nc``.
        """
        # Call the component
        self.forcing.write(
            filename=filename,
            output_frequency=output_frequency,
            time_chunk=time_chunk,
            time_units=time_units,
            decimals=decimals,
            overwrite=overwrite or self.root.mode.value == "w+",
            **kwargs,
        )

    def set_forcing(
        self,
        data: xr.DataArray | xr.Dataset,
        name: str | None = None,
    ):
        """Add data for the forcing component."""
        self.forcing.set(data=data, name=name)

    @hydromt_step
    def read_states(self):
        """
        Read states at <root/dir_input/state.path_input>.

        Checks the path of the file in the config toml using both ``state.path_input``
        and ``dir_input``. If not found uses the default path ``instate/instates.nc``
        in the root folder.
        """
        self.states.read()

    @hydromt_step
    def write_states(self, filename: str | None = None):
        """
        Write states at <root/dir_input/state.path_input> in model ready format.

        Checks the path of the file in the config toml using both ``state.path_input``
        and ``dir_input``. If not found uses the default path ``instate/instates.nc``
        in the root folder.
        If filename is provided, it will be used and config ``state.path_input``
        will be updated accordingly.

        Parameters
        ----------
        filename : str, Path, optional
            Name of the states file, relative to model root and ``dir_input`` if any.
            By default None to use the name as defined in the model config file.
        """
        # Write
        self.states.write(filename=filename)

    def set_states(
        self,
        data: xr.DataArray | xr.Dataset,
        name: str | None = None,
    ):
        """Add data to states.

        All layers of states must have identical spatial coordinates. This is an
        inherited method from HydroMT-core's StatesModel.set_states with some fixes.
        If basin data is available the states will be masked to that upon setting.

        Parameters
        ----------
        data: xarray.DataArray or xarray.Dataset
            new map layer to add to states
        name: str, optional
            Name of new map layer, this is used to overwrite the name of a DataArray and
            ignored if data is a Dataset
        """
        if self._MAPS["basins"] in self.staticmaps.data:
            data = utils.mask_raster_from_layer(
                data, self.staticmaps.data[self._MAPS["basins"]]
            )
        # fall back on default set_states behaviour
        self.states.set(data, name=name)

    @hydromt_step
    def read_outputs(self):
        """
        Read outputs at <root/dir_output>.

        Reads netcdf_grid at ``output.netcdf_grid.path``, netcdf_scalar at
        ``output.netcdf_scalar.path`` and csv outputs at ``output.csv.path``.

        Checks if ``dir_output`` is present.
        """
        self.output_grid.read()
        self.output_scalar.read()
        self.output_csv.read()

    @hydromt_step
    def read_tables(self, **kwargs):
        """Read table files at <root> and parse to dict of dataframes.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed to the `pd.read_csv` method

        Returns
        -------
        None
            The tables are read and stored in the `self.tables.data` attribute.
        """
        self.tables.read(float_precision="round_trip", **kwargs)

    @hydromt_step
    def write_tables(self):
        """Write tables at <root>."""
        self.tables.write()

    def set_tables(self, df: pd.DataFrame, name: str):
        """Add table <pandas.DataFrame> to model."""
        self.tables.set(tables=df, name=name)

    def set_root(self, root: Path | str, mode: ModeLike = "w"):
        """Set the model root folder.

        Parameters
        ----------
        root : Path, str
            Path to the model root folder.
        mode : str, optional
            Mode to open the model root folder, by default 'w'.
            Can be 'r' for read-only or 'r+' for read-write.
        """
        self.root.set(Path(root), mode=mode)

    def get_config(
        self,
        *args,
        fallback: Any = None,
        abs_path: bool = False,
    ) -> str | None:
        """Get a config value at key.

        Parameters
        ----------
        args : tuple, str
            keys can given by multiple args: ('key1', 'key2')
            or a string with '.' indicating a new level: ('key1.key2')
        fallback: Any, optional
            fallback value if key(s) not found in config, by default None.
        abs_path: bool, optional
            If True return the absolute path relative to the model root,
            by default False.
            NOTE: this assumes the config is located in model root!

        Returns
        -------
        value : Any
            dictionary value

        Examples
        --------
        >> # self.config = {'a': 1, 'b': {'c': {'d': 2}}}

        >> get_config('a')
        >> 1

        >> get_config('b', 'c', 'd') # identical to get_config('b.c.d')
        >> 2

        >> get_config('b.c') # # identical to get_config('b','c')
        >> {'d': 2}
        """
        return self.config.get_value(
            *args,
            fallback=fallback,
            abs_path=abs_path,
        )

    def set_config(self, *args):
        """
        Update the config toml at key(s) with values.

        This function is made to maintain the structure of your toml file.
        When adding keys it will look for the most specific header present in
        the toml file and add it under that.

        meaning that if you have a config toml that is empty and you run
        ``wflow_model.set_config("input.forcing.scale", 1)``

        it will result in the following file:

        .. code-block:: toml

            input.forcing.scale = 1


        however if your toml file looks like this before:

        .. code-block:: toml

            [input.forcing]

        (i.e. you have a header in there that has no keys)

        then after the insertion it will look like this:

        .. code-block:: toml

            [input.forcing]
            scale = 1


        .. warning::

            Due to limitations of the underlying library it is currently not possible to
            create new headers (i.e. groups like ``input.forcing`` in the example above)
            programmatically, and they will need to be added to the default config
            toml document


        Parameters
        ----------
        args : str, tuple, list
            if tuple or list, minimal length of two
            keys can given by multiple args: ('key1', 'key2', 'value')
            or a string with '.' indicating a new level: ('key1.key2', 'value')

        Examples
        --------
        .. code-block:: ipython

            >> self.config
            >> {'a': 1, 'b': {'c': {'d': 2}}}

            >> self.set_config('a', 99)
            >> {'a': 99, 'b': {'c': {'d': 2}}}

            >> self.set_config('b', 'c', 'd', 99) # identical to set_config('b.d.e', 99)
            >> {'a': 1, 'b': {'c': {'d': 99}}}
        """
        key = ".".join(args[:-1])
        value = args[-1]
        self.config.set(key, value)

    def _update_naming(self, rename_dict: dict):
        """Update the naming of the model variables.

        Parameters
        ----------
        rename_dict: dict
            Dictionary with the wflow variable and new output name in file.
        """
        _wflow_names_inv = {v: k for k, v in self._WFLOW_NAMES.items()}
        _hydromt_names_inv = {v: k for k, v in self._MAPS.items()}
        for wflow_var, new_name in rename_dict.items():
            if wflow_var is None:
                continue
            # Find the previous name in self._WFLOW_NAMES
            old_name = _wflow_names_inv.get(wflow_var, None)
            if old_name is not None:
                # Rename the variable in self._WFLOW_NAMES
                self._WFLOW_NAMES.pop(old_name)
                self._WFLOW_NAMES[new_name] = wflow_var
                # Rename the variable in self._MAPS
                hydromt_name = _hydromt_names_inv.get(old_name, None)
                if hydromt_name is not None:
                    self._MAPS[hydromt_name] = new_name
            else:
                logger.warning(f"Wflow variable {wflow_var} not found, check spelling.")

    def _update_config_variable_name(
        self, data_vars: str | list[str], data_type: str | None = "static"
    ):
        """Update the variable names in the config file.

        Parameters
        ----------
        data_vars: list of str
            List of variable names to update in the config file.
        data_type: str, optional
            Type of data (static, forcing, cyclic, None), by default "static"
        """
        data_vars = [data_vars] if isinstance(data_vars, str) else data_vars
        _prefix = f"input.{data_type}" if data_type is not None else "input"
        for var in data_vars:
            if var in self._WFLOW_NAMES:
                # Get the name from the Wflow variable name
                wflow_var = self._WFLOW_NAMES[var]
                # Update the config variable name
                self.set_config(f"{_prefix}.{wflow_var}", var)
            # else not a wflow variable
            # (spelling mistakes should have been checked in _update_naming)

    ## WFLOW specific data and method
    # Non model component properties
    @property
    def basins(self) -> gpd.GeoDataFrame | None:
        """Returns a basin(s) geometry as a geopandas.GeoDataFrame."""
        if "basins" in self.geoms.data:
            gdf = self.geoms.get("basins")
        elif self._MAPS["basins"] in self.staticmaps.data:
            gdf = (
                self.staticmaps.data[self._MAPS["basins"]]
                .raster.vectorize()
                .set_index("value")
                .sort_index()
            )
            self.set_geoms(gdf, name="basins")
        else:
            logger.warning(f"Basin map {self._MAPS['basins']} not found in grid.")
            gdf = None
        return gdf

    @property
    def basins_highres(self) -> gpd.GeoDataFrame | None:
        """Returns a high resolution basin(s) geometry."""
        if "basins_highres" in self.geoms.data:
            gdf = self.geoms.get("basins_highres")
        else:
            gdf = self.basins
        return gdf

    @property
    def rivers(self) -> gpd.GeoDataFrame | None:
        """Return a river geometry as a geopandas.GeoDataFrame.

        If available, the stream order and upstream area values are added to
        the geometry properties.
        """
        if "rivers" in self.geoms.data:
            gdf = self.geoms.get("rivers")
        elif self._MAPS["rivmsk"] in self.staticmaps.data:
            rivmsk = self.staticmaps.data[self._MAPS["rivmsk"]].values != 0
            # Check if there are river cells in the model before continuing
            if np.any(rivmsk):
                # add stream order 'strord' column
                strord = self.flwdir.stream_order(mask=rivmsk)
                feats = self.flwdir.streams(mask=rivmsk, strord=strord)
                gdf = gpd.GeoDataFrame.from_features(feats)
                gdf.crs = pyproj.CRS.from_user_input(self.crs)
                self.set_geoms(gdf, name="rivers")
        else:
            logger.warning("No river cells detected in the selected basin.")
            gdf = None
        return gdf

    @property
    def flwdir(self) -> pyflwdir.FlwdirRaster:
        """Return the pyflwdir.FlwdirRaster object parsed from wflow ldd."""
        if self._flwdir is None:
            self.set_flwdir()
        return self._flwdir

    def set_flwdir(self, ftype="infer"):
        """Parse pyflwdir.FlwdirRaster object parsed from the wflow ldd."""
        flwdir_name = self._MAPS["flwdir"]
        self._flwdir = flw.flwdir_from_da(
            self.staticmaps.data[flwdir_name],
            ftype=ftype,
            check_ftype=True,
            mask=(self.staticmaps.data[self._MAPS["basins"]] > 0),
        )
