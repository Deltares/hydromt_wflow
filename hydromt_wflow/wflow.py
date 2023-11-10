"""Implement Wflow model class."""
# Implement model class following model API

import codecs
import glob
import logging
import os
from os.path import basename, dirname, isdir, isfile, join
from pathlib import Path
from typing import List, Optional, Union

import geopandas as gpd

# from dask.distributed import LocalCluster, Client, performance_report
import hydromt
import numpy as np
import pandas as pd
import pyflwdir
import pyproj
import toml
import xarray as xr
from dask.diagnostics import ProgressBar
from hydromt import flw
from hydromt.models.model_grid import GridModel
from pyflwdir import core_conversion, core_d8, core_ldd
from shapely.geometry import box

from . import DATADIR, utils, workflows
from .pcrm import read_staticmaps_pcr as read_pcr_func
from .pcrm import write_staticmaps_pcr as write_pcr_func

__all__ = ["WflowModel"]

logger = logging.getLogger(__name__)


class WflowModel(GridModel):
    """Wflow model class."""

    _NAME = "wflow"
    _CONF = "wflow_sbm.toml"
    _CLI_ARGS = {"region": "setup_basemaps", "res": "setup_basemaps"}
    _DATADIR = DATADIR
    _GEOMS = {}
    _MAPS = {
        "flwdir": "wflow_ldd",
        "elevtn": "wflow_dem",
        "subelv": "dem_subgrid",
        "uparea": "wflow_uparea",
        "strord": "wflow_streamorder",
        "basins": "wflow_subcatch",
        "rivlen": "wflow_riverlength",
        "rivmsk": "wflow_river",
        "rivwth": "wflow_riverwidth",
        "lndslp": "Slope",
        "rivslp": "RiverSlope",
        "rivdph": "RiverDepth",
        "rivman": "N_River",
        "gauges": "wflow_gauges",
        "landuse": "wflow_landuse",
        "resareas": "wflow_reservoirareas",
        "reslocs": "wflow_reservoirlocs",
        "lakeareas": "wflow_lakeareas",
        "lakelocs": "wflow_lakelocs",
        "glacareas": "wflow_glacierareas",
        "glacfracs": "wflow_glacierfrac",
        "glacstore": "wflow_glacierstore",
    }
    _FOLDERS = [
        "staticgeoms",
        "instate",
        "run_default",
    ]
    _CATALOGS = join(_DATADIR, "parameters_data.yml")

    def __init__(
        self,
        root: Optional[str] = None,
        mode: Optional[str] = "w",
        config_fn: Optional[str] = None,
        data_libs: Union[List, str] = None,
        logger=logger,
        **artifact_keys,
    ):
        if data_libs is None:
            data_libs = []
        for lib, version in artifact_keys.items():
            logger.warning(
                "Adding a predefined data catalog as key-word argument is deprecated, "
                f"add the catalog as '{lib}={version}'"
                " to the data_libs list instead."
            )
            if not version:  # False or None
                continue
            elif isinstance(version, str):
                lib += f"={version}"
            data_libs = [lib] + data_libs

        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )

        # wflow specific
        self._intbl = dict()
        self._tables = dict()
        self._flwdir = None
        self.data_catalog.from_yml(self._CATALOGS)
        # To be deprecated from v0.6.0 onwards
        self._staticmaps = None
        self._staticgeoms = None

    # COMPONENTS
    def setup_basemaps(
        self,
        region,
        res=1 / 120.0,
        hydrography_fn="merit_hydro",
        basin_index_fn="merit_hydro_index",
        upscale_method="ihu",
    ):
        """
        Build the DEM and flow direction for a Wflow model.

        Setup basemaps sets the ``region`` of interest and ``res``
        (resolution in degrees) of the model.
        All DEM and flow direction related maps are then build.

        If the model resolution is larger than the source data resolution,
        the flow direction is upscaled using the ``upscale_method``, by default the
        Iterative Hydrography Upscaling (IHU).
        The default ``hydrography_fn`` is "merit_hydro"
        (`MERIT hydro <http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/index.html>`_
        at 3 arcsec resolution).
        Alternative sources include "merit_hydro_1k" at 30 arcsec resolution.
        Users can also supply their own elevation and flow direction data
        in any CRS and not only EPSG:4326.

        Note that in order to define the region, using points or bounding box,
        the coordinates of the points / bounding box
        should be in the same CRS than the hydrography data.
        The wflow model will then also be in the same CRS than the
        hydrography data in order to avoid assumptions and reprojection errors.
        If the user wishes to use a different CRS,
        we recommend first to reproject the hydrography data seperately,
        because calling hydromt build.
        You can find examples on how to reproject or prepare hydrography data in the
        `prepare flow directions example notebok
        <https://deltares.github.io/hydromt_wflow/latest/_examples/prepare_ldd.html>`_.

        Adds model layers:

        * **wflow_ldd** map: flow direction in LDD format [-]
        * **wflow_subcatch** map: basin ID map [-]
        * **wflow_uparea** map: upstream area [km2]
        * **wflow_streamorder** map: Strahler stream order [-]
        * **wflow_dem** map: average elevation [m+REF]
        * **dem_subgrid** map: subgrid outlet elevation [m+REF]
        * **Slope** map: average land surface slope [m/m]
        * **basins** geom: basins boundary vector
        * **region** geom: region boundary vector

        Parameters
        ----------
        hydrography_fn : str
            Name of data source for basemap parameters.

            * Required variables: ['flwdir', 'uparea', 'basins', 'strord', 'elevtn']

            * Optional variables: ['lndslp', 'mask']
        basin_index_fn : str
            Name of data source for basin_index data linked to hydrography_fn.
        region : dict
            Dictionary describing region of interest.
            See :py:meth:`hydromt.workflows.basin_mask.parse_region()` for all options
        res : float
            Output model resolution
        upscale_method : {'ihu', 'eam', 'dmm'}
            Upscaling method for flow direction data, by default 'ihu'.

        See Also
        --------
        hydromt.workflows.parse_region
        hydromt.workflows.get_basin_geometry
        workflows.hydrography
        workflows.topography
        """
        self.logger.info("Preparing base hydrography basemaps.")
        # retrieve global data (lazy!)
        ds_org = self.data_catalog.get_rasterdataset(hydrography_fn)

        # get basin geometry and clip data
        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        xy = None
        if kind in ["basin", "subbasin", "outlet"]:
            if basin_index_fn is not None:
                bas_index = self.data_catalog[basin_index_fn]
            else:
                bas_index = None
            geom, xy = hydromt.workflows.get_basin_geometry(
                ds=ds_org,
                kind=kind,
                basin_index=bas_index,
                logger=self.logger,
                **region,
            )
        elif "bbox" in region:
            bbox = region.get("bbox")
            geom = gpd.GeoDataFrame(geometry=[box(*bbox)], crs=4326)
        elif "geom" in region:
            geom = region.get("geom")
        else:
            raise ValueError(f"wflow region argument not understood: {region}")
        if geom is not None and geom.crs is None:
            raise ValueError("wflow region geometry has no CRS")

        ds_org = ds_org.raster.clip_geom(geom, align=res, buffer=10)
        ds_org.coords["mask"] = ds_org.raster.geometry_mask(geom)
        self.logger.debug("Adding basins vector to geoms.")
        self.set_geoms(geom, name="basins")

        # Check on resolution (degree vs meter) depending on ds_org res/crs
        scale_ratio = int(np.round(res / ds_org.raster.res[0]))
        if scale_ratio < 1:
            raise ValueError(
                f"The model resolution {res} should be \
larger than the {hydrography_fn} resolution {ds_org.raster.res[0]}"
            )
        if ds_org.raster.crs.is_geographic:
            if res > 1:  # 111 km
                raise ValueError(
                    f"The model resolution {res} should be smaller than 1 degree \
(111km) for geographic coordinate systems. "
                    "Make sure you provided res in degree rather than in meters."
                )
        # setup hydrography maps and set staticmap attribute with renamed maps
        ds_base, _ = workflows.hydrography(
            ds=ds_org,
            res=res,
            xy=xy,
            upscale_method=upscale_method,
            logger=self.logger,
        )
        # Convert flow direction from d8 to ldd format
        flwdir_data = ds_base["flwdir"].values.astype(np.uint8)  # force dtype
        # if d8 convert to ldd
        if core_d8.isvalid(flwdir_data):
            data = core_conversion.d8_to_ldd(flwdir_data)
            da_flwdir = xr.DataArray(
                name="flwdir",
                data=data,
                coords=ds_base.raster.coords,
                dims=ds_base.raster.dims,
                attrs=dict(
                    long_name="ldd flow direction",
                    _FillValue=core_ldd._mv,
                ),
            )
            ds_base["flwdir"] = da_flwdir
        # Rename and add to grid
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_base.data_vars}
        self.set_grid(ds_base.rename(rmdict))

        # setup topography maps
        ds_topo = workflows.topography(
            ds=ds_org, ds_like=self.grid, method="average", logger=self.logger
        )
        ds_topo["lndslp"] = np.maximum(ds_topo["lndslp"], 0.0)
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_topo.data_vars}
        self.set_grid(ds_topo.rename(rmdict))
        # set basin geometry
        self.logger.debug("Adding region vector to geoms.")
        self.set_geoms(self.region, name="region")

        # update toml for degree/meters if needed
        if ds_base.raster.crs.is_projected:
            self.set_config("model.sizeinmetres", True)

    def setup_rivers(
        self,
        hydrography_fn,
        river_geom_fn=None,
        river_upa: float = 30,
        rivdph_method: str = "powlaw",
        slope_len: float = 2e3,
        min_rivlen_ratio=0.0,
        min_rivdph: float = 1,
        min_rivwth: float = 30,
        smooth_len: float = 5e3,
        rivman_mapping_fn: str = "roughness_river_mapping_default",
        elevtn_map: str = "wflow_dem",
        river_routing: str = "kinematic-wave",
        connectivity: int = 8,
        **kwargs,
    ):
        """
        Set all river parameter maps.

        The river mask is defined by all cells with a mimimum upstream area threshold
        ``river_upa`` [km2].

        The river length is defined as the distance from the subgrid outlet pixel to
        the next upstream subgrid outlet pixel. The ``min_rivlen_ratio`` is the minimum
        global river length to avg. cell resolution ratio and is used as a threshold in
        window based smoothing of river length.

        The river slope is derived from the subgrid elevation difference between pixels
        at a half distance ``slope_len`` [m] up-
        and downstream from the subgrid outlet pixel.

        The river manning roughness coefficient is derived based on reclassification
        of the streamorder map using a lookup table ``rivman_mapping_fn``.

        The river width is derived from the nearest river segment in ``river_geom_fn``.
        Data gaps are filled by the nearest valid upstream value and averaged along
        the flow directions over a length ``smooth_len`` [m]

        The river depth is calculated using the ``rivdph_method``, by default powlaw:
        h = hc*Qbf**hp, which is based on qbankfull discharge from the nearest river
        segment in ``river_geom_fn`` and takes optional arguments for the hc
        (default = 0.27) and hp (default = 0.30) parameters. For other methods see
        :py:meth:`hydromt.workflows.river_depth`.

        If ``river_routing`` is set to "local-inertial", the bankfull elevantion map
        can be conditioned based on the average cell elevation ("wflow_dem")
        or subgrid outlet pixel elevation ("dem_subgrid").
        The subgrid elevation might provide a better representation
        of the river elevation profile, however in combination with
        local-inertial land routing (see :py:meth:`setup_floodplains`)
        the subgrid elevation will likely overestimate the floodplain storage capacity.
        Note that the same input elevation map should be used for river bankfull
        elevation and land elevation when using local-inertial land routing.

        Adds model layers:

        * **wflow_river** map: river mask [-]
        * **wflow_riverlength** map: river length [m]
        * **wflow_riverwidth** map: river width [m]
        * **RiverDepth** map: bankfull river depth [m]
        * **RiverSlope** map: river slope [m/m]
        * **N_River** map: Manning coefficient for river cells [s.m^1/3]
        * **rivers** geom: river vector based on wflow_river mask
        * **hydrodem** map: hydrologically conditioned elevation [m+REF]

        Parameters
        ----------
        hydrography_fn : str, Path
            Name of data source for hydrography data.
            Must be same as setup_basemaps for consistent results.

            * Required variables: ['flwdir', 'uparea', 'elevtn']
            * Optional variables: ['rivwth', 'qbankfull']
        river_geom_fn : str, Path, optional
            Name of data source for river data.

            * Required variables: ['rivwth', 'qbankfull']
        river_upa : float
            minimum upstream area threshold for the river map [km2]
        slope_len : float
            length over which the river slope is calculated [km]
        min_rivlen_ratio: float
            Ratio of cell resolution used minimum length threshold in a moving
            window based smoothing of river length, by default 0.0
            The river length smoothing is skipped if min_riverlen_ratio = 0.
            For details about the river length smoothing,
            see :py:meth:`pyflwdir.FlwdirRaster.smooth_rivlen`
        rivdph_method : {'gvf', 'manning', 'powlaw'}
            see :py:meth:`hydromt.workflows.river_depth` for details, by default \
                "powlaw"
        river_routing : {'kinematic-wave', 'local-inertial'}
            Routing methodology to be used, by default "kinematic-wave".
        smooth_len : float, optional
            Length [m] over which to smooth the output river width and depth,
            by default 5e3
        min_rivdph : float, optional
            Minimum river depth [m], by default 1.0
        min_rivwth : float, optional
            Minimum river width [m], by default 30.0
        elevtn_map : str, optional
            by default "wflow_dem"

        See Also
        --------
        workflows.river_bathymetry
        hydromt.workflows.river_depth
        pyflwdir.FlwdirRaster.river_depth
        setup_floodplains
        """
        self.logger.info("Preparing river maps.")

        rivdph_methods = ["gvf", "manning", "powlaw"]
        if rivdph_method not in rivdph_methods:
            raise ValueError(f'"{rivdph_method}" unknown. Select from {rivdph_methods}')

        routing_options = ["kinematic-wave", "local-inertial"]
        if river_routing not in routing_options:
            raise ValueError(
                f'river_routing="{river_routing}" unknown. \
Select from {routing_options}.'
            )

        # read data
        ds_hydro = self.data_catalog.get_rasterdataset(
            hydrography_fn, geom=self.region, buffer=10
        )
        ds_hydro.coords["mask"] = ds_hydro.raster.geometry_mask(self.region)

        # get rivmsk, rivlen, rivslp
        # read model maps and revert wflow to hydromt map names
        inv_rename = {v: k for k, v in self._MAPS.items() if v in self.grid}
        ds_riv = workflows.river(
            ds=ds_hydro,
            ds_model=self.grid.rename(inv_rename),
            river_upa=river_upa,
            slope_len=slope_len,
            channel_dir="up",
            min_rivlen_ratio=min_rivlen_ratio,
            logger=self.logger,
        )[0]

        ds_riv["rivmsk"] = ds_riv["rivmsk"].assign_attrs(
            river_upa=river_upa, slope_len=slope_len, min_rivlen_ratio=min_rivlen_ratio
        )
        dvars = ["rivmsk", "rivlen", "rivslp"]
        rmdict = {k: self._MAPS.get(k, k) for k in dvars}
        self.set_grid(ds_riv[dvars].rename(rmdict))

        # TODO make separate workflows.river_manning  method
        # Make N_River map from csv file with mapping
        # between streamorder and N_River value
        strord = self.grid[self._MAPS["strord"]].copy()
        df = self.data_catalog.get_dataframe(rivman_mapping_fn)
        # max streamorder value above which values get the same N_River value
        max_str = df.index[-2]
        # if streamorder value larger than max_str, assign last value
        strord = strord.where(strord <= max_str, max_str)
        # handle missing value (last row of csv is mapping of missing values)
        strord = strord.where(strord != strord.raster.nodata, -999)
        strord.raster.set_nodata(-999)
        ds_nriver = workflows.landuse(
            da=strord,
            ds_like=self.grid,
            df=df,
            logger=self.logger,
        )
        self.set_grid(ds_nriver)

        # get rivdph, rivwth
        # while we still have setup_riverwidth one can skip river_bathymetry here
        # TODO make river_geom_fn required after removing setup_riverwidth
        if river_geom_fn is not None:
            gdf_riv = self.data_catalog.get_geodataframe(
                river_geom_fn, geom=self.region
            )
            # reread model data to get river maps
            inv_rename = {v: k for k, v in self._MAPS.items() if v in self.grid}
            ds_riv1 = workflows.river_bathymetry(
                ds_model=self.grid.rename(inv_rename),
                gdf_riv=gdf_riv,
                method=rivdph_method,
                smooth_len=smooth_len,
                min_rivdph=min_rivdph,
                min_rivwth=min_rivwth,
                logger=self.logger,
                **kwargs,
            )
            rmdict = {k: v for k, v in self._MAPS.items() if k in ds_riv1.data_vars}
            self.set_grid(ds_riv1.rename(rmdict))
            # update config
            self.set_config("input.lateral.river.bankfull_depth", self._MAPS["rivdph"])

        self.logger.debug("Adding rivers vector to geoms.")
        self.geoms.pop("rivers", None)  # remove old rivers if in geoms
        self.rivers  # add new rivers to geoms

        # Add hydrologically conditioned elevation map for the river, if required
        if river_routing == "local-inertial":
            postfix = {"wflow_dem": "_avg", "dem_subgrid": "_subgrid"}.get(
                elevtn_map, ""
            )
            name = f"hydrodem{postfix}"

            ds_out = flw.dem_adjust(
                da_flwdir=self.grid[self._MAPS["flwdir"]],
                da_elevtn=self.grid[elevtn_map],
                da_rivmsk=self.grid[self._MAPS["rivmsk"]],
                flwdir=self.flwdir,
                connectivity=connectivity,
                river_d8=True,
                logger=self.logger,
            ).rename(name)
            self.set_grid(ds_out)

            # update toml model.river_routing
            self.logger.debug(
                f'Update wflow config model.river_routing="{river_routing}"'
            )
            self.set_config("model.river_routing", river_routing)

            self.set_config("input.lateral.river.bankfull_depth", self._MAPS["rivdph"])
            self.set_config("input.lateral.river.bankfull_elevation", name)
        else:
            self.set_config("model.river_routing", river_routing)

    def setup_floodplains(
        self,
        hydrography_fn,
        floodplain_type: str,
        ### Options for 1D floodplains
        river_upa: Optional[float] = None,
        flood_depths: List = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        ### Options for 2D floodplains
        elevtn_map: str = "wflow_dem",
        connectivity: int = 4,
    ):
        """
        Add floodplain information to the model schematistation.

        The user can define what type of floodplains are required (1D or 2D),
        through the ``floodplain_type`` argument.

        If ``floodplain_type`` is set to "1d", a floodplain profile is derived for every
        river cell. It adds a map with floodplain volume per flood depth,
        which is used in the wflow 1D floodplain schematisation.

        Note, it is important to use the same river uparea value as used in the
        :py:meth:`setup_rivers` method.

        If ``floodplain_type`` is set to "2d", this component adds
        a hydrologically conditioned elevation (hydrodem) map for
        land routing (local-inertial). For this options, landcells need to be
        conditioned to D4 flow directions otherwise pits may remain in the land cells.

        The conditioned elevation can be based on the average cell elevation
        ("wflow_dem") or subgrid outlet pixel elevation ("dem_subgrid").
        Note that the subgrid elevation will likely overestimate
        the floodplain storage capacity.

        Additionally, note that the same input elevation map should be used for river
        bankfull elevation and land elevation when using local-inertial land routing.

        Requires :py:meth:`setup_rivers` to be executed beforehand
        (with ``river_routing`` set to "local-inertial").

        Adds model layers:

        * **floodplain_volume** map: map with floodplain volumes, has flood depth as \
            third dimension [m3] (for 1D floodplains)
        * **hydrodem** map: hydrologically conditioned elevation [m+REF] (for 2D \
            floodplains)

        Parameters
        ----------
        floodplain_type: {"1d", "2d"}
            Option defining the type of floodplains, see below what arguments
            are related to the different floodplain types
        hydrography_fn : str, Path
            Name of data source for hydrography data. Must be same as setup_basemaps for
            consistent results.

            * Required variables: ['flwdir', 'uparea', 'elevtn']
        river_upa : float, optional
            (1D floodplains) minimum upstream area threshold for drain in the HAND.
            Optional value, as it is inferred from the grid metadata,
            to be consistent with setup_rivers.
        flood_depths : tuple of float, optional
            (1D floodplains) flood depths at which a volume is derived.
            By default [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

        elevtn_map: {"wflow_dem", "dem_subgrid"}
            (2D floodplains) Name of staticmap to hydrologically condition.
            By default "wflow_dem"

        See Also
        --------
        workflows.river_floodplain_volume
        hydromt.flw.dem_adjust
        pyflwdir.FlwdirRaster.dem_adjust
        setup_rivers
        """
        if self.get_config("model.river_routing") != "local-inertial":
            raise ValueError(
                "Floodplains (1d or 2d) are currently only supported with \
local intertial river routing"
            )

        r_list = ["1d", "2d"]
        if floodplain_type not in r_list:
            raise ValueError(
                f'river_routing="{floodplain_type}" unknown. Select from {r_list}.'
            )

        # Adjust settings based on floodplain_type selection
        if floodplain_type == "1d":
            floodplain_1d = True
            land_routing = "kinematic-wave"

            if not hasattr(pyflwdir.FlwdirRaster, "ucat_volume"):
                self.logger.warning("This method requires pyflwdir >= 0.5.6")
                return

            self.logger.info("Preparing 1D river floodplain_volume map.")

            # read data
            ds_hydro = self.data_catalog.get_rasterdataset(
                hydrography_fn, geom=self.region, buffer=10
            )
            ds_hydro.coords["mask"] = ds_hydro.raster.geometry_mask(self.region)

            # try to get river uparea from grid, throw error if not specified
            # or when found but different from specified value
            new_river_upa = self.grid[self._MAPS["rivmsk"]].attrs.get(
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
            self.logger.debug(f"Using river_upa value value of: {new_river_upa}")

            # get river floodplain volume
            inv_rename = {v: k for k, v in self._MAPS.items() if v in self.grid}
            da_fldpln = workflows.river_floodplain_volume(
                ds=ds_hydro,
                ds_model=self.grid.rename(inv_rename),
                river_upa=new_river_upa,
                flood_depths=flood_depths,
                logger=self.logger,
            )

            # check if the layer already exists, since overwriting with different
            # flood_depth values is not working properly if this is the case
            if "floodplain_volume" in self.grid:
                self.logger.warning(
                    "Layer `floodplain_volume` already in grid, removing layer \
and `flood_depth` dimension to ensure correctly \
setting new flood_depth dimensions"
                )
                self._grid = self._grid.drop_dims("flood_depth")

            self.set_grid(da_fldpln, "floodplain_volume")

        elif floodplain_type == "2d":
            floodplain_1d = False
            land_routing = "local-inertial"

            if elevtn_map not in self.grid:
                raise ValueError(f'"{elevtn_map}" not found in grid')

            postfix = {"wflow_dem": "_avg", "dem_subgrid": "_subgrid"}.get(
                elevtn_map, ""
            )
            name = f"hydrodem{postfix}"

            self.logger.info(f"Preparing {name} map for land routing.")
            name = f"hydrodem{postfix}_D{connectivity}"
            self.logger.info(f"Preparing {name} map for land routing.")
            ds_out = flw.dem_adjust(
                da_flwdir=self.grid[self._MAPS["flwdir"]],
                da_elevtn=self.grid[elevtn_map],
                da_rivmsk=self.grid[self._MAPS["rivmsk"]],
                flwdir=self.flwdir,
                connectivity=connectivity,
                river_d8=True,
                logger=self.logger,
            ).rename(name)

            self.set_grid(ds_out)

        # Update config
        self.logger.debug(f'Update wflow config model.floodplain_1d="{floodplain_1d}"')
        self.set_config("model.floodplain_1d", floodplain_1d)
        self.logger.debug(f'Update wflow config model.land_routing="{land_routing}"')
        self.set_config("model.land_routing", land_routing)

        if floodplain_type == "1d":
            # include new input data
            self.set_config(
                "input.lateral.river.floodplain.volume", "floodplain_volume"
            )
            # Add states
            self.set_config("state.lateral.river.floodplain.q", "q_floodplain")
            self.set_config("state.lateral.river.floodplain.h", "h_floodplain")
            self.set_config("state.lateral.land.q", "q_land")
            # Remove local-inertial land states
            if self.get_config("state.lateral.land.qx") is not None:
                self.config["state"]["lateral"]["land"].pop("qx", None)
            if self.get_config("state.lateral.land.qy") is not None:
                self.config["state"]["lateral"]["land"].pop("qy", None)
            if self.get_config("output.lateral.land.qx") is not None:
                self.config["output"]["lateral"]["land"].pop("qx", None)
            if self.get_config("output.lateral.land.qy") is not None:
                self.config["output"]["lateral"]["land"].pop("qy", None)

        else:
            # include new input data
            self.set_config("input.lateral.river.bankfull_elevation", name)
            self.set_config("input.lateral.land.elevation", name)
            # Add local-inertial land routing states
            self.set_config("state.lateral.land.qx", "qx_land")
            self.set_config("state.lateral.land.qy", "qy_land")
            # Remove kinematic-wave and 1d floodplain states
            if self.get_config("state.lateral.land.q") is not None:
                self.config["state"]["lateral"]["land"].pop("q", None)
            if self.get_config("state.lateral.river.floodplain.q") is not None:
                self.config["state"]["lateral"]["river"]["floodplain"].pop("q", None)
            if self.get_config("state.lateral.river.floodplain.h") is not None:
                self.config["state"]["lateral"]["river"]["floodplain"].pop("h", None)
            if self.get_config("output.lateral.land.q") is not None:
                self.config["output"]["lateral"]["land"].pop("q", None)

    def setup_riverwidth(
        self,
        predictor="discharge",
        fill=False,
        fit=False,
        min_wth=1.0,
        precip_fn="chelsa",
        climate_fn="koppen_geiger",
        **kwargs,
    ):
        """
        Set the river width parameter based on power-lay relationship with a predictor.

        By default the riverwidth is estimated based on discharge as ``predictor``
        and used to set the riverwidth globally based on pre-defined power-law
        parameters per climate class. With ``fit`` set to True,
        the power-law relationsship paramters are set on-the-fly.
        With ``fill`` set to True, the estimated river widths are only used
        to fill gaps in the observed data. Alternative ``predictor`` values
        are precip (accumulated precipitation) and uparea (upstream area).
        For these predictors values ``fit`` default to True.
        By default the predictor is based on discharge which is estimated through
        multiple linear regression with precipitation and upstream area
        per climate zone.

        * **wflow_riverwidth** map: river width [m]

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
        fit : bool, optional kwarg
            If True, the power-law parameters are fitted on the fly
            By default True for all but "discharge" predictor.
            A-priori derived parameters will be overwritten if True.
        a, b : float, optional kwarg
            Manual power-law parameters
        min_wth : float
            minimum river width
        precip_fn : {'chelsa'}
            Source of long term precipitation grid if the predictor
            is set to 'discharge' or 'precip'.
        climate_fn: {'koppen_geiger'}
            Source of long-term climate grid if the predictor is set to 'discharge'.
        """
        self.logger.warning(
            'The "setup_riverwidth" method has been deprecated \
and will soon be removed. '
            'You can now use the "setup_river" method for all river parameters.'
        )
        if self._MAPS["rivmsk"] not in self.grid:
            raise ValueError(
                'The "setup_riverwidth" method requires \
to run setup_river method first.'
            )

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

        inv_rename = {v: k for k, v in self._MAPS.items() if v in self.grid}
        da_rivwth = workflows.river_width(
            ds_like=self.grid.rename(inv_rename),
            flwdir=self.flwdir,
            data=data,
            fill=fill,
            fill_outliers=kwargs.pop("fill_outliers", fill),
            min_wth=min_wth,
            mask_names=["lakeareas", "resareas", "glacareas"],
            predictor=predictor,
            a=kwargs.get("a", None),
            b=kwargs.get("b", None),
            logger=self.logger,
            fit=fit,
            **kwargs,
        )

        self.set_grid(da_rivwth, name=self._MAPS["rivwth"])

    def setup_lulcmaps(
        self,
        lulc_fn="globcover",
        lulc_mapping_fn=None,
        lulc_vars=[
            "landuse",
            "Kext",
            "N",
            "PathFrac",
            "RootingDepth",
            "Sl",
            "Swood",
            "WaterFrac",
        ],
    ):
        """
        Derive several wflow maps are derived based on landuse-landcover (LULC) data.

        Currently, ``lulc_fn`` can be set to the "vito", "globcover", "esa_worldcover"
        or "corine", of which lookup tables are constructed to convert lulc classses to
        model parameters based on literature. The data is remapped at its original
        resolution and then resampled to the model resolution using the average
        value, unless noted differently.

        Adds model layers:

        * **landuse** map: Landuse class [-]
        * **Kext** map: Extinction coefficient in the canopy gap fraction equation [-]
        * **Sl** map: Specific leaf storage [mm]
        * **Swood** map: Fraction of wood in the vegetation/plant [-]
        * **RootingDepth** map: Length of vegetation roots [mm]
        * **PathFrac** map: The fraction of compacted or urban area per grid cell [-]
        * **WaterFrac** map: The fraction of open water per grid cell [-]
        * **N** map: Manning Roughness [-]

        Parameters
        ----------
        lulc_fn : {"globcover", "vito", "corine"}
            Name of data source in data_sources.yml file.
        lulc_mapping_fn : str
            Path to a mapping csv file from landuse in source name to
            parameter values in lulc_vars.
        lulc_vars : list
            List of landuse parameters to keep.
            By default \
["landuse","Kext","N","PathFrac","RootingDepth","Sl","Swood","WaterFrac"]
        """
        self.logger.info("Preparing LULC parameter maps.")
        if lulc_mapping_fn is None:
            fn_map = f"{lulc_fn}_mapping_default"
        else:
            fn_map = lulc_mapping_fn
        if not isfile(fn_map) and fn_map not in self.data_catalog:
            raise ValueError(f"LULC mapping file not found: {fn_map}")
        # read landuse map to DataArray
        da = self.data_catalog.get_rasterdataset(
            lulc_fn, geom=self.region, buffer=2, variables=["landuse"]
        )
        df_map = self.data_catalog.get_dataframe(
            fn_map,
            driver_kwargs={"index_col": 0},  # only used if fn_map is a file path
        )
        # process landuse
        ds_lulc_maps = workflows.landuse(
            da=da,
            ds_like=self.grid,
            df=df_map,
            params=lulc_vars,
            logger=self.logger,
        )
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_lulc_maps.data_vars}
        self.set_grid(ds_lulc_maps.rename(rmdict))

    def setup_laimaps(self, lai_fn="modis_lai"):
        """
        Set leaf area index (LAI) climatology maps per month.

        The values are resampled to the model resolution using the average value.
        The only ``lai_fn`` currently supported is "modis_lai" based on MODIS data.

        Adds model layers:

        * **LAI** map: Leaf Area Index climatology [-]
            Resampled from source data using average. Assuming that missing values
            correspond to bare soil, these are set to zero before resampling.

        Parameters
        ----------
        lai_fn : {'modis_lai'}
            Name of data source for LAI parameters, see data/data_sources.yml.

            * Required variables: ['LAI']
        """
        if lai_fn not in self.data_catalog:
            self.logger.warning(f"Invalid source '{lai_fn}', skipping setup_laimaps.")
            return
        # retrieve data for region
        self.logger.info("Preparing LAI maps.")
        da = self.data_catalog.get_rasterdataset(lai_fn, geom=self.region, buffer=2)
        da_lai = workflows.lai(
            da=da,
            ds_like=self.grid,
            logger=self.logger,
        )
        # Rename the first dimension to time
        rmdict = {da_lai.dims[0]: "time"}
        self.set_grid(da_lai.rename(rmdict), name="LAI")

    def setup_config_output_timeseries(
        self,
        mapname: str,
        toml_output: Optional[str] = "csv",
        header: Optional[List[str]] = ["Q"],
        param: Optional[List[str]] = ["lateral.river.q_av"],
        reducer: Optional[List[str]] = None,
    ):
        """Set the default gauge map based on basin outlets.

        Adds model layers:

        * **csv.column** config: csv timeseries to save based on mapname locations
        * **netcdf.variable** config: netcdf timeseries to save based on mapname \
            locations

        Parameters
        ----------
        mapname : str
            Name of the gauge map (in staticmaps.nc) to use for scalar output.
        toml_output : str, optional
            One of ['csv', 'netcdf', None] to update [csv] or [netcdf] section of wflow
            toml file or do nothing. By default, 'csv'.
        header : list, optional
            Save specific model parameters in csv section. This option defines
            the header of the csv file.
            By default saves Q (for lateral.river.q_av).
        param: list, optional
            Save specific model parameters in csv section. This option defines
            the wflow variable corresponding to the
            names in gauge_toml_header. By default saves lateral.river.q_av (for Q).
        reducer: list, optional
            If map is an area rather than a point location, provides the reducer
            for the parameters to save. By default None.
        """
        # # Add new outputcsv section in the config
        if toml_output == "csv" or toml_output == "netcdf":
            self.logger.info(f"Adding {param} to {toml_output} section of toml.")
            # Add map to the input section of config
            basename = (
                mapname
                if not mapname.startswith("wflow")
                else mapname.replace("wflow_", "")
            )
            self.set_config(f"input.{basename}", mapname)
            # Settings and add csv or netcdf sections if not already in config
            # csv
            if toml_output == "csv":
                header_name = "header"
                var_name = "column"
                if self.get_config("csv") is None:
                    self.set_config("csv.path", "output.csv")
            # netcdf
            if toml_output == "netcdf":
                header_name = "name"
                var_name = "variable"
                if self.get_config("netcdf") is None:
                    self.set_config("netcdf.path", "output_scalar.nc")
            # initialise column / varibale section
            if self.get_config(f"{toml_output}.{var_name}") is None:
                self.set_config(f"{toml_output}.{var_name}", [])

            # Add new output column/variable to config
            for o in range(len(param)):
                gauge_toml_dict = {
                    header_name: header[o],
                    "map": basename,
                    "parameter": param[o],
                }
                if reducer is not None:
                    gauge_toml_dict["reducer"]: reducer[o]
                # If the gauge column/variable already exists skip writting twice
                if gauge_toml_dict not in self.config[toml_output][var_name]:
                    self.config[toml_output][var_name].append(gauge_toml_dict)
        else:
            self.logger.info(
                f"toml_output set to {toml_output}, \
skipping adding gauge specific outputs to the toml."
            )

    def setup_outlets(
        self,
        river_only=True,
        toml_output="csv",
        gauge_toml_header=["Q"],
        gauge_toml_param=["lateral.river.q_av"],
    ):
        """Set the default gauge map based on basin outlets.

        Adds model layers:

        * **wflow_gauges** map: gauge IDs map from catchment outlets [-]
        * **gauges** geom: polygon of catchment outlets

        Parameters
        ----------
        river_only : bool, optional
            Only derive outlet locations if they are located on a river instead of
            locations for all catchments, by default True.
        toml_output : str, optional
            One of ['csv', 'netcdf', None] to update [csv] or [netcdf] section of
            wflow toml file or do nothing. By default, 'csv'.
        gauge_toml_header : list, optional
            Save specific model parameters in csv section. This option defines
            the header of the csv file.
            By default saves Q (for lateral.river.q_av).
        gauge_toml_param: list, optional
            Save specific model parameters in csv section. This option defines
            the wflow variable corresponding to the names in gauge_toml_header.
            By default saves lateral.river.q_av (for Q).
        """
        # read existing geoms; important to get the right basin when updating
        # fix in set_geoms / set_geoms method
        self.geoms

        self.logger.info("Gauges locations set based on river outlets.")
        idxs_out = self.flwdir.idxs_pit
        # Only keep river outlets for gauges
        if river_only:
            idxs_out = idxs_out[
                (self.grid[self._MAPS["rivmsk"]] > 0).values.flat[idxs_out]
            ]
        da_out, idxs_out, ids_out = flw.gauge_map(
            self.grid,
            idxs=idxs_out,
            flwdir=self.flwdir,
            logger=self.logger,
        )
        self.set_grid(da_out, name=self._MAPS["gauges"])
        points = gpd.points_from_xy(*self.grid.raster.idx_to_xy(idxs_out))
        gdf = gpd.GeoDataFrame(
            index=ids_out.astype(np.int32), geometry=points, crs=self.crs
        )
        gdf["fid"] = ids_out.astype(np.int32)
        self.set_geoms(gdf, name="gauges")
        self.logger.info("Gauges map based on catchment river outlets added.")

        self.setup_config_output_timeseries(
            mapname="wflow_gauges",
            toml_output=toml_output,
            header=gauge_toml_header,
            param=gauge_toml_param,
        )

    def setup_gauges(
        self,
        gauges_fn: Union[str, Path, gpd.GeoDataFrame],
        index_col: Optional[str] = None,
        snap_to_river: Optional[bool] = True,
        mask: Optional[np.ndarray] = None,
        snap_uparea: Optional[bool] = False,
        max_dist: Optional[float] = 10e3,
        wdw: Optional[int] = 3,
        rel_error: Optional[float] = 0.05,
        derive_subcatch: Optional[bool] = False,
        basename: Optional[str] = None,
        toml_output: Optional[str] = "csv",
        gauge_toml_header: Optional[List[str]] = ["Q", "P"],
        gauge_toml_param: Optional[List[str]] = [
            "lateral.river.q_av",
            "vertical.precipitation",
        ],
        **kwargs,
    ):
        """Set a gauge map based on ``gauges_fn`` data.

        Supported gauge datasets include "grdc"
        or "<path_to_source>" for user supplied csv or geometry files
        with gauge locations. If a csv file is provided, a "x" or "lon" and
        "y" or "lat" column is required and the first column will be used as
        IDs in the map.

        There are three available methods to prepare the gauge map:

        * no snapping: ``mask=None``, ``snap_to_river=False``, ``snap_uparea=False``.
          The gauge locations are used as is.
        * snapping to mask: the gauge locations are snapped to a boolean mask map:
          either provide ``mask`` or set ``snap_to_river=True``
          to snap to the river (default).
          ``max_dist`` can be used to set the maximum distance to snap to the mask.
        * snapping based on upstream area matching: : ``snap_uparea=True``.
          The gauge locations are snapped to the closest matching upstream area value.
          Requires gauges_fn to have an ``uparea`` [km2] column. The closest value will
          be looked for in a cell window of size ``wdw`` and the difference between
          the gauge and the closest value should be smaller than ``rel_error``.

        If ``derive_subcatch`` is set to True, an additional subcatch map is derived
        from the gauge locations.

        Finally the output locations can be added to wflow TOML file sections [csv]
        or [netcdf] using the ``toml_output`` option. The ``gauge_toml_header`` and
        ``gauge_toml_param`` options can be used to define the header and corresponding
        wflow variable names in the TOML file.

        Adds model layers:

        * **wflow_gauges_source** map: gauge IDs map from source [-] (if gauges_fn)
        * **wflow_subcatch_source** map: subcatchment based on gauge locations [-] \
(if derive_subcatch)
        * **gauges_source** geom: polygon of gauges from source
        * **subcatch_source** geom: polygon of subcatchment based on \
gauge locations [-] (if derive_subcatch)

        Parameters
        ----------
        gauges_fn : str, Path, gpd.GeoDataFrame, optional
            Known source name or path to gauges file geometry file, by default None.

            * Required variables if snap_uparea is True: ["uparea"]
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
            the best fit grid cell, only used if snap_area is True.
        derive_subcatch : bool, optional
            Derive subcatch map for gauges, by default False
        basename : str, optional
            Map name in grid (wflow_gauges_basename)
            if None use the gauges_fn basename.
        toml_output : str, optional
            One of ['csv', 'netcdf', None] to update [csv] or [netcdf] section of
            wflow toml file or do nothing. By default, 'csv'.
        gauge_toml_header : list, optional
            Save specific model parameters in csv section.
            This option defines the header of the csv file.
            By default saves Q (for lateral.river.q_av) and
            P (for vertical.precipitation).
        gauge_toml_param: list, optional
            Save specific model parameters in csv section. This option defines
            the wflow variable corresponding to the names in gauge_toml_header.
            By default saves lateral.river.q_av (for Q) and
            vertical.precipitation (for P).
        """
        # Read data
        kwargs = {}
        if isinstance(gauges_fn, gpd.GeoDataFrame):
            gdf_gauges = gauges_fn
            if not np.all(np.isin(gdf_gauges.geometry.type, "Point")):
                raise ValueError(f"{gauges_fn} contains other geometries than Point")
        elif isfile(gauges_fn):
            # try to get epsg number directly, important when writting back data_catalog
            if hasattr(self.crs, "to_epsg"):
                code = self.crs.to_epsg()
            else:
                code = self.crs
            kwargs.update(crs=code)
            gdf_gauges = self.data_catalog.get_geodataframe(
                gauges_fn, geom=self.basins, assert_gtype="Point", **kwargs
            )
        elif self.data_catalog[gauges_fn].data_type == "GeoDataFrame":
            gdf_gauges = self.data_catalog.get_geodataframe(
                gauges_fn, geom=self.basins, assert_gtype="Point", **kwargs
            )
        elif self.data_catalog[gauges_fn].data_type == "GeoDataset":
            da = self.data_catalog.get_geodataset(gauges_fn, geom=self.basins, **kwargs)
            gdf_gauges = da.vector.to_gdf()
            # Check for point geometry
            if not np.all(np.isin(gdf_gauges.geometry.type, "Point")):
                raise ValueError(f"{gauges_fn} contains other geometries than Point")
        else:
            raise ValueError(
                f"{gauges_fn} data source not found or \
incorrect data_type (GeoDataFrame or GeoDataset)."
            )

        # Create basename
        if basename is None:
            basename = os.path.basename(gauges_fn).split(".")[0].replace("_", "-")

        # Create gauge map, subcatch map and update toml
        if gdf_gauges.index.size == 0:
            self.logger.warning(f"No {gauges_fn} gauge locations found within domain")

            return

        # Create the gauges map
        self.logger.info(
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
        idxs = self.grid.raster.xy_to_idx(xs, ys)
        if index_col is not None and index_col in gdf_gauges.columns:
            gdf_gauges = gdf_gauges.set_index(index_col)
        if np.any(gdf_gauges.index == 0):
            self.logger.warning("Gauge ID 0 is not allowed, setting to 1")
            gdf_gauges.index = gdf_gauges.index.values + 1
        ids = gdf_gauges.index.values

        # if snap_to_river use river map as the mask
        if snap_to_river and mask is None:
            mask = self._MAPS["rivmsk"]
        if mask is not None:
            mask = self.grid[mask].values
        if snap_uparea and "uparea" in gdf_gauges.columns:
            # Derive gauge map based on upstream area snapping
            da, idxs, ids = workflows.gauge_map_uparea(
                self.grid,
                gdf_gauges,
                uparea_name="wflow_uparea",
                wdw=wdw,
                rel_error=rel_error,
                logger=self.logger,
            )
        else:
            # Derive gauge map
            da, idxs, ids = flw.gauge_map(
                self.grid,
                idxs=idxs,
                ids=ids,
                stream=mask,
                flwdir=self.flwdir,
                max_dist=max_dist,
                logger=self.logger,
            )
            # Filter gauges that could not be snapped to rivers
            if snap_to_river:
                ids_old = ids.copy()
                da = da.where(self.grid[self._MAPS["rivmsk"]] != 0, da.raster.nodata)
                ids_new = np.unique(da.values[da.values > 0])
                idxs = idxs[np.isin(ids_old, ids_new)]
                ids = da.values.flat[idxs]

        # Add to grid
        mapname = f'{str(self._MAPS["gauges"])}_{basename}'
        self.set_grid(da, name=mapname)

        # geoms
        points = gpd.points_from_xy(*self.grid.raster.idx_to_xy(idxs))
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
        self.set_geoms(gdf_snapped, name=mapname.replace("wflow_", ""))

        # Add output timeseries for gauges in the toml
        self.setup_config_output_timeseries(
            mapname=mapname,
            toml_output=toml_output,
            header=gauge_toml_header,
            param=gauge_toml_param,
        )

        # add subcatch
        if derive_subcatch:
            da_basins = flw.basin_map(self.grid, self.flwdir, idxs=idxs, ids=ids)[0]
            mapname = self._MAPS["basins"] + "_" + basename
            self.set_grid(da_basins, name=mapname)
            gdf_basins = self.grid[mapname].raster.vectorize()
            self.set_geoms(gdf_basins, name=mapname.replace("wflow_", ""))

    def setup_areamap(
        self,
        area_fn: str,
        col2raster: str,
        nodata: Union[int, float] = -1,
    ):
        """Set area map from vector data to save wflow outputs for specific area.

        Adds model layer:

        * **col2raster** map:  output area data map

        Parameters
        ----------
        area_fn : str
            Name of vector data corresponding to wflow output area.
        col2raster : str
            Name of the column from the vector file to rasterize.
        nodata : int/float, optional
            Nodata value to use when rasterizing. Should match the dtype of col2raster.
            By default -1.
        """
        if area_fn not in self.data_catalog:
            self.logger.warning(f"Invalid source '{area_fn}', skipping setup_areamap.")
            return

        self.logger.info(f"Preparing '{col2raster}' map from '{area_fn}'.")
        gdf_org = self.data_catalog.get_geodataframe(
            area_fn, geom=self.basins, dst_crs=self.crs
        )
        if gdf_org.empty:
            self.logger.warning(
                f"No shapes of {area_fn} found within region, skipping areamap."
            )
            return
        else:
            da_area = self.grid.raster.rasterize(
                gdf=gdf_org,
                col_name=col2raster,
                nodata=nodata,
                all_touched=True,
            )
        self.set_grid(da_area.rename(col2raster))

    def setup_lakes(
        self,
        lakes_fn: Union[str, Path],
        rating_curve_fns: List[Union[str, Path]] = None,
        min_area: float = 10.0,
        add_maxstorage: bool = False,
    ):
        """Generate maps of lake areas and outlets.

        Also meant to generate parameters with average lake area,
        depth and discharge values. The data is generated from features with
        ``min_area`` [km2] (default 1 km2) from a database with lake geometry, IDs and
        metadata. Data required are lake ID 'waterbody_id',
        average area 'Area_avg' [m2], average volume 'Vol_avg' [m3],
        average depth 'Depth_avg' [m] and average discharge 'Dis_avg' [m3/s].

        If rating curve data is available for storage and discharge they can be prepared
        via ``rating_curve_fns`` (see below for syntax and requirements).
        Else the parameters 'Lake_b' and 'Lake_e' will be used for discharge and
        for storage a rectangular profile lake is assumed.
        See Wflow documentation for more information.

        If ``add_maxstorage`` is True, the maximum storage of the lake is added to the
        output (controlled lake) based on 'Vol_max' [m3] column of lakes_fn.

        Adds model layers:

        * **wflow_lakeareas** map: lake IDs [-]
        * **wflow_lakelocs** map: lake IDs at outlet locations [-]
        * **LakeArea** map: lake area [m2]
        * **LakeAvgLevel** map: lake average water level [m]
        * **LakeAvgOut** map: lake average discharge [m3/s]
        * **Lake_b** map: lake rating curve coefficient [-]
        * **LakeOutflowFunc** map: option to compute rating curve [-]
        * **LakeStorFunc** map: option to compute storage curve [-]
        * **LakeMaxStorage** map: optional, maximum storage of lake [m3]
        * **lakes** geom: polygon with lakes and wflow lake parameters

        Parameters
        ----------
        lakes_fn :
            Name of data source for lake parameters, see data/data_sources.yml.

            * Required variables for direct use: \
['waterbody_id', 'Area_avg', 'Depth_avg', 'Dis_avg', 'Lake_b', 'Lake_e', \
'LakeOutflowFunc', 'LakeStorFunc', 'LakeThreshold', 'LinkedLakeLocs']

            * Required variables for parameter estimation: \
['waterbody_id', 'Area_avg', 'Vol_avg', 'Depth_avg', 'Dis_avg']
        rating_curve_fns: str, Path, List[str], List[Path], optional
            Data catalog entry/entries or path(s) containing rating curve values
            for lakes. If None then will be derived from properties of lakes_fn.
            Assumes one file per lake (with all variables) and that the lake ID is
            either in the filename or data catalog entry name (eg using placeholder).
            The ID should be placed at the end separated by an underscore (eg
            'rating_curve_12.csv' or 'rating_curve_12')

            * Required variables: ['elevtn', 'volume'] for storage curve and \
['elevtn', 'discharge'] for discharge rating curve
        min_area : float, optional
            Minimum lake area threshold [km2], by default 10.0 km2.
        add_maxstorage : bool, optional
            If True, maximum storage of the lake is added to the output
            (controlled lake) based on 'Vol_max' [m3] column of lakes_fn.
            By default False (natural lake).
        """
        # Derive lake are and outlet maps
        gdf_org, ds_lakes = self._setup_waterbodies(lakes_fn, "lake", min_area)
        if ds_lakes is None:
            return
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_lakes.data_vars}
        ds_lakes = ds_lakes.rename(rmdict)

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
                    self.logger.warning(
                        f"Could not parse integer lake index from \
rating curve fn {fn}. Skipping."
                    )
            # assume lake index will be in the path
            # Assume one rating curve per lake index
            for id in gdf_org["waterbody_id"].values:
                id = int(id)
                # Find if id is is one of the paths in rating_curve_fns
                if id in fns_ids:
                    # Update path based on current waterbody_id
                    i = fns_ids.index(id)
                    rating_fn = rating_curve_fns[i]
                    # Read data
                    if isfile(rating_fn) or rating_fn in self.data_catalog:
                        self.logger.info(
                            f"Preparing lake rating curve data from {rating_fn}"
                        )
                        df_rate = self.data_catalog.get_dataframe(rating_fn)
                        # Add to dict
                        rating_dict[id] = df_rate
                else:
                    self.logger.warning(
                        f"Rating curve file not found for lake with id {id}. \
Using default storage/outflow function parameters."
                    )
        else:
            self.logger.info(
                "No rating curve data provided. \
Using default storage/outflow function parameters."
            )

        # add waterbody parameters
        ds_lakes, gdf_lakes, rating_curves = workflows.waterbodies.lakeattrs(
            ds_lakes, gdf_org, rating_dict, add_maxstorage=add_maxstorage
        )

        # add to grid
        self.set_grid(ds_lakes)
        # write lakes with attr tables to static geoms.
        self.set_geoms(gdf_lakes, name="lakes")
        # add the tables
        for k, v in rating_curves.items():
            self.set_tables(v, name=k)

        # if there are lakes, change True in toml
        # Lake seetings in the toml to update
        lakes_toml = {
            "model.lakes": True,
            "state.lateral.river.lake.waterlevel": "waterlevel_lake",
            "input.lateral.river.lake.area": "LakeArea",
            "input.lateral.river.lake.areas": "wflow_lakeareas",
            "input.lateral.river.lake.b": "Lake_b",
            "input.lateral.river.lake.e": "Lake_e",
            "input.lateral.river.lake.locs": "wflow_lakelocs",
            "input.lateral.river.lake.outflowfunc": "LakeOutflowFunc",
            "input.lateral.river.lake.storfunc": "LakeStorFunc",
            "input.lateral.river.lake.threshold": "LakeThreshold",
            "input.lateral.river.lake.linkedlakelocs": "LinkedLakeLocs",
            "input.lateral.river.lake.waterlevel": "LakeAvgLevel",
        }
        if "LakeMaxStorage" in ds_lakes:
            lakes_toml["input.lateral.river.lake.maxstorage"] = "LakeMaxStorage"
        for option in lakes_toml:
            self.set_config(option, lakes_toml[option])

    def setup_reservoirs(
        self,
        reservoirs_fn: str,
        timeseries_fn: str = None,
        min_area: float = 1.0,
        **kwargs,
    ):
        """Generate maps of reservoir areas and outlets.

        Also meant to generate parameters with average reservoir area, demand,
        min and max target storage capacities and discharge capacity values.

        The data is generated from features with ``min_area`` [km2] (default is 1 km2)
        from a database with reservoir geometry, IDs and metadata.

        Data requirements for direct use (i.e. wflow parameters are data already present
        in reservoirs_fn) are reservoir ID 'waterbody_id', area 'ResSimpleArea' [m2],
        maximum volume 'ResMaxVolume' [m3], the targeted minimum and maximum fraction of
        water volume in the reservoir 'ResTargetMinFrac' and 'ResTargetMaxFrac' [-],
        the average water demand ResDemand [m3/s] and the maximum release of
        the reservoir before spilling 'ResMaxRelease' [m3/s].

        In case the wflow parameters are not directly available they can be computed by
        HydroMT based on time series of reservoir surface water area.
        These time series can be retreived from either the hydroengine or the gwwapi,
        based on the Hylak_id the reservoir, found in the GrandD database.

        The required variables for computation of the parameters with time series data
        are reservoir ID 'waterbody_id', reservoir ID in the HydroLAKES database
        'Hylak_id', average volume 'Vol_avg' [m3], average depth 'Depth_avg' [m],
        average discharge 'Dis_avg' [m3/s] and dam height 'Dam_height' [m].
        To compute parameters without using time series data, the required varibales in
        reservoirs_fn are reservoir ID 'waterbody_id', average area 'Area_avg' [m2],
        average volume 'Vol_avg' [m3], average depth 'Depth_avg' [m], average discharge
        'Dis_avg' [m3/s] and dam height 'Dam_height' [m]
        and minimum / normal / maximum storage capacity of the dam 'Capacity_min',
        'Capacity_norm', 'Capacity_max' [m3].

        Adds model layers:

        * **wflow_reservoirareas** map: reservoir IDs [-]
        * **wflow_reservoirlocs** map: reservoir IDs at outlet locations [-]
        * **ResSimpleArea** map: reservoir area [m2]
        * **ResMaxVolume** map: reservoir max volume [m3]
        * **ResTargetMinFrac** map: reservoir target min frac [m3/m3]
        * **ResTargetFullFrac** map: reservoir target full frac [m3/m3]
        * **ResDemand** map: reservoir demand flow [m3/s]
        * **ResMaxRelease** map: reservoir max release flow [m3/s]
        * **reservoirs** geom: polygon with reservoirs and wflow reservoir parameters

        Parameters
        ----------
        reservoirs_fn : str
            Name of data source for reservoir parameters, see data/data_sources.yml.

            * Required variables for direct use: \
['waterbody_id', 'ResSimpleArea', 'ResMaxVolume', 'ResTargetMinFrac', \
'ResTargetFullFrac', 'ResDemand', 'ResMaxRelease']

            * Required variables for computation with timeseries_fn: \
['waterbody_id', 'Hylak_id', 'Vol_avg', 'Depth_avg', 'Dis_avg', 'Dam_height']

            * Required variables for computation without timeseries_fn: \
['waterbody_id', 'Area_avg', 'Vol_avg', 'Depth_avg', 'Dis_avg', \
'Capacity_max', 'Capacity_norm', 'Capacity_min', 'Dam_height']
        timeseries_fn : str {'gww', 'hydroengine', 'none'}, optional
            Download and use time series of reservoir surface water area to calculate
            and overwrite the reservoir volume/areas of the data source. Timeseries are
            either downloaded from Global Water Watch 'gww' (using gwwapi package) or
            JRC 'jrc' (using hydroengine package). By default None.
        min_area : float, optional
            Minimum reservoir area threshold [km2], by default 1.0 km2.

        """
        # rename to wflow naming convention
        tbls = {
            "resarea": "ResSimpleArea",
            "resdemand": "ResDemand",
            "resfullfrac": "ResTargetFullFrac",
            "resminfrac": "ResTargetMinFrac",
            "resmaxrelease": "ResMaxRelease",
            "resmaxvolume": "ResMaxVolume",
            "resid": "expr1",
        }

        res_toml = {
            "model.reservoirs": True,
            "state.lateral.river.reservoir.volume": "volume_reservoir",
            "input.lateral.river.reservoir.area": "ResSimpleArea",
            "input.lateral.river.reservoir.areas": "wflow_reservoirareas",
            "input.lateral.river.reservoir.demand": "ResDemand",
            "input.lateral.river.reservoir.locs": "wflow_reservoirlocs",
            "input.lateral.river.reservoir.maxrelease": "ResMaxRelease",
            "input.lateral.river.reservoir.maxvolume": "ResMaxVolume",
            "input.lateral.river.reservoir.targetfullfrac": "ResTargetFullFrac",
            "input.lateral.river.reservoir.targetminfrac": "ResTargetMinFrac",
        }

        gdf_org, ds_res = self._setup_waterbodies(reservoirs_fn, "reservoir", min_area)
        # TODO: check if there are missing values in the above columns of
        # the parameters tbls =
        # if everything is present, skip calculate reservoirattrs() and
        # directly make the maps
        if ds_res is not None:
            rmdict = {k: v for k, v in self._MAPS.items() if k in ds_res.data_vars}
            self.set_grid(ds_res.rename(rmdict))

            # add attributes
            # if present use directly
            resattributes = [
                "waterbody_id",
                "ResSimpleArea",
                "ResMaxVolume",
                "ResTargetMinFrac",
                "ResTargetFullFrac",
                "ResDemand",
                "ResMaxRelease",
            ]
            if np.all(np.isin(resattributes, gdf_org.columns)):
                intbl_reservoirs = gdf_org[resattributes]
                reservoir_accuracy = None
                reservoir_timeseries = None
            # else compute
            else:
                (
                    intbl_reservoirs,
                    reservoir_accuracy,
                    reservoir_timeseries,
                ) = workflows.reservoirattrs(
                    gdf=gdf_org, timeseries_fn=timeseries_fn, logger=self.logger
                )
                intbl_reservoirs = intbl_reservoirs.rename(columns=tbls)

            # create a geodf with id of reservoir and gemoetry at outflow location
            gdf_org_points = gpd.GeoDataFrame(
                gdf_org["waterbody_id"],
                geometry=gpd.points_from_xy(gdf_org.xout, gdf_org.yout),
            )
            intbl_reservoirs = intbl_reservoirs.rename(
                columns={"expr1": "waterbody_id"}
            )
            gdf_org_points = gdf_org_points.merge(
                intbl_reservoirs, on="waterbody_id"
            )  # merge
            # add parameter attributes to polygone gdf:
            gdf_org = gdf_org.merge(intbl_reservoirs, on="waterbody_id")

            # write reservoirs with param values to geoms
            self.set_geoms(gdf_org, name="reservoirs")

            for name in gdf_org_points.columns[2:]:
                gdf_org_points[name] = gdf_org_points[name].astype("float32")
                da_res = ds_res.raster.rasterize(
                    gdf_org_points, col_name=name, dtype="float32", nodata=-999
                )
                self.set_grid(da_res)

            # Save accuracy information on reservoir parameters
            if reservoir_accuracy is not None:
                reservoir_accuracy.to_csv(join(self.root, "reservoir_accuracy.csv"))

            if reservoir_timeseries is not None:
                reservoir_timeseries.to_csv(
                    join(self.root, f"reservoir_timeseries_{timeseries_fn}.csv")
                )

            for option in res_toml:
                self.set_config(option, res_toml[option])

    def _setup_waterbodies(self, waterbodies_fn, wb_type, min_area=0.0):
        """Help with common workflow of setup_lakes and setup_reservoir.

        See specific methods for more info about the arguments.
        """
        # retrieve data for basin
        self.logger.info(f"Preparing {wb_type} maps.")
        gdf_org = self.data_catalog.get_geodataframe(
            waterbodies_fn, geom=self.basins, predicate="contains"
        )
        # skip small size waterbodies
        if "Area_avg" in gdf_org.columns and gdf_org.geometry.size > 0:
            min_area_m2 = min_area * 1e6
            gdf_org = gdf_org[gdf_org.Area_avg >= min_area_m2]
        else:
            self.logger.warning(
                f"{wb_type}'s database has no area attribute. "
                f"All {wb_type}s will be considered."
            )
        # get waterbodies maps and parameters
        nb_wb = gdf_org.geometry.size
        ds_waterbody = None
        if nb_wb > 0:
            self.logger.info(
                f"{nb_wb} {wb_type}(s) of sufficient size found within region."
            )
            # add waterbody maps
            uparea_name = self._MAPS["uparea"]
            if uparea_name not in self.grid.data_vars:
                self.logger.warning(
                    f"Upstream area map for {wb_type} outlet setup not found. "
                    "Database coordinates used instead"
                )
                uparea_name = None
            ds_waterbody, gdf_wateroutlet = workflows.waterbodymaps(
                gdf=gdf_org,
                ds_like=self.grid,
                wb_type=wb_type,
                uparea_name=uparea_name,
                logger=self.logger,
            )
            # update xout and yout in gdf_org from gdf_wateroutlet:
            if "xout" in gdf_org.columns and "yout" in gdf_org.columns:
                gdf_org.loc[:, "xout"] = gdf_wateroutlet["xout"]
                gdf_org.loc[:, "yout"] = gdf_wateroutlet["yout"]

        else:
            self.logger.warning(
                f"No {wb_type}s of sufficient size found within region! "
                f"Skipping {wb_type} procedures!"
            )

        # rasterize points polygons in raster.rasterize --
        # you need grid to know the grid
        return gdf_org, ds_waterbody

    def setup_soilmaps(self, soil_fn="soilgrids", ptf_ksatver="brakensiek"):
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
        (3) the c parameter is computed as weighted average over wflow_sbm soil layers \
in soilgrids_2020 versus at specific depths for soilgrids.

        The following maps are added to grid:

        * **thetaS** map: average saturated soil water content [m3/m3]
        * **thetaR** map: average residual water content [m3/m3]
        * **KsatVer** map: vertical saturated hydraulic conductivity at \
soil surface [mm/day]
        * **SoilThickness** map: soil thickness [mm]
        * **SoilMinThickness** map: minimum soil thickness [mm] (equal to SoilThickness)
        * **M** map: model parameter [mm] that controls exponential decline of \
KsatVer with soil depth (fitted with curve_fit (scipy.optimize)), bounds of M are \
    checked
        * **M_** map: model parameter [mm] that controls exponential decline of \
KsatVer with soil depth (fitted with numpy linalg regression), bounds of `M_` are \
    checked
        * **M_original** map: M without checking bounds
        * **M_original_** map: `M_` without checking bounds
        * **f** map: scaling parameter controlling the decline of KsatVer [mm-1] \
(fitted with curve_fit (scipy.optimize)), bounds are checked
        * **f_** map: scaling parameter controlling the decline of KsatVer [mm-1] \
(fitted with numpy linalg regression), bounds are checked
        * **c_0** map: Brooks Corey coefficient [-] based on pore size distribution \
index at depth of 1st soil layer (100 mm) wflow_sbm
        * **c_1** map: idem c_0 at depth 2nd soil layer (400 mm) wflow_sbm
        * **c_2** map: idem c_0 at depth 3rd soil layer (1200 mm) wflow_sbm
        * **c_3** map: idem c_0 at depth 4th soil layer (> 1200 mm) wflow_sbm
        * **KsatVer_[z]cm** map: KsatVer [mm/day] at soil depths [z] of SoilGrids data \
[0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0]
        * **wflow_soil** map: soil texture based on USDA soil texture triangle \
(mapping: [1:Clay, 2:Silty Clay, 3:Silty Clay-Loam, 4:Sandy Clay, 5:Sandy Clay-Loam, \
6:Clay-Loam, 7:Silt, 8:Silt-Loam, 9:Loam, 10:Sand, 11: Loamy Sand, 12:Sandy Loam])


        Parameters
        ----------
        soil_fn : {'soilgrids', 'soilgrids_2020'}
            Name of data source for soil parameter maps, see data/data_sources.yml.
            Should contain info for the 7 soil depths of soilgrids
            (or 6 depths intervals for soilgrids_2020).
            * Required variables: \
['bd_sl*', 'clyppt_sl*', 'sltppt_sl*', 'oc_sl*', 'ph_sl*', \
'sndppt_sl*', 'soilthickness', 'tax_usda']
        ptf_ksatver : {'brakensiek', 'cosby'}
            Pedotransfer function (PTF) to use for calculation KsatVer
            (vertical saturated hydraulic conductivity [mm/day]).
            By default 'brakensiek'.
        """
        self.logger.info("Preparing soil parameter maps.")
        # TODO add variables list with required variable names
        dsin = self.data_catalog.get_rasterdataset(soil_fn, geom=self.region, buffer=2)
        dsout = workflows.soilgrids(
            dsin,
            self.grid,
            ptf_ksatver,
            soil_fn,
            logger=self.logger,
        ).reset_coords(drop=True)
        self.set_grid(dsout)

    def setup_glaciers(self, glaciers_fn="rgi", min_area=1):
        """
        Generate maps of glacier areas, area fraction and volume fraction.

        Also generates tables with temperature threshold, melting factor and snow-to-ice
        convertion fraction.

        The data is generated from features with ``min_area`` [km2] (default is 1 km2)
        from a database with glacier geometry, IDs and metadata.

        The required variables from glaciers_fn dataset are glacier ID 'simple_id'.
        Optionnally glacier area 'AREA' [km2] can be present to filter the glaciers
        by size. If not present it will be computed on the fly.

        Adds model layers:

        * **wflow_glacierareas** map: glacier IDs [-]
        * **wflow_glacierfrac** map: area fraction of glacier per cell [-]
        * **wflow_glacierstore** map: storage (volume) of glacier per cell [mm]
        * **G_TT** map: temperature threshold for glacier melt/buildup [°C]
        * **G_Cfmax** map: glacier melting factor [mm/°C*day]
        * **G_SIfrac** map: fraction of snowpack on top of glacier converted to ice, \
added to glacierstore [-]

        Parameters
        ----------
        glaciers_fn : {'rgi'}
            Name of data source for glaciers, see data/data_sources.yml.

            * Required variables: ['simple_id']
        min_area : float, optional
            Minimum glacier area threshold [km2], by default 0 (all included)
        """
        glac_toml = {
            "model.glacier": True,
            "state.vertical.glacierstore": "glacierstore",
            "input.vertical.glacierstore": "wflow_glacierstore",
            "input.vertical.glacierfrac": "wflow_glacierfrac",
            "input.vertical.g_cfmax": "G_Cfmax",
            "input.vertical.g_tt": "G_TT",
            "input.vertical.g_sifrac": "G_SIfrac",
        }
        # retrieve data for basin
        self.logger.info("Preparing glacier maps.")
        gdf_org = self.data_catalog.get_geodataframe(
            glaciers_fn, geom=self.basins, predicate="intersects"
        )
        # skip small size glacier
        if "AREA" in gdf_org.columns and gdf_org.geometry.size > 0:
            gdf_org = gdf_org[gdf_org["AREA"] >= min_area]
        # get glacier maps and parameters
        nb_glac = gdf_org.geometry.size
        ds_glac = None
        if nb_glac > 0:
            self.logger.info(
                f"{nb_glac} glaciers of sufficient size found within region."
            )
            # add glacier maps
            ds_glac = workflows.glaciermaps(
                gdf=gdf_org,
                ds_like=self.grid,
                id_column="simple_id",
                elevtn_name=self._MAPS["elevtn"],
                logger=self.logger,
            )

            rmdict = {k: v for k, v in self._MAPS.items() if k in ds_glac.data_vars}
            self.set_grid(ds_glac.rename(rmdict))

            self.set_geoms(gdf_org, name="glaciers")

            for option in glac_toml:
                self.set_config(option, glac_toml[option])
        else:
            self.logger.warning(
                "No glaciers of sufficient size found within region!"
                "Skipping glacier procedures!"
            )

    def setup_constant_pars(self, dtype="float32", nodata=-999, **kwargs):
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

        """
        for key, value in kwargs.items():
            nodata = np.dtype(dtype).type(nodata)
            da_param = xr.where(self.grid[self._MAPS["basins"]], value, nodata).astype(
                dtype
            )
            da_param.raster.set_nodata(nodata)

            da_param = da_param.rename(key)
            self.set_grid(da_param)

    def setup_grid_from_raster(
        self,
        raster_fn: str,
        reproject_method: str,
        variables: Optional[List] = None,
        wflow_variables: Optional[List] = None,
        fill_method: Optional[str] = None,
    ) -> List[str]:
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
            Source name of raster data in data_catalog.
        reproject_method: str
            Reprojection method from rasterio.enums.Resampling.
            Available methods: ['nearest', 'bilinear', 'cubic', 'cubic_spline', \
'lanczos', 'average', 'mode', 'gauss', 'max', 'min', 'med', 'q1', 'q3', \
'sum', 'rms']
        variables: list, optional
            List of variables to add to grid from raster_fn. By default all.
        wflow_variables: list, optional
            List of corresponding wflow variables to update the config toml
            (e.g: ["input.vertical.altitude"]).
            Should match the variables list. variables list should be provided unless
            raster_fn contains a single variable (len 1).
        fill_method : str, optional
            If specified, fills nodata values using fill_nodata method.
            Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.

        Returns
        -------
        list
            Names of added model staticmap layers.
        """
        self.logger.info(f"Preparing grid data from raster source {raster_fn}")
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
        ds_out = ds.raster.reproject_like(self.grid, method=reproject_method)
        # Add to grid
        self.set_grid(ds_out)

        # Update config
        if wflow_variables is not None:
            self.logger.info(
                f"Updating the config for wflow_variables: {wflow_variables}"
            )
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
                    self.set_config(wflow_variables[i], variables[i])

    def setup_precip_forcing(
        self,
        precip_fn: str = "era5",
        precip_clim_fn: Optional[str] = None,
        chunksize: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Generate gridded precipitation forcing at model resolution.

        Adds model layer:

        * **precip**: precipitation [mm]

        Parameters
        ----------
        precip_fn : str, default era5
            Precipitation data source, see data/forcing_sources.yml.

            * Required variable: ['precip']
        precip_clim_fn : str, default None
            High resolution climatology precipitation data source to correct
            precipitation, see data/forcing_sources.yml.

            * Required variable: ['precip']
        chunksize: int, optional
            Chunksize on time dimension for processing data (not for saving to disk!).
            If None the data chunksize is used, this can however be optimized for
            large/small catchments. By default None.
        """
        if precip_fn is None:
            return
        starttime = self.get_config("starttime")
        endtime = self.get_config("endtime")
        freq = pd.to_timedelta(self.get_config("timestepsecs"), unit="s")
        mask = self.grid[self._MAPS["basins"]].values > 0

        precip = self.data_catalog.get_rasterdataset(
            precip_fn,
            geom=self.region,
            buffer=2,
            time_tuple=(starttime, endtime),
            variables=["precip"],
        )

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

        precip_out = hydromt.workflows.forcing.precip(
            precip=precip,
            da_like=self.grid[self._MAPS["elevtn"]],
            clim=clim,
            freq=freq,
            resample_kwargs=dict(label="right", closed="right"),
            logger=self.logger,
            **kwargs,
        )

        # Update meta attributes (used for default output filename later)
        precip_out.attrs.update({"precip_fn": precip_fn})
        if precip_clim_fn is not None:
            precip_out.attrs.update({"precip_clim_fn": precip_clim_fn})
        self.set_forcing(precip_out.where(mask), name="precip")

    def setup_temp_pet_forcing(
        self,
        temp_pet_fn: str = "era5",
        pet_method: str = "debruin",
        press_correction: bool = True,
        temp_correction: bool = True,
        wind_correction: bool = True,
        wind_altitude: int = 10,
        reproj_method: str = "nearest_index",
        dem_forcing_fn: str = "era5_orography",
        skip_pet: str = False,
        chunksize: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Generate gridded reference evapotranspiration forcing at model resolution.

        Adds model layer:

        * **pet**: reference evapotranspiration [mm]
        * **temp**: temperature [°C]

        Parameters
        ----------
        temp_pet_fn : str, optional
            Name or path of data source with variables to calculate temperature
            and reference evapotranspiration, see data/forcing_sources.yml.
            By default 'era5_daily_zarr'.

            * Required variable for temperature: ['temp']

            * Required variables for De Bruin reference evapotranspiration: \
['temp', 'press_msl', 'kin', 'kout']

            * Required variables for Makkink reference evapotranspiration: \
['temp', 'press_msl', 'kin']

            * Required variables for daily Penman-Monteith \
reference evapotranspiration: \
either ['temp', 'temp_min', 'temp_max', 'wind', 'rh', 'kin'] \
for 'penman-monteith_rh_simple' or ['temp', 'temp_min', 'temp_max', 'temp_dew', \
'wind', 'kin', 'press_msl', "wind10_u", "wind10_v"] for 'penman-monteith_tdew' \
(these are the variables available in ERA5)
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
        skip_pet : bool, optional
            If True calculate temp only.
        chunksize: int, optional
            Chunksize on time dimension for processing data (not for saving to disk!).
            If None the data chunksize is used, this can however be optimized for
            large/small catchments. By default None.
        """
        if temp_pet_fn is None:
            return
        starttime = self.get_config("starttime")
        endtime = self.get_config("endtime")
        timestep = self.get_config("timestepsecs")
        freq = pd.to_timedelta(timestep, unit="s")
        mask = self.grid[self._MAPS["basins"]].values > 0

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
                    "makking",
                    "penman-monteith_rh_simple",
                    "penman-monteith_tdew",
                ]
                ValueError(f"Unknown pet method {pet_method}, select from {methods}")

        ds = self.data_catalog.get_rasterdataset(
            temp_pet_fn,
            geom=self.region,
            buffer=1,
            time_tuple=(starttime, endtime),
            variables=variables,
            single_var_as_array=False,  # always return dataset
        )
        if chunksize is not None:
            ds = ds.chunk({"time": chunksize})

        dem_forcing = None
        if dem_forcing_fn is not None:
            dem_forcing = self.data_catalog.get_rasterdataset(
                dem_forcing_fn,
                geom=ds.raster.box,  # clip dem with forcing bbox for full coverage
                buffer=2,
                variables=["elevtn"],
            ).squeeze()

        temp_in = hydromt.workflows.forcing.temp(
            ds["temp"],
            dem_model=self.grid[self._MAPS["elevtn"]],
            dem_forcing=dem_forcing,
            lapse_correction=temp_correction,
            logger=self.logger,
            freq=None,  # resample time after pet workflow
            **kwargs,
        )

        if (
            "penman-monteith" in pet_method
        ):  # also downscaled temp_min and temp_max for Penman needed
            temp_max_in = hydromt.workflows.forcing.temp(
                ds["temp_max"],
                dem_model=self.grid[self._MAPS["elevtn"]],
                dem_forcing=dem_forcing,
                lapse_correction=temp_correction,
                logger=self.logger,
                freq=None,  # resample time after pet workflow
                **kwargs,
            )
            temp_max_in.name = "temp_max"

            temp_min_in = hydromt.workflows.forcing.temp(
                ds["temp_min"],
                dem_model=self.grid[self._MAPS["elevtn"]],
                dem_forcing=dem_forcing,
                lapse_correction=temp_correction,
                logger=self.logger,
                freq=None,  # resample time after pet workflow
                **kwargs,
            )
            temp_min_in.name = "temp_min"

            temp_in = xr.merge([temp_in, temp_max_in, temp_min_in])

        if not skip_pet:
            pet_out = hydromt.workflows.forcing.pet(
                ds[variables[1:]],
                temp=temp_in,
                dem_model=self.grid[self._MAPS["elevtn"]],
                method=pet_method,
                press_correction=press_correction,
                wind_correction=wind_correction,
                wind_altitude=wind_altitude,
                reproj_method=reproj_method,
                freq=freq,
                resample_kwargs=dict(label="right", closed="right"),
                logger=self.logger,
                **kwargs,
            )
            # Update meta attributes with setup opt
            opt_attr = {
                "pet_fn": temp_pet_fn,
                "pet_method": pet_method,
            }
            pet_out.attrs.update(opt_attr)
            self.set_forcing(pet_out.where(mask), name="pet")

        # make sure only temp is written to netcdf
        if "penman-monteith" in pet_method:
            temp_in = temp_in["temp"]
        # resample temp after pet workflow
        temp_out = hydromt.workflows.forcing.resample_time(
            temp_in,
            freq,
            upsampling="bfill",  # we assume right labeled original data
            downsampling="mean",
            label="right",
            closed="right",
            conserve_mass=False,
            logger=self.logger,
        )
        # Update meta attributes with setup opt (used for default naming later)
        opt_attr = {
            "temp_fn": temp_pet_fn,
            "temp_correction": str(temp_correction),
        }
        temp_out.attrs.update(opt_attr)
        self.set_forcing(temp_out.where(mask), name="temp")

    def setup_rootzoneclim(
        self,
        run_fn: Union[str, Path, xr.Dataset],
        forcing_obs_fn: Union[str, Path, xr.Dataset],
        forcing_cc_hist_fn: Optional[Union[str, Path, xr.Dataset]] = None,
        forcing_cc_fut_fn: Optional[Union[str, Path, xr.Dataset]] = None,
        chunksize: Optional[int] = 100,
        return_period: Optional[list] = [2, 3, 5, 10, 15, 20, 25, 50, 60, 100],
        Imax: Optional[float] = 2.0,
        start_hydro_year: Optional[str] = "Sep",
        start_field_capacity: Optional[str] = "Apr",
        LAI: Optional[bool] = False,
        rootzone_storage: Optional[bool] = False,
        correct_cc_deficit: Optional[bool] = False,
        time_tuple: Optional[tuple] = None,
        time_tuple_fut: Optional[tuple] = None,
        missing_days_threshold: Optional[int] = 330,
        update_toml_rootingdepth: Optional[str] = "RootingDepth_obs_20",
    ) -> None:
        """
        Set the RootingDepth.

        Done by estimating the catchment-scale root-zone storage capacity from observed
        hydroclimatic data (and optionally also for climate change historical and
        future periods).

        This presents an alternative approach to determine the RootingDepth
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
        the RootingDepth (rootzone_storage / (thetaS-thetaR)).
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

        * **RootingDepth_{forcing}_{RP}** map: rooting depth [mm of the soil column] \
estimated from hydroclimatic data {forcing: obs, cc_hist or cc_fut} for different \
return periods RP. The translation to RootingDepth is done by dividing \
the rootzone_storage by (thetaS - thetaR).
        * **rootzone_storage_{forcing}_{RP}** geom: polygons of rootzone \
storage capacity [mm of water] for each catchment estimated before filling \
the missings with data from downstream catchments.
        * **rootzone_storage_{forcing}_{RP}** map: rootzone storage capacity \
[mm of water] estimated from hydroclimatic data {forcing: obs, cc_hist or cc_fut} for \
different return periods RP. Only if rootzone_storage is set to True!


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
            List with one or more values indiciating the return period(s) (in
            years) for wich the rootzone storage depth should be calculated. The
            default is [2,3,5,10,15,20,25,50,60,100] years.
        Imax : float, optional
            The maximum interception storage capacity [mm]. The default is 2.0 mm.
        start_hydro_year : str, optional
            The start month (abreviated to the first three letters of the month,
            starting with a capital letter) of the hydrological year. The
            default is 'Sep'.
        start_field_capacity : str, optional
            The end of the wet season / commencement of dry season. This is the
            moment when the soil is at field capacity, i.e. there is no storage
            deficit yet. The default is 'Apr'.
        LAI : bool, optional
            Determine whether the LAI will be used to determine Imax. The
            default is False.
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
        time_tuple: tuple, optional
            Select which time period to read from all the forcing files.
            There should be some overlap between the time period available in the
            forcing files for the historical period and in the observed streamflow data.
        missing_days_threshold: int, optional
            Minimum number of days within a year for that year to be counted in
            the long-term Budyko analysis.
        update_toml_rootingdepth: str, optional
            Update the wflow_sbm model config of the RootingDepth variable with
            the estimated RootingDepth.
            The default is RootingDepth_obs_20,
            which requires to have RP 20 in the list provided for \
the return_period argument.
        """
        self.logger.info("Preparing climate based root zone storage parameter maps.")
        # Open the data sets
        ds_obs = self.data_catalog.get_rasterdataset(
            forcing_obs_fn,
            geom=self.region,
            buffer=2,
            variables=["pet", "precip"],
            time_tuple=time_tuple,
        )
        ds_cc_hist = None
        if forcing_cc_hist_fn is not None:
            ds_cc_hist = self.data_catalog.get_rasterdataset(
                forcing_cc_hist_fn,
                geom=self.region,
                buffer=2,
                variables=["pet", "precip"],
                time_tuple=time_tuple,
            )
        ds_cc_fut = None
        if forcing_cc_fut_fn is not None:
            ds_cc_fut = self.data_catalog.get_rasterdataset(
                forcing_cc_fut_fn,
                geom=self.region,
                buffer=2,
                variables=["pet", "precip"],
                time_tuple=time_tuple_fut,
            )
        # observed streamflow data
        dsrun = self.data_catalog.get_geodataset(
            run_fn, single_var_as_array=False, time_tuple=time_tuple
        )

        # make sure dsrun overlaps with ds_obs, otherwise give error
        if dsrun.time[0] < ds_obs.time[0]:
            dsrun = dsrun.sel(time=slice(ds_obs.time[0], None))
        if dsrun.time[-1] > ds_obs.time[-1]:
            dsrun = dsrun.sel(time=slice(None, ds_obs.time[-1]))
        if len(dsrun.time) == 0:
            self.logger.error(
                "No overlapping period between the meteo and observed streamflow data"
            )

        # check if setup_soilmaps and setup_laimaps were run if LAI =True and
        # if rooting_depth = True"
        if (LAI == True) and ("LAI" not in self.grid):
            self.logger.error(
                "LAI variable not found in grid. \
Set LAI to False or run setup_laimaps first"
            )

        if ("thetaR" not in self.grid) or ("thetaS" not in self.grid):
            self.logger.error(
                "thetaS or thetaR variables not found in grid. \
Run setup_soilmaps first"
            )

        # Run the rootzone clim workflow
        dsout, gdf = workflows.rootzoneclim(
            dsrun=dsrun,
            ds_obs=ds_obs,
            ds_like=self.grid,
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
            logger=self.logger,
        )

        # set nodata value outside basin
        dsout = dsout.where(self.grid[self._MAPS["basins"]] > 0, -999)
        for var in dsout.data_vars:
            dsout[var].raster.set_nodata(-999)
        self.set_grid(dsout)
        self.set_geoms(gdf, name="rootzone_storage")

        # update config
        self.set_config("input.vertical.rootingdepth", update_toml_rootingdepth)

    # I/O
    def read(self):
        """Read the complete model schematization and configuration from file."""
        self.read_config()
        self.read_grid()
        self.read_intbl()
        self.read_tables()
        self.read_geoms()
        self.read_forcing()
        self.logger.info("Model read")

    def write(self):
        """Write the complete model schematization and configuration to file."""
        self.logger.info(f"Write model data to {self.root}")
        # if in r, r+ mode, only write updated components
        if not self._write:
            self.logger.warning("Cannot write in read-only mode")
            return
        self.write_data_catalog()
        if self.config:  # try to read default if not yet set
            self.write_config()
        if self._grid:
            self.write_grid()
        if self._tables:
            self.write_tables()
        if self._geoms:
            self.write_geoms()
        if self._forcing:
            self.write_forcing()

    def read_grid(self, **kwargs):
        """Read grid."""
        fn_default = join(self.root, "staticmaps.nc")
        fn = self.get_config("input.path_static", abs_path=True, fallback=fn_default)

        if self.get_config("dir_input") is not None:
            input_dir = self.get_config("dir_input", abs_path=True)
            fn = join(
                input_dir, self.get_config("input.path_static", fallback=fn_default)
            )
            self.logger.info(f"Input directory found {input_dir}")

        if not self._write:
            # start fresh in read-only mode
            self._grid = xr.Dataset()
        if fn is not None and isfile(fn):
            self.logger.info(f"Read grid from {fn}")
            # FIXME: we need a smarter (lazy) solution for big models which also
            # works when overwriting / appending data in the same source!
            ds = xr.open_dataset(
                fn, mask_and_scale=False, decode_coords="all", **kwargs
            ).load()
            ds.close()
            # make sure internally maps are always North -> South oriented
            if ds.raster.res[1] > 0:
                ds = ds.raster.flipud()
            self.set_grid(ds)
        elif len(glob.glob(join(self.root, "staticmaps", "*.map"))) > 0:
            self.read_staticmaps_pcr()

    def read_staticmaps(self, **kwargs):
        """Read staticmaps."""
        self.logger.warning(
            "read_staticmaps is deprecated. Use 'read_grid' instead. "
            "Will be removed in hydromt_wflow v0.6.0"
        )
        self.read_grid(**kwargs)

    def write_grid(self):
        """Write grid."""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        # clean-up grid and write CRS according to CF-conventions
        # TODO replace later with hydromt.raster.gdal_compliant method
        # after core release
        crs = self.grid.raster.crs
        ds_out = self.grid.reset_coords()
        # TODO?!
        # if ds_out.raster.res[1] < 0: # write data with South -> North orientation
        #     ds_out = ds_out.raster.flipud()
        x_dim, y_dim, x_attrs, y_attrs = hydromt.gis_utils.axes_attrs(crs)
        ds_out = ds_out.rename({ds_out.raster.x_dim: x_dim, ds_out.raster.y_dim: y_dim})
        ds_out[x_dim].attrs.update(x_attrs)
        ds_out[y_dim].attrs.update(y_attrs)
        ds_out = ds_out.drop_vars(["mask", "spatial_ref", "ls"], errors="ignore")
        ds_out.rio.write_crs(crs, inplace=True)
        ds_out.rio.write_transform(self.grid.raster.transform, inplace=True)
        ds_out.raster.set_spatial_dims()

        # Remove FillValue Nan for x_dim, y_dim
        encoding = dict()
        for v in [ds_out.raster.x_dim, ds_out.raster.y_dim]:
            ds_out[v].attrs.pop("_FillValue", None)
            encoding[v] = {"_FillValue": None}

        # filename
        fn_default = join(self.root, "staticmaps.nc")
        fn = self.get_config("input.path_static", abs_path=True, fallback=fn_default)
        # Append inputdir if required
        if self.get_config("dir_input") is not None:
            input_dir = self.get_config("dir_input", abs_path=True)
            fn = join(
                input_dir, self.get_config("input.path_static", fallback=fn_default)
            )
        # Check if all sub-folders in fn exists and if not create them
        if not isdir(dirname(fn)):
            os.makedirs(dirname(fn))
        self.logger.info(f"Write grid to {fn}")
        mask = ds_out[self._MAPS["basins"]] > 0
        for v in ds_out.data_vars:
            # nodata is required for all but boolean fields
            if ds_out[v].dtype != "bool":
                ds_out[v] = ds_out[v].where(mask, ds_out[v].raster.nodata)
        ds_out.to_netcdf(fn, encoding=encoding)
        # self.write_grid_pcr()

    def write_staticmaps(self, **kwargs):
        """Write staticmaps."""
        self.logger.warning(
            "write_staticmaps is deprecated. Use 'write_grid' instead. "
            "Will be removed in hydromt_wflow v0.6.0"
        )
        self.write_grid()

    def read_staticmaps_pcr(self, crs=4326, **kwargs):
        """Read and staticmaps at <root/staticmaps> and parse to xarray."""
        self.logger.warning(
            "read_staticmaps_pcr as a method of this object is deprecated. "
            "Use 'read_staticmaps_pcr' for the 'pcrm' submodule instead. "
            "Will be removed in hydromt_wflow v0.5.0"
        )
        if self._read and "chunks" not in kwargs:
            kwargs.update(chunks={"y": -1, "x": -1})
        read_pcr_func(
            self.root,
            crs=crs,
            obj=self,
            **kwargs,
        )

    def write_staticmaps_pcr(self):
        """Write staticmaps at <root/staticmaps> in PCRaster maps format."""
        self.logger.warning(
            "write_staticmaps_pcr as a method of this object is deprecated. "
            "Use 'write_staticmaps_pcr' for the 'pcrm' submodule instead. "
            "Will be removed in hydromt_wflow v0.5.0"
        )
        if not self._write:
            raise IOError("Model opened in read-only mode")
        write_pcr_func(
            self.staticmaps,
            self.root,
        )

    def read_geoms(
        self,
        geom_fn: str = "staticgeoms",
    ):
        """Read and geoms at <root/geoms> and parse to geopandas."""
        if not self._write:
            self._geoms = dict()  # fresh start in read-only mode
        dir_default = join(self.root, "staticmaps.nc")
        dir_mod = dirname(
            self.get_config("input.path_static", abs_path=True, fallback=dir_default)
        )
        fns = glob.glob(join(dir_mod, geom_fn, "*.geojson"))
        if len(fns) > 1:
            self.logger.info("Reading model staticgeom files.")
        for fn in fns:
            name = basename(fn).split(".")[0]
            if name != "region":
                self.set_geoms(gpd.read_file(fn), name=name)

    def read_staticgeoms(
        self,
        geom_fn: str = "staticgeoms",
    ):
        """Read static geometries."""
        self.logger.warning(
            "read_staticgeoms is deprecated. Use 'read_geoms' instead. "
            "Will be removed in hydromt_wflow v0.6.0"
        )
        self.read_geoms(geom_fn)

    def write_geoms(
        self,
        geom_fn: str = "staticgeoms",
    ):
        """Write grid at <root/geoms> in model ready format."""
        # to write use self.geoms[var].to_file()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self.geoms:
            self.logger.info("Writing model staticgeom to file.")
            for name, gdf in self.geoms.items():
                fn_out = join(self.root, geom_fn, f"{name}.geojson")
                gdf.to_file(fn_out, driver="GeoJSON")

    def write_staticgeoms(
        self,
        geom_fn: str = "staticgeoms",
    ):
        """Write static geometries."""
        self.logger.warning(
            "write_staticgeoms is deprecated. Use 'write_geoms' instead. "
            "Will be removed in hydromt_wflow v0.6.0"
        )
        self.write_geoms(geom_fn)

    def read_forcing(self):
        """Read forcing."""
        fn_default = join(self.root, "inmaps.nc")
        fn = self.get_config("input.path_forcing", abs_path=True, fallback=fn_default)

        if self.get_config("dir_input") is not None:
            input_dir = self.get_config("dir_input", abs_path=True)
            fn = join(
                input_dir, self.get_config("input.path_forcing", fallback=fn_default)
            )
            self.logger.info(f"Input directory found {input_dir}")

        if not self._write:
            # start fresh in read-only mode
            self._forcing = dict()
        if fn is not None and isfile(fn):
            self.logger.info(f"Read forcing from {fn}")
            ds = xr.open_dataset(fn, chunks={"time": 30}, decode_coords="all")
            for v in ds.data_vars:
                self.set_forcing(ds[v])
        elif "*" in str(fn):
            self.logger.info(f"Read multiple forcing files using {fn}")
            fns = list(fn.parent.glob(fn.name))
            if len(fns) == 0:
                raise IOError(f"No forcing files found using {fn}")
            ds = xr.open_mfdataset(fns, chunks={"time": 30}, decode_coords="all")
            for v in ds.data_vars:
                self.set_forcing(ds[v])

    def write_forcing(
        self,
        fn_out=None,
        freq_out=None,
        chunksize=1,
        decimals=2,
        time_units="days since 1900-01-01T00:00:00",
        **kwargs,
    ):
        """Write forcing at ``fn_out`` in model ready format.

        If no ``fn_out`` path is provided and path_forcing from the  wflow toml exists,
        the following default filenames are used:

            * Default name format (with downscaling): \
inmaps_sourcePd_sourceTd_methodPET_freq_startyear_endyear.nc
            * Default name format (no downscaling): \
inmaps_sourceP_sourceT_methodPET_freq_startyear_endyear.nc

        Parameters
        ----------
        fn_out: str, Path, optional
            Path to save output netcdf file; if None the name is read from the wflow
            toml file.
        freq_out: str (Offset), optional
            Write several files for the forcing according to fn_freq. For example 'Y'
            for one file per year or 'M' for one file per month.
            By default writes the one file.
            For more options, \
see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        chunksize: int, optional
            Chunksize on time dimension when saving to disk. By default 1.
        decimals: int, optional
            Round the ouput data to the given number of decimals.
        time_units: str, optional
            Common time units when writting several netcdf forcing files.
            By default "days since 1900-01-01T00:00:00".

        """
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self.forcing:
            self.logger.info("Write forcing file")

            # Get default forcing name from forcing attrs
            yr0 = pd.to_datetime(self.get_config("starttime")).year
            yr1 = pd.to_datetime(self.get_config("endtime")).year
            freq = self.get_config("timestepsecs")
            # get output filename
            if fn_out is not None:
                self.set_config("input.path_forcing", fn_out)
                self.write_config()  # re-write config
            else:
                fn_out = self.get_config("input.path_forcing", abs_path=True)
                if "*" in basename(fn_out):
                    # get rid of * in case model had multiple forcing files and
                    # write to single nc file.
                    self.logger.warning("Writing multiple forcing files to one file")
                    fn_out = join(dirname(fn_out), basename(fn_out).replace("*", ""))
                if self.get_config("dir_input") is not None:
                    input_dir = self.get_config("dir_input", abs_path=True)
                    fn_out = join(input_dir, fn_out)

                # get deafult filename if file exists
                if fn_out is None or isfile(fn_out):
                    self.logger.warning(
                        "Netcdf forcing file from input.path_forcing in the TOML  "
                        "already exists, using default name."
                    )
                    sourceP = ""
                    sourceT = ""
                    methodPET = ""
                    if "precip" in self.forcing:
                        val = self.forcing["precip"].attrs.get("precip_clim_fn", None)
                        Pdown = "d" if val is not None else ""
                        val = self.forcing["precip"].attrs.get("precip_fn", None)
                        if val is not None:
                            sourceP = f"_{val}{Pdown}"
                    if "temp" in self.forcing:
                        val = self.forcing["temp"].attrs.get("temp_correction", "False")
                        Tdown = "d" if val == "True" else ""
                        val = self.forcing["temp"].attrs.get("temp_fn", None)
                        if val is not None:
                            sourceT = f"_{val}{Tdown}"
                    if "pet" in self.forcing:
                        val = self.forcing["pet"].attrs.get("pet_method", None)
                        if val is not None:
                            methodPET = f"_{val}"
                    fn_default = (
                        f"inmaps{sourceP}{sourceT}{methodPET}_{freq}_{yr0}_{yr1}.nc"
                    )
                    fn_default_path = join(self.root, fn_default)
                    if isfile(fn_default_path):
                        self.logger.warning(
                            "Netcdf default forcing file already exists, \
skipping write_forcing. "
                            "To overwrite netcdf forcing file: \
change name input.path_forcing "
                            "in setup_config section of the build inifile."
                        )
                        return
                    else:
                        self.set_config("input.path_forcing", fn_default)
                        self.write_config()  # re-write config
                        fn_out = fn_default_path

            # Check if all dates between (starttime, endtime) are in all da forcing
            # Check if starttime and endtime timestamps are correct
            start = pd.to_datetime(self.get_config("starttime"))
            end = pd.to_datetime(self.get_config("endtime"))
            correct_times = False
            for da in self.forcing.values():
                if "time" in da.coords:
                    # only correct dates in toml for standard calendars:
                    if not hasattr(da.indexes["time"], "to_datetimeindex"):
                        times = da.time.values
                        if (start < pd.to_datetime(times[0])) | (start not in times):
                            start = pd.to_datetime(times[0])
                            correct_times = True
                        if (end > pd.to_datetime(times[-1])) | (end not in times):
                            end = pd.to_datetime(times[-1])
                            correct_times = True
            # merge, process and write forcing
            ds = xr.merge([da.reset_coords(drop=True) for da in self.forcing.values()])
            ds.raster.set_crs(self.crs)
            # Send warning, and update config with new start and end time
            if correct_times:
                self.logger.warning(
                    f"Not all dates found in precip_fn changing starttime to \
{start} and endtime to {end} in the toml."
                )
                self.set_config("starttime", start.to_pydatetime())
                self.set_config("endtime", end.to_pydatetime())
                self.write_config()

            if decimals is not None:
                ds = ds.round(decimals)
            # clean-up forcing and write CRS according to CF-conventions
            ds = ds.raster.gdal_compliant(rename_dims=True, force_sn=False)
            ds = ds.drop_vars(["mask", "idx_out"], errors="ignore")

            # write with output chunksizes with single timestep and complete
            # spatial grid to speed up the reading from wflow.jl
            # dims are always ordered (time, y, x)
            ds.raster._check_dimensions()
            chunksizes = (chunksize, ds.raster.ycoords.size, ds.raster.xcoords.size)
            encoding = {
                v: {"zlib": True, "dtype": "float32", "chunksizes": chunksizes}
                for v in ds.data_vars.keys()
            }
            # make sure no _FillValue is written to the time / x_dim / y_dim dimension
            # For several forcing files add common units attributes to time
            for v in ["time", ds.raster.x_dim, ds.raster.y_dim]:
                ds[v].attrs.pop("_FillValue", None)
                encoding[v] = {"_FillValue": None}

            # Check if all sub-folders in fn_out exists and if not create them
            if not isdir(dirname(fn_out)):
                os.makedirs(dirname(fn_out))

            forcing_list = []

            if freq_out is None:
                # with compute=False we get a delayed object which is executed when
                # calling .compute where we can pass more arguments to
                # the dask.compute method
                forcing_list.append([fn_out, ds])
            else:
                self.logger.info(f"Writting several forcing with freq {freq_out}")
                # For several forcing files add common units attributes to time
                encoding["time"] = {"_FillValue": None, "units": time_units}
                # Updating path forcing in config
                fns_out = os.path.relpath(fn_out, self.root)
                fns_out = f"{str(fns_out)[0:-3]}_*.nc"
                self.set_config("input.path_forcing", fns_out)
                self.write_config()  # re-write config
                for label, ds_gr in ds.resample(time=freq_out):
                    # ds_gr = group[1]
                    start = ds_gr["time"].dt.strftime("%Y%m%d")[0].item()
                    fn_out_gr = f"{str(fn_out)[0:-3]}_{start}.nc"
                    forcing_list.append([fn_out_gr, ds_gr])

            for fn_out_gr, ds_gr in forcing_list:
                self.logger.info(f"Process forcing; saving to {fn_out_gr}")
                delayed_obj = ds_gr.to_netcdf(
                    fn_out_gr, encoding=encoding, mode="w", compute=False
                )
                with ProgressBar():
                    delayed_obj.compute(**kwargs)

            # TO profile uncomment lines below to replace lines above
            # from dask.diagnostics import Profiler, CacheProfiler, ResourceProfiler
            # import cachey
            # with Profiler() as prof, CacheProfiler(metric=cachey.nbytes) as cprof,
            # ResourceProfiler() as rprof:
            #     delayed_obj.compute()
            # visualize([prof, cprof, rprof],
            # file_path=r'c:\Users\eilan_dk\work\profile2.html')

    def read_states(self):
        """Read states at <root/?/> and parse to dict of xr.DataArray."""
        if not self._write:
            # start fresh in read-only mode
            self._states = dict()
        # raise NotImplementedError()

    def write_states(self):
        """Write states at <root/?/> in model ready format."""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        # raise NotImplementedError()

    def read_results(self):
        """Read results at <root/?/> and parse to dict of xr.DataArray/xr.Dataset."""
        if not self._write:
            # start fresh in read-only mode
            self._results = dict()

        output_dir = ""
        if self.get_config("dir_output") is not None:
            output_dir = self.get_config("dir_output")

        # Read gridded netcdf (output section)
        nc_fn = self.get_config("output.path", abs_path=True)
        nc_fn = nc_fn.parent / output_dir / nc_fn.name if nc_fn is not None else nc_fn
        if nc_fn is not None and isfile(nc_fn):
            self.logger.info(f"Read results from {nc_fn}")
            ds = xr.open_dataset(nc_fn, chunks={"time": 30}, decode_coords="all")
            # TODO ? align coords names and values of results nc with grid
            self.set_results(ds, name="output")

        # Read scalar netcdf (netcdf section)
        ncs_fn = self.get_config("netcdf.path", abs_path=True)
        ncs_fn = (
            ncs_fn.parent / output_dir / ncs_fn.name if ncs_fn is not None else ncs_fn
        )
        if ncs_fn is not None and isfile(ncs_fn):
            self.logger.info(f"Read results from {ncs_fn}")
            ds = xr.open_dataset(ncs_fn, chunks={"time": 30})
            self.set_results(ds, name="netcdf")

        # Read csv timeseries (csv section)
        csv_fn = self.get_config("csv.path", abs_path=True)
        csv_fn = (
            csv_fn.parent / output_dir / csv_fn.name if csv_fn is not None else csv_fn
        )
        if csv_fn is not None and isfile(csv_fn):
            csv_dict = utils.read_csv_results(
                csv_fn, config=self.config, maps=self.grid
            )
            for key in csv_dict:
                # Add to results
                self.set_results(csv_dict[f"{key}"])

    def write_results(self):
        """Write results at <root/?/> in model ready format."""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        # raise NotImplementedError()

    def read_intbl(self, **kwargs):
        """Read and intbl files at <root/intbl> and parse to xarray."""
        if not self._write:
            self._intbl = dict()  # start fresh in read-only mode
        if not self._read:
            self.logger.info("Reading default intbl files.")
            fns = glob.glob(join(DATADIR, "wflow", "intbl", "*.tbl"))
        else:
            self.logger.info("Reading model intbl files.")
            fns = glob.glob(join(self.root, "intbl", "*.tbl"))
        if len(fns) > 0:
            for fn in fns:
                name = basename(fn).split(".")[0]
                tbl = pd.read_csv(fn, delim_whitespace=True, header=None)
                tbl.columns = [
                    f"expr{i+1}" if i + 1 < len(tbl.columns) else "value"
                    for i in range(len(tbl.columns))
                ]  # rename columns
                self.set_intbl(tbl, name=name)

    def write_intbl(self):
        """Write intbl at <root/intbl> in PCRaster table format."""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self.intbl:
            self.logger.info("Writing intbl files.")
            for name in self.intbl:
                fn_out = join(self.root, "intbl", f"{name}.tbl")
                self.intbl[name].to_csv(fn_out, sep=" ", index=False, header=False)

    def set_intbl(self, df, name):
        """Add intbl <pandas.DataFrame> to model."""
        if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
            raise ValueError("df type not recognized, should be pandas.DataFrame.")
        if name in self._intbl:
            if not self._write:
                raise IOError(f"Cannot overwrite intbl {name} in read-only mode")
            elif self._read:
                self.logger.warning(f"Overwriting intbl: {name}")
        self._intbl[name] = df

    def read_tables(self, **kwargs):
        """Read table files at <root> and parse to dict of dataframes."""
        if not self._write:
            self._tables = dict()  # start fresh in read-only mode

        self.logger.info("Reading model table files.")
        fns = glob.glob(join(self.root, "*.csv"))
        if len(fns) > 0:
            for fn in fns:
                name = basename(fn).split(".")[0]
                tbl = pd.read_csv(fn)
                self.set_tables(tbl, name=name)

    def write_tables(self):
        """Write tables at <root>."""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self.tables:
            self.logger.info("Writing table files.")
            for name in self.tables:
                fn_out = join(self.root, f"{name}.csv")
                self.tables[name].to_csv(fn_out, sep=",", index=False, header=True)

    def set_tables(self, df, name):
        """Add table <pandas.DataFrame> to model."""
        if not (isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)):
            raise ValueError("df type not recognized, should be pandas.DataFrame.")
        if name in self._tables:
            if not self._write:
                raise IOError(f"Cannot overwrite table {name} in read-only mode")
            elif self._read:
                self.logger.warning(f"Overwriting table: {name}")
        self._tables[name] = df

    def _configread(self, fn):
        with codecs.open(fn, "r", encoding="utf-8") as f:
            fdict = toml.load(f)
        return fdict

    def _configwrite(self, fn):
        with codecs.open(fn, "w", encoding="utf-8") as f:
            toml.dump(self.config, f)

    ## WFLOW specific data and methods

    @property
    def intbl(self):
        """Return a dictionary of pandas.DataFrames representing wflow intbl files."""
        if not self._intbl:
            self.read_intbl()
        return self._intbl

    @property
    # Move to core Model API ?
    def tables(self):
        """Return a dictionary of pandas.DataFrames representing wflow intbl files."""
        if not self._tables:
            self.read_tables()
        return self._tables

    @property
    def flwdir(self):
        """Return the pyflwdir.FlwdirRaster object parsed from wflow ldd."""
        if self._flwdir is None:
            self.set_flwdir()
        return self._flwdir

    def set_flwdir(self, ftype="infer"):
        """Parse pyflwdir.FlwdirRaster object parsed from the wflow ldd."""
        flwdir_name = flwdir_name = self._MAPS["flwdir"]
        self._flwdir = flw.flwdir_from_da(
            self.grid[flwdir_name],
            ftype=ftype,
            check_ftype=True,
            mask=(self.grid[self._MAPS["basins"]] > 0),
        )

    @property
    def basins(self):
        """Returns a basin(s) geometry as a geopandas.GeoDataFrame."""
        if "basins" in self.geoms:
            gdf = self.geoms["basins"]
        elif self._MAPS["basins"] in self.grid:
            gdf = (
                self.grid[self._MAPS["basins"]]
                .raster.vectorize()
                .set_index("value")
                .sort_index()
            )
            gdf.index.name = self._MAPS["basins"]
            self.set_geoms(gdf, name="basins")
        else:
            self.logger.warning(f"Basin map {self._MAPS['basins']} not found in grid.")
            gdf = None
        return gdf

    @property
    def rivers(self):
        """Return a river geometry as a geopandas.GeoDataFrame.

        If available, the stream order and upstream area values are added to
        the geometry properties.
        """
        if "rivers" in self.geoms:
            gdf = self.geoms["rivers"]
        elif self._MAPS["rivmsk"] in self.grid:
            rivmsk = self.grid[self._MAPS["rivmsk"]].values != 0
            # Check if there are river cells in the model before continuing
            if np.any(rivmsk):
                # add stream order 'strord' column
                strord = self.flwdir.stream_order(mask=rivmsk)
                feats = self.flwdir.streams(mask=rivmsk, strord=strord)
                gdf = gpd.GeoDataFrame.from_features(feats)
                gdf.crs = pyproj.CRS.from_user_input(self.crs)
                self.set_geoms(gdf, name="rivers")
        else:
            self.logger.warning("No river cells detected in the selected basin.")
            gdf = None
        return gdf

    @property
    def staticgeoms(self):
        """Return static geometries.

        Note: deprecated and will be removed in hydromt_wflow v0.6.0. Use
        :py:attr:`geoms` instead.
        """
        self.logger.warning(
            "staticgeoms is deprecated. Call 'geoms' instead. "
            "Will be removed in hydromt_wflow v0.6.0"
        )
        return self.geoms

    @property
    def staticmaps(self):
        """Return staticmaps.

        Note: deprecated and will be removed in hydromt_wflow v0.6.0. Use
        :py:attr:`grid` instead.
        """
        self.logger.warning(
            "staticmaps is deprecated. Call 'grid' instead. "
            "Will be removed in hydromt_wflow v0.6.0"
        )
        return self.grid

    ## WFLOW specific modification (clip for now) methods

    def clip_grid(
        self,
        region,
        buffer=0,
        align=None,
        crs=4326,
    ):
        """Clip grid to subbasin.

        Parameters
        ----------
        region : dict
            See :meth:`models.wflow.WflowModel.setup_basemaps`
        buffer : int, optional
            Buffer around subbasin in number of pixels, by default 0
        align : float, optional
            Align bounds of region to raster with resolution <align>, by default None
        crs: int, optional
            Default crs of the grid to clip.

        Returns
        -------
        xarray.DataSet
            Clipped grid.
        """
        basins_name = self._MAPS["basins"]
        flwdir_name = self._MAPS["flwdir"]

        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        # translate basin and outlet kinds to geom
        geom = region.get("geom", None)
        bbox = region.get("bbox", None)
        if kind in ["basin", "outlet", "subbasin"]:
            # supply bbox to avoid getting basin bounds first when clipping subbasins
            if kind == "subbasin" and bbox is None:
                region.update(bbox=self.bounds)
            geom, _ = hydromt.workflows.get_basin_geometry(
                ds=self.grid,
                logger=self.logger,
                kind=kind,
                basins_name=basins_name,
                flwdir_name=flwdir_name,
                **region,
            )
        # clip based on subbasin args, geom or bbox
        if geom is not None:
            ds_grid = self.grid.raster.clip_geom(geom, align=align, buffer=buffer)
            ds_grid.coords["mask"] = ds_grid.raster.geometry_mask(geom)
            ds_grid[basins_name] = ds_grid[basins_name].where(
                ds_grid.coords["mask"], self.grid[basins_name].raster.nodata
            )
            ds_grid[basins_name].attrs.update(
                _FillValue=self.grid[basins_name].raster.nodata
            )
        elif bbox is not None:
            ds_grid = self.grid.raster.clip_bbox(bbox, align=align, buffer=buffer)

        # Update flwdir grid and geoms
        if self.crs is None and crs is not None:
            self.set_crs(crs)

        self._grid = xr.Dataset()
        self.set_grid(ds_grid)

        # add pits at edges after clipping
        self._flwdir = None  # make sure old flwdir object is removed
        self.grid[self._MAPS["flwdir"]].data = self.flwdir.to_array("ldd")

        # Reinitiliase geoms and re-create basins/rivers
        self._geoms = dict()
        # self.basins
        # self.rivers
        # now geoms links to geoms which does not exist in every hydromt version
        # remove when updating wflow to new objects
        # Basin shape
        basins = (
            self.grid[basins_name].raster.vectorize().set_index("value").sort_index()
        )
        basins.index.name = basins_name
        self.set_geoms(basins, name="basins")

        rivmsk = self.grid[self._MAPS["rivmsk"]].values != 0
        # Check if there are river cells in the model before continuing
        if np.any(rivmsk):
            # add stream order 'strord' column
            strord = self.flwdir.stream_order(mask=rivmsk)
            feats = self.flwdir.streams(mask=rivmsk, strord=strord)
            gdf = gpd.GeoDataFrame.from_features(feats)
            gdf.crs = pyproj.CRS.from_user_input(self.crs)
            self.set_geoms(gdf, name="rivers")

        # Update reservoir and lakes
        remove_reservoir = False
        if self._MAPS["resareas"] in self.grid:
            reservoir = self.grid[self._MAPS["resareas"]]
            if not np.any(reservoir > 0):
                remove_reservoir = True
                remove_maps = [
                    self._MAPS["resareas"],
                    self._MAPS["reslocs"],
                    "ResSimpleArea",
                    "ResDemand",
                    "ResTargetFullFrac",
                    "ResTargetMinFrac",
                    "ResMaxRelease",
                    "ResMaxVolume",
                ]
                self._grid = self.grid.drop_vars(remove_maps)

        remove_lake = False
        if self._MAPS["lakeareas"] in self.grid:
            lake = self.grid[self._MAPS["lakeareas"]]
            if not np.any(lake > 0):
                remove_lake = True
                remove_maps = [
                    self._MAPS["lakeareas"],
                    self._MAPS["lakelocs"],
                    "LinkedLakeLocs",
                    "LakeStorFunc",
                    "LakeOutflowFunc",
                    "LakeArea",
                    "LakeAvgLevel",
                    "LakeAvgOut",
                    "LakeThreshold",
                    "Lake_b",
                    "Lake_e",
                ]
                self._grid = self.grid.drop_vars(remove_maps)

            # Update tables
            ids = np.unique(lake)
            self._tables = {
                k: v
                for k, v in self.tables.items()
                if not any([str(x) in k for x in ids])
            }

        # Update config
        # Remove the absolute path and if needed remove lakes and reservoirs
        if remove_reservoir:
            # change reservoirs = true to false
            self.set_config("model.reservoirs", False)
            # remove states
            if self.get_config("state.lateral.river.reservoir") is not None:
                del self.config["state"]["lateral"]["river"]["reservoir"]

        if remove_lake:
            # change lakes = true to false
            self.set_config("model.lakes", False)
            # remove states
            if self.get_config("state.lateral.river.lake") is not None:
                del self.config["state"]["lateral"]["river"]["lake"]

    def clip_staticmaps(
        self,
        region,
        buffer=0,
        align=None,
        crs=4326,
    ):
        """Clip staticmaps to subbasin."""
        self.logger.warning(
            "clip_staticmaps is deprecated. Use 'clip_grid' instead. "
            "Will be removed in hydromt_wflow v0.6.0"
        )
        self.clip_grid(region, buffer, align, crs)

    def clip_forcing(self, crs=4326, **kwargs):
        """Return clippped forcing for subbasin.

        Returns
        -------
        xarray.DataSet
            Clipped forcing.

        """
        if len(self.forcing) > 0:
            self.logger.info("Clipping NetCDF forcing..")
            ds_forcing = xr.merge(self.forcing.values()).raster.clip_bbox(
                self.grid.raster.bounds
            )
            self.set_forcing(ds_forcing)
