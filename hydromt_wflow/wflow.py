"""Implement Wflow model class."""

# Implement model class following model API

import codecs
import glob
import logging
import os
from os.path import basename, dirname, isdir, isfile, join
from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd

# from dask.distributed import LocalCluster, Client, performance_report
import hydromt
import numpy as np
import pandas as pd
import pyflwdir
import pyproj
import shapely
import toml
import xarray as xr
from dask.diagnostics import ProgressBar
from hydromt import flw
from hydromt.models.model_grid import GridModel
from hydromt.nodata import NoDataStrategy
from pyflwdir import core_conversion, core_d8, core_ldd
from shapely.geometry import box

from . import utils, workflows
from .naming import HYDROMT_NAMES, WFLOW_NAMES
from .utils import DATADIR

__all__ = ["WflowModel"]

logger = logging.getLogger(__name__)


class WflowModel(GridModel):
    """Wflow model class."""

    _NAME = "wflow"
    _CONF = "wflow_sbm.toml"
    _CLI_ARGS = {"region": "setup_basemaps", "res": "setup_basemaps"}
    _DATADIR = DATADIR
    _GEOMS = {}
    _MAPS = HYDROMT_NAMES
    _FOLDERS = [
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

    # COMPONENTS
    def setup_basemaps(
        self,
        region: Dict,
        hydrography_fn: Union[str, xr.Dataset],
        basin_index_fn: Union[str, xr.Dataset] = None,
        res: Union[float, int] = 1 / 120.0,
        upscale_method: str = "ihu",
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
        in any CRS and not only EPSG:4326. Both arcgis D8 and pcraster LDD
        conventions are supported (see also `PyFlwDir documentation
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

        # Set the basins geometry
        ds_org = ds_org.raster.clip_geom(geom, align=res, buffer=10)
        ds_org.coords["mask"] = ds_org.raster.geometry_mask(geom)
        self.logger.debug("Adding basins vector to geoms.")

        # Set name based on scale_factor
        if scale_ratio != 1:
            self.set_geoms(geom, name="basins_highres")

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
        # Call basins once to set it
        self.basins

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
        hydrography_fn: Union[str, xr.Dataset],
        river_geom_fn: Union[str, gpd.GeoDataFrame] = None,
        river_upa: float = 30,
        rivdph_method: str = "powlaw",
        slope_len: float = 2e3,
        min_rivlen_ratio: float = 0.0,
        min_rivdph: float = 1,
        min_rivwth: float = 30,
        smooth_len: float = 5e3,
        rivman_mapping_fn: Union[
            str, Path, pd.DataFrame
        ] = "roughness_river_mapping_default",
        elevtn_map: str = "wflow_dem",
        river_routing: str = "kinematic-wave",
        connectivity: int = 8,
        **kwargs,
    ):
        """
        Set all river parameter maps.

        The river mask is defined by all cells with a minimum upstream area threshold
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

        If ``river_routing`` is set to "local-inertial", the bankfull elevation map
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
        hydrography_fn : str, Path, xarray.Dataset
        Name of RasterDataset source for hydrography data.
        Must be same as setup_basemaps for consistent results.

        * Required variables: 'flwdir' [LLD or D8 or NEXTXY], 'uparea' [km2],
              'elevtn'[m+REF]
        * Optional variables: 'rivwth' [m], 'qbankfull' [m3/s]
        river_geom_fn : str, Path, geopandas.GeoDataFrame, optional
        Name of GeoDataFrame source for river data.

        * Required variables: 'rivwth' [m], 'qbankfull' [m3/s]
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
        Name of the elevation map in the current WflowModel.grid.
        By default "wflow_dem"

        See Also
        --------
        workflows.river_bathymetry
        hydromt.workflows.river_depth
        pyflwdir.FlwdirRaster.river_depth
        setup_floodplains
        """
        self.logger.info("Preparing river maps.")

        # Check that river_upa threshold is bigger than the maximum uparea in the grid
        if river_upa > float(self.grid[self._MAPS["uparea"]].max()):
            raise ValueError(
                f"river_upa threshold {river_upa} should be larger than the maximum \
uparea in the grid {float(self.grid[self._MAPS['uparea']].max())} in order to create \
river cells."
            )

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
        nodata = df.index[-1]
        # if streamorder value larger than max_str, assign last value
        strord = strord.where(strord <= max_str, max_str)
        # handle missing value (last row of csv is mapping of missing values)
        strord = strord.where(strord != strord.raster.nodata, nodata)
        strord.raster.set_nodata(nodata)
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
        hydrography_fn: Union[str, xr.Dataset],
        floodplain_type: str,
        ### Options for 1D floodplains
        river_upa: Optional[float] = None,
        flood_depths: List = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        ### Options for 2D floodplains
        elevtn_map: str = "wflow_dem",
        connectivity: int = 4,
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
local inertial river routing"
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
        predictor: str = "discharge",
        fill: bool = False,
        fit: bool = False,
        min_wth: float = 1.0,
        precip_fn: Union[str, xr.DataArray] = "chelsa",
        climate_fn: Union[str, xr.DataArray] = "koppen_geiger",
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
        lulc_fn: Union[str, xr.DataArray],
        lulc_mapping_fn: Union[str, Path, pd.DataFrame] = None,
        lulc_vars: List = [
            "landuse",
            "Kext",
            "N",
            "PathFrac",
            "RootingDepth",
            "Sl",
            "Swood",
            "WaterFrac",
            "kc",
            "alpha_h1",
            "h1",
            "h2",
            "h3_high",
            "h3_low",
            "h4",
        ],
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

        * **landuse** map: Landuse class [-]
        * **Kext** map: Extinction coefficient in the canopy gap fraction equation [-]
        * **Sl** map: Specific leaf storage [mm]
        * **Swood** map: Fraction of wood in the vegetation/plant [-]
        * **RootingDepth** map: Length of vegetation roots [mm]
        * **PathFrac** map: The fraction of compacted or urban area per grid cell [-]
        * **WaterFrac** map: The fraction of open water per grid cell [-]
        * **N** map: Manning Roughness [-]
        * **kc** map: Crop coefficient [-]
        * **alpha_h1** map: Root water uptake reduction at soil water pressure head h1
          (0 or 1) [-]
        * **h1** map: Soil water pressure head h1 at which root water uptake is reduced
          (Feddes) [cm]
        * **h2** map: Soil water pressure head h2 at which root water uptake is reduced
          (Feddes) [cm]
        * **h3_high** map: Soil water pressure head h3 at which root water uptake is
          reduced (Feddes) [cm]
        * **h3_low** map: Soil water pressure head h3 at which root water uptake is
          reduced (Feddes) [cm]
        * **h4** map: Soil water pressure head h4 at which root water uptake is reduced
          (Feddes) [cm]


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
            List of landuse parameters to prepare.
            By default ["landuse","Kext","N","PathFrac","RootingDepth","Sl","Swood",
            "WaterFrac"]
        """
        self.logger.info("Preparing LULC parameter maps.")
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
            ds_like=self.grid,
            df=df_map,
            params=lulc_vars,
            logger=self.logger,
        )
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_lulc_maps.data_vars}
        self.set_grid(ds_lulc_maps.rename(rmdict))

        # Add entries to the config
        for name in ds_lulc_maps.data_vars:
            if name in WFLOW_NAMES and WFLOW_NAMES[name] is not None:
                self.set_config(WFLOW_NAMES[name], name)

    def setup_lulcmaps_from_vector(
        self,
        lulc_fn: Union[str, gpd.GeoDataFrame],
        lulc_mapping_fn: Union[str, Path, pd.DataFrame] = None,
        lulc_vars: List = [
            "landuse",
            "Kext",
            "N",
            "PathFrac",
            "RootingDepth",
            "Sl",
            "Swood",
            "WaterFrac",
            "kc",
            "alpha_h1",
            "h1",
            "h2",
            "h3_high",
            "h3_low",
            "h4",
        ],
        lulc_res: Optional[Union[float, int]] = None,
        all_touched: bool = False,
        buffer: int = 1000,
        save_raster_lulc: bool = False,
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

        * **landuse** map: Landuse class [-]
        * **Kext** map: Extinction coefficient in the canopy gap fraction equation [-]
        * **Sl** map: Specific leaf storage [mm]
        * **Swood** map: Fraction of wood in the vegetation/plant [-]
        * **RootingDepth** map: Length of vegetation roots [mm]
        * **PathFrac** map: The fraction of compacted or urban area per grid cell [-]
        * **WaterFrac** map: The fraction of open water per grid cell [-]
        * **N** map: Manning Roughness [-]
        * **kc** map: Crop coefficient [-]
        * **alpha_h1** map: Root water uptake reduction at soil water pressure head h1
          (0 or 1) [-]
        * **h1** map: Soil water pressure head h1 at which root water uptake is reduced
          (Feddes) [cm]
        * **h2** map: Soil water pressure head h2 at which root water uptake is reduced
          (Feddes) [cm]
        * **h3_high** map: Soil water pressure head h3 at which root water uptake is
          reduced (Feddes) [cm]
        * **h3_low** map: Soil water pressure head h3 at which root water uptake is
          reduced (Feddes) [cm]
        * **h4** map: Soil water pressure head h4 at which root water uptake is reduced
          (Feddes) [cm]

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
            List of landuse parameters to prepare.
            By default ["landuse","Kext","N","PathFrac","RootingDepth","Sl","Swood",
            "WaterFrac"]
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

        See Also
        --------
        workflows.landuse_from_vector
        """
        self.logger.info("Preparing LULC parameter maps.")
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
            bbox=self.grid.raster.bounds,
            buffer=buffer,
            variables=["landuse"],
        )
        if save_raster_lulc:
            lulc_out = join(self.root, "maps", "landuse_raster.tif")
        else:
            lulc_out = None

        # process landuse
        ds_lulc_maps = workflows.landuse_from_vector(
            gdf=gdf,
            ds_like=self.grid,
            df=df_map,
            params=lulc_vars,
            lulc_res=lulc_res,
            all_touched=all_touched,
            buffer=buffer,
            lulc_out=lulc_out,
            logger=self.logger,
        )
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_lulc_maps.data_vars}
        self.set_grid(ds_lulc_maps.rename(rmdict))

        # Add entries to the config
        for name in ds_lulc_maps.data_vars:
            if name in WFLOW_NAMES and WFLOW_NAMES[name] is not None:
                self.set_config(WFLOW_NAMES[name], name)

    def setup_laimaps(
        self,
        lai_fn: Union[str, xr.DataArray],
        lulc_fn: Optional[Union[str, xr.DataArray]] = None,
        lulc_sampling_method: str = "any",
        lulc_zero_classes: List[int] = [],
        buffer: int = 2,
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

        * **LAI** map: Leaf Area Index climatology [-]
            Resampled from source data using average. Assuming that missing values
            correspond to bare soil, these are set to zero before resampling.

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
            Buffer around the region to read the data, by default 2.
        """
        # retrieve data for region
        self.logger.info("Preparing LAI maps.")
        da = self.data_catalog.get_rasterdataset(
            lai_fn, geom=self.region, buffer=buffer
        )
        if lulc_fn is not None:
            self.logger.info("Preparing LULC-LAI mapping table.")
            da_lulc = self.data_catalog.get_rasterdataset(
                lulc_fn, geom=self.region, buffer=buffer
            )
            # derive mapping
            df_lai_mapping = workflows.create_lulc_lai_mapping_table(
                da_lulc=da_lulc,
                da_lai=da.copy(),
                sampling_method=lulc_sampling_method,
                lulc_zero_classes=lulc_zero_classes,
                logger=self.logger,
            )
            # Save to csv
            if isinstance(lulc_fn, str) and not isfile(lulc_fn):
                df_fn = f"lai_per_lulc_{lulc_fn}.csv"
            else:
                df_fn = "lai_per_lulc.csv"
            df_lai_mapping.to_csv(join(self.root, df_fn))

        # Resample LAI data to wflow model resolution
        da_lai = workflows.lai(
            da=da,
            ds_like=self.grid,
            logger=self.logger,
        )
        # Rename the first dimension to time
        rmdict = {da_lai.dims[0]: "time"}
        self.set_grid(da_lai.rename(rmdict), name="LAI")

    def setup_laimaps_from_lulc_mapping(
        self,
        lulc_fn: Optional[Union[str, xr.DataArray]],
        lai_mapping_fn: Union[str, pd.DataFrame],
    ):
        """
        Derive cyclic LAI maps from a LULC data source and a LULC-LAI mapping table.

        Adds model layers:

        * **LAI** map: Leaf Area Index climatology [-]
            Resampled from source data using average. Assuming that missing values
            correspond to bare soil, these are set to zero before resampling.

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
        """
        self.logger.info(
            "Preparing LAI maps from LULC data using LULC-LAI mapping table."
        )

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
            ds_like=self.grid,
            df=df_lai_mapping,
            logger=self.logger,
        )
        # Add to grid
        self.set_grid(da_lai, name="LAI")

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
            # initialise column / variable section
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
                    gauge_toml_dict["reducer"] = reducer[o]
                # If the gauge column/variable already exists skip writing twice
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

        If wflow_subcatch is available, the catchment outlets IDs will be matching the
        wflow_subcatch IDs. If not, then IDs from 1 to number of outlets are used.

        Can also add csv/netcdf output settings in the TOML.

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
        # Use the wflow_subcatch ids
        if self._MAPS["basins"] in self.grid:
            ids = self.grid[self._MAPS["basins"]].values.flat[idxs_out]
        else:
            ids = None
        da_out, idxs_out, ids_out = flw.gauge_map(
            self.grid,
            idxs=idxs_out,
            ids=ids,
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
        snap_to_river: bool = True,
        mask: Optional[np.ndarray] = None,
        snap_uparea: bool = False,
        max_dist: float = 10e3,
        wdw: int = 3,
        rel_error: float = 0.05,
        abs_error: float = 50.0,
        fillna: bool = False,
        derive_subcatch: bool = False,
        basename: Optional[str] = None,
        toml_output: str = "csv",
        gauge_toml_header: List[str] = ["Q", "P"],
        gauge_toml_param: List[str] = [
            "lateral.river.q_av",
            "vertical.precipitation",
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
        kwargs : dict, optional
        Additional keyword arguments to pass to the get_data method ie
        get_geodataframe or get_geodataset depending  on the data_type of gauges_fn.
        """
        # Read data
        kwargs = {}
        if isinstance(gauges_fn, gpd.GeoDataFrame):
            gdf_gauges = gauges_fn
            if not np.all(np.isin(gdf_gauges.geometry.type, "Point")):
                raise ValueError(f"{gauges_fn} contains other geometries than Point")
        elif isfile(gauges_fn):
            # try to get epsg number directly, important when writing back data_catalog
            if hasattr(self.crs, "to_epsg"):
                code = self.crs.to_epsg()
            else:
                code = self.crs
            kwargs.update(crs=code)
            gdf_gauges = self.data_catalog.get_geodataframe(
                gauges_fn,
                geom=self.basins,
                assert_gtype="Point",
                handle_nodata=NoDataStrategy.IGNORE,
                **kwargs,
            )
        elif gauges_fn in self.data_catalog:
            if self.data_catalog[gauges_fn].data_type == "GeoDataFrame":
                gdf_gauges = self.data_catalog.get_geodataframe(
                    gauges_fn,
                    geom=self.basins,
                    assert_gtype="Point",
                    handle_nodata=NoDataStrategy.IGNORE,
                    **kwargs,
                )
            elif self.data_catalog[gauges_fn].data_type == "GeoDataset":
                da = self.data_catalog.get_geodataset(
                    gauges_fn,
                    geom=self.basins,
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
                f"{gauges_fn} data source not found or \
incorrect data_type (GeoDataFrame or GeoDataset)."
            )

        # Create basename
        if basename is None:
            basename = os.path.basename(gauges_fn).split(".")[0].replace("_", "-")

        # Check if there is data found
        if gdf_gauges is None:
            self.logger.info("Skipping method, as no data has been found")
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
                mask=mask,
                wdw=wdw,
                rel_error=rel_error,
                abs_error=abs_error,
                fillna=fillna,
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

        # Check if there are gauges left
        if ids.size == 0:
            self.logger.warning(
                "No gauges found within domain after snapping, skipping method."
            )
            return

        # Add to grid
        mapname = f"{str(self._MAPS['gauges'])}_{basename}"
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
        area_fn: Union[str, gpd.GeoDataFrame],
        col2raster: str,
        nodata: Union[int, float] = -1,
    ):
        """Set area map from vector data to save wflow outputs for specific area.

        Adds model layer:

        * **col2raster** map:  output area data map

        Parameters
        ----------
        area_fn : str, geopandas.GeoDataFrame
            Name of GeoDataFrame data corresponding to wflow output area.
        col2raster : str
            Name of the column from `area_fn` to rasterize.
        nodata : int/float, optional
            Nodata value to use when rasterizing. Should match the dtype of `col2raster`
            . By default -1.
        """
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
        lakes_fn: Union[str, Path, gpd.GeoDataFrame],
        rating_curve_fns: List[Union[str, Path, pd.DataFrame]] = None,
        min_area: float = 10.0,
        add_maxstorage: bool = False,
        **kwargs,
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
        Name of GeoDataFrame source for lake parameters, see data/data_sources.yml.

        * Required variables for direct use: \
'waterbody_id' [-], 'Area_avg' [m2], 'Depth_avg' [m], 'Dis_avg' [m3/s], 'Lake_b' [-], \
'Lake_e' [-], 'LakeOutflowFunc' [-], 'LakeStorFunc' [-], 'LakeThreshold' [m], \
'LinkedLakeLocs' [-]

        * Required variables for parameter estimation: \
'waterbody_id' [-], 'Area_avg' [m2], 'Vol_avg' [m3], 'Depth_avg' [m], 'Dis_avg'[m3/s]
        rating_curve_fns: str, Path, pandas.DataFrame, List, optional
        Data catalog entry/entries, path(s) or pandas.DataFrame containing rating
        curve values for lakes. If None then will be derived from properties of
        `lakes_fn`.
        Assumes one file per lake (with all variables) and that the lake ID is
        either in the filename or data catalog entry name (eg using placeholder).
        The ID should be placed at the end separated by an underscore (eg
        'rating_curve_12.csv' or 'rating_curve_12')

        * Required variables for storage curve: 'elevtn' [m+REF], 'volume' [m3]

        * Required variables for rating curve: 'elevtn' [m+REF], 'discharge' [m3/s]
        min_area : float, optional
        Minimum lake area threshold [km2], by default 10.0 km2.
        add_maxstorage : bool, optional
        If True, maximum storage of the lake is added to the output
        (controlled lake) based on 'Vol_max' [m3] column of lakes_fn.
        By default False (natural lake).
        kwargs: optional
        Keyword arguments passed to the method
        hydromt.DataCatalog.get_rasterdataset()
        """
        # Derive lake are and outlet maps
        gdf_org, ds_lakes = self._setup_waterbodies(
            lakes_fn, "lake", min_area, **kwargs
        )
        if ds_lakes is None:
            self.logger.info("Skipping method, as no data has been found")
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
        # Lake settings in the toml to update
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
        reservoirs_fn: Union[str, gpd.GeoDataFrame],
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
'waterbody_id' [-], 'ResSimpleArea' [m2], 'ResMaxVolume' [m3], 'ResTargetMinFrac' \
[m3/m3], 'ResTargetFullFrac' [m3/m3], 'ResDemand' [m3/s], 'ResMaxRelease' [m3/s]

        * Required variables for computation with timeseries_fn: \
'waterbody_id' [-], 'Hylak_id' [-], 'Vol_avg' [m3], 'Depth_avg' [m], 'Dis_avg' [m3/s], \
'Dam_height' [m]

        * Required variables for computation without timeseries_fn: \
'waterbody_id' [-], 'Area_avg' [m2], 'Vol_avg' [m3], 'Depth_avg' [m], 'Dis_avg' \
[m3/s], 'Capacity_max' [m3], 'Capacity_norm' [m3], 'Capacity_min' [m3], 'Dam_height' [m]
        timeseries_fn : {'gww', 'hydroengine', None}, optional
        Download and use time series of reservoir surface water area to calculate
        and overwrite the reservoir volume/areas of the data source. Timeseries are
        either downloaded from Global Water Watch 'gww' (using gwwapi package) or
        JRC 'jrc' (using hydroengine package). By default None.
        min_area : float, optional
        Minimum reservoir area threshold [km2], by default 1.0 km2.
        kwargs: optional
        Keyword arguments passed to the method
        hydromt.DataCatalog.get_rasterdataset()

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

        gdf_org, ds_res = self._setup_waterbodies(
            reservoirs_fn, "reservoir", min_area, **kwargs
        )
        # TODO: check if there are missing values in the above columns of
        # the parameters tbls =
        # if everything is present, skip calculate reservoirattrs() and
        # directly make the maps

        # Skip method if no data is returned
        if ds_res is None:
            self.logger.info("Skipping method, as no data has been found")
            return

        # Continue method if data has been found
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

        # create a geodf with id of reservoir and geometry at outflow location
        gdf_org_points = gpd.GeoDataFrame(
            gdf_org["waterbody_id"],
            geometry=gpd.points_from_xy(gdf_org.xout, gdf_org.yout),
        )
        intbl_reservoirs = intbl_reservoirs.rename(columns={"expr1": "waterbody_id"})
        gdf_org_points = gdf_org_points.merge(
            intbl_reservoirs, on="waterbody_id"
        )  # merge
        # add parameter attributes to polygon gdf:
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

    def _setup_waterbodies(self, waterbodies_fn, wb_type, min_area=0.0, **kwargs):
        """Help with common workflow of setup_lakes and setup_reservoir.

        See specific methods for more info about the arguments.
        """
        # retrieve data for basin
        self.logger.info(f"Preparing {wb_type} maps.")
        if "predicate" not in kwargs:
            kwargs.update(predicate="contains")
        gdf_org = self.data_catalog.get_geodataframe(
            waterbodies_fn,
            geom=self.basins_highres,
            handle_nodata=NoDataStrategy.IGNORE,
            **kwargs,
        )
        if gdf_org is None:
            # Return two times None (similar to main function output), if there is no
            # data found
            return None, None

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
            # update/replace xout and yout in gdf_org from gdf_wateroutlet:
            gdf_org["xout"] = gdf_wateroutlet["xout"]
            gdf_org["yout"] = gdf_wateroutlet["yout"]

        else:
            self.logger.warning(
                f"No {wb_type}s of sufficient size found within region! "
                f"Skipping {wb_type} procedures!"
            )

        # rasterize points polygons in raster.rasterize --
        # you need grid to know the grid
        return gdf_org, ds_waterbody

    def setup_soilmaps(
        self,
        soil_fn: str = "soilgrids",
        ptf_ksatver: str = "brakensiek",
        wflow_thicknesslayers: List[int] = [100, 300, 800],
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
        (3) the c parameter is computed as weighted average over wflow_sbm soil layers \
defined in ``wflow_thicknesslayers``.

        The required data from soilgrids are soil bulk density 'bd_sl*' [g/cm3], \
clay content 'clyppt_sl*' [%], silt content 'sltppt_sl*' [%], organic carbon content \
'oc_sl*' [%], pH 'ph_sl*' [-], sand content 'sndppt_sl*' [%] and soil thickness \
'soilthickness' [cm].

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
        * **c_n** map: Brooks Corey coefficients [-] based on pore size distribution, \
a map for each of the wflow_sbm soil layers (n in total)
        * **KsatVer_[z]cm** map: KsatVer [mm/day] at soil depths [z] of SoilGrids data \
[0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0]
        * **wflow_soil** map: soil texture based on USDA soil texture triangle \
(mapping: [1:Clay, 2:Silty Clay, 3:Silty Clay-Loam, 4:Sandy Clay, 5:Sandy Clay-Loam, \
6:Clay-Loam, 7:Silt, 8:Silt-Loam, 9:Loam, 10:Sand, 11: Loamy Sand, 12:Sandy Loam])


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
        Pedotransfer function (PTF) to use for calculation KsatVer
        (vertical saturated hydraulic conductivity [mm/day]).
        By default 'brakensiek'.
        wflow_thicknesslayers : list of int, optional
        Thickness of soil layers [mm] for wflow_sbm soil model.
        By default [100, 300, 800] for layers at depths 100, 400, 1200 and >1200 mm.
        Used only for Brooks Corey coefficients.
        """
        self.logger.info("Preparing soil parameter maps.")
        # TODO add variables list with required variable names
        dsin = self.data_catalog.get_rasterdataset(soil_fn, geom=self.region, buffer=2)
        dsout = workflows.soilgrids(
            ds=dsin,
            ds_like=self.grid,
            ptfKsatVer=ptf_ksatver,
            soil_fn=soil_fn,
            wflow_layers=wflow_thicknesslayers,
            logger=self.logger,
        ).reset_coords(drop=True)
        self.set_grid(dsout)

        # Update the toml file
        self.set_config("model.thicknesslayers", wflow_thicknesslayers)

    def setup_ksathorfrac(
        self,
        ksat_fn: Union[str, xr.DataArray],
        variable: Optional[str] = None,
        resampling_method: str = "average",
    ):
        """Set KsatHorFrac parameter values from a predetermined map.

        This predetermined map contains (preferably) 'calibrated' values of \
the KsatHorFrac parameter. This map is either selected from the wflow Deltares data \
or created by a third party/ individual.

        Parameters
        ----------
        ksat_fn : str, xr.DataArray
        The identifier of the KsatHorFrac dataset in the data catalog.
        variable : str, optional
        The variable name for the ksathorfrac map to use in ``ksat_fn`` in case \
``ksat_fn`` contains several variables. By default None.
        resampling_method : str, optional
        The resampling method when up- or downscaled, by default "average"
        """
        self.logger.info("Preparing KsatHorFrac parameter map.")

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
                "The ksathorfrac data contains several variables. \
Select the variable to use for ksathorfrac using 'variable' argument."
            )

        # Create scaled ksathorfrac map
        daout = workflows.ksathorfrac(
            dain,
            ds_like=self.grid,
            resampling_method=resampling_method,
        )

        # Set the output variable name
        if not isinstance(ksat_fn, str):
            bname = ksat_fn.name if ksat_fn.name is not None else "KsatHorFrac"
        else:
            bname = ksat_fn  # base name of the outgoing layer name

        lname = bname
        if variable is not None:
            lname += f"_{variable}"

        # Set the grid
        self.set_grid(daout, name=lname)
        self.set_config("input.lateral.subsurface.ksathorfrac", lname)

    def setup_ksatver_vegetation(
        self,
        soil_fn: str = "soilgrids",
        alfa: float = 4.5,
        beta: float = 5,
    ):
        """Calculate KsatVer values from vegetation in addition to soil characteristics.

        This allows to account for biologically-promoted soil structure and \
        heterogeneities in natural landscapes based on the work of \
        Bonetti et al. (2021) https://www.nature.com/articles/s43247-021-00180-0.

        This method requires to have run setup_soilgrids and setup_lai first.

        The following map is added to grid:

        * **KsatVer_vegetation** map: saturated hydraulic conductivity considering \
        vegetation characteristics [mm/d]

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
        """
        self.logger.info("Modifying ksatver based on vegetation characteristics")

        # open soil dataset to get sand percentage
        sndppt = self.data_catalog.get_rasterdataset(
            soil_fn, geom=self.region, buffer=2, variables=["sndppt_sl1"]
        )

        # in function get_ksatver_vegetation KsatVer should be provided in mm/d
        KSatVer_vegetation = workflows.ksatver_vegetation(
            ds_like=self.grid,
            sndppt=sndppt,
            alfa=alfa,
            beta=beta,
        )

        map_name = "KsatVer_vegetation"

        # add to grid
        self.set_grid(KSatVer_vegetation, map_name)
        # update config file
        self.set_config("input.vertical.kv_0", map_name)

    def setup_lulcmaps_with_paddy(
        self,
        lulc_fn: Union[str, Path, xr.DataArray],
        paddy_class: int,
        output_paddy_class: Optional[int] = None,
        lulc_mapping_fn: Union[str, Path, pd.DataFrame] = None,
        paddy_fn: Optional[Union[str, Path, xr.DataArray]] = None,
        paddy_mapping_fn: Optional[Union[str, Path, pd.DataFrame]] = None,
        soil_fn: Union[str, Path, xr.DataArray] = "soilgrids",
        wflow_thicknesslayers: List[int] = [50, 100, 50, 200, 800],
        target_conductivity: List[Union[None, int, float]] = [
            None,
            None,
            5,
            None,
            None,
        ],
        lulc_vars: List = [
            "landuse",
            "Kext",
            "N",
            "PathFrac",
            "RootingDepth",
            "Sl",
            "Swood",
            "WaterFrac",
            "kc",
            "alpha_h1",
            "h1",
            "h2",
            "h3_high",
            "h3_low",
            "h4",
        ],
        paddy_waterlevels: Dict = {"h_min": 20, "h_opt": 50, "h_max": 80},
        save_high_resolution_lulc: bool = False,
    ):
        """Set up landuse maps and parameters including for paddy fields.

        THIS FUNCTION SHOULD BE RUN AFTER setup_soilmaps.

        Lookup table `lulc_mapping_fn` columns are converted to lulc classes model
        parameters based on literature. The data is remapped at its original resolution
        and then resampled to the model resolution using the average value, unless noted
        differently.

        If paddies are present either directly as a class in the landuse_fn or in a
        separate paddy_fn, the paddy class is used to derive the paddy parameters.

        To allow for water to pool on the surface (for paddy/rice fields), the layers in
        the model can be updated to new depths, such that we can allow a thin layer with
        limited vertical conductivity. These updated layers means that the ``c``
        parameter needs to be calculated again. Next, the kvfrac layer corrects the
        vertical conductivity (by multiplying) such that the bottom of the layer
        corresponds to the ``target_conductivity`` for that layer. This currently
        assumes the wflow models to have an exponential declining vertical conductivity
        (using the ``f`` parameter). If no target_conductivity is specified for a layer
        (``None``), the kvfrac value is set to 1.

        The different values for the minimum/optimal/maximum water levels for paddy
        fields will be added as constant values in the toml file, through the
        ``vertical.paddy.h_min.value = 20`` interface.

        Adds model layers:

        * **landuse** map: Landuse class [-]
        * **Kext** map: Extinction coefficient in the canopy gap fraction equation [-]
        * **Sl** map: Specific leaf storage [mm]
        * **Swood** map: Fraction of wood in the vegetation/plant [-]
        * **RootingDepth** map: Length of vegetation roots [mm]
        * **PathFrac** map: The fraction of compacted or urban area per grid cell [-]
        * **WaterFrac** map: The fraction of open water per grid cell [-]
        * **N** map: Manning Roughness [-]
        * **kc** map: Crop coefficient [-]
        * **alpha_h1** map: Root water uptake reduction at soil water pressure head h1
          (0 or 1) [-]
        * **h1** map: Soil water pressure head h1 at which root water uptake is reduced
          (Feddes) [cm]
        * **h2** map: Soil water pressure head h2 at which root water uptake is reduced
          (Feddes) [cm]
        * **h3_high** map: Soil water pressure head h3 at which root water uptake is
          reduced (Feddes) [cm]
        * **h3_low** map: Soil water pressure head h3 at which root water uptake is
          reduced (Feddes) [cm]
        * **h4** map: Soil water pressure head h4 at which root water uptake is reduced
          (Feddes) [cm]
        * **h_min** map: Minimum required water depth for paddy fields [mm]
        * **h_opt** map: Optimal water depth for paddy fields [mm]
        * **h_max** map: Maximum water depth for paddy fields [mm]
        * **kvfrac**: Map with a multiplication factor for the vertical conductivity [-]

        Updates model layers:

        * **c**: Brooks Corey coefficients [-] based on pore size distribution, a map
          for each of the wflow_sbm soil layers (updated based on the newly specified
          layers)


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
            Soil data to be used to recalculate the Brooks-Corey coefficients (`c`
            parameter), based on the provided ``wflow_thicknesslayers``, by default
            "soilgrids", but should ideally be equal to the data used in
            :py:meth:`setup_soilmaps`

            * Required variables: 'bd_sl*' [g/cm3], 'clyppt_sl*' [%], 'sltppt_sl*' [%],
              'ph_sl*' [-].
        wflow_thicknesslayers: list
            List of soil thickness per layer [mm], by default [50, 100, 50, 200, 800, ]
        target_conductivity: list
            List of target vertical conductivities [mm/day] for each layer in
            ``wflow_thicknesslayers``. Set value to `None` if no specific value is
            required, by default [None, None, 5, None, None].
        lulc_vars : list
            List of landuse parameters to prepare.
            By default ["landuse","Kext","N","PathFrac","RootingDepth","Sl","Swood",
            "WaterFrac"]
        paddy_waterlevels : dict
            Dictionary with the minimum, optimal and maximum water levels for paddy
            fields [mm]. By default {"h_min": 20, "h_opt": 50, "h_max": 80}
        save_high_resolution_lulc : bool
            Save the high resolution landuse map merged with the paddies to the static
            folder. By default False.
        """
        self.logger.info("Preparing LULC parameter maps including paddies.")
        # Check if soil data is available
        if "KsatVer" not in self.grid.data_vars:
            raise ValueError(
                "KsatVer and f are required to update the soil parameters with paddies."
                "Please run setup_soilmaps first."
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
                output_dir = join(self.root, "maps")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                landuse.raster.to_raster(join(output_dir, "landuse_with_paddy.tif"))
                df_mapping.to_csv(join(output_dir, "landuse_with_paddy_mapping.csv"))

        # Prepare landuse parameters
        landuse_maps = workflows.landuse(
            da=landuse,
            ds_like=self.grid,
            df=df_mapping,
            params=lulc_vars,
            logger=self.logger,
        )
        rmdict = {k: v for k, v in self._MAPS.items() if k in landuse_maps.data_vars}
        self.set_grid(landuse_maps.rename(rmdict))

        # Update soil parameters if there are paddies in the domain
        # Get paddy pixels at model resolution
        wflow_paddy = landuse_maps["landuse"] == output_paddy_class
        if wflow_paddy.any():
            if self.get_config("model.thicknesslayers") == len(wflow_thicknesslayers):
                self.logger.info(
                    "same thickness already present, skipping updating `c` parameter"
                )
                update_c = False
            else:
                self.logger.info(
                    "Different thicknesslayers requested, updating `c` parameter"
                )
                update_c = True
            # Read soil data
            soil = self.data_catalog.get_rasterdataset(
                soil_fn, geom=self.region, buffer=2
            )
            # update soil parameters c and kvfrac
            soil_maps = workflows.update_soil_with_paddy(
                ds=soil,
                ds_like=self.grid,
                paddy_mask=wflow_paddy,
                soil_fn=soil_fn,
                update_c=update_c,
                wflow_layers=wflow_thicknesslayers,
                target_conductivity=target_conductivity,
                logger=self.logger,
            )
            self.set_grid(soil_maps["kvfrac"], name="kvfrac")
            if "c" in soil_maps:
                self.set_grid(soil_maps["c"], name="c")
                self.set_config("model.thicknesslayers", wflow_thicknesslayers)
            # Add paddy water levels to the config
            for key, value in paddy_waterlevels.items():
                self.set_config(f"input.vertical.paddy.{key}.value", value)
            # Update the states
            self.set_config("state.vertical.paddy.h", "h_paddy")
        else:
            self.logger.info("No paddy fields found, skipping updating soil parameters")

        # Add entries to the config
        for name in landuse_maps.data_vars:
            if name in WFLOW_NAMES and WFLOW_NAMES[name] is not None:
                self.set_config(WFLOW_NAMES[name], name)

    def setup_glaciers(self, glaciers_fn="rgi", min_area=1):
        """
        Generate maps of glacier areas, area fraction and volume fraction.

        Also generates tables with temperature threshold, melting factor and snow-to-ice
        conversion fraction.

        The data is generated from features with ``min_area`` [km2] (default is 1 km2)
        from a database with glacier geometry, IDs and metadata.

        The required variables from glaciers_fn dataset are glacier ID 'simple_id'.
        Optionally glacier area 'AREA' [km2] can be present to filter the glaciers
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
            glaciers_fn,
            geom=self.basins_highres,
            predicate="intersects",
            handle_nodata=NoDataStrategy.IGNORE,
        )
        # Check if there are glaciers found
        if gdf_org is None:
            self.logger.info("Skipping method, as no data has been found")
            return

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

    def setup_constant_pars(
        self, dtype: str = "float32", nodata: Union[int, float] = -999, **kwargs
    ):
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
        raster_fn: Union[str, xr.Dataset],
        reproject_method: str,
        variables: Optional[List[str]] = None,
        wflow_variables: Optional[List[str]] = None,
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
        precip_fn: Union[str, xr.DataArray],
        precip_clim_fn: Optional[Union[str, xr.DataArray]] = None,
        chunksize: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Generate gridded precipitation forcing at model resolution.

        Adds model layer:

        * **precip**: precipitation [mm]

        Parameters
        ----------
        precip_fn : str, xarray.DataArray
            Precipitation RasterDataset source, see data/forcing_sources.yml.

            * Required variable: 'precip' [mm]

            * Required dimension: 'time'  [timestamp]
        precip_clim_fn : str, xarray.DataArray, optional
            High resolution climatology precipitation RasterDataset source to correct
            precipitation, see data/forcing_sources.yml.

            * Required variable: 'precip' [mm]

            * Required dimension: 'time'  [cyclic month]
        chunksize: int, optional
            Chunksize on time dimension for processing data (not for saving to disk!).
            If None the data chunksize is used, this can however be optimized for
            large/small catchments. By default None.
        """
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
        temp_pet_fn: Union[str, xr.Dataset],
        pet_method: str = "debruin",
        press_correction: bool = True,
        temp_correction: bool = True,
        wind_correction: bool = True,
        wind_altitude: int = 10,
        reproj_method: str = "nearest_index",
        fillna_method: Optional[str] = None,
        dem_forcing_fn: Union[str, xr.DataArray] = None,
        skip_pet: str = False,
        chunksize: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Generate gridded temperature and reference evapotranspiration forcing.

        If `temp_correction` is True, the temperature will be reprojected and then
        downscaled to model resolution using the elevation lapse rate. For better
        accuracy, you can provide the elevation grid of the climate data in
        `dem_forcing_fn`. If not present, the upscaled elevation grid of the wflow model
        is used ('wflow_dem').

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

        Parameters
        ----------
        temp_pet_fn : str, xarray.Dataset
        Name or path of RasterDataset source with variables to calculate temperature
        and reference evapotranspiration, see data/forcing_sources.yml.

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
            time_tuple=(starttime, endtime),
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
        if fillna_method is not None:
            temp_out = temp_out.raster.interpolate_na(
                dim=temp_out.raster.x_dim,
                method=fillna_method,
                fill_value="extrapolate",
            )
        self.set_forcing(temp_out.where(mask), name="temp")

    def setup_pet_forcing(
        self,
        pet_fn: Union[str, xr.DataArray],
        chunksize: Optional[int] = None,
    ):
        """
        Prepare PET forcing from existig PET data.

        Adds model layer:

        * **pet**: reference evapotranspiration [mm]

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
        self.logger.info("Preparing potential evapotranspiration forcing maps.")

        starttime = self.get_config("starttime")
        endtime = self.get_config("endtime")
        freq = pd.to_timedelta(self.get_config("timestepsecs"), unit="s")

        pet = self.data_catalog.get_rasterdataset(
            pet_fn,
            geom=self.region,
            buffer=2,
            variables=["pet"],
            time_tuple=(starttime, endtime),
        )
        pet = pet.astype("float32")

        pet_out = workflows.forcing.pet(
            pet=pet,
            ds_like=self.grid,
            freq=freq,
            mask_name=self._MAPS["basins"],
            chunksize=chunksize,
            logger=self.logger,
        )

        # Update meta attributes (used for default output filename later)
        pet_out.attrs.update({"pet_fn": pet_fn})
        self.set_forcing(pet_out, name="pet")

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
the missing with data from downstream catchments.
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

    def setup_1dmodel_connection(
        self,
        river1d_fn: Union[str, Path, gpd.GeoDataFrame],
        connection_method: str = "subbasin_area",
        area_max: float = 30.0,
        add_tributaries: bool = True,
        include_river_boundaries: bool = True,
        mapname: str = "1dmodel",
        update_toml: bool = True,
        toml_output: str = "netcdf",
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
        tributaries using :py:meth:`hydromt_wflow.wflow.setup_config_output_timeseries`.

        Adds model layer:

        * **wflow_subcatch_{mapname}** map/geom:  connection subbasins between
          wflow and the 1D model.
        * **wflow_subcatch_riv_{mapname}** map/geom:  connection subbasins between
          wflow and the 1D model for river cells only.
        * **wflow_gauges_{mapname}** map/geom, optional: outlets of the tributaries
          flowing into the 1D model.

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
        add_tributaries : bool, default True
            If True, derive tributaries for the subbasins larger than area_max. Always
            True for **subbasin_area** method.
        include_river_boundaries : bool, default True
            If True, include the upstream boundary(ies) of the 1d river as an
            additional tributary(ies).
        mapname : str, default 1dmodel
            Name of the map to save the subcatchments and tributaries in the wflow model
            staticmaps and geoms (wflow_subcatch_{mapname}).
        update_toml : bool, default True
            If True, updates the wflow configuration file to save the required outputs
            for the 1D model.
        toml_output : str, optional
            One of ['csv', 'netcdf', None] to update [csv] or [netcdf] section of wflow
            toml file or do nothing. By default, 'netcdf'.
        **kwargs
            Additional keyword arguments passed to the snapping method
            hydromt.flw.gauge_map. See its documentation for more information.

        See Also
        --------
        hydromt.flw.gauge_map
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
        inv_rename = {v: k for k, v in self._MAPS.items() if v in self.grid}
        ds_out = workflows.wflow_1dmodel_connection(
            gdf_riv,
            ds_model=self.grid.rename(inv_rename),
            connection_method=connection_method,
            area_max=area_max,
            add_tributaries=add_tributaries,
            include_river_boundaries=include_river_boundaries,
            logger=self.logger,
            **kwargs,
        )

        # Derive tributary gauge map
        if "gauges" in ds_out.data_vars:
            self.set_grid(ds_out["gauges"], name=f"wflow_gauges_{mapname}")
            # Derive the gauges staticgeoms
            gdf_tributary = ds_out["gauges"].raster.vectorize()
            gdf_tributary["geometry"] = gdf_tributary["geometry"].centroid
            gdf_tributary["value"] = gdf_tributary["value"].astype(
                ds_out["gauges"].dtype
            )
            self.set_geoms(gdf_tributary, name=f"gauges_{mapname}")

            # Add a check that all gauges are on the river
            if (
                self.grid[self._MAPS["rivmsk"]].raster.sample(gdf_tributary)
                == self.grid[self._MAPS["rivmsk"]].raster.nodata
            ).any():
                river_upa = self.grid[self._MAPS["rivmsk"]].attrs.get("river_upa", "")
                self.logger.warning(
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
                    mapname=f"wflow_gauges_{mapname}",
                    toml_output=toml_output,
                    header=["Q"],
                    param=["lateral.river.q_av"],
                    reducer=None,
                )

        # Derive subcatchment map
        self.set_grid(ds_out["subcatch"], name=f"wflow_subcatch_{mapname}")
        gdf_subcatch = ds_out["subcatch"].raster.vectorize()
        gdf_subcatch["value"] = gdf_subcatch["value"].astype(ds_out["subcatch"].dtype)
        self.set_geoms(gdf_subcatch, name=f"subcatch_{mapname}")
        # Subcatchment map for river cells only (to be able to save river outputs
        # in wflow)
        self.set_grid(ds_out["subcatch_riv"], name=f"wflow_subcatch_riv_{mapname}")
        gdf_subcatch_riv = ds_out["subcatch_riv"].raster.vectorize()
        gdf_subcatch_riv["value"] = gdf_subcatch_riv["value"].astype(
            ds_out["subcatch"].dtype
        )
        self.set_geoms(gdf_subcatch_riv, name=f"subcatch_riv_{mapname}")

        # Update toml
        if update_toml:
            self.setup_config_output_timeseries(
                mapname=f"wflow_subcatch_riv_{mapname}",
                toml_output=toml_output,
                header=["Qlat"],
                param=["lateral.river.inwater"],
                reducer=["sum"],
            )

    def setup_allocation_areas(
        self,
        waterareas_fn: Union[str, gpd.GeoDataFrame],
        priority_basins: bool = True,
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

        Parameters
        ----------
        waterareas_fn : Union[str, gpd.GeoDataFrame]
            Administrative boundaries GeoDataFrame data, this could be
            e.g. water management areas by water boards or the administrative
            boundaries of countries.
        priority_basins : bool, optional
            If True, merge the basins with the closest downstream basin, else merge
            with any large enough basin in the same water area, by default True.
        """
        self.logger.info("Preparing water demand allocation map.")

        # Read the data
        waterareas = self.data_catalog.get_geodataframe(
            waterareas_fn,
            geom=self.region,
        )

        # Create the allocation grid
        da_alloc, gdf_alloc = workflows.demand.allocation_areas(
            ds_like=self.grid,
            waterareas=waterareas,
            basins=self.basins,
            priority_basins=priority_basins,
        )
        self.set_grid(da_alloc, name="allocation_areas")

        # Update the settings toml
        self.set_config("input.vertical.allocation.areas", "allocation_areas")

        # Add alloc to geoms
        self.set_geoms(gdf_alloc, name="allocation_areas")

    def setup_allocation_surfacewaterfrac(
        self,
        gwfrac_fn: Union[str, xr.DataArray],
        waterareas_fn: Optional[Union[str, xr.DataArray]] = None,
        gwbodies_fn: Optional[Union[str, xr.DataArray]] = None,
        ncfrac_fn: Optional[Union[str, xr.DataArray]] = None,
        interpolate_nodata: bool = False,
        mask_and_scale_gwfrac: bool = True,
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

        * **frac_sw_used**: fraction of water allocated from surface water [0-1]

        Parameters
        ----------
        gwfrac_fn : Union[str, xr.DataArray]
            The raw groundwater fraction per grid cell. The values of these cells need
            to be between 0 and 1.
        waterareas_fn : Union[str, xr.DataArray]
            The areas over which the water has to be distributed. This may either be
            a global (or more local map). If not provided, the source areas created by
            the `setup_allocation_areas` will be used.
        gwbodies_fn : Union[str, xr.DataArray], optional
            The presence of groundwater bodies per grid cell. The values are ought to
            be binary (either 0 or 1). If they are not provided, we assume groundwater
            bodies are present where gwfrac is more than 0.
        ncfrac_fn : Union[str, xr.DataArray], optional
            The non-conventional fraction. Same types of values apply as for
            `gwfrac_fn`. If not provided, we assume no non-conventional sources are
            used.
        interpolate_nodata : bool, optional
            If True, nodata values in the resulting frac_sw_used map will be linearly
            interpolated. Else a default value of 1 will be used for nodata values
            (default).
        mask_and_scale_gwfrac : bool, optional
            If True, gwfrac will be masked for areas with no groundwater bodies. To keep
            the average gwfrac used over waterareas similar after the masking, gwfrac
            for areas with groundwater bodies can increase. If False, gwfrac will be
            used as is. By default True.
        """
        self.logger.info("Preparing surface water fraction map.")
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
            self.logger.info("Using wflow model allocation areas.")
            if "allocation_areas" not in self.grid:
                self.logger.error(
                    "No allocation areas found. Run setup_allocation_areas first "
                    "or provide a waterareas_fn."
                )
                return
            waterareas = self.grid["allocation_areas"]
        else:
            waterareas = self.data_catalog.get_rasterdataset(
                waterareas_fn,
                geom=self.region,
                buffer=2,
            )

        # Call the workflow
        w_frac = workflows.demand.surfacewaterfrac_used(
            gwfrac_raw=gwfrac_raw,
            da_like=self.grid["wflow_dem"],
            waterareas=waterareas,
            gwbodies=gwbodies,
            ncfrac=ncfrac,
            interpolate=interpolate_nodata,
            mask_and_scale_gwfrac=mask_and_scale_gwfrac,
        )

        # Update the settings toml
        self.set_config(
            "input.vertical.allocation.frac_sw_used",
            "frac_sw_used",
        )

        # Set the dataarray to the wflow grid
        self.set_grid(w_frac, name="frac_sw_used")

    def setup_domestic_demand(
        self,
        domestic_fn: Union[str, xr.Dataset],
        population_fn: Optional[Union[str, xr.Dataset]] = None,
        domestic_fn_original_res: Optional[float] = None,
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

        * **domestic_gross**: gross domestic water demand [mm/day]
        * **domestic_net**: net domestic water demand [mm/day]

        Parameters
        ----------
        domestic_fn : Union[str, xr.Dataset]
            The domestic dataset. This can either be the dataset directly (xr.Dataset),
            a string referring to an entry in the data catalog or a dictionary
            containing the name of the dataset (keyword: `source`) and any optional
            keyword arguments (e.g. `version`). The data can be cyclic
            (with a `time` dimension) or non-cyclic. Allowed cyclic data can be monthly
            (12) or dayofyear (365 or 366).

            * Required variables: 'dom_gross' [mm/day], 'dom_net' [mm/day]
        population_fn : Union[str, xr.Dataset]
            The population dataset in capita. Either provided as a dataset directly or
            as a string referring to an entry in the data catalog.
        domestic_fn_original_res : Optional[float], optional
            The original resolution of the domestic dataset, by default None to skip
            upscaling before downsampling with population.
        """
        self.logger.info("Preparing domestic demand maps.")
        # Set flag for cyclic data
        _cyclic = False

        # Read data
        domestic_raw = self.data_catalog.get_rasterdataset(
            domestic_fn,
            geom=self.region,
            buffer=2,
            variables=["dom_gross", "dom_net"],
        )
        # Increase the buffer if original resolution is provided
        if domestic_fn_original_res is not None:
            buffer = np.ceil(domestic_fn_original_res / abs(domestic_raw.raster.res[0]))
            domestic_raw = self.data_catalog.get_rasterdataset(
                domestic_fn,
                geom=self.region,
                buffer=buffer,
                variables=["dom_gross", "dom_net"],
            )
        # Check if data is time dependent
        if "time" in domestic_raw.coords:
            # Check that this is indeed cyclic data
            if len(domestic_raw.time) in [12, 365, 366]:
                _cyclic = True
                domestic_raw["time"] = domestic_raw.time.astype("int32")
            else:
                self.logger.error(
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
            ds_like=self.grid,
            popu=pop_raw,
            original_res=domestic_fn_original_res,
        )
        # Add to grid
        rmdict = {k: self._MAPS.get(k, k) for k in domestic.data_vars}
        self.set_grid(domestic.rename(rmdict))
        if population_fn is not None:
            self.set_grid(pop, name="population")

        # Update toml
        if _cyclic and self.get_config("input.cyclic") is None:
            self.set_config("input.cyclic", [])
        self.set_config("model.water_demand.domestic", True)

        for demand_type in ["gross", "net"]:
            self.set_config(
                f"input.vertical.domestic.demand_{demand_type}",
                f"domestic_{demand_type}",
            )
            if _cyclic:
                self.config["input"]["cyclic"].append(
                    f"vertical.domestic.demand_{demand_type}"
                )

    def setup_other_demand(
        self,
        demand_fn: Union[str, dict, xr.Dataset],
        variables: list = ["ind_gross", "ind_net", "lsk_gross", "lsk_net"],
        resampling_method: str = "average",
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

        Parameters
        ----------
        demand_fn : Union[str, dict, xr.Dataset]
            The water demand dataset. This can either be the dataset directly
            (xr.Dataset), a string referring to an entry in the data catalog or
            a dictionary containing the name of the dataset (keyword: `source`) and
            any optional keyword arguments (e.g. `version`). The data can be cyclic
            (with a `time` dimension) or non-cyclic. Allowed cyclic data can be monthly
            (12) or dayofyear (365 or 366).

            * Required variables: variables listed in `variables` in [mm/day].
        variables : list, optional
            The variables to be processed. Supported variables are ['dom_gross',
            'dom_net', 'ind_gross', 'ind_net', 'lsk_gross', 'lsk_net'] where 'dom' is
            domestic, 'ind' is industrial and 'lsk' is livestock. By default gross and
            net demand for industry and livestock are processed.
        resampling_method : str, optional
            Resampling method for the demand maps, by default "average"
        """
        self.logger.info(f"Preparing water demand maps for {variables}.")
        # Set flag for cyclic data
        _cyclic = False

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
                self.logger.error(
                    "The provided demand data is cyclic but the time dimension does "
                    "not match the expected length of 12, 365 or 366."
                )

        # Create static water demand rasters
        demand = workflows.demand.other_demand(
            demand_raw,
            ds_like=self.grid,
            ds_method=resampling_method,
        )
        rmdict = {k: self._MAPS.get(k, k) for k in demand.data_vars}
        demand = demand.rename(rmdict)
        self.set_grid(demand)

        # Update the settings toml
        if _cyclic and self.get_config("input.cyclic") is None:
            self.set_config("input.cyclic", [])
        for var in demand.data_vars:
            sname, suffix = var.split("_")
            self.set_config(
                f"input.vertical.{sname}.demand_{suffix}",
                var,
            )
            # Set flag
            self.set_config(f"model.water_demand.{sname}", True)

            # Also for the fact that these parameters are cyclic
            if _cyclic:
                self.config["input"]["cyclic"].append(
                    f"vertical.{sname}.demand_{suffix}",
                )

    def setup_irrigation(
        self,
        irrigated_area_fn: Union[str, Path, xr.DataArray],
        irrigation_value: List[int],
        cropland_class: List[int],
        paddy_class: List[int] = [],
        area_threshold: float = 0.6,
        lai_threshold: float = 0.2,
    ):
        """
        Add required information to simulate irrigation water demand.

        THIS FUNCTION SHOULD BE RUN AFTER LANDUSE AND LAI MAPS ARE CREATED.

        The function requires data that contains information about the location of the
        irrigated areas (``irrigated_area_fn``). This, combined with the wflow landuse
        map that contains classes for cropland (``cropland_class) and optionally for
        paddy (rice) (``paddy_class``), determines which locations are considered to be
        paddy irrigation, and which locations are considered to be non-paddy irrigation.

        Next, the irrigated are map is reprojected to the model resolution, where a
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

        * **paddy_irrigation_areas**: Irrigated (paddy) mask [-]
        * **nonpaddy_irrigation_areas**: Irrigated (non-paddy) mask [-]
        * **irrigation_trigger**: Map with monthly values, indicating whether irrigation
          is allowed (1) or not (0) [-]

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
        self.logger.info("Preparing irrigation maps.")

        # Extract irrigated area dataset
        irrigated_area = self.data_catalog.get_rasterdataset(
            irrigated_area_fn, bbox=self.grid.raster.bounds, buffer=3
        )

        # Get irrigation areas for paddy, non paddy and irrigation trigger
        ds_irrigation = workflows.demand.irrigation(
            da_irrigation=irrigated_area,
            ds_like=self.grid,
            irrigation_value=irrigation_value,
            cropland_class=cropland_class,
            paddy_class=paddy_class,
            area_threshold=area_threshold,
            lai_threshold=lai_threshold,
            logger=self.logger,
        )

        # Check if paddy and non paddy are present
        cyclic_lai = len(self.grid["LAI"].dims) > 2
        if (
            "paddy_irrigation_areas" in ds_irrigation.data_vars
            and ds_irrigation["paddy_irrigation_areas"]
            .raster.mask_nodata()
            .sum()
            .values
            != 0
        ):
            self.set_config("model.water_demand.paddy", True)
            self.set_grid(ds_irrigation["paddy_irrigation_areas"])
            self.set_config(
                "input.vertical.paddy.irrigation_areas", "paddy_irrigation_areas"
            )
            # Irrigation trigger
            self.set_grid(ds_irrigation["paddy_irrigation_trigger"])
            self.set_config(
                "input.vertical.paddy.irrigation_trigger", "paddy_irrigation_trigger"
            )
            if cyclic_lai:
                self.config["input"]["cyclic"].append(
                    "vertical.paddy.irrigation_trigger"
                )
        else:
            self.set_config("model.water_demand.paddy", False)

        if (
            ds_irrigation["nonpaddy_irrigation_areas"].raster.mask_nodata().sum().values
            != 0
        ):
            self.set_config("model.water_demand.nonpaddy", True)
            self.set_grid(ds_irrigation["nonpaddy_irrigation_areas"])
            self.set_config(
                "input.vertical.nonpaddy.irrigation_areas", "nonpaddy_irrigation_areas"
            )
            # Irrigation trigger
            self.set_grid(ds_irrigation["nonpaddy_irrigation_trigger"])
            self.set_config(
                "input.vertical.nonpaddy.irrigation_trigger",
                "nonpaddy_irrigation_trigger",
            )
            if cyclic_lai:
                self.config["input"]["cyclic"].append(
                    "vertical.nonpaddy.irrigation_trigger"
                )
        else:
            self.set_config("model.water_demand.nonpaddy", False)

    def setup_cold_states(
        self,
        timestamp: str = None,
    ) -> None:
        """Prepare cold states for Wflow.

        To be run last as this requires some soil parameters or constant_pars to be
        computed already.

        To be run after setup_lakes, setup_reservoirs and setup_glaciers to also create
        cold states for them if they are present in the basin.

        This function is mainly useful in case the wflow model is read into Delft-FEWS.

        Adds model layers:

        * **satwaterdepth**: saturated store [mm]
        * **snow**: snow storage [mm]
        * **tsoil**: top soil temperature [°C]
        * **ustorelayerdepth**: amount of water in the unsaturated store, per layer [mm]
        * **snowwater**: liquid water content in the snow pack [mm]
        * **canopystorage**: canopy storage [mm]
        * **q_river**: river discharge [m3/s]
        * **h_river**: river water level [m]
        * **h_av_river**: river average water level [m]
        * **ssf**: subsurface flow [m3/d]
        * **h_land**: land water level [m]
        * **h_av_land**: land average water level[m]
        * **q_land** or **qx_land**+**qy_land**: overland flow for kinwave [m3/s] or
          overland flow in x/y directions for local-inertial [m3/s]

        If lakes, also adds:

        * **waterlevel_lake**: lake water level [m]

        If reservoirs, also adds:

        * **volume_reservoir**: reservoir volume [m3]

        If glaciers, also adds:

        * **glacierstore**: water within the glacier [mm]

        If paddy, also adds:

        * **h_paddy**: water on the paddy fields [mm]

        Parameters
        ----------
        timestamp : str, optional
            Timestamp of the cold states. By default uses the (starttime - timestepsecs)
            from the config.
        """
        states, states_config = workflows.prepare_cold_states(
            self.grid,
            config=self.config,
            timestamp=timestamp,
        )

        self.set_states(states)

        # Update config to read the states
        self.set_config("model.reinit", False)
        # Update states variables names in config
        for option in states_config:
            self.set_config(option, states_config[option])

    # I/O
    def read(
        self,
    ):
        """Read the complete model schematization and configuration from file."""
        self.read_config()
        self.read_grid()
        self.read_geoms()
        self.read_forcing()
        self.read_intbl()
        self.read_tables()
        self.read_states()
        self.logger.info("Model read")

    def write(
        self,
        config_fn: str = None,
        grid_fn: Union[Path, str] = "staticmaps.nc",
        geoms_fn: Union[Path, str] = "staticgeoms",
        forcing_fn: Union[Path, str] = None,
        states_fn: Union[Path, str] = None,
    ):
        """
        Write the complete model schematization and configuration to file.

        From this function, the output filenames/folder of the different components can
        be set. If not set, the default filenames/folder are used.
        To change more advanced settings, use the specific write methods directly.

        Parameters
        ----------
        config_fn : str, optional
            Name of the config file, relative to model root. By default None.
        grid_fn : str, optional
            Name of the grid file, relative to model root/dir_input. By default
            'staticmaps.nc'.
        geoms_fn : str, optional
            Name of the geoms folder relative to grid_fn (ie model root/dir_input). By
            default 'staticgeoms'.
        forcing_fn : str, optional
            Name of the forcing file relative to model root/dir_input. By default None
            to use the name as defined in the model config file.
        states_fn : str, optional
            Name of the states file relative to model root/dir_input. By default None
            to use the name as defined in the model config file.
        """
        self.logger.info(f"Write model data to {self.root}")
        # if in r, r+ mode, only write updated components
        if not self._write:
            self.logger.warning("Cannot write in read-only mode")
            return
        self.write_data_catalog()
        _ = self.config  # try to read default if not yet set
        if self._grid:
            self.write_grid(fn_out=grid_fn)
        if self._geoms:
            self.write_geoms(geoms_fn=geoms_fn)
        if self._forcing:
            self.write_forcing(fn_out=forcing_fn)
        if self._tables:
            self.write_tables()
        if self._states:
            self.write_states(fn_out=states_fn)
        # Write the config last as variables can get set in other write methods
        self.write_config(config_name=config_fn)

    def write_config(
        self,
        config_name: Optional[str] = None,
        config_root: Optional[str] = None,
    ):
        """
        Write config to <root/config_fn>.

        Parameters
        ----------
        config_name : str, optional
            Name of the config file. By default None to use the default name
            wflow_sbm.toml.
        config_root : str, optional
            Root folder to write the config file if different from model root (default).
        """
        self._assert_write_mode()
        if config_name is not None:
            self._config_fn = config_name
        elif self._config_fn is None:
            self._config_fn = self._CONF
        if config_root is None:
            config_root = self.root
        fn = join(config_root, self._config_fn)
        # Create the folder if it does not exist
        if not isdir(dirname(fn)):
            os.makedirs(dirname(fn))
        self.logger.info(f"Writing model config to {fn}")
        self._configwrite(fn)

    def read_grid(self, **kwargs):
        """
        Read wflow static input and add to ``grid``.

        Checks the path of the file in the config toml using both ``input.path_static``
        and ``dir_input``. If not found uses the default path ``staticmaps.nc`` in the
        root folder.

        For reading old PCRaster maps, see the pcrm submodule.

        See Also
        --------
        pcrm.read_staticmaps_pcr
        """
        fn_default = "staticmaps.nc"
        fn = self.get_config(
            "input.path_static", abs_path=True, fallback=join(self.root, fn_default)
        )

        if self.get_config("dir_input") is not None:
            input_dir = self.get_config("dir_input", abs_path=True)
            fn = join(
                input_dir,
                self.get_config("input.path_static", fallback=fn_default),
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

    def write_grid(
        self,
        fn_out: Optional[Union[Path, str]] = None,
    ):
        """
        Write grid to wflow static data file.

        Checks the path of the file in the config toml using both ``input.path_static``
        and ``dir_input``. If not found uses the default path ``staticmaps.nc`` in the
        root folder.

        Parameters
        ----------
        fn_out : Path, str, optional
            Name or path to the outgoing grid file (including extension). This is the
            path/name relative to the root folder and if present the ``dir_input``
            folder.
        """
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
        if fn_out is not None:
            fn = join(self.root, fn_out)
            self.set_config("input.path_static", fn_out)
        else:
            fn_out = "staticmaps.nc"
            fn = self.get_config(
                "input.path_static", abs_path=True, fallback=join(self.root, fn_out)
            )
        # Append inputdir if required
        if self.get_config("dir_input") is not None:
            input_dir = self.get_config("dir_input", abs_path=True)
            fn = join(
                input_dir,
                self.get_config("input.path_static", fallback=fn_out),
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

    def set_grid(
        self,
        data: Union[xr.DataArray, xr.Dataset, np.ndarray],
        name: Optional[str] = None,
    ):
        """Add data to grid.

        All layers of grid must have identical spatial coordinates. This is an inherited
        method from HydroMT-core's GridModel.set_grid with some fixes.

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
        if "time" in data.dims:
            # Raise error if the dimension does not have a supported length
            if len(data.time) not in [12, 365, 366]:
                raise ValueError(
                    f"Length of cyclic dataset ({len(data)}) is not supported by "
                    "Wflow.jl. Ensure the data has length 12, 365, or 366"
                )
            tname = "time"
            time_axes = {
                k: v for k, v in dict(self.grid.dims).items() if k.startswith("time")
            }
            if data["time"].size not in time_axes.values():
                tname = f"time_{data['time'].size}" if "time" in time_axes else tname
            else:
                k = list(
                    filter(lambda x: time_axes[x] == data["time"].size, time_axes)
                )[0]
                tname = k

            if tname != "time":
                data = data.rename_dims({"time": tname})
        if "layer" in data.dims and "layer" in self.grid:
            if len(data["layer"]) != len(self.grid["layer"]):
                vars_to_drop = [
                    var for var in self.grid.variables if "layer" in self.grid[var].dims
                ]
                # Drop variables
                self.logger.info(
                    "Dropping these variables, as they depend on the layer "
                    f"dimension: {vars_to_drop}"
                )
                # Use `_grid` as `grid` cannot be set
                self._grid = self.grid.drop_vars(vars_to_drop)

        # fall back on default set_grid behaviour
        GridModel.set_grid(self, data, name)

    def read_geoms(
        self,
        geoms_fn: str = "staticgeoms",
    ):
        """
        Read static geometries and adds to ``geoms``.

        Assumes that the `geoms_fn` folder is located relative to root and ``dir_input``
        if defined in the toml. If not found, uses assumes they are in <root/geoms_fn>.

        Parameters
        ----------
        geoms_fn : str, optional
            Folder name/path where the static geometries are stored relative to the
            model root and ``dir_input`` if any. By default "staticgeoms".
        """
        if not self._write:
            self._geoms = dict()  # fresh start in read-only mode
        # Check if dir_input is set and add
        if self.get_config("dir_input") is not None:
            dir_mod = join(self.get_config("dir_input", abs_path=True), geoms_fn)
        else:
            dir_mod = join(self.root, geoms_fn)

        fns = glob.glob(join(dir_mod, "*.geojson"))
        if len(fns) > 1:
            self.logger.info("Reading model staticgeom files.")
        for fn in fns:
            name = basename(fn).split(".")[0]
            if name != "region":
                self.set_geoms(gpd.read_file(fn), name=name)

    def write_geoms(
        self,
        geoms_fn: str = "staticgeoms",
        precision: Optional[int] = None,
    ):
        """
        Write geoms in GeoJSON format.

        Checks the path of ``geoms_fn`` using both model root and
        ``dir_input``. If not found uses the default path ``staticgeoms`` in the root
        folder.

        Parameters
        ----------
        geoms_fn : str, optional
            Folder name/path where the static geometries are stored relative to the
            model root and ``dir_input`` if any. By default "staticgeoms".
        precision : int, optional
            Decimal precision to write the geometries. By default None to use 1 decimal
            for projected crs and 6 for non-projected crs.
        """
        # to write use self.geoms[var].to_file()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self.geoms:
            self.logger.info("Writing model staticgeom to file.")
            # Set projection to 1 decimal if projected crs
            _precision = precision
            if precision is None:
                if self.crs.is_projected:
                    _precision = 1
                else:
                    _precision = 6
            grid_size = 10 ** (-_precision)
            # Prepare the output folder
            if self.get_config("dir_input") is not None:
                geoms_dir = join(
                    self.get_config("dir_input", abs_path=True),
                    geoms_fn,
                )
            else:
                geoms_dir = join(self.root, geoms_fn)
            # Create the geoms dir if it does not already exist
            if not isdir(geoms_dir):
                os.makedirs(geoms_dir)

            for name, gdf in self.geoms.items():
                # TODO change to geopandas functionality once geopandas 1.0.0 comes
                # See https://github.com/geopandas/geopandas/releases/tag/v1.0.0-alpha1
                gdf.geometry = shapely.set_precision(
                    gdf.geometry,
                    grid_size=grid_size,
                )
                fn_out = join(geoms_dir, f"{name}.geojson")
                gdf.to_file(fn_out, driver="GeoJSON")

    def read_forcing(self):
        """
        Read forcing.

        Checks the path of the file in the config toml using both ``input.path_forcing``
        and ``dir_input``. If not found uses the default path ``inmaps.nc`` in the
        root folder.

        If several files are used using '*' in ``input.path_forcing``, all corresponding
        files are read and merged into one xarray dataset before being split to one
        xarray dataaray per forcing variable in the hydromt ``forcing`` dictionary.
        """
        fn_default = "inmaps.nc"
        fn = self.get_config(
            "input.path_forcing", abs_path=True, fallback=join(self.root, fn_default)
        )

        if self.get_config("dir_input") is not None:
            input_dir = self.get_config("dir_input", abs_path=True)
            fn = join(
                input_dir,
                self.get_config(
                    "input.path_forcing",
                    fallback=fn_default,
                ),
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
        Round the output data to the given number of decimals.
        time_units: str, optional
        Common time units when writing several netcdf forcing files.
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
            else:
                fn_name = self.get_config("input.path_forcing", abs_path=False)
                if fn_name is not None:
                    if "*" in basename(fn_name):
                        # get rid of * in case model had multiple forcing files and
                        # write to single nc file.
                        self.logger.warning(
                            "Writing multiple forcing files to one file"
                        )
                        fn_name = join(
                            dirname(fn_name), basename(fn_name).replace("*", "")
                        )
                    if self.get_config("dir_input") is not None:
                        input_dir = self.get_config("dir_input", abs_path=True)
                        fn_out = join(input_dir, fn_name)
                    else:
                        fn_out = join(self.root, fn_name)
                else:
                    fn_out = None

                # get default filename if file exists
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
                    if self.get_config("dir_input") is not None:
                        input_dir = self.get_config("dir_input", abs_path=True)
                        fn_default_path = join(input_dir, fn_default)
                    else:
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
                        if (start < pd.to_datetime(times[0])) or (start not in times):
                            start = pd.to_datetime(times[0])
                            correct_times = True
                        if (end > pd.to_datetime(times[-1])) or (end not in times):
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
                # Set the strings first
                self.set_config("starttime", start.strftime("%Y-%m-%dT%H:%M:%S"))
                self.set_config("endtime", end.strftime("%Y-%m-%dT%H:%M:%S"))

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
                self.logger.info(f"Writing several forcing with freq {freq_out}")
                # For several forcing files add common units attributes to time
                encoding["time"] = {"_FillValue": None, "units": time_units}
                # Updating path forcing in config
                fns_out = os.path.relpath(fn_out, self.root)
                fns_out = f"{str(fns_out)[0:-3]}_*.nc"
                self.set_config("input.path_forcing", fns_out)
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
        """Read states at <root/instate/> and parse to dict of xr.DataArray."""
        fn_default = join("instate", "instates.nc")
        fn = self.get_config(
            "state.path_input", abs_path=True, fallback=join(self.root, fn_default)
        )

        if self.get_config("dir_input") is not None:
            input_dir = self.get_config("dir_input", abs_path=True)
            fn = join(
                input_dir,
                self.get_config("state.path_input", fallback=fn_default),
            )
            self.logger.info(f"Input directory found {input_dir}")

        if not self._write:
            # start fresh in read-only mode
            self._states = dict()

        if fn is not None and isfile(fn):
            self.logger.info(f"Read states from {fn}")
            ds = xr.open_dataset(fn, mask_and_scale=False)
            for v in ds.data_vars:
                self.set_states(ds[v])

    def write_states(self, fn_out: Union[str, Path] = None):
        """Write states at <root/instate/> in model ready format."""
        if not self._write:
            raise IOError("Model opened in read-only mode")

        if self.states:
            self.logger.info("Writing states file")

            # get output filename and if needed update and re-write the config
            if fn_out is not None:
                self.set_config("state.path_input", fn_out)
                self.write_config()  # re-write config
            else:
                fn_name = self.get_config(
                    "state.path_input", abs_path=False, fallback=None
                )
                if fn_out is None:
                    fn_name = join("instate", "instates.nc")
                    self.set_config("state.path_input", fn_name)
                    self.write_config()  # re-write config
                if self.get_config("dir_input") is not None:
                    input_dir = self.get_config("dir_input", abs_path=True)
                    fn_out = join(input_dir, fn_name)
                else:
                    fn_out = join(self.root, fn_name)

            # merge, process and write forcing
            ds = xr.merge(self.states.values())

            # make sure no _FillValue is written to the time dimension
            ds["time"].attrs.pop("_FillValue", None)

            # Check if all sub-folders in fn_out exists and if not create them
            if not isdir(dirname(fn_out)):
                os.makedirs(dirname(fn_out))

            # write states
            ds.to_netcdf(fn_out, mode="w")

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
                    f"expr{i + 1}" if i + 1 < len(tbl.columns) else "value"
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
                tbl = pd.read_csv(fn, float_precision="round_trip")
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
            self.set_geoms(gdf, name="basins")
        else:
            self.logger.warning(f"Basin map {self._MAPS['basins']} not found in grid.")
            gdf = None
        return gdf

    @property
    def basins_highres(self):
        """Returns a high resolution basin(s) geometry."""
        if "basins_highres" in self.geoms:
            gdf = self.geoms["basins_highres"]
        else:
            gdf = self.basins
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

    def clip_states(self, crs=4326, **kwargs):
        """Return clippped states for subbasin.

        Returns
        -------
        xarray.DataSet
            Clipped states.
        """
        if len(self.states) > 0:
            self.logger.info("Clipping NetCDF states..")
            ds_states = xr.merge(self.states.values()).raster.clip_bbox(
                self.grid.raster.bounds
            )
            # Check for reservoirs/lakes presence in the clipped model
            remove_maps = []
            if self._MAPS["resareas"] not in self.grid:
                if "volume_reservoir" in ds_states:
                    remove_maps.extend(["volume_reservoir"])
            if self._MAPS["lakeareas"] not in self.grid:
                if "waterlevel_lake" in ds_states:
                    remove_maps.extend(["waterlevel_lake"])
            ds_states = ds_states.drop_vars(remove_maps)
            self.set_states(ds_states)
