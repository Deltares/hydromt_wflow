"""Implement wflow model class"""
# Implement model class following model API

import os
from os.path import join, dirname, basename, isfile, isdir
from typing import Union, Optional, List
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
import pyproj
import toml
import codecs
from pyflwdir import core_d8, core_ldd, core_conversion
from dask.diagnostics import ProgressBar
import logging
import rioxarray  # required for rio accessor

# from dask.distributed import LocalCluster, Client, performance_report
import hydromt
from hydromt.models.model_api import Model
from hydromt import flw
from hydromt.io import open_mfraster

from . import utils, workflows, DATADIR


__all__ = ["WflowModel"]

logger = logging.getLogger(__name__)

# specify pcraster map types
# NOTE non scalar (float) data types only
PCR_VS_MAP = {
    "wflow_ldd": "ldd",
    "wflow_river": "bool",
    "wflow_streamorder": "ordinal",
    "wflow_gauges": "nominal",  # to avoid large memory usage in pcraster.aguila
    "wflow_subcatch": "nominal",  # idem.
    "wflow_landuse": "nominal",
    "wflow_soil": "nominal",
    "wflow_reservoirareas": "nominal",
    "wflow_reservoirlocs": "nominal",
    "wflow_lakeareas": "nominal",
    "wflow_lakelocs": "nominal",
    "wflow_glacierareas": "nominal",
}


class WflowModel(Model):
    """This is the wflow model class"""

    _NAME = "wflow"
    _CONF = "wflow_sbm.toml"
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

    def __init__(
        self,
        root=None,
        mode="w",
        config_fn=None,
        data_libs=None,
        deltares_data=False,
        logger=logger,
    ):
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            deltares_data=deltares_data,
            logger=logger,
        )

        # wflow specific
        self._intbl = dict()
        self._flwdir = None

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
        This component sets the ``region`` of interest and ``res`` (resolution in degrees) of the
        model. All DEM and flow direction related maps are then build.

        If the model resolution is larger than the source data resolution,
        the flow direction is upscaled using the ``upscale_method``, by default the
        Iterative Hydrography Upscaling (IHU).
        The default ``hydrography_fn`` is "merit_hydro" (`MERIT hydro <http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/index.html>`_
        at 3 arcsec resolution) Alternative sources include "merit_hydro_1k" at 30 arcsec resolution.
        Users can also supply their own elevation and flow direction data. Note that only EPSG:4326 base data supported.

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
            See :py:function:~basin_mask.parse_region for all options
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
        self.logger.info(f"Preparing base hydrography basemaps.")
        # retrieve global data (lazy!)
        ds_org = self.data_catalog.get_rasterdataset(hydrography_fn)
        # TODO support and test (user) data from other sources with other crs!
        if ds_org.raster.crs is None or ds_org.raster.crs.to_epsg() != 4326:
            raise ValueError("Only EPSG:4326 base data supported.")
        # get basin geometry and clip data
        kind, region = hydromt.workflows.parse_region(region, logger=self.logger)
        xy = None
        if kind in ["basin", "subbasin", "outlet"]:
            bas_index = self.data_catalog[basin_index_fn]
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
        self.logger.debug(f"Adding basins vector to staticgeoms.")
        self.set_staticgeoms(geom, name="basins")

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
        # Rename and add to staticmaps
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_base.data_vars}
        self.set_staticmaps(ds_base.rename(rmdict))

        # setup topography maps
        ds_topo = workflows.topography(
            ds=ds_org, ds_like=self.staticmaps, method="average", logger=self.logger
        )
        ds_topo["lndslp"] = np.maximum(ds_topo["lndslp"], 0.0)
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_topo.data_vars}
        self.set_staticmaps(ds_topo.rename(rmdict))
        # set basin geometry
        self.logger.debug(f"Adding region vector to staticgeoms.")
        self.set_staticgeoms(self.region, name="region")

    def setup_rivers(
        self,
        hydrography_fn,
        river_geom_fn=None,
        river_upa=30,
        rivdph_method="powlaw",
        slope_len=2e3,
        min_rivlen_ratio=0.0,
        min_rivdph=1,
        min_rivwth=30,
        smooth_len=5e3,
        rivman_mapping_fn=join(DATADIR, "wflow", "N_river_mapping.csv"),
        **kwargs,
    ):
        """
        This component sets the all river parameter maps.

        The river mask is defined by all cells with a mimimum upstream area threshold
        `river_upa` [km2].

        The river length is defined as the distance from the subgrid outlet pixel to
        the next upstream subgrid outlet pixel. The `min_rivlen_ratio` is the minimum
        global river length to avg. cell resolution ratio and is used as a threshold in
        window based smoothing of river length.

        The river slope is derived from the subgrid elevation difference between pixels at a
        half distance `slope_len` [m] up- and downstream from the subgrid outlet pixel.

        The river manning roughness coefficient is derived based on reclassification
        of the streamorder map using a lookup table `rivman_mapping_fn`.

        The river width is derived from the nearest river segment in `river_geom_fn`.
        Data gaps are filled by the nearest valid upstream value and averaged along
        the flow directions over a length `smooth_len` [m]

        The river depth is calculated using the `rivdph_method`, by default powlaw:
        h = hc*Qbf**hp, which is based on qbankfull discharge from the nearest river
        segment in `river_geom_fn` and takes optional arguments for the hc
        (default = 0.27) and hp (default = 0.30) parameters. For other methods see
        :py:meth:`hydromt.workflows.river_depth`.

        Adds model layers:

        * **wflow_river** map: river mask [-]
        * **wflow_riverlength** map: river length [m]
        * **wflow_riverwidth** map: river width [m]
        * **RiverDepth** map: bankfull river depth [m]
        * **RiverSlope** map: river slope [m/m]
        * **N_River** map: Manning coefficient for river cells [s.m^1/3]
        * **rivers** geom: river vector based on wflow_river mask

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
            For details about the river length smoothing, see :py:meth:`pyflwdir.FlwdirRaster.smooth_rivlen`
        rivdph_method : {'gvf', 'manning', 'powlaw'}
            see py:meth:`hydromt.workflows.river_depth` for details, by default "powlaw"
        smooth_len : float, optional
            Length [m] over which to smooth the output river width and depth, by default 5e3
        min_rivdph : float, optional
            Minimum river depth [m], by default 1.0
        min_rivwth : float, optional
            Minimum river width [m], by default 30.0

        See Also
        --------
        workflows.river_bathymetry
        hydromt.workflows.river_depth
        pyflwdir.FlwdirRaster.river_depth
        """
        self.logger.info(f"Preparing river maps.")

        # read data
        ds_hydro = self.data_catalog.get_rasterdataset(
            hydrography_fn, geom=self.region, buffer=10
        )

        # get rivmsk, rivlen, rivslp
        # read model maps and revert wflow to hydromt map names
        inv_rename = {v: k for k, v in self._MAPS.items() if v in self.staticmaps}
        ds_riv = workflows.river(
            ds=ds_hydro,
            ds_model=self.staticmaps.rename(inv_rename),
            river_upa=river_upa,
            slope_len=slope_len,
            channel_dir="up",
            min_rivlen_ratio=min_rivlen_ratio,
            logger=self.logger,
        )[0]
        dvars = ["rivmsk", "rivlen", "rivslp"]
        rmdict = {k: self._MAPS.get(k, k) for k in dvars}
        self.set_staticmaps(ds_riv[dvars].rename(rmdict))

        # TODO make separate workflows.river_manning  method
        # Make N_River map from csv file with mapping between streamorder and N_River value
        strord = self.staticmaps[self._MAPS["strord"]].copy()
        df = pd.read_csv(rivman_mapping_fn, index_col=0, sep=",|;", engine="python")
        # max streamorder value above which values get the same N_River value
        max_str = df.index[-2]
        # if streamorder value larger than max_str, assign last value
        strord = strord.where(strord <= max_str, max_str)
        # handle missing value (last row of csv is mapping of missing values)
        strord = strord.where(strord != strord.raster.nodata, -999)
        strord.raster.set_nodata(-999)
        ds_nriver = workflows.landuse(
            da=strord,
            ds_like=self.staticmaps,
            fn_map=rivman_mapping_fn,
            logger=self.logger,
        )
        self.set_staticmaps(ds_nriver)

        # get rivdph, rivwth
        # while we still have setup_riverwidth one can skip river_bathymetry here
        # TODO make river_geom_fn required after removing setup_riverwidth
        if river_geom_fn is not None:
            gdf_riv = self.data_catalog.get_geodataframe(
                river_geom_fn, geom=self.region
            )
            # reread model data to get river maps
            inv_rename = {v: k for k, v in self._MAPS.items() if v in self.staticmaps}
            ds_riv1 = workflows.river_bathymetry(
                ds_model=self.staticmaps.rename(inv_rename),
                gdf_riv=gdf_riv,
                method=rivdph_method,
                smooth_len=smooth_len,
                min_rivdph=min_rivdph,
                min_rivwth=min_rivwth,
                logger=self.logger,
                **kwargs,
            )
            rmdict = {k: v for k, v in self._MAPS.items() if k in ds_riv1.data_vars}
            self.set_staticmaps(ds_riv1.rename(rmdict))
            # update config
            self.set_config("input.lateral.river.bankfull_depth", self._MAPS["rivdph"])

        self.logger.debug(f"Adding rivers vector to staticgeoms.")
        self.staticgeoms.pop("rivers", None)  # remove old rivers if in staticgeoms
        self.rivers  # add new rivers to staticgeoms

    def setup_hydrodem(
        self,
        elevtn_map="wflow_dem",
        river_routing="local-inertial",
        land_routing="kinematic-wave",
    ):
        """This component adds a hydrologically conditioned elevation (hydrodem) map for
        river and/or land local-inertial routing.

        River cells are always conditioned to D8 flow directions.  Land cells are conditioned
        to D4 flow directions if land_routing="local-inertial", else to D8. If local inertial
        is selected for land routing, it is required to have a D4 conditioning, otherwise pits
        may remain in the land cells.

        The conditioned elevation can be based on the average cell elevation ("wflow_dem")
        or subgrid outlet pixel elevation ("dem_subgrid"). For local-inertial river
        routing the subgrid elevation might provide a better representation of the river
        elevation profile, however in combination with local-inertial land routing the
        subgrid elevation will likely overestimate the floodplain storage capacity.

        Note that the same input elevation map should be used for river bankfull elevation
        and land elevation when using local-inertial land routing.

        Adds model layers:

        * **hydrodem** map: hydrologically conditioned elevation [m+REF].

        Parameters
        ----------
        elevtn_map: {"wflow_dem", "dem_subgrid"}
            Name of staticmap to hydrologically condition, by default "wflow_dem".
        river_routing, land_routing : {"kinematic-wave", "local-inertial"}
            changes wflow config model.river_routing (by default "local-inertial")
            and model.land_routing (default "kinematic-wave")

        See Also
        --------
        hydromt.flw.dem_adjust
        pyflwdir.FlwdirRaster.dem_adjust
        """
        r_list = ["kinematic-wave", "local-inertial"]
        if not elevtn_map in self.staticmaps:
            raise ValueError(f'"{elevtn_map}" not found in staticmaps')
        if river_routing not in r_list:
            raise ValueError(
                f'river_routing="{river_routing}" unknown. Select from {r_list}.'
            )
        if land_routing not in r_list:
            raise ValueError(
                f'land_routing="{land_routing}" unknown. Select from {r_list}.'
            )

        postfix = {"wflow_dem": "_avg", "dem_subgrid": "_subgrid"}.get(elevtn_map, "")
        name = f"hydrodem{postfix}"

        self.logger.info(f"Preparing {name} map for routing.")
        connectivity = {"local-inertial": 4, "kinematic-wave": 8}[land_routing]
        name = f"hydrodem{postfix}_D{connectivity}"
        self.logger.info(f"Preparing {name} map for routing.")
        ds_out = flw.dem_adjust(
            da_flwdir=self.staticmaps[self._MAPS["flwdir"]],
            da_elevtn=self.staticmaps[elevtn_map],
            da_rivmsk=self.staticmaps[self._MAPS["rivmsk"]],
            flwdir=self.flwdir,
            connectivity=connectivity,
            river_d8=True,
            logger=self.logger,
        ).rename(name)
        self.set_staticmaps(ds_out)

        # update toml model.river_routing
        self.logger.debug(f'Update wflow config model.river_routing="{river_routing}"')
        self.set_config("model.river_routing", river_routing)
        self.logger.debug(f'Update wflow config model.land_routing="{land_routing}"')
        self.set_config("model.land_routing", land_routing)
        if river_routing == "local-inertial":
            self.set_config("input.lateral.river.bankfull_depth", self._MAPS["rivdph"])
            self.set_config("input.lateral.river.bankfull_elevation", name)
        if land_routing == "local-inertial":
            self.set_config("input.lateral.land.elevation", name)
            self.config["state"]["lateral"]["land"].pop("q", None)
            self.config["output"]["lateral"]["land"].pop("q", None)
            self.set_config("state.lateral.land.qx", "qx_land")
            self.set_config("state.lateral.land.qy", "qy_land")
        else:
            self.config["state"]["lateral"]["land"].pop("qx", None)
            self.config["state"]["lateral"]["land"].pop("qy", None)
            self.config["output"]["lateral"]["land"].pop("qx", None)
            self.config["output"]["lateral"]["land"].pop("qy", None)
            self.set_config("state.lateral.land.q", "q_land")

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
        This component sets the river width parameter based on a power-lay relationship with a predictor.

        By default the riverwidth is estimated based on discharge as ``predictor`` and used to
        set the riverwidth globally based on pre-defined power-law parameters per climate class.
        With ``fit`` set to True, the power-law relationsship paramters are set on-the-fly.
        With ``fill`` set to True, the estimated river widths are only used fill gaps in the
        observed data. Alternative ``predictor`` values are precip (accumulated precipitation)
        and uparea (upstream area). For these predictors values ``fit`` default to True.
        By default the predictor is based on discharge which is estimated through multiple linear
        regression with precipitation and upstream area per climate zone.

        * **wflow_riverwidth** map: river width [m]

        Parameters
        ----------
        predictor : {"discharge", "precip", "uparea"}
            Predictor used in the power-law equation: width = a * predictor ^ b.
            Discharge is based on multiple linear regression per climate zone.
            Precip is based on the 10x the daily average accumulated precipitation [m3/s].
            Uparea is based on the upstream area grid [km2].
            Other variables, e.g. bankfull discharge, can also be provided if present in the staticmaps
        fill : bool, optional
            If True (default), use estimate to fill gaps, outliers and lake/res areas in observed width data (if present);
            if False, set all riverwidths based on predictor (automatic choice if no observations found)
        fit : bool, optional kwarg
            If True, the power-law parameters are fitted on the fly
            By default True for all but "discharge" predictor.
            A-priori derived parameters will be overwritten if True.
        a, b : float, optional kwarg
            Manual power-law parameters
        min_wth : float
            minimum river width
        precip_fn : {'chelsa'}
            Source of long term precipitation grid if the predictor is set to 'discharge' or 'precip'.
        climate_fn: {'koppen_geiger'}
            Source of long-term climate grid if the predictor is set to 'discharge'.
        """
        self.logger.warning(
            'The "setup_riverwidth" method has been deprecated and will soon be removed. '
            'You can now use the "setup_river" method for all river parameters.'
        )
        if not self._MAPS["rivmsk"] in self.staticmaps:
            raise ValueError(
                'The "setup_riverwidth" method requires to run setup_river method first.'
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

        inv_rename = {v: k for k, v in self._MAPS.items() if v in self.staticmaps}
        da_rivwth = workflows.river_width(
            ds_like=self.staticmaps.rename(inv_rename),
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

        self.set_staticmaps(da_rivwth, name=self._MAPS["rivwth"])

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
        This component derives several wflow maps are derived based on landuse-
        landcover (LULC) data.

        Currently, ``lulc_fn`` can be set to the "vito", "globcover"
        or "corine", fo which lookup tables are constructed to convert lulc classses to
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
            Path to a mapping csv file from landuse in source name to parameter values in lulc_vars.
        lulc_vars : list
            List of landuse parameters to keep.\
            By default ["landuse","Kext","N","PathFrac","RootingDepth","Sl","Swood","WaterFrac"]
        """
        self.logger.info(f"Preparing LULC parameter maps.")
        if lulc_mapping_fn is None:
            fn_map = join(DATADIR, "lulc", f"{lulc_fn}_mapping.csv")
        else:
            fn_map = lulc_mapping_fn
        if not isfile(fn_map):
            self.logger.error(f"LULC mapping file not found: {fn_map}")
            return
        # read landuse map to DataArray
        da = self.data_catalog.get_rasterdataset(
            lulc_fn, geom=self.region, buffer=2, variables=["landuse"]
        )
        # process landuse
        ds_lulc_maps = workflows.landuse(
            da=da,
            ds_like=self.staticmaps,
            fn_map=fn_map,
            params=lulc_vars,
            logger=self.logger,
        )
        rmdict = {k: v for k, v in self._MAPS.items() if k in ds_lulc_maps.data_vars}
        self.set_staticmaps(ds_lulc_maps.rename(rmdict))

    def setup_laimaps(self, lai_fn="modis_lai"):
        """
        This component sets leaf area index (LAI) climatology maps per month.

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
        self.logger.info(f"Preparing LAI maps.")
        da = self.data_catalog.get_rasterdataset(lai_fn, geom=self.region, buffer=2)
        da_lai = workflows.lai(
            da=da,
            ds_like=self.staticmaps,
            logger=self.logger,
        )
        # Rename the first dimension to time
        rmdict = {da_lai.dims[0]: "time"}
        self.set_staticmaps(da_lai.rename(rmdict), name="LAI")

    def setup_gauges(
        self,
        gauges_fn="grdc",
        source_gdf=None,
        index_col=None,
        snap_to_river=True,
        mask=None,
        derive_subcatch=False,
        derive_outlet=True,
        basename=None,
        update_toml=True,
        gauge_toml_header=None,
        gauge_toml_param=None,
        **kwargs,
    ):
        """This components sets the default gauge map based on basin outlets and additional
        gauge maps based on ``gauges_fn`` data.

        Supported gauge datasets include "grdc"
        or "<path_to_source>" for user supplied csv or geometry files with gauge locations.
        If a csv file is provided, a "x" or "lon" and "y" or "lat" column is required
        and the first column will be used as IDs in the map. If ``snap_to_river`` is set
        to True, the gauge location will be snapped to the boolean river mask. If
        ``derive_subcatch`` is set to True, an additional subcatch map is derived from
        the gauge locations.

        Adds model layers:

        * **wflow_gauges** map: gauge IDs map from catchment outlets [-]
        * **wflow_gauges_source** map: gauge IDs map from source [-] (if gauges_fn)
        * **wflow_subcatch_source** map: subcatchment based on gauge locations [-] (if derive_subcatch)
        * **gauges** geom: polygon of catchment outlets
        * **gauges_source** geom: polygon of gauges from source
        * **subcatch_source** geom: polygon of subcatchment based on gauge locations [-] (if derive_subcatch)

        Parameters
        ----------
        gauges_fn : str, {"grdc"}, optional
            Known source name or path to gauges file geometry file, by default None.
        source_gdf : geopandas.GeoDataFame, optional
            Direct gauges file geometry, by default None.
        index_col : str, optional
            Column in gauges_fn to use for ID values, by default None (use the default index column)
        snap_to_river : bool, optional
            Snap point locations to the closest downstream river cell, by default True
        mask : np.boolean, optional
            If provided snaps to the mask, else snaps to the river (default).
        derive_subcatch : bool, optional
            Derive subcatch map for gauges, by default False
        derive_outlet : bool, optional
            Derive gaugemap based on catchment outlets, by default True
        basename : str, optional
            Map name in staticmaps (wflow_gauges_basename), if None use the gauges_fn basename.
        update_toml : boolean, optional
            Update [outputcsv] section of wflow toml file.
        gauge_toml_header : list, optional
            Save specific model parameters in csv section. This option defines the header of the csv file./
            By default saves Q (for lateral.river.q_av) and P (for vertical.precipitation).
        gauge_toml_param: list, optional
            Save specific model parameters in csv section. This option defines the wflow variable corresponding to the/
            names in gauge_toml_header. By default saves lateral.river.q_av (for Q) and vertical.precipitation (for P).
        """
        # read existing staticgeoms; important to get the right basin when updating
        self.staticgeoms

        if derive_outlet:
            self.logger.info(f"Gauges locations set based on river outlets.")
            da, idxs, ids = flw.gauge_map(self.staticmaps, idxs=self.flwdir.idxs_pit)
            # Only keep river outlets for gauges
            da = da.where(self.staticmaps[self._MAPS["rivmsk"]] != 0, da.raster.nodata)
            ids_da = np.unique(da.values[da.values > 0])
            idxs_da = idxs[np.isin(ids, ids_da)]
            self.set_staticmaps(da, name=self._MAPS["gauges"])
            points = gpd.points_from_xy(*self.staticmaps.raster.idx_to_xy(idxs_da))
            gdf = gpd.GeoDataFrame(
                index=ids_da.astype(np.int32), geometry=points, crs=self.crs
            )
            gdf["fid"] = ids_da.astype(np.int32)
            self.set_staticgeoms(gdf, name="gauges")
            self.logger.info(f"Gauges map based on catchment river outlets added.")

        if gauges_fn is not None or source_gdf is not None:
            # append location from geometry
            # TODO check snapping locations based on upstream area attribute of the gauge data
            if gauges_fn is not None:
                kwargs = {}
                if isfile(gauges_fn):
                    # try to get epsg number directly, important when writting back data_catalog
                    if hasattr(self.crs, "to_epsg"):
                        code = self.crs.to_epsg()
                    else:
                        code = self.crs
                    kwargs.update(crs=code)
                gdf = self.data_catalog.get_geodataframe(
                    gauges_fn, geom=self.basins, assert_gtype="Point", **kwargs
                )
                gdf = gdf.to_crs(self.crs)
            elif source_gdf is not None and basename is None:
                raise ValueError(
                    "Basename is required when setting gauges based on source_gdf"
                )
            elif source_gdf is not None:
                self.logger.info(f"Gauges locations read from source_gdf")
                gdf = source_gdf.to_crs(self.crs)

            if gdf.index.size == 0:
                self.logger.warning(
                    f"No {gauges_fn} gauge locations found within domain"
                )
            else:
                if basename is None:
                    basename = (
                        os.path.basename(gauges_fn).split(".")[0].replace("_", "-")
                    )
                self.logger.info(
                    f"{gdf.index.size} {basename} gauge locations found within domain"
                )
                # Set index to index_col
                if index_col is not None and index_col in gdf:
                    gdf = gdf.set_index(index_col)
                xs, ys = np.vectorize(lambda p: (p.xy[0][0], p.xy[1][0]))(
                    gdf["geometry"]
                )
                idxs = self.staticmaps.raster.xy_to_idx(xs, ys)
                ids = gdf.index.values

                if snap_to_river and mask is None:
                    mask = self.staticmaps[self._MAPS["rivmsk"]].values
                da, idxs, ids = flw.gauge_map(
                    self.staticmaps,
                    idxs=idxs,
                    ids=ids,
                    stream=mask,
                    flwdir=self.flwdir,
                    logger=self.logger,
                )
                # Filter gauges that could not be snapped to rivers
                if snap_to_river:
                    ids_old = ids.copy()
                    da = da.where(
                        self.staticmaps[self._MAPS["rivmsk"]] != 0, da.raster.nodata
                    )
                    ids_new = np.unique(da.values[da.values > 0])
                    idxs = idxs[np.isin(ids_old, ids_new)]
                    ids = da.values.flat[idxs]
                # Add to staticmaps
                mapname = f'{str(self._MAPS["gauges"])}_{basename}'
                self.set_staticmaps(da, name=mapname)

                # geoms
                points = gpd.points_from_xy(*self.staticmaps.raster.idx_to_xy(idxs))
                # if csv contains additional columns, these are also written in the staticgeoms
                gdf_snapped = gpd.GeoDataFrame(
                    index=ids.astype(np.int32), geometry=points, crs=self.crs
                )
                # Set the index name of gdf snapped based on original gdf
                if gdf.index.name is not None:
                    gdf_snapped.index.name = gdf.index.name
                else:
                    gdf_snapped.index.name = "fid"
                    gdf.index.name = "fid"
                # Add gdf attributes to gdf_snapped (filter on snapped index before merging)
                df_attrs = pd.DataFrame(gdf.drop(columns="geometry"))
                df_attrs = df_attrs[np.isin(df_attrs.index, gdf_snapped.index)]
                gdf_snapped = gdf_snapped.merge(
                    df_attrs, how="inner", on=gdf.index.name
                )
                # Add gdf_snapped to staticgeoms
                self.set_staticgeoms(gdf_snapped, name=mapname.replace("wflow_", ""))

                # # Add new outputcsv section in the config
                if gauge_toml_param is None and update_toml:
                    gauge_toml_header = ["Q", "P"]
                    gauge_toml_param = ["lateral.river.q_av", "vertical.precipitation"]

                if update_toml:
                    self.set_config(f"input.gauges_{basename}", f"{mapname}")
                    if self.get_config("csv") is not None:
                        for o in range(len(gauge_toml_param)):
                            gauge_toml_dict = {
                                "header": gauge_toml_header[o],
                                "map": f"gauges_{basename}",
                                "parameter": gauge_toml_param[o],
                            }
                            # If the gauge outcsv column already exists skip writting twice
                            if gauge_toml_dict not in self.config["csv"]["column"]:
                                self.config["csv"]["column"].append(gauge_toml_dict)
                    self.logger.info(f"Gauges map from {basename} added.")

                # add subcatch
                if derive_subcatch:
                    da_basins = flw.basin_map(
                        self.staticmaps, self.flwdir, idxs=idxs, ids=ids
                    )[0]
                    mapname = self._MAPS["basins"] + "_" + basename
                    self.set_staticmaps(da_basins, name=mapname)
                    gdf_basins = self.staticmaps[mapname].raster.vectorize()
                    self.set_staticgeoms(gdf_basins, name=mapname.replace("wflow_", ""))

    def setup_areamap(
        self,
        area_fn: str,
        col2raster: str,
        nodata: Union[int, float] = -1,
    ):
        """Setup area map from vector data to save wflow outputs for specific area.
        Adds model layer:
        * **area_fn** map:  output area data map
        Parameters
        ----------
        area_fn : str
            Name of vector data corresponding to wflow output area.
        col2raster : str
            Name of the column from the vector file to rasterize.
        nodata : int/float, optional
            Nodata value to use when rasterizing. Should match the dtype of col2raster. By default -1.
        """
        if area_fn not in self.data_catalog:
            self.logger.warning(f"Invalid source '{area_fn}', skipping setup_areamap.")
            return

        self.logger.info(f"Preparing '{area_fn}' map.")
        gdf_org = self.data_catalog.get_geodataframe(
            area_fn, geom=self.basins, dst_crs=self.crs
        )
        if gdf_org.empty:
            self.logger.warning(
                f"No shapes of {area_fn} found within region, skipping areamap."
            )
            return
        else:
            da_area = self.staticmaps.raster.rasterize(
                gdf=gdf_org,
                col_name=col2raster,
                nodata=nodata,
                all_touched=True,
            )
        self.set_staticmaps(da_area.rename(area_fn))

    def setup_lakes(self, lakes_fn="hydro_lakes", min_area=10.0):
        """This component generates maps of lake areas and outlets as well as parameters
        with average lake area, depth a discharge values.

        The data is generated from features with ``min_area`` [km2] (default 1 km2) from a database with
        lake geometry, IDs and metadata. Data required are lake ID 'waterbody_id', average area 'Area_avg' [m2],
        average volume 'Vol_avg' [m3], average depth 'Depth_avg' [m] and average discharge 'Dis_avg' [m3/s].

        Adds model layers:

        * **wflow_lakeareas** map: lake IDs [-]
        * **wflow_lakelocs** map: lake IDs at outlet locations [-]
        * **LakeArea** map: lake area [m2]
        * **LakeAvgLevel** map: lake average water level [m]
        * **LakeAvgOut** map: lake average discharge [m3/s]
        * **Lake_b** map: lake rating curve coefficient [-]
        * **lakes** geom: polygon with lakes and wflow lake parameters

        Parameters
        ----------
        lakes_fn : {'hydro_lakes'}
            Name of data source for lake parameters, see data/data_sources.yml.

            * Required variables: ['waterbody_id', 'Area_avg', 'Vol_avg', 'Depth_avg', 'Dis_avg']
        min_area : float, optional
            Minimum lake area threshold [km2], by default 1.0 km2.
        """

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

        gdf_org, ds_lakes = self._setup_waterbodies(lakes_fn, "lake", min_area)
        if ds_lakes is not None:
            rmdict = {k: v for k, v in self._MAPS.items() if k in ds_lakes.data_vars}
            self.set_staticmaps(ds_lakes.rename(rmdict))
            # add waterbody parameters
            # rename to param values
            gdf_org = gdf_org.rename(
                columns={
                    "Area_avg": "LakeArea",
                    "Depth_avg": "LakeAvgLevel",
                    "Dis_avg": "LakeAvgOut",
                }
            )
            # Minimum value for LakeAvgOut
            LakeAvgOut = gdf_org["LakeAvgOut"].copy()
            gdf_org["LakeAvgOut"] = np.maximum(gdf_org["LakeAvgOut"], 0.01)
            gdf_org["Lake_b"] = gdf_org["LakeAvgOut"].values / (
                gdf_org["LakeAvgLevel"].values
            ) ** (2)
            gdf_org["Lake_e"] = 2
            gdf_org["LakeStorFunc"] = 1
            gdf_org["LakeOutflowFunc"] = 3
            gdf_org["LakeThreshold"] = 0.0
            gdf_org["LinkedLakeLocs"] = 0

            # Check if some LakeAvgOut values have been replaced
            if not np.all(LakeAvgOut == gdf_org["LakeAvgOut"]):
                self.logger.warning(
                    "Some values of LakeAvgOut have been replaced by a minimum value of 0.01m3/s"
                )

            lake_params = [
                "waterbody_id",
                "LakeArea",
                "LakeAvgLevel",
                "LakeAvgOut",
                "Lake_b",
                "Lake_e",
                "LakeStorFunc",
                "LakeOutflowFunc",
                "LakeThreshold",
                "LinkedLakeLocs",
            ]

            gdf_org_points = gpd.GeoDataFrame(
                gdf_org[lake_params],
                geometry=gpd.points_from_xy(gdf_org.xout, gdf_org.yout),
            )

            # write lakes with attr tables to static geoms.
            self.set_staticgeoms(gdf_org, name="lakes")

            for name in lake_params[1:]:
                da_lake = ds_lakes.raster.rasterize(
                    gdf_org_points, col_name=name, dtype="float32", nodata=-999
                )
                self.set_staticmaps(da_lake)

            # if there are lakes, change True in toml
            for option in lakes_toml:
                self.set_config(option, lakes_toml[option])

    def setup_reservoirs(
        self,
        reservoirs_fn="hydro_reservoirs",
        min_area=1.0,
        priority_jrc=True,
        **kwargs,
    ):
        """This component generates maps of reservoir areas and outlets as well as parameters
        with average reservoir area, demand, min and max target storage capacities and
        discharge capacity values.

        The data is generated from features with ``min_area`` [km2] (default is 1 km2)
        from a database with reservoir geometry, IDs and metadata.

        Data requirements for direct use (ie wflow parameters are data already present in reservoirs_fn)
        are reservoir ID 'waterbody_id', area 'ResSimpleArea' [m2], maximum volume 'ResMaxVolume' [m3],
        the targeted minimum and maximum fraction of water volume in the reservoir 'ResTargetMinFrac'
        and 'ResTargetMaxFrac' [-], the average water demand ResDemand [m3/s] and the maximum release of
        the reservoir before spilling 'ResMaxRelease' [m3/s].

        In case the wflow parameters are not directly available they can be computed by HydroMT using other
        reservoir characteristics. If not enough characteristics are available, the hydroengine tool will be
        used to download additionnal timeseries from the JRC database.
        The required variables for computation of the parameters with hydroengine are reservoir ID 'waterbody_id',
        reservoir ID in the HydroLAKES database 'Hylak_id', average volume 'Vol_avg' [m3], average depth 'Depth_avg'
        [m], average discharge 'Dis_avg' [m3/s] and dam height 'Dam_height' [m].
        To compute parameters without using hydroengine, the required varibales in reservoirs_fn are reservoir ID 'waterbody_id',
        average area 'Area_avg' [m2], average volume 'Vol_avg' [m3], average depth 'Depth_avg' [m], average discharge 'Dis_avg'
        [m3/s] and dam height 'Dam_height' [m] and minimum / normal / maximum storage capacity of the dam 'Capacity_min',
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
        reservoirs_fn : {'hydro_reservoirs'}
            Name of data source for reservoir parameters, see data/data_sources.yml.

            * Required variables for direct use: ['waterbody_id', 'ResSimpleArea', 'ResMaxVolume', 'ResTargetMinFrac', 'ResTargetFullFrac', 'ResDemand', 'ResMaxRelease']

            * Required variables for computation with hydroengine: ['waterbody_id', 'Hylak_id', 'Vol_avg', 'Depth_avg', 'Dis_avg', 'Dam_height']

            * Required variables for computation without hydroengine: ['waterbody_id', 'Area_avg', 'Vol_avg', 'Depth_avg', 'Dis_avg', 'Capacity_max', 'Capacity_norm', 'Capacity_min', 'Dam_height']
        min_area : float, optional
            Minimum reservoir area threshold [km2], by default 1.0 km2.
        priority_jrc : boolean, optional
            If True, use JRC water occurrence (Pekel,2016) data from GEE to calculate
            and overwrite the reservoir volume/areas of the data source.
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
        # TODO: check if there are missing values in the above columns of the parameters tbls =
        # if everything is present, skip calculate reservoirattrs() and directly make the maps
        if ds_res is not None:
            rmdict = {k: v for k, v in self._MAPS.items() if k in ds_res.data_vars}
            self.set_staticmaps(ds_res.rename(rmdict))

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
            # else compute
            else:
                intbl_reservoirs, reservoir_accuracy = workflows.reservoirattrs(
                    gdf=gdf_org,
                    priorityJRC=priority_jrc,
                    usehe=kwargs.get("usehe", True),
                    logger=self.logger,
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

            # write reservoirs with param values to staticgeoms
            self.set_staticgeoms(gdf_org, name="reservoirs")

            for name in gdf_org_points.columns[2:]:
                gdf_org_points[name] = gdf_org_points[name].astype("float32")
                da_res = ds_res.raster.rasterize(
                    gdf_org_points, col_name=name, dtype="float32", nodata=-999
                )
                self.set_staticmaps(da_res)

            # Save accuracy information on reservoir parameters
            if reservoir_accuracy is not None:
                reservoir_accuracy.to_csv(join(self.root, "reservoir_accuracy.csv"))

            for option in res_toml:
                self.set_config(option, res_toml[option])

    def _setup_waterbodies(self, waterbodies_fn, wb_type, min_area=0.0):
        """Helper method with the common workflow of setup_lakes and setup_reservoir.
        See specific methods for more info about the arguments."""
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
            if uparea_name not in self.staticmaps.data_vars:
                self.logger.warning(
                    f"Upstream area map for {wb_type} outlet setup not found. "
                    "Database coordinates used instead"
                )
                uparea_name = None
            ds_waterbody, gdf_wateroutlet = workflows.waterbodymaps(
                gdf=gdf_org,
                ds_like=self.staticmaps,
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

        # rasterize points polygons in raster.rasterize -- you need staticmaps to nkow the grid
        return gdf_org, ds_waterbody

    def setup_soilmaps(self, soil_fn="soilgrids", ptf_ksatver="brakensiek"):
        """
        This component derives several (layered) soil parameters based on a database with
        physical soil properties using available point-scale (pedo)transfer functions (PTFs)
        from literature with upscaling rules to ensure flux matching across scales.

        Currently, supported ``soil_fn`` is "soilgrids" and "soilgrids_2020".
        ``ptf_ksatver`` (PTF for the vertical hydraulic conductivity) options are "brakensiek" and "cosby".
        "soilgrids" provides data at 7 specific depths, while "soilgrids_2020" provides data averaged over 6 depth intervals.
        This leads to small changes in the workflow: (1) M parameter uses midpoint depths in soilgrids_2020 versus specific depths in soilgrids,
        (2) weighted average of soil properties over soil thickness is done with the trapezoidal rule in soilgrids versus simple block weighted average in soilgrids_2020,
        (3) the c parameter is computed as weighted average over wflow_sbm soil layers in soilgrids_2020 versus at specific depths for soilgrids.

        The following maps are added to staticmaps:

        * **thetaS** map: average saturated soil water content [m3/m3]
        * **thetaR** map: average residual water content [m3/m3]
        * **KsatVer** map: vertical saturated hydraulic conductivity at soil surface [mm/day]
        * **SoilThickness** map: soil thickness [mm]
        * **SoilMinThickness** map: minimum soil thickness [mm] (equal to SoilThickness)
        * **M** map: model parameter [mm] that controls exponential decline of KsatVer with soil depth
            (fitted with curve_fit (scipy.optimize)), bounds of M are checked
        * **`M_`** map: model parameter [mm] that controls exponential decline of KsatVer with soil depth
            (fitted with numpy linalg regression), bounds of `M_` are checked
        * **M_original** map: M without checking bounds
        * **`M_original_`** map: `M_` without checking bounds
        * **f** map: scaling parameter controlling the decline of KsatVer [mm-1] (fitted with curve_fit (scipy.optimize)), bounds are checked
        * **`f_`** map: scaling parameter controlling the decline of KsatVer [mm-1] (fitted with numpy linalg regression), bounds are checked
        * **c_0** map: Brooks Corey coefficient [-] based on pore size distribution index at
            depth of 1st soil layer (100 mm) wflow_sbm
        * **c_1** map: idem c_0 at depth 2nd soil layer (400 mm) wflow_sbm
        * **c_2** map: idem c_0 at depth 3rd soil layer (1200 mm) wflow_sbm
        * **c_3** map: idem c_0 at depth 4th soil layer (> 1200 mm) wflow_sbm
        * **KsatVer_[z]cm** map: KsatVer [mm/day] at soil depths [z] of SoilGrids data [0.0, 5.0, 15.0, 30.0, 60.0, 100.0, 200.0]
        * **wflow_soil** map: soil texture based on USDA soil texture triangle (mapping: [1:Clay, 2:Silty Clay, 3:Silty Clay-Loam, 4:Sandy Clay, 5:Sandy Clay-Loam, 6:Clay-Loam, 7:Silt, 8:Silt-Loam, 9:Loam, 10:Sand, 11: Loamy Sand, 12:Sandy Loam])


        Parameters
        ----------
        soil_fn : {'soilgrids', 'soilgrids_2020'}
            Name of data source for soil parameter maps, see data/data_sources.yml. Should contain info for the
            7 soil depths of soilgrids (or 6 depths intervals for soilgrids_2020).
            * Required variables: ['bd_sl*', 'clyppt_sl*', 'sltppt_sl*', 'oc_sl*', 'ph_sl*', 'sndppt_sl*', 'soilthickness', 'tax_usda']
        ptf_ksatver : {'brakensiek', 'cosby'}
            Pedotransfer function (PTF) to use for calculation KsatVer (vertical saturated
            hydraulic conductivity [mm/day]). By default 'brakensiek'.
        """
        self.logger.info(f"Preparing soil parameter maps.")
        # TODO add variables list with required variable names
        dsin = self.data_catalog.get_rasterdataset(soil_fn, geom=self.region, buffer=2)
        dsout = workflows.soilgrids(
            dsin,
            self.staticmaps,
            ptf_ksatver,
            soil_fn,
            logger=self.logger,
        ).reset_coords(drop=True)
        self.set_staticmaps(dsout)

    def setup_glaciers(self, glaciers_fn="rgi", min_area=1):
        """
        This component generates maps of glacier areas, area fraction and volume fraction,
        as well as tables with temperature threshold, melting factor and snow-to-ice
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
        * **G_SIfrac** map: fraction of snowpack on top of glacier converted to ice, added to glacierstore [-]

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
        self.logger.info(f"Preparing glacier maps.")
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
                ds_like=self.staticmaps,
                id_column="simple_id",
                elevtn_name=self._MAPS["elevtn"],
                logger=self.logger,
            )

            rmdict = {k: v for k, v in self._MAPS.items() if k in ds_glac.data_vars}
            self.set_staticmaps(ds_glac.rename(rmdict))

            self.set_staticgeoms(gdf_org, name="glaciers")

            for option in glac_toml:
                self.set_config(option, glac_toml[option])
        else:
            self.logger.warning(
                f"No glaciers of sufficient size found within region!"
                f"Skipping glacier procedures!"
            )

    def setup_constant_pars(self, dtype="float32", nodata=-999, **kwargs):
        """Setup constant parameter maps for all active model cells.

        Adds model layer:

        * **param_name** map: constant parameter map.

        Parameters
        ----------
        dtype: str
            data type
        nodata: int or float
            nodata value
        kwargs
            "param_name: value" pairs for constant staticmaps.

        """
        for key, value in kwargs.items():
            nodata = np.dtype(dtype).type(nodata)
            da_param = xr.where(
                self.staticmaps[self._MAPS["basins"]], value, nodata
            ).astype(dtype)
            da_param.raster.set_nodata(nodata)

            da_param = da_param.rename(key)
            self.set_staticmaps(da_param)

    def setup_staticmaps_from_raster(
        self,
        raster_fn: str,
        reproject_method: str,
        variables: Optional[List] = None,
        wflow_variables: Optional[List] = None,
        fill_method: Optional[str] = None,
    ) -> List[str]:
        """
        This component adds data variable(s) from ``raster_fn`` to staticmaps object.
        If raster is a dataset, all variables will be added unless ``variables`` list is specified.
        The config toml can also be updated to include the new maps using ``wflow_variables``.

        Adds model layers:
        * **raster.name** or **variables** staticmaps: data from raster_fn

        Parameters
        ----------
        raster_fn: str
            Source name of raster data in data_catalog.
        reproject_method: str
            Reprojection method from rasterio.enums.Resampling.
            Available methods: ['nearest', 'bilinear', 'cubic', 'cubic_spline', 'lanczos', 'average', 'mode',
            'gauss', 'max', 'min', 'med', 'q1', 'q3', 'sum', 'rms']
        variables: list, optional
            List of variables to add to staticmaps from raster_fn. By default all.
        wflow_variables: list, optional
            List of corresponding wflow variables to update the config toml (e.g: ["input.vertical.altitude"]).
            Should match the variables list. variables list should be provided unless raster_fn contains
            a single variable (len 1).
        fill_method : str, optional
            If specified, fills nodata values using fill_nodata method.
            Available methods are {'linear', 'nearest', 'cubic', 'rio_idw'}.

        Returns
        -------
        list
            Names of added model staticmap layers.
        """
        self.logger.info(f"Preparing staticmaps data from raster source {raster_fn}")
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
        ds_out = ds.raster.reproject_like(self.staticmaps, method=reproject_method)
        # Add to staticmaps
        self.set_staticmaps(ds_out)

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
                        f"Cannot update the toml if raster_fn has more than one variable and variables list is not provided."
                    )

            # Check on len
            if len(wflow_variables) != len(variables):
                raise ValueError(
                    f"Length of variables {variables} do not match wflow_variables {wflow_variables}. Cannot update the toml."
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
        """Setup gridded precipitation forcing at model resolution.

        Adds model layer:

        * **precip**: precipitation [mm]

        Parameters
        ----------
        precip_fn : str, default era5
            Precipitation data source, see data/forcing_sources.yml.

            * Required variable: ['precip']
        precip_clim_fn : str, default None
            High resolution climatology precipitation data source to correct precipitation,
            see data/forcing_sources.yml.

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
        mask = self.staticmaps[self._MAPS["basins"]].values > 0

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
        if precip_clim_fn != None:
            clim = self.data_catalog.get_rasterdataset(
                precip_clim_fn,
                geom=precip.raster.box,
                buffer=2,
                variables=["precip"],
            )

        precip_out = hydromt.workflows.forcing.precip(
            precip=precip,
            da_like=self.staticmaps[self._MAPS["elevtn"]],
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
        dem_forcing_fn: str = "era5_orography",
        skip_pet: str = False,
        chunksize: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Setup gridded reference evapotranspiration forcing at model resolution.

        Adds model layer:

        * **pet**: reference evapotranspiration [mm]
        * **temp**: temperature [°C]

        Parameters
        ----------
        temp_pet_fn : str, optional
            Name or path of data source with variables to calculate temperature and reference evapotranspiration,
            see data/forcing_sources.yml, by default 'era5'.

            * Required variable for temperature: ['temp']

            * Required variables for De Bruin reference evapotranspiration: ['temp', 'press_msl', 'kin', 'kout']

            * Required variables for Makkink reference evapotranspiration: ['temp', 'press_msl', 'kin']
        pet_method : {'debruin', 'makkink'}, optional
            Reference evapotranspiration method, by default 'debruin'.
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
        mask = self.staticmaps[self._MAPS["basins"]].values > 0

        variables = ["temp"]
        if not skip_pet:
            if pet_method == "debruin":
                variables += ["press_msl", "kin", "kout"]
            elif pet_method == "makkink":
                variables += ["press_msl", "kin"]
            else:
                methods = ["debruin", "makking"]
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
        if dem_forcing_fn != None:
            dem_forcing = self.data_catalog.get_rasterdataset(
                dem_forcing_fn,
                geom=ds.raster.box,  # clip dem with forcing bbox for full coverage
                buffer=2,
                variables=["elevtn"],
            ).squeeze()

        temp_in = hydromt.workflows.forcing.temp(
            ds["temp"],
            dem_model=self.staticmaps[self._MAPS["elevtn"]],
            dem_forcing=dem_forcing,
            lapse_correction=temp_correction,
            logger=self.logger,
            freq=None,  # resample time after pet workflow
            **kwargs,
        )

        if not skip_pet:
            pet_out = hydromt.workflows.forcing.pet(
                ds[variables[1:]],
                dem_model=self.staticmaps[self._MAPS["elevtn"]],
                temp=temp_in,
                method=pet_method,
                press_correction=press_correction,
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

    # I/O
    def read(self):
        """Method to read the complete model schematization and configuration from file."""
        self.read_config()
        self.read_staticmaps()
        self.read_intbl()
        self.read_staticgeoms()
        self.read_forcing()
        self.logger.info("Model read")

    def write(self):
        """Method to write the complete model schematization and configuration to file."""
        self.logger.info(f"Write model data to {self.root}")
        # if in r, r+ mode, only write updated components
        if not self._write:
            self.logger.warning("Cannot write in read-only mode")
            return
        self.write_data_catalog()
        if self.config:  # try to read default if not yet set
            self.write_config()
        if self._staticmaps:
            self.write_staticmaps()
        if self._staticgeoms:
            self.write_staticgeoms()
        if self._forcing:
            self.write_forcing()

    def read_staticmaps(self, **kwargs):
        """Read staticmaps"""
        fn_default = join(self.root, "staticmaps.nc")
        fn = self.get_config("input.path_static", abs_path=True, fallback=fn_default)
        if not self._write:
            # start fresh in read-only mode
            self._staticmaps = xr.Dataset()
        if fn is not None and isfile(fn):
            self.logger.info(f"Read staticmaps from {fn}")
            # FIXME: we need a smarter (lazy) solution for big models which also
            # works when overwriting / appending data in the same source!
            ds = xr.open_dataset(
                fn, mask_and_scale=False, decode_coords="all", **kwargs
            ).load()
            ds.close()
            # make sure internally maps are always North -> South oriented
            if ds.raster.res[1] > 0:
                ds = ds.raster.flipud()
            self.set_staticmaps(ds)
        elif len(glob.glob(join(self.root, "staticmaps", "*.map"))) > 0:
            self.read_staticmaps_pcr()

    def write_staticmaps(self):
        """Write staticmaps"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        # clean-up staticmaps and write CRS according to CF-conventions
        # TODO replace later with hydromt.raster.gdal_compliant method after core release
        crs = self.staticmaps.raster.crs
        ds_out = self.staticmaps.reset_coords()
        # TODO?!
        # if ds_out.raster.res[1] < 0: # write data with South -> North orientation
        #     ds_out = ds_out.raster.flipud()
        x_dim, y_dim, x_attrs, y_attrs = hydromt.gis_utils.axes_attrs(crs)
        ds_out = ds_out.rename({ds_out.raster.x_dim: x_dim, ds_out.raster.y_dim: y_dim})
        ds_out[x_dim].attrs.update(x_attrs)
        ds_out[y_dim].attrs.update(y_attrs)
        ds_out = ds_out.drop_vars(["mask", "spatial_ref", "ls"], errors="ignore")
        ds_out.rio.write_crs(crs, inplace=True)
        ds_out.rio.write_transform(self.staticmaps.raster.transform, inplace=True)
        ds_out.raster.set_spatial_dims()

        # filename
        fn_default = join(self.root, "staticmaps.nc")
        fn = self.get_config("input.path_static", abs_path=True, fallback=fn_default)
        # Check if all sub-folders in fn exists and if not create them
        if not isdir(dirname(fn)):
            os.makedirs(dirname(fn))
        self.logger.info(f"Write staticmaps to {fn}")
        mask = ds_out[self._MAPS["basins"]] > 0
        for v in ds_out.data_vars:
            # nodata is required for all but boolean fields
            if ds_out[v].dtype != "bool":
                ds_out[v] = ds_out[v].where(mask, ds_out[v].raster.nodata)
        ds_out.to_netcdf(fn)
        # self.write_staticmaps_pcr()

    def read_staticmaps_pcr(self, crs=4326, **kwargs):
        """Read and staticmaps at <root/staticmaps> and parse to xarray"""
        if self._read and "chunks" not in kwargs:
            kwargs.update(chunks={"y": -1, "x": -1})
        fn = join(self.root, "staticmaps", f"*.map")
        fns = glob.glob(fn)
        if len(fns) == 0:
            self.logger.warning(f"No staticmaps found at {fn}")
            return
        self._staticmaps = open_mfraster(fns, **kwargs)
        path = join(self.root, "staticmaps", "clim", f"LAI*")
        if len(glob.glob(path)) > 0:
            da_lai = open_mfraster(
                path, concat=True, concat_dim="time", logger=self.logger, **kwargs
            )
            self.set_staticmaps(da_lai, "LAI")

        # reorganize c_0 etc maps
        da_c = []
        list_c = [v for v in self._staticmaps if str(v).startswith("c_")]
        if len(list_c) > 0:
            for i, v in enumerate(list_c):
                da_c.append(self._staticmaps[f"c_{i:d}"])
            da = xr.concat(
                da_c, pd.Index(np.arange(len(list_c), dtype=int), name="layer")
            ).transpose("layer", ...)
            self.set_staticmaps(da, "c")

        if self.crs is None:
            if crs is None:
                crs = 4326  # default to 4326
            self.set_crs(crs)

    def write_staticmaps_pcr(self):
        """Write staticmaps at <root/staticmaps> in PCRaster maps format."""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        ds_out = self.staticmaps
        if "LAI" in ds_out.data_vars:
            ds_out = ds_out.rename_vars({"LAI": "clim/LAI"})
        if "c" in ds_out.data_vars:
            for layer in ds_out["layer"]:
                ds_out[f"c_{layer.item():d}"] = ds_out["c"].sel(layer=layer)
                ds_out[f"c_{layer.item():d}"].raster.set_nodata(
                    ds_out["c"].raster.nodata
                )
            ds_out = ds_out.drop_vars(["c", "layer"])
        self.logger.info("Writing (updated) staticmap files.")
        # add datatypes for maps with same basenames, e.g. wflow_gauges_grdc
        pcr_vs_map = PCR_VS_MAP.copy()
        for var_name in ds_out.raster.vars:
            base_name = "_".join(var_name.split("_")[:-1])  # clip _<postfix>
            if base_name in PCR_VS_MAP:
                pcr_vs_map.update({var_name: PCR_VS_MAP[base_name]})
        ds_out.raster.to_mapstack(
            root=join(self.root, "staticmaps"),
            mask=True,
            driver="PCRaster",
            pcr_vs_map=pcr_vs_map,
            logger=self.logger,
        )

    def read_staticgeoms(self):
        """Read and staticgeoms at <root/staticgeoms> and parse to geopandas"""
        if not self._write:
            self._staticgeoms = dict()  # fresh start in read-only mode
        dir_default = join(self.root, "staticmaps.nc")
        dir_mod = dirname(
            self.get_config("input.path_static", abs_path=True, fallback=dir_default)
        )
        fns = glob.glob(join(dir_mod, "staticgeoms", "*.geojson"))
        if len(fns) > 1:
            self.logger.info("Reading model staticgeom files.")
        for fn in fns:
            name = basename(fn).split(".")[0]
            if name != "region":
                self.set_staticgeoms(gpd.read_file(fn), name=name)

    def write_staticgeoms(self):
        """Write staticmaps at <root/staticgeoms> in model ready format"""
        # to write use self.staticgeoms[var].to_file()
        if not self._write:
            raise IOError("Model opened in read-only mode")
        if self.staticgeoms:
            self.logger.info("Writing model staticgeom to file.")
            for name, gdf in self.staticgeoms.items():
                fn_out = join(self.root, "staticgeoms", f"{name}.geojson")
                gdf.to_file(fn_out, driver="GeoJSON")

    def read_forcing(self):
        """Read forcing"""
        fn_default = join(self.root, "inmaps.nc")
        fn = self.get_config("input.path_forcing", abs_path=True, fallback=fn_default)
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
        """write forcing at `fn_out` in model ready format.

        If no `fn_out` path is provided and path_forcing from the  wflow toml exists,
        the following default filenames are used:

            * Default name format (with downscaling): inmaps_sourcePd_sourceTd_methodPET_freq_startyear_endyear.nc
            * Default name format (no downscaling): inmaps_sourceP_sourceT_methodPET_freq_startyear_endyear.nc

        Parameters
        ----------
        fn_out: str, Path, optional
            Path to save output netcdf file; if None the name is read from the wflow
            toml file.
        freq_out: str (Offset), optional
            Write several files for the forcing according to fn_freq. For example 'Y' for one file per year or 'M'
            for one file per month. By default writes the one file.
            For more options, see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        chunksize: int, optional
            Chunksize on time dimension when saving to disk. By default 1.
        decimals: int, optional
            Round the ouput data to the given number of decimals.
        time_units: str, optional
            Common time units when writting several netcdf forcing files. By default "days since 1900-01-01T00:00:00".

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
                            "Netcdf default forcing file already exists, skipping write_forcing. "
                            "To overwrite netcdf forcing file: change name input.path_forcing "
                            "in setup_config section of the build inifile."
                        )
                        return
                    else:
                        self.set_config("input.path_forcing", fn_default)
                        self.write_config()  # re-write config
                        fn_out = fn_default_path

            # Check if all dates between (starttime, endtime) are in all da forcing
            start = pd.to_datetime(self.get_config("starttime"))
            end = pd.to_datetime(self.get_config("endtime"))
            missings = False
            for da in self.forcing.values():
                if "time" in da.coords:
                    if hasattr(da.indexes["time"], "to_datetimeindex"):
                        times = da.indexes["time"].to_datetimeindex().values
                    else:
                        times = da.time.values
                    if start < pd.to_datetime(times[0]):
                        start = pd.to_datetime(times[0])
                        missings = True
                    if end > pd.to_datetime(times[-1]):
                        end = pd.to_datetime(times[-1])
                        missings = True
            # merge, process and write forcing
            ds = xr.merge([da.reset_coords(drop=True) for da in self.forcing.values()])
            ds.raster.set_crs(self.crs)
            # Send warning, slice ds and update config with new start and end time
            if missings:
                self.logger.warning(
                    f"Not all dates found in precip_fn changing starttime to {start} and endtime to {end} in the toml."
                )
                self.set_config("starttime", start.to_pydatetime())
                self.set_config("endtime", end.to_pydatetime())
                self.write_config()
                ds = ds.sel({"time": slice(start, end)})

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
            # make sure no _FillValue is written to the time dimension
            # For several forcing files add common units attributes to time
            ds["time"].attrs.pop("_FillValue", None)
            encoding["time"] = {"_FillValue": None}

            # Check if all sub-folders in fn_out exists and if not create them
            if not isdir(dirname(fn_out)):
                os.makedirs(dirname(fn_out))

            forcing_list = []

            if freq_out is None:
                # with compute=False we get a delayed object which is executed when
                # calling .compute where we can pass more arguments to the dask.compute method
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
            # with Profiler() as prof, CacheProfiler(metric=cachey.nbytes) as cprof, ResourceProfiler() as rprof:
            #     delayed_obj.compute()
            # visualize([prof, cprof, rprof], file_path=r'c:\Users\eilan_dk\work\profile2.html')

    def read_states(self):
        """Read states at <root/?/> and parse to dict of xr.DataArray"""
        if not self._write:
            # start fresh in read-only mode
            self._states = dict()
        # raise NotImplementedError()

    def write_states(self):
        """write states at <root/?/> in model ready format"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        # raise NotImplementedError()

    def read_results(self):
        """Read results at <root/?/> and parse to dict of xr.DataArray/xr.Dataset"""
        if not self._write:
            # start fresh in read-only mode
            self._results = dict()

        # Read gridded netcdf (output section)
        nc_fn = self.get_config("output.path", abs_path=True)
        if nc_fn is not None and isfile(nc_fn):
            self.logger.info(f"Read results from {nc_fn}")
            ds = xr.open_dataset(nc_fn, chunks={"time": 30}, decode_coords="all")
            # TODO ? align coords names and values of results nc with staticmaps
            self.set_results(ds, name="output")

        # Read scalar netcdf (netcdf section)
        ncs_fn = self.get_config("netcdf.path", abs_path=True)
        if ncs_fn is not None and isfile(ncs_fn):
            self.logger.info(f"Read results from {ncs_fn}")
            ds = xr.open_dataset(ncs_fn, chunks={"time": 30})
            self.set_results(ds, name="netcdf")

        # Read csv timeseries (csv section)
        csv_fn = self.get_config("csv.path", abs_path=True)
        if csv_fn is not None and isfile(csv_fn):
            csv_dict = utils.read_csv_results(
                csv_fn, config=self.config, maps=self.staticmaps
            )
            for key in csv_dict:
                # Add to results
                self.set_results(csv_dict[f"{key}"])

    def write_results(self):
        """write results at <root/?/> in model ready format"""
        if not self._write:
            raise IOError("Model opened in read-only mode")
        # raise NotImplementedError()

    def read_intbl(self, **kwargs):
        """Read and intbl files at <root/intbl> and parse to xarray"""
        if not self._write:
            self._intbl = dict()  # start fresh in read-only mode
        if not self._read:
            self.logger.info("Reading default intbl files.")
            fns = glob.glob(join(DATADIR, "wflow", "intbl", f"*.tbl"))
        else:
            self.logger.info("Reading model intbl files.")
            fns = glob.glob(join(self.root, "intbl", f"*.tbl"))
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
        """Returns a dictionary of pandas.DataFrames representing the wflow intbl files."""
        if not self._intbl:
            self.read_intbl()
        return self._intbl

    @property
    def flwdir(self):
        """Returns the pyflwdir.FlwdirRaster object parsed from the wflow ldd."""
        if self._flwdir is None:
            self.set_flwdir()
        return self._flwdir

    def set_flwdir(self, ftype="infer"):
        """Parse pyflwdir.FlwdirRaster object parsed from the wflow ldd"""
        flwdir_name = flwdir_name = self._MAPS["flwdir"]
        self._flwdir = flw.flwdir_from_da(
            self.staticmaps[flwdir_name],
            ftype=ftype,
            check_ftype=True,
            mask=(self.staticmaps[self._MAPS["basins"]] > 0),
        )

    @property
    def basins(self):
        """Returns a basin(s) geometry as a geopandas.GeoDataFrame."""
        if "basins" in self.staticgeoms:
            gdf = self.staticgeoms["basins"]
        else:
            gdf = flw.basin_shape(
                self.staticmaps, self.flwdir, basin_name=self._MAPS["basins"]
            )
            self.set_staticgeoms(gdf, name="basins")
        return gdf

    @property
    def rivers(self):
        """Returns a river geometry as a geopandas.GeoDataFrame. If available, the
        stream order and upstream area values are added to the geometry properties.
        """
        if "rivers" in self.staticgeoms:
            gdf = self.staticgeoms["rivers"]
        elif self._MAPS["rivmsk"] in self.staticmaps:
            rivmsk = self.staticmaps[self._MAPS["rivmsk"]].values != 0
            # Check if there are river cells in the model before continuing
            if np.any(rivmsk):
                # add stream order 'strord' column
                strord = self.flwdir.stream_order(mask=rivmsk)
                feats = self.flwdir.streams(mask=rivmsk, strord=strord)
                gdf = gpd.GeoDataFrame.from_features(feats)
                gdf.crs = pyproj.CRS.from_user_input(self.crs)
                self.set_staticgeoms(gdf, name="rivers")
        else:
            self.logger.warning("No river cells detected in the selected basin.")
            gdf = None
        return gdf

    def clip_staticmaps(
        self,
        region,
        buffer=0,
        align=None,
        crs=4326,
    ):
        """Clip staticmaps to subbasin.

        Parameters
        ----------
        region : dict
            See :meth:`models.wflow.WflowModel.setup_basemaps`
        buffer : int, optional
            Buffer around subbasin in number of pixels, by default 0
        align : float, optional
            Align bounds of region to raster with resolution <align>, by default None
        crs: int, optional
            Default crs of the staticmaps to clip.

        Returns
        -------
        xarray.DataSet
            Clipped staticmaps.
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
                ds=self.staticmaps,
                logger=self.logger,
                kind=kind,
                basins_name=basins_name,
                flwdir_name=flwdir_name,
                **region,
            )
        # clip based on subbasin args, geom or bbox
        if geom is not None:
            ds_staticmaps = self.staticmaps.raster.clip_geom(
                geom, align=align, buffer=buffer
            )
            ds_staticmaps[basins_name] = ds_staticmaps[basins_name].where(
                ds_staticmaps["mask"], self.staticmaps[basins_name].raster.nodata
            )
            ds_staticmaps[basins_name].attrs.update(
                _FillValue=self.staticmaps[basins_name].raster.nodata
            )
        elif bbox is not None:
            ds_staticmaps = self.staticmaps.raster.clip_bbox(
                bbox, align=align, buffer=buffer
            )

        # Update flwdir staticmaps and staticgeoms
        if self.crs is None and crs is not None:
            self.set_crs(crs)

        self._staticmaps = xr.Dataset()
        self.set_staticmaps(ds_staticmaps)

        # add pits at edges after clipping
        self._flwdir = None  # make sure old flwdir object is removed
        self.staticmaps[self._MAPS["flwdir"]].data = self.flwdir.to_array("ldd")

        # Reinitiliase staticgeoms and re-create basins/rivers
        self._staticgeoms = dict()
        # self.basins
        # self.rivers
        # now staticgeoms links to geoms which does not exist in every hydromt version
        # remove when updating wflow to new objects
        basins = flw.basin_shape(
            self.staticmaps, self.flwdir, basin_name=self._MAPS["basins"]
        )
        self.set_staticgeoms(basins, name="basins")
        rivmsk = self.staticmaps[self._MAPS["rivmsk"]].values != 0
        # Check if there are river cells in the model before continuing
        if np.any(rivmsk):
            # add stream order 'strord' column
            strord = self.flwdir.stream_order(mask=rivmsk)
            feats = self.flwdir.streams(mask=rivmsk, strord=strord)
            gdf = gpd.GeoDataFrame.from_features(feats)
            gdf.crs = pyproj.CRS.from_user_input(self.crs)
            self.set_staticgeoms(gdf, name="rivers")

        # Update reservoir and lakes
        remove_reservoir = False
        if self._MAPS["resareas"] in self.staticmaps:
            reservoir = self.staticmaps[self._MAPS["resareas"]]
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
                self._staticmaps = self.staticmaps.drop_vars(remove_maps)

        remove_lake = False
        if self._MAPS["lakeareas"] in self.staticmaps:
            lake = self.staticmaps[self._MAPS["lakeareas"]]
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
                self._staticmaps = self.staticmaps.drop_vars(remove_maps)

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
                self.staticmaps.raster.bounds
            )
            self.set_forcing(ds_forcing)
