"""Implement the Wflow Sediment model class."""

import logging
from os.path import isfile

import numpy as np
import xarray as xr

from hydromt_wflow.wflow import WflowModel

from . import DATADIR
from .workflows import landuse, soilgrids_sediment

__all__ = ["WflowSedimentModel"]

logger = logging.getLogger(__name__)


class WflowSedimentModel(WflowModel):
    """The wflow sediment model class, a subclass of WflowModel."""

    _NAME = "wflow_sediment"
    _CONF = "wflow_sediment.toml"
    _DATADIR = DATADIR
    _GEOMS = {}
    _MAPS = WflowModel._MAPS.copy()
    _MAPS.update(
        {
            "soil": "wflow_soil",
        }
    )
    _FOLDERS = WflowModel._FOLDERS

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

    def setup_rivers(self, *args, **kwargs):
        """Copy the functionality of WflowModel.

        It however removes the river_routing key from the config.

        See Also
        --------
        WflowModel.setup_rivers
        """
        super().setup_rivers(*args, **kwargs)

        self.config["model"].pop("river_routing", None)

    def setup_lakes(self, lakes_fn="hydro_lakes", min_area=1.0):
        """Generate maps of lake areas and outlets.

        Also generates well as parameters with average lake area,
        depth a discharge values.

        The data is generated from features with ``min_area`` [km2] from a database with
        lake geometry, IDs and metadata. Currently, "hydro_lakes" (hydroLakes) is the
        only supported ``lakes_fn`` data source and
        we use a default minimum area of 1 km2.

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

            * Required variables: \
['waterbody_id', 'Area_avg', 'Vol_avg', 'Depth_avg', 'Dis_avg']
        min_area : float, optional
            Minimum lake area threshold [km2], by default 1.0 km2.
        """
        super().setup_lakes(lakes_fn=lakes_fn, min_area=min_area)
        # Update the toml to match wflow_sediment and not wflow_sbm
        lakes_toml_add = {
            "model.dolake": True,
            "model.lakes": False,
        }
        if "wflow_lakeareas" in self.grid:
            for option in lakes_toml_add:
                self.set_config(option, lakes_toml_add[option])
            if self.get_config("state.lateral.river.lake") is not None:
                del self.config["state"]["lateral"]["river"]["lake"]
            if self.get_config("input.lateral.river.lake") is not None:
                del self.config["input"]["lateral"]["river"]["lake"]

    def setup_reservoirs(
        self,
        reservoirs_fn: str,
        timeseries_fn: str = None,
        min_area: float = 1.0,
        **kwargs,
    ):
        """Generate maps of lake areas and outlets.

        Also generates well as parameters with average reservoir area,
        demand, min and max target storage capacities and discharge capacity values.

        The data is generated from features with ``min_area`` [km2] (default is 1 km2)
        from a database with reservoir geometry, IDs and metadata.

        Data requirements for direct use \
(ie wflow parameters are data already present in reservoirs_fn)
        are reservoir ID 'waterbody_id', area 'ResSimpleArea' [m2],
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
        'Dis_avg' [m3/s] and dam height 'Dam_height' [m] and
        minimum / normal / maximum storage capacity of the dam 'Capacity_min',
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
['waterbody_id', 'Area_avg', 'Vol_avg', 'Depth_avg', 'Dis_avg', 'Capacity_max', \
'Capacity_norm', 'Capacity_min', 'Dam_height']
        timeseries_fn : str {'gww', 'hydroengine', 'none'}, optional
            Download and use time series of reservoir surface water area to calculate
            and overwrite the reservoir volume/areas of the data source.
            Timeseries are either downloaded from Global Water Watch 'gww'
            (using gwwapi package) or JRC 'jrc' (using hydroengine package).
            By default None.
        min_area : float, optional
            Minimum reservoir area threshold [km2], by default 1.0 km2.
        """
        super().setup_reservoirs(
            reservoirs_fn=reservoirs_fn,
            timeseries_fn=timeseries_fn,
            min_area=min_area,
            **kwargs,
        )
        # Update the toml to match wflow_sediment and not wflow_sbm
        res_toml_add = {
            "model.doreservoir": True,
            "model.reservoirs": False,
        }
        if "wflow_reservoirareas" in self.grid:
            for option in res_toml_add:
                self.set_config(option, res_toml_add[option])
            if self.get_config("state.lateral.river.reservoir") is not None:
                del self.config["state"]["lateral"]["river"]["reservoir"]
            if self.get_config("input.lateral.river.reservoir") is not None:
                del self.config["input"]["lateral"]["river"]["reservoir"]

    def setup_outlets(
        self,
        river_only=True,
        toml_output="csv",
        gauge_toml_header=["TSS"],
        gauge_toml_param=["lateral.river.SSconc"],
    ):
        """Set the default gauge map based on basin outlets.

        Adds model layers:

        * **wflow_gauges** map: gauge IDs map from catchment outlets [-]
        * **gauges** geom: polygon of catchment outlets

        Parameters
        ----------
        river_only : bool, optional
            Only derive outlet locations if they are located on a river instead of
            locations for all catchments.
            By default True.
        toml_output : str, optional
            One of ['csv', 'netcdf', None] to update [csv] or [netcdf] section of
            wflow toml file or do nothing.
            By default, 'csv'.
        gauge_toml_header : list, optional
            Save specific model parameters in csv section.
            This option defines the header of the csv file.
            By default saves TSS (for lateral.river.SSconc).
        gauge_toml_param: list, optional
            Save specific model parameters in csv section.
            This option defines the wflow variable corresponding to the
            names in gauge_toml_header.
            By default saves lateral.river.SSconc (for TSS).
        """
        super().setup_outlets(
            river_only=river_only,
            toml_output=toml_output,
            gauge_toml_header=gauge_toml_header,
            gauge_toml_param=gauge_toml_param,
        )

    def setup_gauges(
        self,
        gauges_fn=None,
        source_gdf=None,
        snap_to_river=True,
        mask=None,
        derive_subcatch=False,
        basename=None,
        toml_output="csv",
        gauge_toml_header=["Q", "TSS"],
        gauge_toml_param=["lateral.river.q_riv", "lateral.river.SSconc"],
        **kwargs,
    ):
        """Set a gauge map based on ``gauges_fn`` data.

        Supported gauge datasets include "grdc"
        or "<path_to_source>" for user supplied csv or geometry files \
with gauge locations.
        If a csv file is provided, a "x" or "lon" and "y" or "lat" column is required
        and the first column will be used as IDs in the map. If ``snap_to_river`` is set
        to True, the gauge location will be snapped to the boolean river mask. If
        ``derive_subcatch`` is set to True, an additonal subcatch map is derived from
        the gauge locations.

        Adds model layers:

        * **wflow_gauges_source** map: gauge IDs map from source [-] (if gauges_fn)
        * **wflow_subcatch_source** map: subcatchment based on gauge locations [-] \
(if derive_subcatch)
        * **gauges_source** geom: polygon of gauges from source
        * **subcatch_source** geom: polygon of subcatchment based on \
gauge locations [-] (if derive_subcatch)

        Parameters
        ----------
        gauges_fn : str, {"grdc"}, optional
            Known source name or path to gauges file geometry file, by default None.
        source_gdf : geopandas.GeoDataFame, optional
            Direct gauges file geometry, by default None.
        snap_to_river : bool, optional
            Snap point locations to the closest downstream river cell, by default True
        mask : np.boolean, optional
            If provided snaps to the mask, else snaps to the river (default).
        derive_subcatch : bool, optional
            Derive subcatch map for gauges, by default False
        derive_outlet : bool, optional
            Derive gaugemap based on catchment outlets, by default True
        basename : str, optional
            Map name in grid (wflow_gauges_basename).
            If None use the gauges_fn basename.
        toml_output : str, optional
            One of ['csv', 'netcdf', None] to update [csv] or [netcdf] section of \
wflow toml file or do nothing.
            By default, 'csv'.
        gauge_toml_header : list, optional
            Save specific model parameters in csv section.
            This option defines the header of the csv file.
            By default saves Q (for lateral.river.q_riv) and
            TSS (for lateral.river.SSconc).
        gauge_toml_param: list, optional
            Save specific model parameters in csv section.
            This option defines the wflow variable corresponding to the
            names in gauge_toml_header.
            By default saves lateral.river.q_riv (for Q) and
            lateral.river.SSconc (for TSS).
        """
        # # Add new outputcsv section in the config
        super().setup_gauges(
            gauges_fn=gauges_fn,
            source_gdf=source_gdf,
            snap_to_river=snap_to_river,
            mask=mask,
            derive_subcatch=derive_subcatch,
            basename=basename,
            toml_output=toml_output,
            gauge_toml_header=gauge_toml_header,
            gauge_toml_param=gauge_toml_param,
        )

    def setup_lulcmaps(
        self,
        lulc_fn="globcover",
        lulc_mapping_fn=None,
        lulc_vars=[
            "landuse",
            "Cov_River",
            "Kext",
            "N",
            "PathFrac",
            "Sl",
            "Swood",
            "USLE_C",
            "WaterFrac",
        ],
    ):
        """Derive several wflow maps based on landuse-landcover (LULC) data.

        Currently, ``lulc_fn`` can be set to the "vito", "globcover"
        or "corine", fo which lookup tables are constructed to convert lulc classses to
        model parameters based on literature. The data is remapped at its original
        resolution and then resampled to the model resolution using the average
        value, unless noted differently.

        Adds model layers:

        * **landuse** map: Landuse class [-]
            Original source dependent LULC class, resampled using nearest neighbour.
        * **Cov_river** map: vegetation coefficent reducing stream bank erosion [-].
        * **Kext** map: Extinction coefficient in the canopy gap fraction equation [-]
        * **Sl** map: Specific leaf storage [mm]
        * **Swood** map: Fraction of wood in the vegetation/plant [-]
        * **USLE_C** map: Cover managment factor from the USLE equation [-]
        * **PathFrac** map: The fraction of compacted or urban area per grid cell [-]
        * **WaterFrac** map: The fraction of open water per grid cell [-]
        * **N** map: Manning Roughness [-]

        Parameters
        ----------
        lulc_fn : {"globcover", "vito", "corine"}
            Name of data source in data_sources.yml file.
        lulc_mapping_fn : str
            Path to a mapping csv file from landuse in source name to parameter values \
in lulc_vars.
        lulc_vars : list
            List of landuse parameters to keep.
            By default: \
["landuse","Cov_river","Kext","N","PathFrac","USLE_C","Sl","Swood","WaterFrac"]
        """
        super().setup_lulcmaps(
            lulc_fn=lulc_fn, lulc_mapping_fn=lulc_mapping_fn, lulc_vars=lulc_vars
        )

    def setup_riverbedsed(self, bedsed_mapping_fn=None, **kwargs):
        """Generate sediments based river bed characteristics maps.

        Adds model layers:

        * **D50_River** map: median sediment diameter of the river bed [mm]
        * **ClayF_River** map: fraction of clay material in the river bed [-]
        * **SiltF_River** map: fraction of silt material in the river bed [-]
        * **SandF_River** map: fraction of sand material in the river bed [-]
        * **GravelF_River** map: fraction of gravel material in the river bed [-]

        Parameters
        ----------
        bedsed_mapping_fn : str
            Path to a mapping csv file from streamorder to river bed \
particles characteristics.

            * Required variable: \
['strord','D50_River', 'ClayF_River', 'SiltF_River', 'SandF_River', 'GravelF_River']

        """
        self.logger.info("Preparing riverbedsed parameter maps.")
        # Make D50_River map from csv file with mapping between streamorder and
        # D50_River value
        if bedsed_mapping_fn is None:
            fn_map = "riverbedsed_mapping_default"
        else:
            fn_map = bedsed_mapping_fn

        if not isfile(fn_map) and fn_map not in self.data_catalog:
            raise ValueError(f"Riverbed sediment mapping file not found: {fn_map}")
        df = self.data_catalog.get_dataframe(fn_map, driver_kwargs=kwargs)

        strord = self.grid[self._MAPS["strord"]].copy()
        # max streamorder value above which values get the same N_River value
        max_str = df.index[-2]
        # if streamroder value larger than max_str, assign last value
        strord = strord.where(strord <= max_str, max_str)
        # handle missing value (last row of csv is mapping of nan values)
        strord = strord.where(strord != strord.raster.nodata, -999)
        strord.raster.set_nodata(-999)

        ds_riversed = landuse(
            da=strord,
            ds_like=self.grid,
            df=df,
            logger=self.logger,
        )

        self.set_grid(ds_riversed)

    def setup_canopymaps(
        self,
        canopy_fn="simard",
    ):
        """Generate sediments based canopy height maps.

        Adds model layers:

        * **CanopyHeight** map: height of the vegetation canopy [m]

        Parameters
        ----------
        canopy_fn : {"simard"}
            Name of canopy height data source in data_sources.yml file.
        """
        self.logger.info("Preparing canopy height map.")

        # Canopy height
        if canopy_fn not in ["simard"]:
            self.logger.warning(
                f"Invalid source '{canopy_fn}', skipping setup canopy map for sediment."
            )
            return

        dsin = self.data_catalog.get_rasterdataset(
            canopy_fn, geom=self.region, buffer=2
        )
        dsout = xr.Dataset(coords=self.grid.raster.coords)
        ds_out = dsin.raster.reproject_like(self.grid, method="average")
        dsout["CanopyHeight"] = ds_out.astype(np.float32)
        dsout["CanopyHeight"] = dsout["CanopyHeight"].fillna(-9999.0)
        dsout["CanopyHeight"].raster.set_nodata(-9999.0)
        self.set_grid(dsout)

    def setup_soilmaps(
        self,
        soil_fn="soilgrids",
        usleK_method="renard",
    ):
        """Generate sediments based soil parameter maps.

        Adds model layers:

        * **PercentClay** map: clay content of the topsoil [%]
        * **PercentSilt** map: silt content of the topsoil [%]
        * **PercentOC** map: organic carbon in the topsoil [%]
        * **ErosK** map: mean detachability of the soil (Morgan et al., 1998) [g/J]
        * **USLE_K** map: soil erodibility factor from the USLE equation [-]


        Parameters
        ----------
        soil_fn : {"soilgrids"}
            Name of soil data source in data_sources.yml file.

            * Required variables: ['clyppt_sl1', 'sltppt_sl1', 'oc_sl1']
        usleK_method: {"renard", "epic"}
            Method to compute the USLE K factor, by default renard.
        """
        self.logger.info("Preparing soil parameter maps.")

        # Soil related maps
        if soil_fn not in ["soilgrids"]:
            self.logger.warning(
                f"Invalid source '{soil_fn}', skipping setup_soilmaps for sediment."
            )
            return

        dsin = self.data_catalog.get_rasterdataset(soil_fn, geom=self.region, buffer=2)
        dsout = soilgrids_sediment(
            dsin,
            self.grid,
            usleK_method,
            logger=self.logger,
        )
        self.set_grid(dsout)
