"""Implement the Wflow Sediment model class."""

import logging
from pathlib import Path
from typing import List, Optional, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hydromt_wflow.utils import (
    DATADIR,
    convert_to_wflow_v1_sediment,
)
from hydromt_wflow.wflow import WflowModel

from .naming import _create_hydromt_wflow_mapping_sediment
from .workflows import add_planted_forest_to_landuse, landuse, soilgrids_sediment

__all__ = ["WflowSedimentModel"]

logger = logging.getLogger(__name__)


class WflowSedimentModel(WflowModel):
    """The wflow sediment model class, a subclass of WflowModel."""

    _NAME = "wflow_sediment"
    _CONF = "wflow_sediment.toml"
    _DATADIR = DATADIR
    _GEOMS = {}
    _FOLDERS = WflowModel._FOLDERS

    def __init__(
        self,
        root: Optional[str] = None,
        mode: Optional[str] = "w",
        config_fn: Optional[str] = None,
        data_libs: List | str = [],
        wflow_version: str = "1.0.0",
        logger=logger,
    ):
        super().__init__(
            root=root,
            mode=mode,
            config_fn=config_fn,
            data_libs=data_libs,
            logger=logger,
        )
        # Update compared to wflow sbm
        self._MAPS, self._WFLOW_NAMES = _create_hydromt_wflow_mapping_sediment(
            self.config
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

    def setup_lakes(
        self,
        lakes_fn: str | Path | gpd.GeoDataFrame = "hydro_lakes",
        min_area: float = 1.0,
        output_names: Dict = {
            "lake_area__count": "wflow_lakeareas",
            "lake_location__count": "wflow_lakelocs",
            "lake_surface__area": "LakeArea",
        },
        geom_name: str = "lakes",
        **kwargs,
    ):
        """Generate maps of lake areas and outlets.

        Also generates average lake area.

        The data is generated from features with ``min_area`` [km2] from a database with
        lake geometry, IDs and metadata. Data required are lake ID 'waterbody_id',
        average area 'Area_avg' [m2].

        Adds model layers:

        * **wflow_lakeareas** map: lake IDs [-]
        * **wflow_lakelocs** map: lake IDs at outlet locations [-]
        * **LakeArea** map: lake area [m2]
        * **lakes** geom: polygon with lakes and wflow lake parameters

        Parameters
        ----------
        lakes_fn :
            Name of GeoDataFrame source for lake parameters.

            * Required variables: ['waterbody_id', 'Area_avg']
        min_area : float, optional
            Minimum lake area threshold [km2], by default 1.0 km2.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        geom_name : str, optional
            Name of the lakes geometry in the staticgeoms folder, by default 'lakes'
            for lakes.geojson.
        kwargs: optional
            Keyword arguments passed to the method
            hydromt.DataCatalog.get_rasterdataset()
        """
        # Derive lake are and outlet maps
        gdf_lakes, ds_lakes = self._setup_waterbodies(
            lakes_fn, "lake", min_area, **kwargs
        )
        if ds_lakes is None:
            self.logger.info("Skipping method, as no data has been found")
            return
        self._update_naming(output_names)

        # add lake area
        gdf_points = gpd.GeoDataFrame(
            gdf_lakes[["waterbody_id", "Area_avg"]],
            geometry=gpd.points_from_xy(gdf_lakes.xout, gdf_lakes.yout),
        )
        ds_lakes["LakeArea"] = self.grid.raster.rasterize(
            gdf_points, col_name="Area_avg", dtype="float32", nodata=-999
        )

        # add to grid
        rmdict = {k: self._MAPS.get(k, k) for k in ds_lakes.data_vars}
        self.set_grid(ds_lakes.rename(rmdict))
        # write lakes with attr tables to static geoms.
        self.set_geoms(gdf_lakes.rename({"Area_avg": "LakeArea"}), name=geom_name)

        # Lake settings in the toml to update
        self.set_config("model.lakes", True)
        for dvar in ds_lakes.data_vars:
            if dvar == "lakeareas" or dvar == "lakelocs":
                self._update_config_variable_name(self._MAPS[dvar], data_type=None)
            elif dvar in self._WFLOW_NAMES:
                self._update_config_variable_name(self._MAPS[dvar])

    def setup_reservoirs(
        self,
        reservoirs_fn: str | Path | gpd.GeoDataFrame,
        min_area: float = 1.0,
        trapping_default: float = 1.0,
        output_names: Dict = {
            "reservoir_area__count": "wflow_reservoirareas",
            "reservoir_location__count": "wflow_reservoirlocs",
            "reservoir_surface__area": "ResSimpleArea",
            "reservoir_sediment~bedload__trapping_efficiency_coefficient": "ResTrapEff",
        },
        geom_name: str = "reservoirs",
        **kwargs,
    ):
        """Generate maps of reservoir areas and outlets.

        Also generates well as parameters with average reservoir area,
        and trapping efficiency for large particles.

        The data is generated from features with ``min_area`` [km2] (default is 1 km2)
        from a database with reservoir geometry, IDs and metadata.

        Adds model layers:

        * **wflow_reservoirareas** map: reservoir IDs [-]
        * **wflow_reservoirlocs** map: reservoir IDs at outlet locations [-]
        * **ResSimpleArea** map: reservoir area [m2]
        * **ResTrapEff** map: reservoir trapping efficiency coefficient [-]

        Parameters
        ----------
        reservoirs_fn : str
            Name of data source for reservoir parameters, see data/data_sources.yml.

            * Required variables: ['waterbody_id', 'Area_avg']

            * Optional variables: ['ResTrapEff']
        min_area : float, optional
            Minimum reservoir area threshold [km2], by default 1.0 km2.
        trapping_default : float, optional
            Default trapping efficiency coefficient for large particles [between 0 and
            1], by default 1 to trap 100% of large particles (sand to gravel) for
            example for gravity dam. For the others the natural deposition in
            waterbodies from Camp is used.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        geom_name : str, optional
            Name of the reservoirs geometry in the staticgeoms folder, by default
            "reservoirs" for reservoirs.geojson.
        kwargs: optional
            Keyword arguments passed to the method
            hydromt.DataCatalog.get_rasterdataset()
        """
        # Derive lake are and outlet maps
        gdf_res, ds_res = self._setup_waterbodies(
            reservoirs_fn, "reservoir", min_area, **kwargs
        )
        if ds_res is None:
            self.logger.info("Skipping method, as no data has been found")
            return
        self._update_naming(output_names)

        # Add default trapping efficiency coefficient if not in data source
        if "ResTrapEff" not in gdf_res.columns:
            gdf_res["ResTrapEff"] = trapping_default
        # add reservoirs parameters to grid
        gdf_points = gpd.GeoDataFrame(
            gdf_res[["waterbody_id", "Area_avg", "ResTrapEff"]],
            geometry=gpd.points_from_xy(gdf_res.xout, gdf_res.yout),
        )
        ds_res["ResSimpleArea"] = self.grid.raster.rasterize(
            gdf_points, col_name="Area_avg", dtype="float32", nodata=-999
        )
        ds_res["ResTrapEff"] = self.grid.raster.rasterize(
            gdf_points, col_name="ResTrapEff", dtype="float32", nodata=-999
        )

        # add to grid
        rmdict = {k: self._MAPS.get(k, k) for k in ds_res.data_vars}
        self.set_grid(ds_res.rename(rmdict))
        # write lakes with attr tables to static geoms.
        self.set_geoms(gdf_res.rename({"Area_avg": "ResSimpleArea"}), name=geom_name)

        # Lake settings in the toml to update
        self.set_config("model.reservoirs", True)
        for dvar in ds_res.data_vars:
            if dvar == "resareas" or dvar == "reslocs":
                self._update_config_variable_name(self._MAPS[dvar], data_type=None)
            elif dvar in self._WFLOW_NAMES:
                self._update_config_variable_name(self._MAPS[dvar])

    def setup_outlets(
        self,
        river_only: bool = True,
        toml_output: str = "csv",
        gauge_toml_header: List[str] = ["TSS"],
        gauge_toml_param: List[str] = [
            "river_water_sediment~suspended__mass_concentration",
        ],
    ):
        """Set the default gauge map based on basin outlets.

        If wflow_subcatch is available, the catchment outlets IDs will be matching the
        wflow_subcatch IDs. If not, then IDs from 1 to number of outlets are used.

        Can also add csv/netcdf_scalar output settings in the TOML.

        Adds model layers:

        * **wflow_gauges** map: gauge IDs map from catchment outlets [-]
        * **gauges** geom: polygon of catchment outlets

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
            By default saves TSS (for
            river_water_sediment~suspended__mass_concentration).
        gauge_toml_param: list, optional
            Save specific model parameters in csv section. This option defines
            the wflow variable corresponding to the names in gauge_toml_header.
            By default saves river_water_sediment~suspended__mass_concentration (for
            TSS).
        """
        super().setup_outlets(
            river_only=river_only,
            toml_output=toml_output,
            gauge_toml_header=gauge_toml_header,
            gauge_toml_param=gauge_toml_param,
        )

    def setup_gauges(
        self,
        gauges_fn: str | Path | gpd.GeoDataFrame,
        index_col: Optional[str] = None,
        snap_to_river: Optional[bool] = True,
        mask: Optional[np.ndarray] = None,
        snap_uparea: Optional[bool] = False,
        max_dist: Optional[float] = 10e3,
        wdw: Optional[int] = 3,
        rel_error: Optional[float] = 0.05,
        abs_error: float = 50.0,
        fillna: bool = False,
        derive_subcatch: Optional[bool] = False,
        basename: Optional[str] = None,
        toml_output: Optional[str] = "csv",
        gauge_toml_header: Optional[List[str]] = ["Q", "TSS"],
        gauge_toml_param: Optional[List[str]] = [
            "river_water__volume_flow_rate",
            "river_water_sediment~suspended__mass_concentration",
        ],
        **kwargs,
    ):
        """Set a gauge map based on ``gauges_fn`` data.

        This function directly calls the ``setup_gauges`` function of the WflowModel,
        see py:meth:`hydromt_wflow.wflow.WflowModel.setup_gauges` for more details.

        The only differences are the default values for the arguments:

        - ``gauge_toml_header`` defaults to ["Q", "TSS"]
        - ``gauge_toml_param`` defaults to ["river_water__volume_flow_rate",
            "river_water_sediment~suspended__mass_concentration"]

        See Also
        --------
        WflowModel.setup_gauges
        """
        # # Add new outputcsv section in the config
        super().setup_gauges(
            gauges_fn=gauges_fn,
            index_col=index_col,
            snap_to_river=snap_to_river,
            mask=mask,
            snap_uparea=snap_uparea,
            max_dist=max_dist,
            wdw=wdw,
            rel_error=rel_error,
            abs_error=abs_error,
            fillna=fillna,
            derive_subcatch=derive_subcatch,
            basename=basename,
            toml_output=toml_output,
            gauge_toml_header=gauge_toml_header,
            gauge_toml_param=gauge_toml_param,
            **kwargs,
        )

    def setup_lulcmaps(
        self,
        lulc_fn: str | Path | xr.DataArray,
        lulc_mapping_fn: str | Path | pd.DataFrame | None = None,
        planted_forest_fn: str | Path | gpd.GeoDataFrame | None = None,
        lulc_vars: Dict = {
            "landuse": None,
            "PathFrac": "soil~compacted__area_fraction",  # compacted_fraction
            "USLE_C": "soil_erosion__usle_c_factor",  # usle_c
        },
        planted_forest_c: float = 0.0881,
        orchard_name: str = "Orchard",
        orchard_c: float = 0.2188,
        output_names_suffix: Optional[str] = None,
    ):
        """Derive several wflow maps based on landuse-landcover (LULC) data.

        Lookup table `lulc_mapping_fn` columns are converted to lulc classes model
        parameters based on literature. The data is remapped at its original resolution
        and then resampled to the model resolution using the average value, unless noted
        differently.

        Currently, if `lulc_fn` is set to the "vito", "globcover", "esa_worldcover"
        "corine" or "glmnco", default lookup tables are available and will be used if
        `lulc_mapping_fn` is not provided.

        The USLE C factor map can be refined for planted forests using the planted
        forest data source. The planted forest data source is a polygon layer with
        planted forest polygons and optionally a column with the forest type to
        identify orchards. The default value for orchards is 0.2188, the default value
        for other planted forests is 0.0881.

        Adds model layers:

        * **landuse** map: Landuse class [-]
            Original source dependent LULC class, resampled using nearest neighbour.
        * **USLE_C** map: Cover management factor from the USLE equation [-]
        * **PathFrac** map: The fraction of compacted or urban area per grid cell [-]

        Parameters
        ----------
        lulc_fn : str, xarray.DataArray
            Name of RasterDataset source.
        lulc_mapping_fn : str
            Path to a mapping csv file from landuse in source name to parameter values
            in lulc_vars.
        planted_forest_fn : str, Path, gpd.GeoDataFrame
            GeoDataFrame source with polygons of planted forests.

            * Optional variable: ["forest_type"]

        lulc_vars : Dict
            Dictionnary of landuse parameters to prepare. The names are the
            the columns of the mapping file and the values are the corresponding
            Wflow.jl variables if any.
        planted_forest_c : float, optional
            Value of USLE C factor for planted forest, by default 0.0881.
        orchard_name : str, optional
            Name of orchard landuse class in the "forest_type" column of
            ``planted_forest_fn``, by default "Orchard".
        orchard_c : float, optional
            Value of USLE C factor for orchards, by default 0.2188.
        output_names_suffix : str, optional
            Suffix to be added to the output names to avoid having to rename all the
            columns of the mapping tables. For example if the suffix is "vito", all
            variables in lulc_vars will be renamed to "landuse_vito", "USLE_C_vito",
            etc.

        See Also
        --------
        workflows.setup_lulcmaps_from_vector
        workflows.add_planted_forest_to_landuse
        """
        # Prepare all default parameters
        super().setup_lulcmaps(
            lulc_fn=lulc_fn,
            lulc_mapping_fn=lulc_mapping_fn,
            lulc_vars=lulc_vars,
            output_names_suffix=output_names_suffix,
        )

        # If available, improve USLE C map with planted forest data
        if "USLE_C" in lulc_vars and planted_forest_fn is not None:
            # Read forest data
            planted_forest = self.data_catalog.get_geodataframe(
                planted_forest_fn,
                geom=self.basins,
                buffer=1,
                predicate="intersects",
                handle_nodata="IGNORE",
            )
            if planted_forest is None:
                self.logger.warning("No Planted forest data found within domain.")
                return
            usle_c = add_planted_forest_to_landuse(
                planted_forest,
                self.grid,  # TODO should have USLE_C in the grid already
                planted_forest_c=planted_forest_c,
                orchard_name=orchard_name,
                orchard_c=orchard_c,
                logger=self.logger,
            )

            # Add to grid
            self.set_grid(usle_c)

    def setup_lulcmaps_from_vector(
        self,
        lulc_fn: str | gpd.GeoDataFrame,
        lulc_mapping_fn: str | Path | pd.DataFrame | None = None,
        planted_forest_fn: str | Path | gpd.GeoDataFrame | None = None,
        lulc_vars: Dict = {
            "landuse": None,
            "PathFrac": "soil~compacted__area_fraction",  # compacted_fraction
            "USLE_C": "soil_erosion__usle_c_factor",  # usle_c
        },
        lulc_res: float | int | None = None,
        all_touched: bool = False,
        buffer: int = 1000,
        save_raster_lulc: bool = False,
        planted_forest_c: float = 0.0881,
        orchard_name: str = "Orchard",
        orchard_c: float = 0.2188,
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

        The USLE C factor map can be refined for planted forests using the planted
        forest data source. The planted forest data source is a polygon layer with
        planted forest polygons and optionally a column with the forest type to
        identify orchards. The default value for orchards is 0.2188, the default value
        for other planted forests is 0.0881.

        Adds model layers:

        * **landuse** map: Landuse class [-]
            Original source dependent LULC class, resampled using nearest neighbour.
        * **USLE_C** map: Cover management factor from the USLE equation [-]
        * **PathFrac** map: The fraction of compacted or urban area per grid cell [-]

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
        planted_forest_fn : str, Path, gpd.GeoDataFrame
            GeoDataFrame source with polygons of planted forests.

            * Optional variable: ["forest_type"]
        lulc_vars : dict
            List of landuse parameters to prepare.
            By default ["landuse","Kext","Sl","Swood","USLE_C","PathFrac"]
        lulc_vars : Dict
            Dictionnary of landuse parameters to prepare. The names are the
            the columns of the mapping file and the values are the corresponding
            Wflow.jl variables.
        all_touched : bool, optional
            If True, all pixels touched by the vector will be burned in the raster,
            by default False.
        buffer : int, optional
            Buffer around the bounding box of the vector data to ensure that all
            landuse classes are included in the rasterized map, by default 1000.
        save_raster_lulc : bool, optional
            If True, the high resolution rasterized landuse map will be saved to
            maps/landuse_raster.tif, by default False.
        planted_forest_c : float, optional
            Value of USLE C factor for planted forest, by default 0.0881.
        orchard_name : str, optional
            Name of orchard landuse class in the "forest_type" column of
            ``planted_forest_fn``, by default "Orchard".
        orchard_c : float, optional
            Value of USLE C factor for orchards, by default 0.2188.
        output_names_suffix : str, optional
            Suffix to be added to the output names to avoid having to rename all the
            columns of the mapping tables. For example if the suffix is "vito", all
            variables in lulc_vars will be renamed to "landuse_vito", "USLE_C_vito",
            etc.

        See Also
        --------
        workflows.setup_lulcmaps_from_vector
        workflows.add_planted_forest_to_landuse
        """
        # Prepare all default parameters
        super().setup_lulcmaps_from_vector(
            lulc_fn=lulc_fn,
            lulc_mapping_fn=lulc_mapping_fn,
            lulc_vars=lulc_vars,
            lulc_res=lulc_res,
            all_touched=all_touched,
            buffer=buffer,
            save_raster_lulc=save_raster_lulc,
            output_names_suffix=output_names_suffix,
        )

        # If available, improve USLE C map with planted forest data
        if "USLE_C" in lulc_vars and planted_forest_fn is not None:
            # Read forest data
            planted_forest = self.data_catalog.get_geodataframe(
                planted_forest_fn,
                geom=self.basins,
                buffer=1,
                predicate="intersects",
                handle_nodata="IGNORE",
            )
            if planted_forest is None:
                self.logger.warning("No Planted forest data found within domain.")
                return
            usle_c = add_planted_forest_to_landuse(
                planted_forest,
                self.grid,
                planted_forest_c=planted_forest_c,
                orchard_name=orchard_name,
                orchard_c=orchard_c,
                logger=self.logger,
            )

            # Add to grid
            self.set_grid(usle_c)

    def setup_riverbedsed(
        self,
        bedsed_mapping_fn: str | Path | pd.DataFrame | None = None,
        output_names: Dict = {
            "river_bottom-and-bank_sediment__d50_diameter": "D50_River",
            "river_bottom-and-bank_clay__mass_fraction": "ClayF_River",
            "river_bottom-and-bank_silt__mass_fraction": "SiltF_River",
            "river_bottom-and-bank_sand__mass_fraction": "SandF_River",
            "river_bottom-and-bank_gravel__mass_fraction": "GravelF_River",
            "river_water_sediment__kodatie_transport_capacity_a-coefficient": "a_kodatie",  # noqa: E501
            "river_water_sediment__kodatie_transport_capacity_b-coefficient": "b_kodatie",  # noqa: E501
            "river_water_sediment__kodatie_transport_capacity_c-coefficient": "c_kodatie",  # noqa: E501
            "river_water_sediment__kodatie_transport_capacity_d-coefficient": "d_kodatie",  # noqa: E501
        },
    ):
        """Generate sediments based river bed characteristics maps.

        Kodatie transport capacity coefficients can also be derived from such mapping
        table based on the mean sediment diameter of the river bed.

        Adds model layers:

        * **D50_River** map: median sediment diameter of the river bed [mm]
        * **ClayF_River** map: fraction of clay material in the river bed [-]
        * **SiltF_River** map: fraction of silt material in the river bed [-]
        * **SandF_River** map: fraction of sand material in the river bed [-]
        * **GravelF_River** map: fraction of gravel material in the river bed [-]
        * **a_kodatie** map: Kodatie transport capacity coefficient a [-]
        * **b_kodatie** map: Kodatie transport capacity coefficient b [-]
        * **c_kodatie** map: Kodatie transport capacity coefficient c [-]
        * **d_kodatie** map: Kodatie transport capacity coefficient d [-]

        Parameters
        ----------
        bedsed_mapping_fn : str
            Path to a mapping csv file from streamorder to river bed particles
            characteristics. If None reverts to default values.

            * Required variable: ['strord','D50_River', 'ClayF_River', 'SiltF_River',
              'SandF_River', 'GravelF_River']
            * Optional variable: ['a_kodatie', 'b_kodatie', 'c_kodatie', 'd_kodatie']
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        """
        self.logger.info("Preparing riverbedsed parameter maps.")
        if self._MAPS["strord"] not in self.grid.data_vars:
            raise ValueError(
                "Streamorder map is not available, please run setup_rivers first."
            )
        # update self._MAPS and self._WFLOW_NAMES with user defined output names
        self._update_naming(output_names)

        # Make D50_River map from csv file with mapping between streamorder and
        # D50_River value
        if bedsed_mapping_fn is None:
            fn_map = "riverbedsed_mapping_default"
        else:
            fn_map = bedsed_mapping_fn

        df = self.data_catalog.get_dataframe(fn_map)

        strord = self.grid[self._MAPS["strord"]].copy()
        # max streamorder value above which values get the same D50 value
        max_str = df.index[-2]
        nodata = df.index[-1]
        # if streamroder value larger than max_str, assign last value
        strord = strord.where(strord <= max_str, max_str)
        # handle missing value (last row of csv is mapping of nan values)
        strord = strord.where(strord != strord.raster.nodata, nodata)
        strord.raster.set_nodata(nodata)

        ds_riversed = landuse(
            da=strord,
            ds_like=self.grid,
            df=df,
            logger=self.logger,
        )

        rmdict = {k: self._MAPS.get(k, k) for k in ds_riversed.data_vars}
        self.set_grid(ds_riversed.rename(rmdict))
        # update config
        self._update_config_variable_name(ds_riversed.rename(rmdict).data_vars)

    def setup_canopymaps(
        self,
        canopy_fn: str | Path | xr.DataArray,
        output_name: str = "CanopyHeight",
    ):
        """Generate sediments based canopy height maps.

        Adds model layers:

        * **CanopyHeight** map: height of the vegetation canopy [m]

        Parameters
        ----------
        canopy_fn :
            Canopy height data source (DataArray).
        output_name : dict, optional
            Name of the output map. By default 'CanopyHeight'.
        """
        self.logger.info("Preparing canopy height map.")

        # Canopy height
        dsin = self.data_catalog.get_rasterdataset(
            canopy_fn, geom=self.region, buffer=2
        )
        dsout = xr.Dataset(coords=self.grid.raster.coords)
        ds_out = dsin.raster.reproject_like(self.grid, method="average")
        dsout["CanopyHeight"] = ds_out.astype(np.float32)
        dsout["CanopyHeight"] = dsout["CanopyHeight"].fillna(-9999.0)
        dsout["CanopyHeight"].raster.set_nodata(-9999.0)

        # update name
        wflow_var = self._WFLOW_NAMES[self._MAPS["CanopyHeight"]]
        self._update_naming({wflow_var: output_name})
        self.set_grid(dsout["CanopyHeigt"], name=output_name)
        # update config
        self._update_config_variable_name(output_name)

    def setup_soilmaps(
        self,
        soil_fn: str = "soilgrids",
        usleK_method: str = "renard",
        add_aggregates: bool = True,
        output_names: Dict = {
            "soil_clay__mass_fraction": "fclay_soil",
            "soil_silt__mass_fraction": "fsilt_soil",
            "soil_sand__mass_fraction": "fsand_soil",
            "soil_aggregates~small__mass_fraction": "fsagg_soil",
            "soil_aggregates~large__mass_fraction": "flagg_soil",
            "soil_erosion__rainfall_soil_detachability_factor": "soil_detachability",
            "soil_erosion__usle_k_factor": "usle_k",
            "land_surface_sediment__d50_diameter": "d50_soil",
            "land_surface_water_sediment__govers_transport_capacity_coefficient": "c_govers",  # noqa: E501
            "land_surface_water_sediment__govers_transport_capacity_exponent": "n_govers",  # noqa: E501
        },
    ):
        """Generate sediments based soil parameter maps.

        Sediment size distribution and addition of small and large aggregates can be
        estimated from primary particle size distribution with Foster et al. (1980).
        USLE K factor can be computed from the soil data using Renard or EPIC methods.
        Calculation of D50 and fraction of fine and very fine sand (fvfs) from
        Fooladmand et al, 2006.

        Adds model layers:

        * **fclay_soil**: clay content of the topsoil [g/g]
        * **fsilt_soil**: silt content of the topsoil [g/g]
        * **fsand_soil**: sand content of the topsoil [g/g]
        * **fsagg_soil**: small aggregate content of the topsoil [g/g]
        * **flagg_soil**: large aggregate content of the topsoil [g/g]
        * **soil_detachability** map: mean detachability of the soil (Morgan et al.,
          1998) [g/J]
        * **usle_k** map: soil erodibility factor from the USLE equation [-]
        * **d50_soil** map: median sediment diameter of the soil [mm]
        * **c_govers** map: Govers factor for overland flow transport capacity [-]
        * **n_govers** map: Govers exponent for overland flow transport capacity [-]


        Parameters
        ----------
        soil_fn : {"soilgrids"}
            Name of soil data source in data_sources.yml file.

            * Required variables: ['clyppt_sl1', 'sltppt_sl1', 'oc_sl1']
        usleK_method: {"renard", "epic"}
            Method to compute the USLE K factor, by default renard.
        add_aggregates: bool, optional
            Add small and large aggregates based on soil texture, by default True.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        """
        self.logger.info("Preparing soil parameter maps.")

        # Soil related maps
        if soil_fn not in ["soilgrids"]:
            self.logger.warning(
                f"Invalid source '{soil_fn}', skipping setup_soilmaps for sediment."
            )
            return

        self._update_naming(output_names)
        dsin = self.data_catalog.get_rasterdataset(soil_fn, geom=self.region, buffer=2)
        dsout = soilgrids_sediment(
            dsin,
            self.grid,
            usleK_method=usleK_method,
            add_aggregates=add_aggregates,
            logger=self.logger,
        )
        rmdict = {k: self._MAPS.get(k, k) for k in dsout.data_vars}
        self.set_grid(dsout)
        self._update_config_variable_name(dsout.rename(rmdict).data_vars)

    def upgrade_to_v1_wflow(
        self,
        soil_fn: str = "soilgrids",
        usleK_method: str = "renard",
    ):
        """
        Upgrade the model to wflow v1 format.

        The function reads a TOML from wflow v0x and converts it to wflow v1x format.
        The other components stay the same.
        A few variables that used to be computed within Wflow.jl are now moved to
        HydroMT to allow more flexibility for the users to update if they do get local
        data or calibrate some of the parameters specifically. For this, the
        ``setup_soilmaps`` and ``setup_riverbedsed`` functions are called again.

        This function should be followed by ``write_config`` to write the upgraded TOML
        file and by ``write_grid`` to write the upgraded static netcdf input file.
        """
        config_out = convert_to_wflow_v1_sediment(self.config, logger=self.logger)
        self._config = dict()
        for option in config_out:
            self.set_config(option, config_out[option])

        # Rerun setup_soilmaps
        self.setup_soilmaps(
            soil_fn=soil_fn,
            usleK_method=usleK_method,
            add_aggregates=True,
        )

        # Rerun setup_riverbedsed
        self.setup_riverbedsed(bedsed_mapping_fn=None)
