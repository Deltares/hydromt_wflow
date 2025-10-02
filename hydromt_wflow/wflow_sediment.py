"""Implement the Wflow Sediment model class."""

import logging
import tomllib
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from hydromt import hydromt_step
from hydromt.error import NoDataStrategy

import hydromt_wflow.utils as utils
from hydromt_wflow import workflows
from hydromt_wflow.naming import _create_hydromt_wflow_mapping_sediment
from hydromt_wflow.version_upgrade import (
    convert_reservoirs_to_wflow_v1_sediment,
    convert_to_wflow_v1_sediment,
)
from hydromt_wflow.wflow_base import WflowBaseModel

__all__ = ["WflowSedimentModel"]
__hydromt_eps__ = ["WflowSedimentModel"]
logger = logging.getLogger(f"hydromt.{__name__}")


class WflowSedimentModel(WflowBaseModel):
    """The wflow sediment model class, a subclass of WflowBaseModel."""

    name: str = "wflow_sediment"

    def __init__(
        self,
        root: str | None = None,
        config_filename: str | None = None,
        mode: str = "w",
        data_libs: list | str = [],
        **catalog_keys,
    ):
        super().__init__(
            root=root,
            config_filename=config_filename,
            mode=mode,
            data_libs=data_libs,
            **catalog_keys,
        )
        # Update compared to wflow sbm
        self._MAPS, self._WFLOW_NAMES = _create_hydromt_wflow_mapping_sediment(
            self.config.data
        )

    @hydromt_step
    def setup_rivers(
        self,
        hydrography_fn: str | xr.Dataset,
        river_geom_fn: str | gpd.GeoDataFrame | None = None,
        river_upa: float = 30,
        slope_len: float = 2e3,
        min_rivlen_ratio: float = 0.0,
        min_rivwth: float = 30,
        smooth_len: float = 5e3,
        output_names: dict = {
            "river_location__mask": "river_mask",
            "river__length": "river_length",
            "river__width": "river_width",
            "river__slope": "river_slope",
        },
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
        the flow directions over a length ``smooth_len`` [m].

        Adds model layers:

            * **river_mask** map: river mask [-]
            * **river_length** map: river length [m]
            * **river_width** map: river width [m]
            * **river_slope** map: river slope [m/m]
            * **rivers** geom: river vector based on river_mask

        Required setup methods:

        * :py:meth:`~WflowSedimentModel.setup_basemaps`

        Parameters
        ----------
        hydrography_fn : str, Path, xarray.Dataset
            Name of RasterDataset source for hydrography data.
            Must be same as setup_basemaps for consistent results.

            * **Required variables**: 'flwdir' [LLD or D8 or NEXTXY], 'uparea' [km2],
              'elevtn' [m+REF]

            * **Optional variables**: 'rivwth' [m]

        river_geom_fn : str, Path, geopandas.GeoDataFrame, optional
            Name of GeoDataFrame source for river data.

            * **Required variables**: 'rivwth' [m]

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
        smooth_len : float, optional
            Length [m] over which to smooth the output river width,
            by default 5e3
        min_rivwth : float, optional
            Minimum river width [m], by default 30.0
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        """
        super().setup_rivers(
            hydrography_fn=hydrography_fn,
            river_geom_fn=river_geom_fn,
            river_upa=river_upa,
            slope_len=slope_len,
            min_rivlen_ratio=min_rivlen_ratio,
            smooth_len=smooth_len,
            min_rivwth=min_rivwth,
            rivdph_method=None,
            min_rivdph=None,
            output_names=output_names,
        )

    @hydromt_step
    def setup_riverbedsed(
        self,
        bedsed_mapping_fn: str | Path | pd.DataFrame | None = None,
        strord_name: str = "meta_streamorder",
        output_names: dict = {
            "river_bottom_and_bank_sediment__median_diameter": "river_bed_sediment_d50",
            "river_bottom_and_bank_clay__mass_fraction": "river_bed_clay_fraction",
            "river_bottom_and_bank_silt__mass_fraction": "river_bed_silt_fraction",
            "river_bottom_and_bank_sand__mass_fraction": "river_bed_sand_fraction",
            "river_bottom_and_bank_gravel__mass_fraction": "river_bed_gravel_fraction",
            "river_water_sediment__kodatie_transport_capacity_a_coefficient": "river_kodatie_a",  # noqa: E501
            "river_water_sediment__kodatie_transport_capacity_b_coefficient": "river_kodatie_b",  # noqa: E501
            "river_water_sediment__kodatie_transport_capacity_c_coefficient": "river_kodatie_c",  # noqa: E501
            "river_water_sediment__kodatie_transport_capacity_d_coefficient": "river_kodatie_d",  # noqa: E501
        },
    ):
        """Generate sediments based river bed characteristics maps.

        Kodatie transport capacity coefficients can also be derived from such mapping
        table based on the mean sediment diameter of the river bed.

        Adds model layers:

        * **river_bed_sediment_d50** map: median sediment diameter of the river bed [mm]
        * **river_bed_clay_fraction** map: fraction of clay material in the river bed [-]
        * **river_bed_silt_fraction** map: fraction of silt material in the river bed [-]
        * **river_bed_sand_fraction** map: fraction of sand material in the river bed [-]
        * **river_bed_gravel_fraction** map: fraction of gravel material in the river bed [-]
        * **river_kodatie_a** map: Kodatie transport capacity coefficient a [-]
        * **river_kodatie_b** map: Kodatie transport capacity coefficient b [-]
        * **river_kodatie_c** map: Kodatie transport capacity coefficient c [-]
        * **river_kodatie_d** map: Kodatie transport capacity coefficient d [-]

        Parameters
        ----------
        bedsed_mapping_fn : str
            Path to a mapping csv file from streamorder to river bed particles
            characteristics. If None reverts to default values.

            * Required variable: ['strord','river_bed_sediment_d50', \
'river_bed_clay_fraction', 'river_bed_silt_fraction', 'river_bed_sand_fraction', \
'river_bed_gravel_fraction']
            * Optional variable: ['river_kodatie_a', 'river_kodatie_b', \
'river_kodatie_c', 'river_kodatie_d']

        strord_name : str, optional
            Name of the stream order map in the grid, by default 'meta_streamorder'.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        """  # noqa: E501
        logger.info("Preparing riverbedsed parameter maps.")
        # check for streamorder
        if self._MAPS["strord"] not in self.staticmaps.data:
            if strord_name not in self.staticmaps.data:
                raise ValueError(
                    f"Streamorder map {strord_name} not found in grid. "
                    "Please run setup_basemaps or update the strord_name argument."
                )
            else:
                self._MAPS["strord"] = strord_name
        # update self._MAPS and self._WFLOW_NAMES with user defined output names
        self._update_naming(output_names)

        # Make river_bed_sediment_d50 map from csv file with mapping between streamorder
        #  and river_bed_sediment_d50 value
        if bedsed_mapping_fn is None:
            fn_map = "riverbedsed_mapping_default"
        else:
            fn_map = bedsed_mapping_fn

        df = self.data_catalog.get_dataframe(fn_map)

        strord = self.staticmaps.data[self._MAPS["strord"]].copy()
        # max streamorder value above which values get the same D50 value
        max_str = df.index[-2]
        nodata = df.index[-1]
        # if streamroder value larger than max_str, assign last value
        strord = strord.where(strord <= max_str, max_str)
        # handle missing value (last row of csv is mapping of nan values)
        strord = strord.where(strord != strord.raster.nodata, nodata)
        strord.raster.set_nodata(nodata)

        ds_riversed = workflows.landuse(
            da=strord,
            ds_like=self.staticmaps.data,
            df=df,
        )

        rmdict = {k: self._MAPS.get(k, k) for k in ds_riversed.data_vars}
        self.set_grid(ds_riversed.rename(rmdict))
        # update config
        self._update_config_variable_name(ds_riversed.rename(rmdict).data_vars)

    @hydromt_step
    def setup_natural_reservoirs(
        self,
        reservoirs_fn: str | Path | gpd.GeoDataFrame,
        overwrite_existing: bool = False,
        duplicate_id: str = "error",
        min_area: float = 1.0,
        output_names: dict = {
            "reservoir_area__count": "reservoir_area_id",
            "reservoir_location__count": "reservoir_outlet_id",
            "reservoir_surface__area": "reservoir_area",
            "reservoir_water_sediment__bedload_trapping_efficiency": "reservoir_trapping_efficiency",  # noqa : E501
        },
        geom_name: str = "meta_natural_reservoirs",
        **kwargs,
    ):
        """Generate maps of natural reservoir areas (lakes) and outlets.

        This method is a specialized version of the `setup_reservoirs` method
        that is specifically designed for lakes, which are natural water bodies
        without artificial dams.

        This means the trapping efficiency is set to 0.0 by default and the output
        staticgeoms will be called meta_natural_reservoirs.geojson by default.

        For a description of the parameters and functionality, see
        py:meth:`setup_reservoirs`.

        Required setup methods:

        * :py:meth:`~WflowSedimentModel.setup_rivers`

        """
        self.setup_reservoirs(
            reservoirs_fn=reservoirs_fn,
            overwrite_existing=overwrite_existing,
            duplicate_id=duplicate_id,
            min_area=min_area,
            trapping_default=0.0,  # lakes have no trapping efficiency
            output_names=output_names,
            geom_name=geom_name,
            **kwargs,
        )

    @hydromt_step
    def setup_reservoirs(
        self,
        reservoirs_fn: str | Path | gpd.GeoDataFrame,
        overwrite_existing: bool = False,
        duplicate_id: str = "error",
        min_area: float = 1.0,
        trapping_default: float = 1.0,
        output_names: dict = {
            "reservoir_area__count": "reservoir_area_id",
            "reservoir_location__count": "reservoir_outlet_id",
            "reservoir_surface__area": "reservoir_area",
            "reservoir_water_sediment__bedload_trapping_efficiency": "reservoir_trapping_efficiency",  # noqa : E501
        },
        geom_name: str = "meta_reservoirs",
        **kwargs,
    ):
        """Generate maps of reservoir areas and outlets.

        Also generates parameters with average reservoir area,
        and trapping efficiency for large particles.

        The data is generated from features with ``min_area`` [km2] (default is 1 km2)
        from a database with reservoir geometry, IDs and metadata.

        Adds model layers:

        * **reservoir_area_id** map: reservoir IDs [-]
        * **reservoir_outlet_id** map: reservoir IDs at outlet locations [-]
        * **reservoir_area** map: reservoir area [m2]
        * **reservoir_trapping_efficiency** map: reservoir bedload trapping efficiency
          coefficient [-] (0 for natural lakes, 0-1 depending on the type of dam)
        * **meta_reservoirs** geom: polygon with reservoirs and parameters
        * **reservoirs** geom: polygon with all reservoirs as in the model

        Required setup methods:

        * :py:meth:`~WflowSedimentModel.setup_rivers`

        Parameters
        ----------
        reservoirs_fn : str
            Name of data source for reservoir parameters, see ``data/data_sources.yml``.

            * Required variables: ``['waterbody_id', 'Area_avg']``

            * Optional variables: ``['reservoir_trapping_efficiency']``

        overwrite_existing : bool, optional
            If True, overwrite existing reservoirs in the model grid. Default is False.
        duplicate_id : {"error", "skip"}, optional
            Action to take if duplicate reservoir IDs are found when merging with
            existing reservoirs. Options are ``"error"`` (default) to raise an error,
            or ``"skip"`` to skip adding new reservoirs.
        min_area : float, optional
            Minimum reservoir area threshold [km2]. Default is 1.0.
        trapping_default : float, optional
            Default trapping efficiency coefficient for large particles [0-1].
            Default is 1.0, meaning 100% of large particles (sand to gravel)
            are trapped (e.g. in a gravity dam). For other dam types, the
            natural deposition rates from Camp are used.
        output_names : dict, optional
            Dictionary with output names to be used in the model NetCDF input
            files. Keys should be Wflow.jl variable names, values the names in
            the NetCDF file.
        geom_name : str, optional
            Name of the reservoirs geometry in the ``staticgeoms`` folder.
            Default is ``"meta_reservoirs"`` (for meta_reservoirs.geojson).
        kwargs : dict, optional
            Additional keyword arguments passed to
            ``hydromt.DataCatalog.get_rasterdataset()``.
        """
        # retrieve data for basin
        logger.info("Preparing reservoir maps.")
        kwargs.setdefault("predicate", "contains")
        gdf_res = self.data_catalog.get_geodataframe(
            reservoirs_fn,
            geom=self.basins_highres,
            handle_nodata=NoDataStrategy.IGNORE,
            **kwargs,
        )
        # Skip method if no data is returned
        if gdf_res is None:
            logger.info("Skipping method, as no data has been found")
            return

        # Derive reservoir area and outlet maps
        ds_res, gdf_res = workflows.reservoir_id_maps(
            gdf=gdf_res,
            ds_like=self.staticmaps.data,
            min_area=min_area,
            uparea_name=self._MAPS["uparea"],
        )
        if ds_res is None:
            # No reservoir of sufficient size found
            return
        self._update_naming(output_names)

        # Add default trapping efficiency coefficient if not in data source
        if "reservoir_trapping_efficiency" not in gdf_res.columns:
            gdf_res["reservoir_trapping_efficiency"] = trapping_default
        # add reservoirs parameters to grid
        gdf_points = gpd.GeoDataFrame(
            gdf_res[["waterbody_id", "Area_avg", "reservoir_trapping_efficiency"]],
            geometry=gpd.points_from_xy(gdf_res.xout, gdf_res.yout),
        )
        ds_res["reservoir_area"] = self.staticmaps.data.raster.rasterize(
            gdf_points, col_name="Area_avg", dtype="float32", nodata=-999
        )
        ds_res["reservoir_trapping_efficiency"] = self.staticmaps.data.raster.rasterize(
            gdf_points,
            col_name="reservoir_trapping_efficiency",
            dtype="float32",
            nodata=-999,
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
            ds_res = workflows.reservoirs.merge_reservoirs_sediment(
                ds_res,
                self.staticmaps.data.rename(inv_rename),
                duplicate_id=duplicate_id,
            )
            # Check if ds_res is None ie duplicate IDs
            if ds_res is None:
                logger.warning(
                    "Duplicate reservoir IDs found. Skip adding the new reservoirs."
                )
                return

        # add to grid
        rmdict = {k: self._MAPS.get(k, k) for k in ds_res.data_vars}
        self.set_grid(ds_res.rename(rmdict))

        # write reservoirs with attr tables to static geoms.
        self.set_geoms(gdf_res.rename({"Area_avg": "reservoir_area"}), name=geom_name)
        # Prepare a combined geoms of all reservoirs
        gdf_res_all = workflows.reservoirs.create_reservoirs_geoms_sediment(
            ds_res.rename(rmdict),
        )
        self.set_geoms(gdf_res_all, name="reservoirs")

        # Reservoir settings in the toml to update
        self.set_config("model.reservoir__flag", True)
        for dvar in ds_res.data_vars:
            if dvar in ["reservoir_area_id", "reservoir_outlet_id"]:
                self._update_config_variable_name(self._MAPS[dvar], data_type=None)
            elif dvar in self._WFLOW_NAMES:
                self._update_config_variable_name(self._MAPS[dvar])

    @hydromt_step
    def setup_lulcmaps(
        self,
        lulc_fn: str | Path | xr.DataArray,
        lulc_mapping_fn: str | Path | pd.DataFrame | None = None,
        planted_forest_fn: str | Path | gpd.GeoDataFrame | None = None,
        lulc_vars: dict = {
            "landuse": None,
            "soil_compacted_fraction": "compacted_soil__area_fraction",
            "erosion_usle_c": "soil_erosion__usle_c_factor",
            "land_water_fraction": "land_water_covered__area_fraction",
        },
        planted_forest_c: float = 0.0881,
        orchard_name: str = "Orchard",
        orchard_c: float = 0.2188,
        output_names_suffix: str | None = None,
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
            * **erosion_usle_c** map: Cover management factor from the USLE equation [-]
            * **soil_compacted_fraction** map: The fraction of compacted or urban area
                per grid cell [-]
            * **land_water_fraction** map: The fraction of water covered area per
                grid cell [-]

        Required setup methods:

        * :py:meth:`~WflowSedimentModel.setup_basemaps`

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

        lulc_vars : dict
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
            variables in lulc_vars will be renamed to "landuse_vito",
            "erosion_usle_c_vito", etc.

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
        if self._MAPS["usle_c"] in lulc_vars and planted_forest_fn is not None:
            # Read forest data
            planted_forest = self.data_catalog.get_geodataframe(
                planted_forest_fn,
                geom=self.basins,
                buffer=1,
                predicate="intersects",
                handle_nodata="IGNORE",
            )
            if planted_forest is None:
                logger.warning("No Planted forest data found within domain.")
                return
            rename_dict = {
                v: k
                for k, v in self._MAPS.items()
                if v in self.staticmaps.data.data_vars
            }
            usle_c = workflows.add_planted_forest_to_landuse(
                planted_forest,
                self.staticmaps.data.rename(rename_dict),
                planted_forest_c=planted_forest_c,
                orchard_name=orchard_name,
                orchard_c=orchard_c,
            )

            # Add to grid
            self.set_grid(usle_c, name=self._MAPS["usle_c"])

    @hydromt_step
    def setup_lulcmaps_from_vector(
        self,
        lulc_fn: str | gpd.GeoDataFrame,
        lulc_mapping_fn: str | Path | pd.DataFrame | None = None,
        planted_forest_fn: str | Path | gpd.GeoDataFrame | None = None,
        lulc_vars: dict = {
            "landuse": None,
            "soil_compacted_fraction": "compacted_soil__area_fraction",
            "erosion_usle_c": "soil_erosion__usle_c_factor",
            "land_water_fraction": "land_water_covered__area_fraction",
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
            * **erosion_usle_c** map: Cover management factor from the USLE equation [-]
            * **soil_compacted_fraction** map: The fraction of compacted or urban area
                per grid cell [-]
            * **land_water_fraction** map: The fraction of water covered area per grid
              cell [-]

        Required setup methods:

        * :py:meth:`~WflowSedimentModel.setup_basemaps`

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
            Dictionary of landuse parameters to prepare. The names are the
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
            variables in lulc_vars will be renamed to "landuse_vito",
            "erosion_usle_c_vito", etc.

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
        if self._MAPS["usle_c"] in lulc_vars and planted_forest_fn is not None:
            # Read forest data
            planted_forest = self.data_catalog.get_geodataframe(
                planted_forest_fn,
                geom=self.basins,
                buffer=1,
                predicate="intersects",
                handle_nodata="IGNORE",
            )
            if planted_forest is None:
                logger.warning("No Planted forest data found within domain.")
                return
            rename_dict = {
                v: k
                for k, v in self._MAPS.items()
                if v in self.staticmaps.data.data_vars
            }
            usle_c = workflows.add_planted_forest_to_landuse(
                planted_forest,
                self.staticmaps.data.rename(rename_dict),
                planted_forest_c=planted_forest_c,
                orchard_name=orchard_name,
                orchard_c=orchard_c,
            )

            # Add to grid
            self.set_grid(usle_c, name=self._MAPS["usle_c"])

    @hydromt_step
    def setup_canopymaps(
        self,
        canopy_fn: str | Path | xr.DataArray,
        output_name: str = "vegetation_height",
    ):
        """Generate sediments based canopy height maps.

        Adds model layers:

            * **vegetation_height** map: height of the vegetation canopy [m]

        Parameters
        ----------
        canopy_fn :
            Canopy height data source (DataArray).
        output_name : dict, optional
            Name of the output map. By default 'vegetation_height'.
        """
        logger.info("Preparing canopy height map.")

        # Canopy height
        dsin = self.data_catalog.get_rasterdataset(
            canopy_fn, geom=self.region, buffer=2
        )
        dsout = xr.Dataset(coords=self.staticmaps.data.raster.coords)
        ds_out = dsin.raster.reproject_like(self.staticmaps.data, method="average")
        dsout["vegetation_height"] = ds_out.astype(np.float32)
        dsout["vegetation_height"] = dsout["vegetation_height"].fillna(-9999.0)
        dsout["vegetation_height"].raster.set_nodata(-9999.0)

        # update name
        wflow_var = self._WFLOW_NAMES[self._MAPS["vegetation_height"]]
        self._update_naming({wflow_var: output_name})
        self.set_grid(dsout["vegetation_height"], name=output_name)
        # update config
        self._update_config_variable_name(output_name)

    @hydromt_step
    def setup_soilmaps(
        self,
        soil_fn: str = "soilgrids",
        usle_k_method: str = "renard",
        add_aggregates: bool = True,
        output_names: dict = {
            "soil_clay__mass_fraction": "soil_clay_fraction",
            "soil_silt__mass_fraction": "soil_silt_fraction",
            "soil_sand__mass_fraction": "soil_sand_fraction",
            "soil_small_aggregates__mass_fraction": "soil_sagg_fraction",
            "soil_large_aggregates__mass_fraction": "soil_lagg_fraction",
            "soil_erosion__rainfall_soil_detachability_factor": "erosion_soil_detachability",  # noqa: E501
            "soil_erosion__usle_k_factor": "erosion_usle_k",
            "land_surface_sediment__median_diameter": "soil_sediment_d50",
            "land_surface_water_sediment__govers_transport_capacity_coefficient": "land_govers_c",  # noqa: E501
            "land_surface_water_sediment__govers_transport_capacity_exponent": "land_govers_n",  # noqa: E501
        },
    ):
        """Generate sediments based soil parameter maps.

        Sediment size distribution and addition of small and large aggregates can be
        estimated from primary particle size distribution with Foster et al. (1980).
        USLE K factor can be computed from the soil data using Renard or EPIC methods.
        Calculation of D50 and fraction of fine and very fine sand (fvfs) from
        Fooladmand et al, 2006.

        Adds model layers:

            * **soil_clay_fraction**: clay content of the topsoil [g/g]
            * **soil_silt_fraction**: silt content of the topsoil [g/g]
            * **soil_sand_fraction**: sand content of the topsoil [g/g]
            * **soil_sagg_fraction**: small aggregate content of the topsoil [g/g]
            * **soil_lagg_fraction**: large aggregate content of the topsoil [g/g]
            * **erosion_soil_detachability** map: mean detachability of the soil
                (Morgan et al., 1998) [g/J]
            * **erosion_usle_k** map: soil erodibility factor from the USLE equation
                [-]
            * **soil_sediment_d50** map: median sediment diameter of the soil [mm]
            * **land_govers_c** map: Govers factor for overland flow transport
                capacity [-]
            * **land_govers_n** map: Govers exponent for overland flow transport
                capacity [-]

        Required setup methods:

        * :py:meth:`~WflowSedimentModel.setup_basemaps`


        Parameters
        ----------
        soil_fn : {"soilgrids"}
            Name of soil data source in data_sources.yml file.
                * Required variables: ['clyppt_sl1', 'sltppt_sl1', 'oc_sl1']
        usle_k_method: {"renard", "epic"}
            Method to compute the USLE K factor, by default renard.
        add_aggregates: bool, optional
            Add small and large aggregates based on soil texture, by default True.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        """
        logger.info("Preparing soil parameter maps.")

        # Soil related maps
        if soil_fn not in ["soilgrids"]:
            logger.warning(
                f"Invalid source '{soil_fn}', skipping setup_soilmaps for sediment."
            )
            return

        self._update_naming(output_names)
        dsin = self.data_catalog.get_rasterdataset(soil_fn, geom=self.region, buffer=2)
        dsout = workflows.soilgrids_sediment(
            dsin,
            self.staticmaps.data,
            usle_k_method=usle_k_method,
            add_aggregates=add_aggregates,
        )
        rmdict = {k: self._MAPS.get(k, k) for k in dsout.data_vars}
        self.set_grid(dsout.rename(rmdict))
        self._update_config_variable_name(dsout.rename(rmdict).data_vars)

    @hydromt_step
    def setup_outlets(
        self,
        river_only: bool = True,
        toml_output: str = "csv",
        gauge_toml_header: list[str] = ["suspended_solids"],
        gauge_toml_param: list[str] = [
            "river_water_sediment__suspended_mass_concentration",
        ],
    ):
        """Set the default gauge map based on basin outlets.

        If the subcatchment map is available, the catchment outlets IDs will be matching
        the subcatchment IDs. If not, then IDs from 1 to number of outlets are used.

        Can also add csv/netcdf_scalar output settings in the TOML.

        Adds model layers:

        * **outlets** map: IDs map from catchment outlets [-]
        * **outlets** geom: polygon of catchment outlets

        Required setup methods:

        * :py:meth:`~WflowSedimentModel.setup_rivers`

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
            By default saves suspended_solids (for
            river_water_sediment__suspended_mass_concentration).
        gauge_toml_param: list, optional
            Save specific model parameters in csv section. This option defines
            the wflow variable corresponding to the names in gauge_toml_header.
            By default saves river_water_sediment__suspended_mass_concentration (for
            suspended_solids).
        """
        super().setup_outlets(
            river_only=river_only,
            toml_output=toml_output,
            gauge_toml_header=gauge_toml_header,
            gauge_toml_param=gauge_toml_param,
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
        toml_output: str | None = "csv",
        gauge_toml_header: list[str] | None = ["river_q", "suspended_solids"],
        gauge_toml_param: list[str] | None = [
            "river_water__volume_flow_rate",
            "river_water_sediment__suspended_mass_concentration",
        ],
        **kwargs,
    ):
        """Set a gauge map based on ``gauges_fn`` data.

        This function directly calls the ``setup_gauges`` function of the
        WflowBaseModel, see
        py:meth:`hydromt_wflow.wflow_base.WflowBaseModel.setup_gauges` for more details.

        The only differences are the default values for the arguments:

        - ``gauge_toml_header`` defaults to ["river_q", "suspended_solids"]
        - ``gauge_toml_param`` defaults to ["river_water__volume_flow_rate",
            "river_water_sediment__suspended_mass_concentration"]

        Required setup methods:

        * :py:meth:`~WflowSedimentModel.setup_rivers`

        See Also
        --------
        WflowBaseModel.setup_gauges
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
            self._MAPS["reservoir_area"],
            self._MAPS["reservoir_trapping_efficiency"],
        ]
        reservoir_maps = {k: self._WFLOW_NAMES.get(k, None) for k in reservoir_maps}

        super().clip(
            region,
            inverse_clip=inverse_clip,
            clip_forcing=clip_forcing,
            clip_states=clip_states,
            reservoir_maps=reservoir_maps,
            crs=crs,
            **kwargs,
        )

    @hydromt_step
    def upgrade_to_v1_wflow(
        self,
        soil_fn: str = "soilgrids",
        usle_k_method: str = "renard",
        strord_name: str = "wflow_streamorder",
    ):
        """
        Upgrade the model to wflow v1 format.

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
        soil_fn : str, optional
            soil_fn argument of setup_soilmaps method.
        usle_k_method : str, optional
            usle_k_method argument of setup_soilmaps method.
        strord_name : str, optional
            strord_name argument of setup_riverbedsed method.
        """
        config_v0 = self.config.data.copy()
        config_out = convert_to_wflow_v1_sediment(self.config.data)

        # Update the config
        with open(utils.DATADIR / "default_config_headers.toml", "rb") as file:
            self.config._data = tomllib.load(file)
        for option in config_out:
            self.set_config(option, config_out[option])

        # Rerun setup_soilmaps
        self.setup_soilmaps(
            soil_fn=soil_fn,
            usle_k_method=usle_k_method,
            add_aggregates=True,
        )

        # Rerun setup_riverbedsed
        self.setup_riverbedsed(bedsed_mapping_fn=None, strord_name=strord_name)

        # Merge lakes and reservoirs layers
        ds_res, vars_to_remove, config_opt = convert_reservoirs_to_wflow_v1_sediment(
            self.staticmaps.data, config_v0
        )
        if ds_res is not None:
            # Remove older maps from grid
            self.staticmaps.drop_vars(vars_to_remove)
            # Add new reservoir maps to grid
            self.set_grid(ds_res)
            # Update the config with the new names
            for option in config_opt:
                self.set_config(option, config_opt[option])
