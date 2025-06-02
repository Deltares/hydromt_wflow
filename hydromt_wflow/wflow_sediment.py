"""Implement the Wflow Sediment model class."""

import logging
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import tomlkit
import xarray as xr

from hydromt_wflow.utils import (
    DATADIR,
    convert_to_wflow_v1_sediment,
)
from hydromt_wflow.wflow import WflowModel

from . import workflows
from .naming import _create_hydromt_wflow_mapping_sediment

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
        root: str | None = None,
        mode: str | None = "w",
        config_fn: str | None = None,
        data_libs: List | str = [],
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

    def setup_rivers(
        self,
        hydrography_fn: str | xr.Dataset,
        river_geom_fn: str | gpd.GeoDataFrame | None = None,
        river_upa: float = 30,
        slope_len: float = 2e3,
        min_rivlen_ratio: float = 0.0,
        min_rivwth: float = 30,
        smooth_len: float = 5e3,
        output_names: Dict = {
            "river_location__mask": "river_mask",
            "river__length": "river_length",
            "river__width": "river_width",
            "river__slope": "river_slope",
        },
    ):
        """Set all river parameter maps.

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

        Adds model layers:

        * **river_mask** map: river mask [-]
        * **river_length** map: river length [m]
        * **river_width** map: river width [m]
        * **river_slope** map: river slope [m/m]
        * **rivers** geom: river vector based on wflow_river mask

        Parameters
        ----------
        hydrography_fn : str, Path, xarray.Dataset
            Name of RasterDataset source for hydrography data.
            Must be same as setup_basemaps for consistent results.

            * Required variables: 'flwdir' [LLD or D8 or NEXTXY], 'uparea' [km2],
              'elevtn'[m+REF]
            * Optional variables: 'rivwth' [m]
        river_geom_fn : str, Path, geopandas.GeoDataFrame, optional
            Name of GeoDataFrame source for river data.

            * Required variables: 'rivwth' [m]
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
            Length [m] over which to smooth the output river width and depth,
            by default 5e3
        min_rivwth : float, optional
            Minimum river width [m], by default 30.0
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.

        See Also
        --------
        workflows.river_bathymetry
        """
        self.logger.info("Preparing river maps.")
        # update self._MAPS and self._WFLOW_NAMES with user defined output names
        self._update_naming(output_names)

        # Check that river_upa threshold is bigger than the maximum uparea in the grid
        if river_upa > float(self.grid[self._MAPS["uparea"]].max()):
            raise ValueError(
                f"river_upa threshold {river_upa} should be larger than the maximum \
uparea in the grid {float(self.grid[self._MAPS['uparea']].max())} in order to create \
river cells."
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
        # update config
        for dvar in dvars:
            if dvar == "rivmsk":
                self._update_config_variable_name(self._MAPS[dvar], data_type=None)
            else:
                self._update_config_variable_name(self._MAPS[dvar])

        # get rivwth
        if river_geom_fn is not None:
            gdf_riv = self.data_catalog.get_geodataframe(
                river_geom_fn, geom=self.region
            )
            # re-read model data to get river maps
            inv_rename = {v: k for k, v in self._MAPS.items() if v in self.grid}
            ds_riv1 = workflows.river_bathymetry(
                ds_model=self.grid.rename(inv_rename),
                gdf_riv=gdf_riv,
                smooth_len=smooth_len,
                min_rivwth=min_rivwth,
                logger=self.logger,
            )
            # only add river width
            self.set_grid(ds_riv1["rivwth"], name=self._MAPS["rivwth"])
            # update config
            self._update_config_variable_name(self._MAPS["rivwth"])

        self.logger.debug("Adding rivers vector to geoms.")
        self.geoms.pop("rivers", None)  # remove old rivers if in geoms
        self.rivers  # add new rivers to geoms

    def setup_lakes(
        self,
        lakes_fn: str | Path | gpd.GeoDataFrame = "hydro_lakes",
        min_area: float = 1.0,
        output_names: Dict = {
            "lake_area__count": "lake_area_id",
            "lake_location__count": "lake_outlet_id",
            "lake_surface__area": "lake_area",
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

        * **lake_area_id** map: lake IDs [-]
        * **lake_outlet_id** map: lake IDs at outlet locations [-]
        * **lake_area** map: lake area [m2]
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
        ds_lakes["lake_area"] = self.grid.raster.rasterize(
            gdf_points, col_name="Area_avg", dtype="float32", nodata=-999
        )

        # add to grid
        rmdict = {k: self._MAPS.get(k, k) for k in ds_lakes.data_vars}
        self.set_grid(ds_lakes.rename(rmdict))
        # write lakes with attr tables to static geoms.
        self.set_geoms(gdf_lakes.rename({"Area_avg": "lake_area"}), name=geom_name)

        # Lake settings in the toml to update
        self.set_config("model.lake__flag", True)
        for dvar in ds_lakes.data_vars:
            if dvar == "lake_area_id" or dvar == "lake_outlet_id":
                self._update_config_variable_name(self._MAPS[dvar], data_type=None)
            elif dvar in self._WFLOW_NAMES:
                self._update_config_variable_name(self._MAPS[dvar])

    def setup_reservoirs(
        self,
        reservoirs_fn: str | Path | gpd.GeoDataFrame,
        min_area: float = 1.0,
        trapping_default: float = 1.0,
        output_names: Dict = {
            "reservoir_area__count": "reservoir_area_id",
            "reservoir_location__count": "reservoir_outlet_id",
            "reservoir_surface__area": "reservoir_area",
            "reservoir_water_sediment~bedload__trapping_efficiency": "reservoir_trapping_efficiency",  # noqa : E501
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

        * **reservoir_area_id** map: reservoir IDs [-]
        * **reservoir_outlet_id** map: reservoir IDs at outlet locations [-]
        * **reservoir_area** map: reservoir area [m2]
        * **reservoir_trapping_efficiency** map: reservoir trapping efficiency
         coefficient [-]

        Parameters
        ----------
        reservoirs_fn : str
            Name of data source for reservoir parameters, see data/data_sources.yml.

            * Required variables: ['waterbody_id', 'Area_avg']

            * Optional variables: ['reservoir_trapping_efficiency']
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
        if "reservoir_trapping_efficiency" not in gdf_res.columns:
            gdf_res["reservoir_trapping_efficiency"] = trapping_default
        # add reservoirs parameters to grid
        gdf_points = gpd.GeoDataFrame(
            gdf_res[["waterbody_id", "Area_avg", "reservoir_trapping_efficiency"]],
            geometry=gpd.points_from_xy(gdf_res.xout, gdf_res.yout),
        )
        ds_res["reservoir_area"] = self.grid.raster.rasterize(
            gdf_points, col_name="Area_avg", dtype="float32", nodata=-999
        )
        ds_res["reservoir_trapping_efficiency"] = self.grid.raster.rasterize(
            gdf_points,
            col_name="reservoir_trapping_efficiency",
            dtype="float32",
            nodata=-999,
        )

        # add to grid
        rmdict = {k: self._MAPS.get(k, k) for k in ds_res.data_vars}
        self.set_grid(ds_res.rename(rmdict))
        # write lakes with attr tables to static geoms.
        self.set_geoms(gdf_res.rename({"Area_avg": "reservoir_area"}), name=geom_name)

        # Lake settings in the toml to update
        self.set_config("model.reservoir__flag", True)
        for dvar in ds_res.data_vars:
            if dvar in ["reservoir_area_id", "reservoir_outlet_id"]:
                self._update_config_variable_name(self._MAPS[dvar], data_type=None)
            elif dvar in self._WFLOW_NAMES:
                self._update_config_variable_name(self._MAPS[dvar])

    def setup_outlets(
        self,
        river_only: bool = True,
        toml_output: str = "csv",
        gauge_toml_header: List[str] = ["suspended_solids"],
        gauge_toml_param: List[str] = [
            "river_water_sediment~suspended__mass_concentration",
        ],
    ):
        """Set the default gauge map based on basin outlets.

        If the subcatchment map is available, the catchment outlets IDs will be matching
        the subcatchment IDs. If not, then IDs from 1 to number of outlets are used.

        Can also add csv/netcdf_scalar output settings in the TOML.

        Adds model layers:

        * **outlets** map: IDs map from catchment outlets [-]
        * **outlets** geom: polygon of catchment outlets

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
            river_water_sediment~suspended__mass_concentration).
        gauge_toml_param: list, optional
            Save specific model parameters in csv section. This option defines
            the wflow variable corresponding to the names in gauge_toml_header.
            By default saves river_water_sediment~suspended__mass_concentration (for
            suspended_solids).
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
        gauge_toml_header: List[str] | None = ["river_q", "suspended_solids"],
        gauge_toml_param: List[str] | None = [
            "river_water__volume_flow_rate",
            "river_water_sediment~suspended__mass_concentration",
        ],
        **kwargs,
    ):
        """Set a gauge map based on ``gauges_fn`` data.

        This function directly calls the ``setup_gauges`` function of the WflowModel,
        see py:meth:`hydromt_wflow.wflow.WflowModel.setup_gauges` for more details.

        The only differences are the default values for the arguments:

        - ``gauge_toml_header`` defaults to ["river_q", "suspended_solids"]
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
            "soil_compacted_fraction": "soil~compacted__area_fraction",
            "erosion_usle_c": "soil_erosion__usle_c_factor",
            "land_water_fraction": "land~water-covered__area_fraction",
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
        * **erosion_usle_c ** map: Cover management factor from the USLE equation [-]
        * **soil_compacted_fraction** map: The fraction of compacted or urban area per
          grid cell [-]
        * **land_water_fraction** map: The fraction of water covered area per grid
        cell [-]

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
                self.logger.warning("No Planted forest data found within domain.")
                return
            rename_dict = {
                v: k for k, v in self._MAPS.items() if v in self.grid.data_vars
            }
            usle_c = workflows.add_planted_forest_to_landuse(
                planted_forest,
                self.grid.rename(rename_dict),
                planted_forest_c=planted_forest_c,
                orchard_name=orchard_name,
                orchard_c=orchard_c,
                logger=self.logger,
            )

            # Add to grid
            self.set_grid(usle_c, name=self._MAPS["usle_c"])

    def setup_lulcmaps_from_vector(
        self,
        lulc_fn: str | gpd.GeoDataFrame,
        lulc_mapping_fn: str | Path | pd.DataFrame | None = None,
        planted_forest_fn: str | Path | gpd.GeoDataFrame | None = None,
        lulc_vars: Dict = {
            "landuse": None,
            "soil_compacted_fraction": "soil~compacted__area_fraction",
            "erosion_usle_c": "soil_erosion__usle_c_factor",
            "land_water_fraction": "land~water-covered__area_fraction",
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
        * **soil_compacted_fraction** map: The fraction of compacted or urban area per
          grid cell [-]
        * **land_water_fraction** map: The fraction of water covered area per grid
          cell [-]

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
        lulc_vars : Dict
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
                self.logger.warning("No Planted forest data found within domain.")
                return
            rename_dict = {
                v: k for k, v in self._MAPS.items() if v in self.grid.data_vars
            }
            usle_c = workflows.add_planted_forest_to_landuse(
                planted_forest,
                self.grid.rename(rename_dict),
                planted_forest_c=planted_forest_c,
                orchard_name=orchard_name,
                orchard_c=orchard_c,
                logger=self.logger,
            )

            # Add to grid
            self.set_grid(usle_c, name=self._MAPS["usle_c"])

    def setup_riverbedsed(
        self,
        bedsed_mapping_fn: str | Path | pd.DataFrame | None = None,
        strord_name: str = "meta_streamorder",
        output_names: Dict = {
            "river_bottom-and-bank_sediment__median_diameter": "river_bed_sediment_d50",
            "river_bottom-and-bank_clay__mass_fraction": "river_bed_clay_fraction",
            "river_bottom-and-bank_silt__mass_fraction": "river_bed_silt_fraction",
            "river_bottom-and-bank_sand__mass_fraction": "river_bed_sand_fraction",
            "river_bottom-and-bank_gravel__mass_fraction": "river_bed_gravel_fraction",
            "river_water_sediment__kodatie_transport_capacity_a-coefficient": "river_kodatie_a",  # noqa: E501
            "river_water_sediment__kodatie_transport_capacity_b-coefficient": "river_kodatie_b",  # noqa: E501
            "river_water_sediment__kodatie_transport_capacity_c-coefficient": "river_kodatie_c",  # noqa: E501
            "river_water_sediment__kodatie_transport_capacity_d-coefficient": "river_kodatie_d",  # noqa: E501
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

            * Required variable: ['strord','river_bed_sediment_d50',
              'river_bed_clay_fraction', 'river_bed_silt_fraction',
                'river_bed_sand_fraction', 'river_bed_gravel_fraction']
            * Optional variable: ['river_kodatie_a', 'river_kodatie_b',
              'river_kodatie_c', 'river_kodatie_d']
        strord_name : str, optional
            Name of the stream order map in the grid, by default 'meta_streamorder'.
        output_names : dict, optional
            Dictionary with output names that will be used in the model netcdf input
            files. Users should provide the Wflow.jl variable name followed by the name
            in the netcdf file.
        """  # noqa: E501
        self.logger.info("Preparing riverbedsed parameter maps.")
        # check for streamorder
        if self._MAPS["strord"] not in self.grid:
            if strord_name not in self.grid:
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

        strord = self.grid[self._MAPS["strord"]].copy()
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
        self.logger.info("Preparing canopy height map.")

        # Canopy height
        dsin = self.data_catalog.get_rasterdataset(
            canopy_fn, geom=self.region, buffer=2
        )
        dsout = xr.Dataset(coords=self.grid.raster.coords)
        ds_out = dsin.raster.reproject_like(self.grid, method="average")
        dsout["vegetation_height"] = ds_out.astype(np.float32)
        dsout["vegetation_height"] = dsout["vegetation_height"].fillna(-9999.0)
        dsout["vegetation_height"].raster.set_nodata(-9999.0)

        # update name
        wflow_var = self._WFLOW_NAMES[self._MAPS["vegetation_height"]]
        self._update_naming({wflow_var: output_name})
        self.set_grid(dsout["vegetation_height"], name=output_name)
        # update config
        self._update_config_variable_name(output_name)

    def setup_soilmaps(
        self,
        soil_fn: str = "soilgrids",
        usle_k_method: str = "renard",
        add_aggregates: bool = True,
        output_names: Dict = {
            "soil_clay__mass_fraction": "soil_clay_fraction",
            "soil_silt__mass_fraction": "soil_silt_fraction",
            "soil_sand__mass_fraction": "soil_sand_fraction",
            "soil_aggregates~small__mass_fraction": "soil_sagg_fraction",
            "soil_aggregates~large__mass_fraction": "soil_lagg_fraction",
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
        * **erosion_usle_k** map: soil erodibility factor from the USLE equation [-]
        * **soil_sediment_d50** map: median sediment diameter of the soil [mm]
        * **land_govers_c** map: Govers factor for overland flow transport capacity [-]
        * **land_govers_n** map: Govers exponent for overland flow transport
        capacity [-]


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
        self.logger.info("Preparing soil parameter maps.")

        # Soil related maps
        if soil_fn not in ["soilgrids"]:
            self.logger.warning(
                f"Invalid source '{soil_fn}', skipping setup_soilmaps for sediment."
            )
            return

        self._update_naming(output_names)
        dsin = self.data_catalog.get_rasterdataset(soil_fn, geom=self.region, buffer=2)
        dsout = workflows.soilgrids_sediment(
            dsin,
            self.grid,
            usle_k_method=usle_k_method,
            add_aggregates=add_aggregates,
            logger=self.logger,
        )
        rmdict = {k: self._MAPS.get(k, k) for k in dsout.data_vars}
        self.set_grid(dsout.rename(rmdict))
        self._update_config_variable_name(dsout.rename(rmdict).data_vars)

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
        self.read()
        config_out = convert_to_wflow_v1_sediment(self.config, logger=self.logger)
        # tomlkit loads errors on this file so we have to do it in two steps
        with open(DATADIR / "default_config_headers.toml", "r") as file:
            default_header_str = file.read()

        self._config = tomlkit.parse(default_header_str)

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
