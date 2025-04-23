==========
What's new
==========
All notable changes to this project will be documented in this page.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

Unreleased
==========
These are the unreleased changes.

Added
-----

Changed
-------
- Changed name of `g_tt` parameter to `g_ttm`, to align with the changes in https://github.com/Deltares/Wflow.jl/pull/512
- **setup_soilmaps** [sediment]: add small and large aggregates to soil composition (additional to clay/silt/sand). Composition is now in fraction and not percentage.
- **setup_soilmaps** [sediment]: additional parameters are prepared by the method (e.g. soil mean diameter, Govers transport capacity parameters).
- **setup_constant_pars** [sediment]: added additional default values for sediment density and particle diameters.
- **setup_riverbedsed** [sediment]: added option to derive Kodatie transport capacity parameters based on streamorder mapping.
- Grid data is masked to subcatchment on `set_grid` now instead of on `write_grid` (#349)

Fixed
-----
- Updated installation guide (#376)

Deprecated
----------

v0.8.0 (9 April 2025)
=====================
Precipitation from point data and other new features.

Added
-----
- **setup_precip_from_point_timeseries**: method to interpolate rainfall station data as model forcing. PR #315
- Added support for inverse clipping by using the `inverse_clip=True` flag in the `clip_grid` method. PR #336
- **setup_domestic_demand_from_population**: method to compute domestic demand from population and water use per capita. PR #334
- **setup_irrigation_from_vector**: method to add irrigation areas from a vector file. PR #334
- **setup_soilmaps**: possibility to derive parameters based on soil texture. Added defaults for InfiltCapSoil. PR #334

Changed
-------
- Changed name of `g_tt` parameter to `g_ttm`, to align with the changes in https://github.com/Deltares/Wflow.jl/pull/512
- **setup_allocation_areas**: added a minimum area threshold (50 km2) to filter too small allocation areas. PR #334
- **setup_soilmaps** [sediment]: add small and large aggregates to soil composition (additional to clay/silt/sand). Composition is now in fraction and not percentage.
- **setup_soilmaps** [sediment]: additional parameters are prepared by the method (e.g. soil mean diameter, Govers transport capacity parameters).
- **setup_constant_pars** [sediment]: added additional default values for sediment density and particle diameters.
- **setup_riverbedsed** [sediment]: added option to derive Kodatie transport capacity parameters based on streamorder mapping.
- Grid data is masked to subcatchment on `set_grid` now instead of on `write_grid` (#349)

Fixed
-----
- **setup_rivers**: fixed bug if manning or gvf methods are used to compute river depth. PR #334
- **setup_lulcmaps_with_paddy**: input.vertical.kvfrac is set to kvfrac in config. PR #362
- **create_lulc_lai_mapping_table**: hardcoded x and y dim names are now set to raster.y_dim and raster.x_dim. PR #362

v0.7.1 (29 January 2025)
========================
Officially drop support for python 3.9.

Added
-----
- **setup_ksatver_vegetation**: method to calculate KsatVer_vegetation to account for biologically-enhanced soil structure in KsatVer. PR #313
- **setup_lulcmaps_from_vector**: method to prepare LULC map and params from a vector input rather than raster. PR #320

Deprecated
----------
- Support for python 3.9 (already not supported in previous releases).

v0.7.0 (8 November 2024)
========================
This release adds support to create water demand and allocation related data (available since Wflow.jl version 0.8.0).
For now, the new methods for demands support is limited to already gridded input datasets.
The release also includes support for the paddy land use type and additional landuse parameters (crop coefficient and root uptake).

Added
-----
- **setup_lulcmaps_with_paddy**: method to add paddy to the model. Adding paddies leads to changes in landuse and soil parameters. PR #226
- **setup_domestic_demand** and **setup_other_demand**: methods to prepare water demands for different sectors using gridded datasets. PR #226
- **setup_irrigation**: method to prepare irrigation areas and parameters. PR #226
- **setup_allocation_areas**: method to prepare allocation areas for water allocation. PR #226
- **setup_allocation_surfacewaterfrac**: method to prepare surface water fraction for water allocation. PR #226

Changed
-------
- **setup_lulcmaps** prepares new vegetation parameters (crop coefficient kc and h values). PR #226
- **set_grid** supports several cyclic time dimensions. PR #226

Fixed
-----
- Error in computation of LAI values from mapping to landuse in **setup_laimaps**. PR #297
- IO error for write_states in write. PR #297
- Creating the staticgeoms folder if it does not already exist (eg when dir_input is provided). PR #297
- Pedo-transfer function for estimation of residual water content. PR #300

v0.6.1 (16 September 2024)
==========================
This release mainly contains small bugfixes and limits xarray version to 2024.03.0

Added
-----
- Added "fillna_method" option for **setup_temp_pet_forcing** by @tnlim
- Output filenames can now be specified in the model.write function. More detailed arguments should still be specified in each individual write* methods. PR #286

Changed
-------
- Individual methods like write_forcing will not longer write the config file if config settings get updated. Always call write_config as the last write method. PR #286
- More uniform handling of the date typing when reading/writing dates from the wflow toml files. PR #286

Fixed
-----
- Wrong dtype for columns when reading a mapping table in **setup_laimaps_from_lulc_mapping** . PR #290
- Read/write staticgeoms if dir_input folder is present in the wflow toml file. PR #286
- Creating subfolders for the config file of wflow in **write_config**. PR #286
- Fixed access to functions in the **pcrm** module (read_staticmaps_pcr, write_staticmaps_pcr). PR #293
- Bug in **setup_pet_forcing** when doing time resampling. PR #294

v0.6.0 (7 June 2024)
====================
Copious amounts of new features and fixes!

Added
-----
- If applicable, basins geometry based on the higher resolution DEM is stored seperately under **basins_highres** `PR #266 <https://github.com/Deltares/hydromt_wflow/pull/266>`_
- New function **setup_1dmodel_connection** to connect wflow to 1D river model (eg Delft3D FM 1D, HEC-RAS, etc.) `PR #210 <https://github.com/Deltares/hydromt_wflow/pull/210>`_
- New setup method for the **KsatHorFrac** parameter **setup_ksathorfarc** to up-downscale existing ksathorfrac maps. `PR #255 <https://github.com/Deltares/hydromt_wflow/pull/255>`_
- New function **setup_pet_forcing** to reproject existing pet data rather than computing from other meteo data. PR #257
- Workflow to compute brooks corey c for the wflow layers based on soilgrids data, soilgrids_brooks_corey. PR #242
- Better support for WflowModel states with new methods: **read_states**, **write_states** and **clip_states**. PR #252
- **setup_lulcmaps** for wflow_sediment: if planted forest data is available, it can be used to update the values of the USLE C parameter. PR #234
- New function **setup_cold_states** to prepare cold states for WflowModel. PR #252
- New utils method **get_grid_from_config** to get the right wflow staticmaps variable based on the TOML configuration (e.g. detects name in netcdf, value, scale and offset). Only applied now to prepare cold states (e.g. not yet in read_grid). PR #252
- Added support for the "GLCNMO" land-use dataset, with a default parameter mapping table (similar to the existing tables). PR #272
- Added the `alpha_h1` parameter (based on land use maps). This parameter represents whether root water uptake reduction at soil water pressure head h1 occurs or not. By default, it is set  to 0.0 for all "non-natural" vegetation (crops) and to 1.0 for all "natural vegetation" PR #272
- Parameter for output filename in **write_grid** (`fn_out`). PR #278
- New function **setup_laimaps_from_lulc_mapping** to set leaf area index (LAI) climatology maps per month based on landuse mapping. PR #273


Changed
-------
- Basins geometry (**basins**) is now based on the actual wflow model resolution basins, instead of based on the supplied DEM `PR #266 <https://github.com/Deltares/hydromt_wflow/pull/266>`
- **setup_soilmaps**: the user can now supply variable sbm soil layer thicknesses. The Brooks Corey coefficient is then computed as weighted average over the sbm layers. PR #242
- **setup_outlets**: the IDs of wflow_subcatch are used to define the outlets IDs rather than [1:n]. PR #247
- wflow forcing data type should always be float32. PR #268
- **setup_laimaps**: if a landuse map if provided, setup_laimaps can also prepare landuse mapping tables for LAI. PR #273

Fixed
-----
- Wrong dtype for wflow_subcatch map. PR #247
- **setup_gauges**: Allow snapping to river/mask for snap_uparea method. PR #248
- Removed building a wflow model without a config file in the build notebook.
- Deprecated np.bool and earlier error message for subbasin delination. PR #263

Deprecated
----------
- **clip_staticmaps** in favour of **clip_grid**
- **read_staticmaps** and **write_staticmaps**, superseded by **read_grid** and **write_grid**
- **read_staticgeoms** and **write_staticgeoms**, superseded by **read_geoms** and **write_geoms**

v0.5.0 (13 February 2024)
=========================
Better handling of nodata and a switch from ini to yaml for configuration.

Added
-----
- **setup_rivers**: Add river depth based on rivdph columns in river_geom_fn rather than only computed from qbankfull column.

Changed
-------
- Remove default values for data sources in the different setup methods. (PR #227)

Fixed
-----
- **setup_reservoirs**: Fix error if optional columns 'Capacity_norm', 'Capacity_min', 'xout', 'yout' are not in reservoir_fn. Allow to pass kwargs to the get_data method.
- **setup_lulcmaps**: Fix error when looking for mapping_fn in self.data_catalog
- **setup_config_output_timeseries**: bugfix for reducer.
- update hydromt configuration files from ini to yml format. PR #230
- remove or update calls to check if source in self.data_catalog `Issue #501 <https://github.com/Deltares/hydromt/issues/501>`_
- Included NoDataStrategy from hydromt-core: setup functions for lakes, reservoirs, glaciers, and gauges are skipped when no data is found withing the model region (same behavior as before) PR #229

Deprecated
----------
- **read_staticmaps_pcr** in favour of same method in **pcrm** submodule
- **write_staticmaps_pcr** in favour of same method in **pcrm** submodule

Documentation
-------------
- Extra information for most of the setup methods of **WflowModel** and **WflowSedimentModel**

v0.4.1 (22 November 2023)
=========================
Small update

Fixed
-----
- Make HydroMT-Wflow **v0.4.0** conda installable

v0.4.0 (21 November 2023)
=========================
Small overhaul of internal methods and stability fixes. This version works with HydroMT **v0.9.1** onwards.

Changed
-------
- **WflowModel** and **WflowSedimentModel** now rely on `GridModel` from HydroMT
- PCRaster methods are moved to `pcrm` submodule and are deprecated as methods for the **WflowModel** class
- **read_staticgeoms**, **write_staticgeoms** and **staticgeoms** are now deprecated
- Staticgeoms methods are superseded by **read_geoms**, **write_geoms** and **geoms**
- **read_staticmaps**, **write_staticmaps** and **staticmaps** are now deprecated
- Staticmaps methods are superseded by **read_grid**, **write_grid** and **grid**

Fixed
-----
- Mainly stability fixes

v0.3.0 (27 July 2023)
=====================
Various new features and bugfixes in support of Wflow.jl v0.7.1. This version works with HydroMT v0.8.0.

Added
-----
- Support for models in CRS other than 4326. `PR #161 <https://github.com/Deltares/hydromt_wflow/pull/161>`_
- Support for elevation data other than MERIT Hydro in **setup_basemaps**.
- Add options to calculate daily Penman-Monteith potential evaporation using the pyet package. Depending on the available variables, two options are defined ``penman-monteith_tdew`` (inputs: ['temp', 'temp_min', 'temp_max', 'wind_u', 'wind_v', 'temp_dew', 'kin', 'press_msl']) and ``penman-monteith_rh_simple`` (inputs: ['temp', 'temp_min', 'temp_max', 'wind', 'rh', 'kin']).
- Support in toml for dir_input and dir_output options. `PR #140 <https://github.com/Deltares/hydromt_wflow/pull/140>`_
- Add options to calculate daily Penman-Monteith potential evaporation using the pyet package. Depending on the available variables, two options are defined ``penman-monteith_tdew`` (inputs: ['temp', 'temp_min', 'temp_max', 'wind_u', 'wind_v', 'temp_dew', 'kin', 'press_msl']) and ``penman-monteith_rh_simple`` (inputs: ['temp', 'temp_min', 'temp_max', 'wind', 'rh', 'kin']).
- In **setup_reservoirs**: Global Water Watch compatibility for determining reservoir parameters.
- In **setup_reservoirs**: All downloaded reservoir timeseries are saved to root in 1 csv file. Column headers indicate reservoir id.
- **setup_oulets**: Add map/geom of basin outlets (on river or all) and optionally updates outputs in toml file.
- **setup_config_output_timeseries**: add new variable/column to the netcf/csv output section of the toml based on a selected gauge/area map.
- **setup_gauges**: support for snapping based on a user defined max distance and snapping based on upstream area attribute.
- **setup_gauges**: gauges_fn can be both GeoDataFrame or GeoDataset (new) data_type.
- New **setup_floodplains** method, that allows the user the choose either 1D or 2D floodplains. Note: requires pyflwdir v0.5.7. `PR #123 <https://github.com/Deltares/hydromt_wflow/pull/123>`_
- In **setup_lakes**: Add option to prepare rating curve tables for lake Q-V and Q-H curves. Also updated LakeOutFlowFunc and LakeStorFunc accordingly. `PR #158 <https://github.com/Deltares/hydromt_wflow/pull/158>`_
- In **setup_lakes**: Support setting lake parameters from direct value in the lake_fn columns. `PR #158 <https://github.com/Deltares/hydromt_wflow/pull/158>`_
- In **setup_lakes**: Option to prepare controlled lake parameter maxstorage (new in Wflow.jl 0.7.0).
- New workflow **waterbodies.lakeattrs** to prepare lake parameters from lake_fn attribute and rating curve data.
- New **tables** model property including read/write: dictionnary of pandas.DataFrame with model tables (e.g. rating curves of lakes, etc.). `PR #158 <https://github.com/Deltares/hydromt_wflow/pull/158>`_
- Removed hardcoded mapping tables, and added those files an additional .yml file, which is by default read when creating a WflowModel. `PR #168 <https://github.com/Deltares/hydromt_wflow/pull/168>`_

Changed
-------
- Default tomls are now using the dir_output option to specify *run_default* folder.
- in **setup_reservoirs**: options 'usehe' and 'priorityjrc' are removed and replaced with 'timeseries_fn'. Options are ['jrc', 'gww']. By default None to use reservoir_fn data directly.
- in **setup_areamap**: name of the added map is based on column name of the vector data (col2raster) instead of name of the vector data file (area_fn). Allows to add several maps from one vector data file.

Fixed
-----
- Bugfix with wrong nodata value in the hydrography method which caused errors for model which where not based on (sub)basins `PR #144 <https://github.com/Deltares/hydromt_wflow/pull/144>`_
- Bugfix with wrong indexing in the river method that could cause memory issues `PR #147 <https://github.com/Deltares/hydromt_wflow/pull/147>`_
- fix error in **setup_reservoirs** when gdf contains no data in np.nanmax calculation for i.e. damheight #35
- write_forcing with time cftime.DatetimeNoLeap #138 by removing slicing forcing if missings (not needed)
- write_forcing automatic adjustment of starttime and endtime based on forcing content
- When clipping a model from a model with multiple forcing files, a single netcdf is made in write_forcing and the * is removed from the filename.
- Remove deprecated basin_shape method `PR #183 <https://github.com/Deltares/hydromt_wflow/pull/183>`_
- Remove FillValue Nan for lat/lon in staticmaps and forcing `PR #183 <https://github.com/Deltares/hydromt_wflow/pull/183>`_
- Fix compatibility with HydroMT v0.8.0, with updated `clip_geom/mask` functionality `PR #189 <https://github.com/Deltares/hydromt_wflow/pull/189>`_

Deprecated
----------
- The **setup_hydrodem** function has been removed, and the functionality are moved to **setup_rivers** and **setup_floodplains**

Documentation
-------------
- New **prepare_ldd** example notebook to demonstrate how to prepare flow directions and other elevation related data.


v0.2.1 (22 November 2022)
=========================
New setup_staticmaps_from_raster method and river smoothing algorithm. Correct some bugs linked to soon
deprecated staticmaps and staticgeoms objects in hydromt core to work with the new 0.6.0 release.

Added
-----
- Parameters for landuse esa_worlcover. `PR #111 <https://github.com/Deltares/hydromt_wflow/pull/111>`_
- New **setup_staticmaps_from_raster** method. `PR #128 <https://github.com/Deltares/hydromt_wflow/issues/111>`_

Changed
-------
- update forcing example with multiple forcing files #122
- New window smoothing algorithm in `setup_rivers` to avoid cells with small river length.
  Set the min_rivlen_ratio argument to a value larger than zero to apply the smoothing.
  Note: requires pyflwdir v0.5.6 `PR #92 <https://github.com/Deltares/hydromt_wflow/pull/92>`_

Fixed
-----
- write_forcing with time of type cftime.DatetimeNoLeap #109
- write_forcing: re-write config in case of multiple forcing files
- read_forcing with multiple files (* key in toml)
- bug in setup_gauges in update mode with crs.is_epsg_code #108
- bug in self.rivers if no staticgeoms and rivmsk is found #113
- bug in wflow_build_sediment.ini template in examples
- wrong defaults in wflow_build.ini teamplate in examples #116
- temporary fix to update staticgeoms basins+rivers in clip_staticmaps (update when moving away from deprecated staticgeoms).
- fix wrong default value for lai_fn in setup_laimaps #119

Deprecated
----------

v0.2.0 (5 August 2022)
======================
We now use rioxarray to read raster data. We recommend reinstalling your hydromt and hydromt_wflow environment including the rioxarray package.
This enables the writting of CF compliant netcdf files for wflow staticmaps.nc and inmaps.nc.
Following an update in xarray, hydromt version should be >= 0.5.0.

Fixed
-----
- correct float32 dtype for all landuse based maps (by changing values in all lookup tables to floats)
- write **CF-compliant** staticmaps.nc and inmaps.nc
- CRS issue when deriving subcatch for user defined gauges in setup_gauges
- update times in config depending on forcing date range availability in **write_forcing** methods #97

Changed
-------
- In the naming of the generated hydrodem map, it is now specified if a D4 or D8 conditionning has been applied for land cells.
- uint8 dtype *wflow_rivers* and *wflow_streamorder* maps
- except for coordinates (incl *x_out* and *y_out*) all variables are saved with at most 32 bit depth
- new dtype and nodata arguments in **setup_constant_pars**
- read boolean PCRaster maps with int type to be consistent with netcdf based maps
- use latest hydromt github version for the test environment files.
- in **setup_glaciers** predicate to intersects glacier data with model region is 'intersects' (the old 'contains' was not used anyway due to a bug in core).
- in **setup_reservoirs** and **setup_lakes** the predicate 'contains' to open data is now officially used after a bugfix in hydromt core (cf #150).

Added
-----
- nodata argument to **setup_areamap** with a default of -1 (was 0 and not user defined).

v0.1.4 (18 February 2022)
=========================

Changed
-------
- **setup_riverwidth** method **deprecated** (will be removed in future versions) in favour of setup_rivers. We suggest to remove the setup_riverwidth component from your ini files.
- **setup_rivers** calculate river width and depth based on the attributes of the new **river_geom_fn** river geometry file. We suggest adding "river_geom_fn = rivers_lin2019_v1" to the setup_rivers component of your ini files.
- In **setup_soilmaps** the interpolation of missing values (interpolate_na function) is executed on the model parameters at the model resolution, rather than on the original raw soilgrids data at higher resolution. This change will generate small differences in the parameter values, but (largely) improve memory usage.
- Possibility to use any dataset and not just the default ones for setup_laimaps, setup_lakes, setup_glaciers. See the documentation for data requirements.

Added
-----
- Possibility to write_forcing in several files based on time frequency (fn_freq argument).
- setup_hydrodem method for hydrological conditioned elevation used with "local-inertial" routing
- workflow.river.river_bathymetry method to derive river width and depth estimates.
  Note that the new river width estimates are different and result in different model results.
- moved basemaps workflows (hydrography and topography) from HydroMT core. Note that HydroMT_Wflow v0.1.3 there should be used together with HydroMT v0.4.4 (not newer!)
- new ID columns for the outlets staticgeoms
- new ``index_col`` attribute to setup_gauges to choose a specific column of gauges_fn as ID for Wflow_gauges

Fixed
-----
- Calculation of lake_b parameter in setup_lakes.
- Add a minimum averaged discharge to lakes to avoid division by zero when computing lake_b.
- When writting several forcing files instead of one, their time_units should be the same to get one Wflow run (time_units option in write_forcing)
- Filter gauges that could not be snapped to river (if snap_to_river is True) in setup_gauges
- Avoid duplicates in the toml csv column for gauges
- Fill missing values in landslope with zeros within the basin mask
- prevent writing a _FillValue on the time coordinate of forcing data


v0.1.3 (4 October 2021)
=======================
This release adds pyflwdir v0.5 compatibility and a data_catalog of the used data to the write_method.

Added
-----

 - write data_catalog with the used data when writing model
 - tests on staticmaps dtype

Changed
-------

- TOML files only contains reservoir/lake/glacier lines when they are setup and present in the model region.

Fixed
-----
 - pyflwdir v0.5 compatibility: changes from stream order bugfix and improved river slope
 - Fixed docs with rtd v1.0
 - Wrong dtype for Wflow_gauges
 - Removed unnecessary glacier/lake/reservoir lines from the TOML, fixes a bug if missing glacier

v0.1.2 (1 September 2021)
=========================
This release implements the new results attributes for Wflow.

Added
-----

- Add results attributes for Wflow and read_results method (including test+example).
- Add `f_` parameter in soilgrids
- Support soilgrids version 2020
- Setup_areamap component to prepare maps of areas of interest to save Wflow outputs at.
- Support Wflow_sediment with vito landuse.
- New utils.py script for low_level Wflow methods.

Changed
-------

- wfow_sbm.toml remove netcdf output.
- Wflow_soil map is now based on soil texture calculated directly from soilgrids data
- test cases change toml and Wflow_soil.map
- Wflow_sbm.toml now includes links to staticmaps of glacier parameters and outstate of glacierstore is added.

Fixed
-----

- Fix f parameter in soilgrids
- Full reading and writing of Wflow filepaths depending on the toml file (including subfolders).
- The Wflow_gauges now contains river outlets only (instead of all outlets).

Documentation
-------------

- Added Wflow_plot_results example.
- Fixed staticmaps_to_mapstack example.

v0.1.1 (21 May 2021)
====================
This release adds more functionnality for saving forcing data for Wflow and fixes several bugs for some parameter values and soilgrids workflow.

Added
-----

- Write the forcing with user defined chunking on time (default is 1) and none on the lat/lon dimensions (makes Wflow.jl run much faster).
- Rounding of the forcing data with user defined number of decimals (by default 2).
- Progress bar when writing the forcing file.

Changed
-------

- Remove unused imports.

Fixed
-----

- Fixed a mistake in the computation of the lake_b parameter for Wflow.
- Missing no data values for soilgrids workflows.
- Streamorder reclass function for Manning roughness.
- New behavior of apply_ufunc from an update of xarray for passing attributes (need to specify keep_attrs=True).

Documentation
-------------

- Added changelog.

Tests
-----

- Tests without hydroengine for the reservoirs (too long).

v0.1.0 (28 April 2021)
======================
Initial open source release of HydroMT Wflow plugin, also published on pypi. Noticeable changes are listed below.

Added
-----

- Minimum HydroMT plugin template in the **plugin-boilerplate** branch.
- Default filename for the forcing file created by HydroMT (when the one in config already exists).

Changed
-------

- Implement new get_basin_geometry from HydroMT core.
- Consistent setup functions arguments for data sources ('_fn').
- Rename **hydrom_merit** source to **merit_hydro** (updated version of data-artifacts).

Fixed
-----

- Bugs using the clip functions

Documentation
-------------

- Initial version of the documentation on github-pages.
- **Latest** and **stable** version of the documentation.
- Setup Binder environment.
- Add examples notebooks for the documentation.

Tests
-----

- Initial tests for Wflow and Wflow_sediment.

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html
