==========
What's new
==========
All notable changes to this project will be documented in this page.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

Unreleased
==========

Added
-----
- New function **setup_1dmodel_connection** to connect wflow to 1D river model (eg Delft3D FM 1D, HEC-RAS, etc.) `PR #210 <https://github.com/Deltares/hydromt_wflow/pull/210>`_
- New setup method for the **KsatHorFrac** parameter **setup_ksathorfarc** to up-downscale existing ksathorfrac maps. `PR #255 <https://github.com/Deltares/hydromt_wflow/pull/255>`_
- new function **setup_pet_forcing** to reproject existing pet data rather than computing from other meteo data. PR #257
- Workflow to compute brooks corey c for the wflow layers based on soilgrids data, soilgrids_brooks_corey. PR #242

Changed
-------
- **setup_soilmaps**: the user can now supply variable sbm soil layer thicknesses. The Brooks Corey coefficient is then computed as weighted average over the sbm layers. PR #242
- **setup_outlets**: the IDs of wflow_subcatch are used to define the outlets IDs rather than [1:n]. PR #247

Fixed
-----
- Wrong dtype for wflow_subcatch map. PR #247,
- Removed building a wflow model without a config file in the build notebook.

v0.5.0 (February 2024)
======================
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

v0.4.1 (November 2023)
======================
Small update

Fixed
-----
- Make HydroMT-Wflow **v0.4.0** conda installable

v0.4.0 (November 2023)
======================
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

v0.3.0 (July 2023)
==================
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
