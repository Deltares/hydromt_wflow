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
- Parameters for landuse esa_worlcover. `PR #111 <https://github.com/Deltares/hydromt_wflow/pull/111>`_
- In **setup_reservoirs**: Global Water Watch compatibility for determining reservoir parameters.
- In **setup_reservoirs**: All dowloaded reservoir timeseries are saved to root in 1 csv file. Column headers indicate reservoir id.

Changed
-------
- in [setup_reservoirs]: options 'usehe' and 'priorityjrc' are removed and replaced with 'timeseries_fn'. Options are ['jrc', 'gww']. By default Global Water Watch is used for downloading reservoir timeseries.

Fixed
-----
- write_forcing with time of type cftime.DatetimeNoLeap #109
- bug in setup_gauges in update mode with crs.is_epsg_code #108
- bug in self.rivers if no staticgeoms and rivmsk is found #113
- bug in wflow_build_sediment.ini template in examples
- temporary fix to update staticgeoms basins+rivers in clip_staticmaps (update when moving away from deprecated staticgeoms). 
- fix wrong default value for lai_fn in setup_laimaps #119
- fix error in **setup_reservoirs** when gdf contains no data in np.nanmax calculation for i.e. damheight #35

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
