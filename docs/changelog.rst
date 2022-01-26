What's new
==========
All notable changes to this project will be documented in this page.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

[Unreleased]
------------

Added
^^^^^
- Possibility to write_forcing in several files based on time frequency (fn_freq argument).
- setup_hydrodem method for hydrological conditioned elevation used with "local-inertial" routing
- workflow.river.river_bathymetry method to derive river width and depth estimates. 
  Note that the new river width estimates are different and result in different model results.
- moved basemaps workflows (hydrography and topography) from hydromt core.

Fixed
^^^^^
- Calculation of lake_b parameter in setup_lakes.

Changed
^^^^^^^^
- setup_riverwidth method deprecated (will be removed in future versions) in favour of setup_rivers.
- setup_rivers takes an additional river_geom_fn argument with a river segment geometry file to calculate river width and depth from its attributes

v0.1.3 (4 October 2021)
-------------------------
This release adds pyflwdir v0.5 compatibility and a data_catalog of the used data to the write_method.

Added
^^^^^

 - write data_catalog with the used data when writing model
 - tests on staticmaps dtype

Changed
^^^^^^^

- TOML files only contains reservoir/lake/glacier lines when they are setup and present in the model region.

Fixed
^^^^^
 - pyflwdir v0.5 compatibility: changes from stream order bugfix and improved river slope
 - Fixed docs with rtd v1.0
 - Wrong dtype for wflow_gauges
 - Removed unnecessary glacier/lake/reservoir lines from the TOML, fixes a bug if missing glacier

v0.1.2 (1 September 2021)
-------------------------
This release implements the new results attributes for Wflow.

Added
^^^^^

- Add results attributes for wflow and read_results method (including test+example).
- Add f_ parameter in soilgrids 
- Support soilgrids version 2020
- Setup_areamap component to prepare maps of areas of interest to save wflow outputs at.
- Support wflow_sediment with vito landuse.
- New utils.py script for low_level wflow methods.

Changed
^^^^^^^

- wfow_sbm.toml remove netcdf output.
- wflow_soil map is now based on soil texture calculated directly from soilgrids data
- test cases change toml and wflow_soil.map
- wflow_sbm.toml now includes links to staticmaps of glacier parameters and outstate of glacierstore is added.

Fixed
^^^^^

- Fix f parameter in soilgrids
- Full reading and writing of wflow filepaths depending on the toml file (including subfolders).
- The wflow_gauges now contains river outlets only (instead of all outlets).

Documentation
^^^^^^^^^^^^^

- Added wflow_plot_results example.
- Fixed staticmaps_to_mapstack example.

v0.1.1 (21 May 2021)
--------------------
This release adds more functionnality for saving forcing data for wflow and fixes several bugs for some parameter values and soilgrids workflow.

Added
^^^^^

- Write the forcing with user defined chunking on time (default is 1) and none on the lat/lon dimensions (makes Wflow.jl run much faster).
- Rounding of the forcing data with user defined number of decimals (by default 2).
- Progress bar when writing the forcing file.

Changed
^^^^^^^

- Remove unused imports.

Fixed
^^^^^

- Fixed a mistake in the computation of the lake_b parameter for wflow.
- Missing no data values for soilgrids workflows.
- Streamorder reclass function for Manning roughness.
- New behavior of apply_ufunc from an update of xarray for passing attributes (need to specify keep_attrs=True).

Documentation
^^^^^^^^^^^^^

- Added changelog.

Tests
^^^^^

- Tests without hydroengine for the reservoirs (too long).

v0.1.0 (28 April 2021)
----------------------
Initial open source release of hydroMT wflow plugin, also published on pypi. Noticeable changes are listed below.

Added
^^^^^

- Minimum hydroMT plugin template in the **plugin-boilerplate** branch.
- Default filename for the forcing file created by hydromt (when the one in config already exists).

Changed
^^^^^^^

- Implement new get_basin_geometry from hydromt core.
- Consistent setup functions arguments for data sources ('_fn').
- Rename **hydrom_merit** source to **merit_hydro** (updated version of data-artifacts).

Fixed
^^^^^

- Bugs using the clip functions

Documentation
^^^^^^^^^^^^^

- Initial version of the documentation on github-pages.
- **Latest** and **stable** version of the documentation.
- Setup Binder environment.
- Add examples notebooks for the documentation.

Tests
^^^^^

- Initial tests for wflow and wflow_sediment.

.. _Keep a Changelog: https://keepachangelog.com/en/1.0.0/
.. _Semantic Versioning: https://semver.org/spec/v2.0.0.html