What's new
==========
All notable changes to this project will be documented in this page.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

[Unreleased]
------------

Added
^^^^^

- Add f_ parameter in soilgrids 

Fixed
^^^^^

- Fix f parameter in soilgrids

v0.1.1 (21 May 2021)
--------------------
This release adds more functionnality for saving forcing data for wflow and fixes several bugs for some parameter values and soilgrids workflow.

Added
^^^^^

- Write the forcing with user defined chunking on time (default is 1) and none on the lat/lon dimensions (makes Wflow.jl run much faster).
- Rounding of the forcing data with user defined number of decimals (by default 2).
- Progress bar when writting the forcing file.

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
- Consistent setup fonctions arguments for data sources ('_fn').
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