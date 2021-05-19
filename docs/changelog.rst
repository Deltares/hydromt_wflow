What's new
==========
All notable changes to this project will be documented in this page.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

[Unreleased]
------------

Fixed
^^^^^

- Fixed a mistake in the computation of the lake_b parameter for wflow.

Documentation
^^^^^^^^^^^^^

- Added changelog.

v0.1.0 (28 April 2021)
--------------------
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