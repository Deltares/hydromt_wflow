.. currentmodule:: hydromt_wflow

.. _api_wflowsbmmodel:

=============
WflowSbmModel
=============

The ``WflowSbmModel`` class represents the main hydrological model implementation using the SBM concept.

.. note::
   Note that this class inherits from :class:`~hydromt_wflow.WflowBaseModel` and thereby includes all base functionalities, I/O routines, and setup methods from the base model.
   Also, note that some setup methods are overridden or extended to include SBM-specific parameters and data.
   To know more about this, see :ref:`WflowBaseModel <api_base_model>`.

.. autosummary::
   :toctree: _generated

   WflowSbmModel


Setup methods
-------------

Configuration
=============
Defines and manages model configuration, global parameters, and output settings.

.. autosummary::
   :toctree: _generated

   WflowSbmModel.setup_config
   WflowSbmModel.setup_config_output_timeseries
   WflowSbmModel.setup_constant_pars


Topography and Rivers
=====================
Prepares elevation maps, drainage networks, and river-related features used to simulate flow routing and floodplain processes.

.. autosummary::
   :toctree: _generated

   WflowSbmModel.setup_basemaps
   WflowSbmModel.setup_rivers
   WflowSbmModel.setup_riverwidth
   WflowSbmModel.setup_river_roughness
   WflowSbmModel.setup_floodplains


Reservoirs and Glaciers
=======================
Adds reservoirs and glaciers, and defines their impact on hydrological storage and flow regulation.

.. autosummary::
   :toctree: _generated

   WflowSbmModel.setup_reservoirs_no_control
   WflowSbmModel.setup_reservoirs_simple_control
   WflowSbmModel.setup_glaciers


Land Use and Vegetation
=======================
Defines land use and vegetation properties, including LULC and LAI maps, which influence evapotranspiration and interception processes.

.. autosummary::
   :toctree: _generated

   WflowSbmModel.setup_lulcmaps
   WflowSbmModel.setup_lulcmaps_from_vector
   WflowSbmModel.setup_lulcmaps_with_paddy
   WflowSbmModel.setup_laimaps
   WflowSbmModel.setup_laimaps_from_lulc_mapping
   WflowSbmModel.setup_rootzoneclim


Soil
====
Sets up soil-related data including soil maps and hydraulic properties.

.. autosummary::
   :toctree: _generated

   WflowSbmModel.setup_soilmaps
   WflowSbmModel.setup_ksathorfrac
   WflowSbmModel.setup_ksatver_vegetation


Water Demands and Allocation
============================
Defines domestic, irrigation, and other water demand maps and allocation parameters.

.. autosummary::
   :toctree: _generated

   WflowSbmModel.setup_allocation_areas
   WflowSbmModel.setup_allocation_surfacewaterfrac
   WflowSbmModel.setup_domestic_demand
   WflowSbmModel.setup_domestic_demand_from_population
   WflowSbmModel.setup_other_demand
   WflowSbmModel.setup_irrigation
   WflowSbmModel.setup_irrigation_from_vector


Forcing
=======
Sets up meteorological forcing inputs such as precipitation, temperature, and potential evapotranspiration.

.. autosummary::
   :toctree: _generated

   WflowSbmModel.setup_precip_forcing
   WflowSbmModel.setup_precip_from_point_timeseries
   WflowSbmModel.setup_temp_pet_forcing
   WflowSbmModel.setup_pet_forcing


States
======
Defines initial hydrological state variables such as soil moisture and groundwater storage.

.. autosummary::
   :toctree: _generated

   WflowSbmModel.setup_cold_states


Output Locations
================
Defines outlets, gauges, and spatial masks used for reporting model results.

.. autosummary::
   :toctree: _generated

   WflowSbmModel.setup_outlets
   WflowSbmModel.setup_gauges
   WflowSbmModel.setup_areamap


Other Setup Methods
===================
Additional high-level utilities to modify model geometry, link external models, or upgrade model versions.

.. autosummary::
   :toctree: _generated

   WflowSbmModel.setup_grid_from_raster
   WflowSbmModel.setup_1dmodel_connection
   WflowSbmModel.upgrade_to_v1_wflow
   WflowSbmModel.clip


Components
----------

If you are using python, you can access and update the model data using the components such as ``config``, ``staticmaps`` etc.
This is for example useful for computing statictics, plotting etc.
The components data are usually xarray, dictionnary or geopandas objects and can be accessed via the data property: `model.staticmaps.data`.
The components of the WflowSbmModel are:


+--------------------------+-----------------------------------------------------------------+
| **Model Component**      | **Component class**                                             |
+==========================+=================================================================+
| ``model.config``         | :class:`~hydromt_wflow.components.WflowConfigComponent`         |
+--------------------------+-----------------------------------------------------------------+
| ``model.staticmaps``     | :class:`~hydromt_wflow.components.WflowStaticmapsComponent`     |
+--------------------------+-----------------------------------------------------------------+
| ``model.geoms``          | :class:`~hydromt_wflow.components.WflowGeomsComponent`          |
+--------------------------+-----------------------------------------------------------------+
| ``model.forcing``        | :class:`~hydromt_wflow.components.WflowForcingComponent`        |
+--------------------------+-----------------------------------------------------------------+
| ``model.states``         | :class:`~hydromt_wflow.components.WflowStatesComponent`         |
+--------------------------+-----------------------------------------------------------------+
| ``model.tables``         | :class:`~hydromt_wflow.components.WflowTablesComponent`         |
+--------------------------+-----------------------------------------------------------------+
| ``model.output_grid``    | :class:`~hydromt_wflow.components.WflowOutputGridComponent`     |
+--------------------------+-----------------------------------------------------------------+
| ``model.output_scalar``  | :class:`~hydromt_wflow.components.WflowOutputScalarComponent`   |
+--------------------------+-----------------------------------------------------------------+
| ``model.output_csv``     | :class:`~hydromt_wflow.components.WflowOutputCsvComponent`      |
+--------------------------+-----------------------------------------------------------------+

I/O methods
-------------

If you are using python, you can read and write the different model components using the methods below.

.. autosummary::
   :toctree: _generated

   WflowSbmModel.build
   WflowSbmModel.update

   WflowSbmModel.read
   WflowSbmModel.write

Component-level API
-------------------

The table below summarizes the important methods and attributes for each component.
These allow for fine-grained reading, writing, modification, and inspection of component data.
They are particularly useful when working interactively in Python, for example when updating
specific configuration parameters, clipping static maps, or inspecting the forcing data.

Each component exposes a ``data`` attribute, which holds the underlying model data
(e.g. :class:`dict`, :class:`xarray.Dataset`, or :class:`geopandas.GeoDataFrame`),
and supports a common set of I/O and manipulation methods such as
:meth:`read`, :meth:`write`, and :meth:`set`.

For general I/O at the model level, refer to:
:class:`~hydromt_wflow.model.WflowSbmModel` and its
:meth:`~hydromt_wflow.model.WflowSbmModel.read` and
:meth:`~hydromt_wflow.model.WflowSbmModel.write` methods.

The following table provides a detailed overview of the component-level APIs.

+-----------------------------------------------------------------+-----------------------------------------------------------------------------+-------------------------------------------------------------------+
| **Component class**                                             | **Methods**                                                                 | **Attributes**                                                    |
+=================================================================+=============================================================================+===================================================================+
| :class:`~hydromt_wflow.components.WflowConfigComponent`         | :meth:`~hydromt_wflow.components.WflowConfigComponent.read`,                | :attr:`~hydromt_wflow.components.WflowConfigComponent.data`       |
|                                                                 | :meth:`~hydromt_wflow.components.WflowConfigComponent.write`,               |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowConfigComponent.get_value`,           |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowConfigComponent.set`,                 |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowConfigComponent.update`,              |                                                                   |
+-----------------------------------------------------------------+-----------------------------------------------------------------------------+-------------------------------------------------------------------+
| :class:`~hydromt_wflow.components.WflowStaticmapsComponent`     | :meth:`~hydromt_wflow.components.WflowStaticmapsComponent.read`,            | :attr:`~hydromt_wflow.components.WflowStaticmapsComponent.data`   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowStaticmapsComponent.write`,           |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowStaticmapsComponent.set`,             |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowStaticmapsComponent.clip`,            |                                                                   |
+-----------------------------------------------------------------+-----------------------------------------------------------------------------+-------------------------------------------------------------------+
| :class:`~hydromt_wflow.components.WflowGeomsComponent`          | :meth:`~hydromt_wflow.components.WflowGeomsComponent.read`,                 | :attr:`~hydromt_wflow.components.WflowGeomsComponent.data`        |
|                                                                 | :meth:`~hydromt_wflow.components.WflowGeomsComponent.write`,                |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowGeomsComponent.get`,                  |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowGeomsComponent.set`,                  |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowGeomsComponent.pop`,                  |                                                                   |
+-----------------------------------------------------------------+-----------------------------------------------------------------------------+-------------------------------------------------------------------+
| :class:`~hydromt_wflow.components.WflowForcingComponent`        | :meth:`~hydromt_wflow.components.WflowForcingComponent.read`,               | :attr:`~hydromt_wflow.components.WflowForcingComponent.data`      |
|                                                                 | :meth:`~hydromt_wflow.components.WflowForcingComponent.write`,              |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowForcingComponent.set`,                |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowForcingComponent.clip`,               |                                                                   |
+-----------------------------------------------------------------+-----------------------------------------------------------------------------+-------------------------------------------------------------------+
| :class:`~hydromt_wflow.components.WflowStatesComponent`         | :meth:`~hydromt_wflow.components.WflowStatesComponent.read`,                | :attr:`~hydromt_wflow.components.WflowStatesComponent.data`       |
|                                                                 | :meth:`~hydromt_wflow.components.WflowStatesComponent.write`,               |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowStatesComponent.set`,                 |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowStatesComponent.clip`,                |                                                                   |
+-----------------------------------------------------------------+-----------------------------------------------------------------------------+-------------------------------------------------------------------+
| :class:`~hydromt_wflow.components.WflowTablesComponent`         | :meth:`~hydromt_wflow.components.WflowTablesComponent.read`,                | :attr:`~hydromt_wflow.components.WflowTablesComponent.data`       |
|                                                                 | :meth:`~hydromt_wflow.components.WflowTablesComponent.write`,               |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowTablesComponent.set`,                 |                                                                   |
+-----------------------------------------------------------------+-----------------------------------------------------------------------------+-------------------------------------------------------------------+
| :class:`~hydromt_wflow.components.WflowOutputGridComponent`     | :meth:`~hydromt_wflow.components.WflowOutputGridComponent.read`,            | :attr:`~hydromt_wflow.components.WflowOutputGridComponent.data`   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowOutputGridComponent.write`,           |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowOutputGridComponent.set`,             |                                                                   |
+-----------------------------------------------------------------+-----------------------------------------------------------------------------+-------------------------------------------------------------------+
| :class:`~hydromt_wflow.components.WflowOutputScalarComponent`   | :meth:`~hydromt_wflow.components.WflowOutputScalarComponent.read`,          | :attr:`~hydromt_wflow.components.WflowOutputScalarComponent.data` |
|                                                                 | :meth:`~hydromt_wflow.components.WflowOutputScalarComponent.write`,         |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowOutputScalarComponent.set`,           |                                                                   |
+-----------------------------------------------------------------+-----------------------------------------------------------------------------+-------------------------------------------------------------------+
| :class:`~hydromt_wflow.components.WflowOutputCsvComponent`      | :meth:`~hydromt_wflow.components.WflowOutputCsvComponent.read`,             | :attr:`~hydromt_wflow.components.WflowOutputCsvComponent.data`    |
|                                                                 | :meth:`~hydromt_wflow.components.WflowOutputCsvComponent.write`,            |                                                                   |
|                                                                 | :meth:`~hydromt_wflow.components.WflowOutputCsvComponent.set`,              |                                                                   |
+-----------------------------------------------------------------+-----------------------------------------------------------------------------+-------------------------------------------------------------------+

Attributes
----------

Other useful model attributes if you are using python:

.. autosummary::
   :toctree: _generated

   WflowSbmModel.crs
   WflowSbmModel.root
   WflowSbmModel.flwdir
   WflowSbmModel.basins
   WflowSbmModel.rivers
