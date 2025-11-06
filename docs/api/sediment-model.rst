.. currentmodule:: hydromt_wflow

.. _api_wflowsedimentmodel:

==================
WflowSedimentModel
==================

The ``WflowSedimentModel`` class extends Wflow functionality with sediment transport modeling.

.. note::
   Note that this class inherits from :class:`~hydromt_wflow.WflowBaseModel` and thereby includes all base functionalities, I/O routines, and setup methods from the base model.
   Also, note that some setup methods are overridden or extended to include sediment-specific parameters and data.
   For a full list of inherited methods and attributes, see :ref:`WflowBaseModel <api_base_model>`.

.. autosummary::
   :toctree: _generated

   WflowSedimentModel


Setup methods
=============
These methods are used to add data, parameters, and components to your Wflow Sediment model and to build or update it step by step.


Configuration
-------------
Defines and manages model configuration, global parameters, and output settings.

.. autosummary::
   :toctree: _generated

   WflowSedimentModel.setup_config
   WflowSedimentModel.setup_config_output_timeseries
   WflowSedimentModel.setup_constant_pars


Topography and Rivers
---------------------
Prepares elevation maps, drainage networks, and sediment-related river properties used to route flow and sediment.

.. autosummary::
   :toctree: _generated

   WflowSedimentModel.setup_basemaps
   WflowSedimentModel.setup_rivers
   WflowSedimentModel.setup_riverwidth
   WflowSedimentModel.setup_riverbedsed


Reservoirs
----------
Adds natural and man-made reservoirs and defines their impact on sediment storage and transport.

.. autosummary::
   :toctree: _generated

   WflowSedimentModel.setup_natural_reservoirs
   WflowSedimentModel.setup_reservoirs


Land Use and Vegetation
-----------------------
Defines land use and vegetation properties that influence sediment erosion and deposition processes.

.. autosummary::
   :toctree: _generated

   WflowSedimentModel.setup_lulcmaps
   WflowSedimentModel.setup_lulcmaps_from_vector
   WflowSedimentModel.setup_canopymaps

Soil
----
Sets up soil-related data including soil maps.

.. autosummary::
   :toctree: _generated

   WflowSedimentModel.setup_soilmaps


Output Locations
----------------
Defines model output points and areas such as outlets, gauges, and spatial masks for reporting results.

.. autosummary::
   :toctree: _generated

   WflowSedimentModel.setup_outlets
   WflowSedimentModel.setup_gauges
   WflowSedimentModel.setup_areamap


Other Methods
-------------
Additional high-level utilities to manage model geometry, upgrade versions, or modify spatial extent.

.. autosummary::
   :toctree: _generated

   WflowSedimentModel.setup_grid_from_raster
   WflowSedimentModel.upgrade_to_v1_wflow
   WflowSedimentModel.clip

Components
==========

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


Component-level API
===================

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
