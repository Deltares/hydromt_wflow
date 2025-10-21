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
   :toctree: ../_generated

   WflowSedimentModel


Setup methods
=============
These methods are used to add data, parameters, and components to your Wflow Sediment model and to build or update it step by step.


Configuration
=============
Defines and manages model configuration, global parameters, and output settings.

.. autosummary::
   :toctree: ../_generated

   WflowSedimentModel.setup_config
   WflowSedimentModel.setup_config_output_timeseries
   WflowSedimentModel.setup_constant_pars


Topography and Rivers
========================
Prepares elevation maps, drainage networks, and sediment-related river properties used to route flow and sediment.

.. autosummary::
   :toctree: ../_generated

   WflowSedimentModel.setup_basemaps
   WflowSedimentModel.setup_rivers
   WflowSedimentModel.setup_riverwidth
   WflowSedimentModel.setup_riverbedsed


Reservoirs
=================
Adds natural and man-made reservoirs and defines their impact on sediment storage and transport.

.. autosummary::
   :toctree: ../_generated

   WflowSedimentModel.setup_natural_reservoirs
   WflowSedimentModel.setup_reservoirs


Land Cover and Soils
========================
Defines vegetation, soil, and land cover maps that influence sediment erosion and deposition processes.

.. autosummary::
   :toctree: ../_generated

   WflowSedimentModel.setup_lulcmaps
   WflowSedimentModel.setup_lulcmaps_from_vector
   WflowSedimentModel.setup_canopymaps
   WflowSedimentModel.setup_soilmaps


Output Locations
====================
Defines model output points and areas such as outlets, gauges, and spatial masks for reporting results.

.. autosummary::
   :toctree: ../_generated

   WflowSedimentModel.setup_outlets
   WflowSedimentModel.setup_gauges
   WflowSedimentModel.setup_areamap


Other Methods
====================
Additional high-level utilities to manage model geometry, upgrade versions, or modify spatial extent.

.. autosummary::
   :toctree: ../_generated

   WflowSedimentModel.setup_grid_from_raster
   WflowSedimentModel.upgrade_to_v1_wflow
   WflowSedimentModel.clip

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



.. autoclass:: hydromt_wflow.WflowSedimentModel
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:
   :no-index:
