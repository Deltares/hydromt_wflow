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

   WflowSbmModel.read_config
   WflowSbmModel.write_config

   WflowSbmModel.read_grid
   WflowSbmModel.write_grid

   WflowSbmModel.read_geoms
   WflowSbmModel.write_geoms

   WflowSbmModel.read_forcing
   WflowSbmModel.write_forcing

   WflowSbmModel.read_states
   WflowSbmModel.write_states

   WflowSbmModel.read_outputs

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

Other general methods
---------------------

If you are using python, you can also use the general methods below to set or get model components and their data.

.. autosummary::
   :toctree: _generated

   WflowSbmModel.get_config
   WflowSbmModel.set_config
   WflowSbmModel.set_forcing
   WflowSbmModel.set_grid
   WflowSbmModel.set_geoms
   WflowSbmModel.set_states
   WflowSbmModel.set_tables
   WflowSbmModel.set_flwdir


.. autoclass:: hydromt_wflow.WflowSbmModel
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:
   :no-index:
