.. currentmodule:: hydromt_wflow

.. _api_wflowsbmmodel:

=============
WflowSbmModel
=============

The ``WflowSbmModel`` class represents the main hydrological model implementation.

.. note::
   Note that this class inherits from :class:`~hydromt_wflow.WflowBaseModel` and thereby includes all base functionalities, I/O routines, and setup methods from the base model.
   Also, note that some setup methods are overridden or extended to include soil-based-specific parameters and data.
   For a full list of inherited methods and attributes, see :ref:`WflowBaseModel <api_base_model>`.

.. autosummary::
   :toctree: ../_generated

   WflowSbmModel


Setup methods
-------------

.. autosummary::
   :toctree: ../_generated

   WflowSbmModel.setup_config
   WflowSbmModel.setup_basemaps
   WflowSbmModel.setup_rivers
   WflowSbmModel.setup_floodplains
   WflowSbmModel.setup_reservoirs_no_control
   WflowSbmModel.setup_reservoirs_simple_control
   WflowSbmModel.setup_glaciers
   WflowSbmModel.setup_lulcmaps
   WflowSbmModel.setup_lulcmaps_from_vector
   WflowSbmModel.setup_lulcmaps_with_paddy
   WflowSbmModel.setup_laimaps
   WflowSbmModel.setup_laimaps_from_lulc_mapping
   WflowSbmModel.setup_allocation_areas
   WflowSbmModel.setup_allocation_surfacewaterfrac
   WflowSbmModel.setup_domestic_demand
   WflowSbmModel.setup_domestic_demand_from_population
   WflowSbmModel.setup_other_demand
   WflowSbmModel.setup_irrigation
   WflowSbmModel.setup_irrigation_from_vector
   WflowSbmModel.setup_ksathorfrac
   WflowSbmModel.setup_ksatver_vegetation
   WflowSbmModel.setup_rootzoneclim
   WflowSbmModel.setup_soilmaps
   WflowSbmModel.setup_outlets
   WflowSbmModel.setup_gauges
   WflowSbmModel.setup_areamap
   WflowSbmModel.setup_config_output_timeseries
   WflowSbmModel.setup_precip_forcing
   WflowSbmModel.setup_precip_from_point_timeseries
   WflowSbmModel.setup_temp_pet_forcing
   WflowSbmModel.setup_pet_forcing
   WflowSbmModel.setup_constant_pars
   WflowSbmModel.setup_1dmodel_connection
   WflowSbmModel.setup_grid_from_raster
   WflowSbmModel.setup_cold_states
   WflowSbmModel.upgrade_to_v1_wflow
   WflowSbmModel.clip

.. autoclass:: hydromt_wflow.WflowSbmModel
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:
