.. currentmodule:: hydromt_wflow

.. _api_reference:

#############
API reference
#############

.. _api_model:

Wflow model classes
===================

Initialize
----------

.. autosummary::
   :toctree: _generated

   WflowBaseModel

.. _components_base:


High level and I/O methods
--------------------------

.. autosummary::
   :toctree: _generated

   WflowBaseModel.build
   WflowBaseModel.update

   WflowBaseModel.read
   WflowBaseModel.write


Components
----------

.. autosummary::
   :toctree: _generated

   WflowBaseModel.config
   WflowBaseModel.staticmaps
   WflowBaseModel.forcing
   WflowBaseModel.states
   WflowBaseModel.tables
   WflowBaseModel.geoms
   WflowBaseModel.output_grid
   WflowBaseModel.output_scalar
   WflowBaseModel.output_csv

Attributes
----------

.. autosummary::
   :toctree: _generated

   WflowBaseModel.crs
   WflowBaseModel.root
   WflowBaseModel.flwdir
   WflowBaseModel.basins
   WflowBaseModel.rivers

Setup methods
-------------

.. autosummary::
   :toctree: _generated

   WflowBaseModel.setup_config
   WflowBaseModel.setup_config_output_timeseries
   WflowBaseModel.setup_basemaps
   WflowBaseModel.setup_rivers
   WflowBaseModel.setup_riverwidth
   WflowBaseModel.setup_lulcmaps
   WflowBaseModel.setup_lulcmaps_from_vector
   WflowBaseModel.setup_outlets
   WflowBaseModel.setup_gauges
   WflowBaseModel.setup_constant_pars
   WflowBaseModel.setup_grid_from_raster
   WflowBaseModel.setup_areamap


Initialize
----------

.. autosummary::
   :toctree: _generated

   WflowSbmModel

.. _components_sbm:

Setup methods
-------------

.. autosummary::
   :toctree: _generated

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

.. _api_model_sediment:

WflowSediment model class
=========================

Initialize
----------

.. autosummary::
   :toctree: _generated

   WflowSedimentModel

.. _components_sediment:

Setup components
----------------

.. autosummary::
   :toctree: _generated

   WflowSedimentModel.setup_config
   WflowSedimentModel.setup_basemaps
   WflowSedimentModel.setup_rivers
   WflowSedimentModel.setup_natural_reservoirs
   WflowSedimentModel.setup_reservoirs
   WflowSedimentModel.setup_lulcmaps
   WflowSedimentModel.setup_lulcmaps_from_vector
   WflowSedimentModel.setup_canopymaps
   WflowSedimentModel.setup_soilmaps
   WflowSedimentModel.setup_riverwidth
   WflowSedimentModel.setup_riverbedsed
   WflowSedimentModel.setup_outlets
   WflowSedimentModel.setup_gauges
   WflowSedimentModel.setup_areamap
   WflowSedimentModel.setup_config_output_timeseries
   WflowSedimentModel.setup_constant_pars
   WflowSedimentModel.setup_grid_from_raster
   WflowSedimentModel.upgrade_to_v1_wflow
   WflowSedimentModel.clip

.. _data_containers:

WflowBaseModel components
=========================

WflowConfigComponent
--------------------

.. autosummary::
   :toctree: _generated

   components.WflowConfigComponent
   components.WflowConfigComponent.data
   components.WflowConfigComponent.get_value
   components.WflowConfigComponent.remove

WflowStaticmapsComponent
------------------------

.. autosummary::
   :toctree: _generated

   components.WflowStaticmapsComponent
   components.WflowStaticmapsComponent.data
   components.WflowStaticmapsComponent.drop_vars

WflowForcingComponent
------------------------

.. autosummary::
   :toctree: _generated

   components.WflowForcingComponent
   components.WflowForcingComponent.data

WflowGeomsComponent
-------------------

.. autosummary::
   :toctree: _generated

   components.WflowGeomsComponent
   components.WflowGeomsComponent.data
   components.WflowGeomsComponent.get

WflowStatesComponent
--------------------

.. autosummary::
   :toctree: _generated

   components.WflowStatesComponent
   components.WflowStatesComponent.data

WflowTablesComponent
--------------------

.. autosummary::
   :toctree: _generated

   components.WflowTablesComponent
   components.WflowTablesComponent.data

WflowOutputGridComponent
------------------------

.. autosummary::
   :toctree: _generated

   components.WflowOutputGridComponent
   components.WflowOutputGridComponent.data
   components.WflowOutputGridComponent.read

WflowOutputScalarComponent
---------------------------

.. autosummary::
   :toctree: _generated

   components.WflowOutputScalarComponent
   components.WflowOutputScalarComponent.data
   components.WflowOutputScalarComponent.read

WflowOutputCsvComponent
------------------------

.. autosummary::
   :toctree: _generated

   components.WflowOutputCsvComponent
   components.WflowOutputCsvComponent.data
   components.WflowOutputCsvComponent.read

.. _workflows:

Wflow workflows
===============

.. autosummary::
   :toctree: _generated

   workflows.allocation_areas
   workflows.surfacewaterfrac_used
   workflows.domestic
   workflows.domestic_from_population
   workflows.other_demand
   workflows.irrigation
   workflows.irrigation_from_vector
   workflows.hydrography
   workflows.topography
   workflows.river
   workflows.river_bathymetry
   workflows.pet
   workflows.spatial_interpolation
   workflows.landuse
   workflows.landuse_from_vector
   workflows.lai
   workflows.create_lulc_lai_mapping_table
   workflows.lai_from_lulc_mapping
   workflows.add_paddy_to_landuse
   workflows.add_planted_forest_to_landuse
   workflows.ksatver_vegetation
   workflows.soilgrids
   workflows.soilgrids_sediment
   workflows.soilgrids_brooks_corey
   workflows.update_soil_with_paddy
   workflows.reservoir_id_maps
   workflows.reservoir_simple_control_parameters
   workflows.reservoir_parameters
   workflows.merge_reservoirs
   workflows.merge_reservoirs_sediment
   workflows.create_reservoirs_geoms
   workflows.create_reservoirs_geoms_sediment
   workflows.glaciermaps
   workflows.glacierattrs
   workflows.rootzoneclim
   workflows.wflow_1dmodel_connection
   workflows.prepare_cold_states


.. _methods:

Wflow low-level methods
=======================

Input/Output methods
---------------------

.. autosummary::
   :toctree: _generated

   utils.read_csv_output

Utility methods
---------------

.. autosummary::
   :toctree: _generated

   utils.get_grid_from_config
