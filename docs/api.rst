.. currentmodule:: hydromt_wflow

.. _api_reference:

#############
API reference
#############

.. _api_model:

Wflow model class
=================

Initialize
----------

.. autosummary::
   :toctree: _generated

   WflowModel

.. _components:

Setup methods
-------------

.. autosummary::
   :toctree: _generated

   WflowModel.setup_config
   WflowModel.setup_basemaps
   WflowModel.setup_rivers
   WflowModel.setup_floodplains
   WflowModel.setup_reservoirs_no_control
   WflowModel.setup_reservoirs_simple_control
   WflowModel.setup_glaciers
   WflowModel.setup_lulcmaps
   WflowModel.setup_lulcmaps_from_vector
   WflowModel.setup_lulcmaps_with_paddy
   WflowModel.setup_laimaps
   WflowModel.setup_laimaps_from_lulc_mapping
   WflowModel.setup_allocation_areas
   WflowModel.setup_allocation_surfacewaterfrac
   WflowModel.setup_domestic_demand
   WflowModel.setup_domestic_demand_from_population
   WflowModel.setup_other_demand
   WflowModel.setup_irrigation
   WflowModel.setup_irrigation_from_vector
   WflowModel.setup_ksathorfrac
   WflowModel.setup_ksatver_vegetation
   WflowModel.setup_rootzoneclim
   WflowModel.setup_soilmaps
   WflowModel.setup_outlets
   WflowModel.setup_gauges
   WflowModel.setup_areamap
   WflowModel.setup_config_output_timeseries
   WflowModel.setup_precip_forcing
   WflowModel.setup_precip_from_point_timeseries
   WflowModel.setup_temp_pet_forcing
   WflowModel.setup_pet_forcing
   WflowModel.setup_constant_pars
   WflowModel.setup_1dmodel_connection
   WflowModel.setup_grid_from_raster
   WflowModel.setup_cold_states
   WflowModel.upgrade_to_v1_wflow

High level and I/O methods
--------------------------

.. autosummary::
   :toctree: _generated

   WflowModel.build
   WflowModel.update

   WflowModel.read
   WflowModel.write

   WflowModel.read_config
   WflowModel.write_config

   WflowModel.read_grid
   WflowModel.write_grid
   WflowModel.clip_grid

   WflowModel.read_geoms
   WflowModel.write_geoms

   WflowModel.read_forcing
   WflowModel.write_forcing
   WflowModel.clip_forcing

   WflowModel.read_states
   WflowModel.write_states
   WflowModel.clip_states

   WflowModel.read_outputs

   WflowModel.read_tables
   WflowModel.write_tables

Components
----------

.. autosummary::
   :toctree: _generated

   WflowModel.config
   WflowModel.staticmaps
   WflowModel.forcing
   WflowModel.states
   WflowModel.tables
   WflowModel.geoms
   WflowModel.output_grid
   WflowModel.output_scalar
   WflowModel.output_csv

Attributes
----------

.. autosummary::
   :toctree: _generated

   WflowModel.crs
   WflowModel.root
   WflowModel.flwdir
   WflowModel.basins
   WflowModel.rivers

Other general methods
---------------------

.. autosummary::
   :toctree: _generated

   WflowModel.set_config
   WflowModel.get_config
   WflowModel.set_forcing
   WflowModel.set_grid
   WflowModel.set_tables
   WflowModel.set_states

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
   WflowSedimentModel.setup_laimaps
   WflowSedimentModel.setup_laimaps_from_lulc_mapping
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

High level and I/O methods
--------------------------

.. autosummary::
   :toctree: _generated

   WflowSedimentModel.build
   WflowSedimentModel.update

   WflowSedimentModel.read
   WflowSedimentModel.write

   WflowSedimentModel.read_config
   WflowSedimentModel.write_config

   WflowSedimentModel.read_grid
   WflowSedimentModel.write_grid
   WflowSedimentModel.clip_grid

   WflowSedimentModel.read_geoms
   WflowSedimentModel.write_geoms

   WflowSedimentModel.read_forcing
   WflowSedimentModel.write_forcing
   WflowSedimentModel.clip_forcing

   WflowSedimentModel.read_states
   WflowSedimentModel.write_states

   WflowSedimentModel.read_outputs

Components
----------

.. autosummary::
   :toctree: _generated

   WflowSedimentModel.config
   WflowSedimentModel.staticmaps
   WflowSedimentModel.forcing
   WflowSedimentModel.states
   WflowSedimentModel.tables
   WflowSedimentModel.geoms
   WflowSedimentModel.output_grid
   WflowSedimentModel.output_scalar
   WflowSedimentModel.output_csv

Attributes
----------

.. autosummary::
   :toctree: _generated

   WflowSedimentModel.crs
   WflowSedimentModel.root
   WflowSedimentModel.flwdir
   WflowSedimentModel.basins
   WflowSedimentModel.rivers

Other general methods
---------------------

.. autosummary::
   :toctree: _generated

   WflowSedimentModel.set_config
   WflowSedimentModel.get_config
   WflowSedimentModel.set_forcing
   WflowSedimentModel.set_grid
   WflowSedimentModel.set_states

.. _data_containers:

WflowModel components
=====================

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

WflowOutputCSVComponent
------------------------

.. autosummary::
   :toctree: _generated

   components.WflowOutputCSVComponent
   components.WflowOutputCSVComponent.data
   components.WflowOutputCSVComponent.read

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
