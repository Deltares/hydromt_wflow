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
   :toctree: ../generated/

   WflowModel

.. _components:

Setup components
----------------

.. autosummary::
   :toctree: ../generated/

   WflowModel.setup_config
   WflowModel.setup_basemaps
   WflowModel.setup_rivers
   WflowModel.setup_lakes
   WflowModel.setup_reservoirs
   WflowModel.setup_glaciers
   WflowModel.setup_lulcmaps
   WflowModel.setup_laimaps
   WflowModel.setup_soilmaps
   WflowModel.setup_hydrodem
   WflowModel.setup_gauges
   WflowModel.setup_areamap
   WflowModel.setup_precip_forcing
   WflowModel.setup_temp_pet_forcing
   WflowModel.setup_constant_pars

Attributes
----------

.. autosummary::
   :toctree: ../generated/

   WflowModel.region
   WflowModel.crs
   WflowModel.res
   WflowModel.root
   WflowModel.config
   WflowModel.staticmaps
   WflowModel.staticgeoms
   WflowModel.forcing
   WflowModel.states
   WflowModel.results
   WflowModel.flwdir
   WflowModel.basins
   WflowModel.rivers

High level methods
------------------

.. autosummary::
   :toctree: ../generated/

   WflowModel.read
   WflowModel.write
   WflowModel.build
   WflowModel.update
   WflowModel.set_root

General methods
---------------

.. autosummary::
   :toctree: ../generated/

   WflowModel.setup_config
   WflowModel.get_config
   WflowModel.set_config
   WflowModel.read_config
   WflowModel.write_config

   WflowModel.set_staticmaps
   WflowModel.read_staticmaps
   WflowModel.write_staticmaps
   WflowModel.clip_staticmaps

   WflowModel.set_staticgeoms
   WflowModel.read_staticgeoms
   WflowModel.write_staticgeoms

   WflowModel.set_forcing
   WflowModel.read_forcing
   WflowModel.write_forcing
   WflowModel.clip_forcing

   WflowModel.set_states
   WflowModel.read_states
   WflowModel.write_states

   WflowModel.set_results
   WflowModel.read_results

   WflowModel.set_flwdir

.. _api_model_sediment:

WflowSediment model class
=========================

Initialize
----------

.. autosummary::
   :toctree: ../generated/

   WflowSedimentModel

.. _components_sediment:

Setup components
----------------

.. autosummary::
   :toctree: ../generated/

   WflowSedimentModel.setup_config
   WflowSedimentModel.setup_basemaps
   WflowSedimentModel.setup_rivers
   WflowSedimentModel.setup_lakes
   WflowSedimentModel.setup_reservoirs
   WflowSedimentModel.setup_lulcmaps
   WflowSedimentModel.setup_laimaps
   WflowSedimentModel.setup_canopymaps
   WflowSedimentModel.setup_soilmaps
   WflowSedimentModel.setup_riverwidth
   WflowSedimentModel.setup_riverbedsed
   WflowSedimentModel.setup_gauges
   WflowSedimentModel.setup_areamap
   WflowSedimentModel.setup_constant_pars

Attributes
----------

.. autosummary::
   :toctree: ../generated/

   WflowSedimentModel.region
   WflowSedimentModel.crs
   WflowSedimentModel.res
   WflowSedimentModel.root
   WflowSedimentModel.config
   WflowSedimentModel.staticmaps
   WflowSedimentModel.staticgeoms
   WflowSedimentModel.forcing
   WflowSedimentModel.states
   WflowSedimentModel.results
   WflowSedimentModel.flwdir
   WflowSedimentModel.basins
   WflowSedimentModel.rivers

High level methods
------------------

.. autosummary::
   :toctree: ../generated/

   WflowSedimentModel.read
   WflowSedimentModel.write
   WflowSedimentModel.build
   WflowSedimentModel.update
   WflowSedimentModel.set_root

General methods
---------------

.. autosummary::
   :toctree: ../generated/

   WflowSedimentModel.setup_config
   WflowSedimentModel.get_config
   WflowSedimentModel.set_config
   WflowSedimentModel.read_config
   WflowSedimentModel.write_config

   WflowSedimentModel.set_staticmaps
   WflowSedimentModel.read_staticmaps
   WflowSedimentModel.write_staticmaps
   WflowSedimentModel.clip_staticmaps

   WflowSedimentModel.set_staticgeoms
   WflowSedimentModel.read_staticgeoms
   WflowSedimentModel.write_staticgeoms

   WflowSedimentModel.set_forcing
   WflowSedimentModel.read_forcing
   WflowSedimentModel.write_forcing
   WflowSedimentModel.clip_forcing

   WflowSedimentModel.set_states
   WflowSedimentModel.read_states
   WflowSedimentModel.write_states

   WflowSedimentModel.set_results
   WflowSedimentModel.read_results

   WflowSedimentModel.set_flwdir

.. _workflows:

Wflow workflows
===============

.. autosummary::
   :toctree: ../generated/

   workflows.hydrography
   workflows.topography
   workflows.river
   workflows.river_bathymetry
   workflows.landuse
   workflows.soilgrids
   workflows.soilgrids_sediment
   workflows.waterbodymaps
   workflows.reservoirattrs
   workflows.glaciermaps
   workflows.glacierattrs


.. _methods:

Wflow low-level methods
=======================

Input/Output methods
---------------------

.. autosummary::
   :toctree: ../generated/

   read_csv_results
