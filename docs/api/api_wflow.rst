.. currentmodule:: hydromt_wflow.wflow
.. _api_model:

Wflow model
===========

Initialize
----------

.. autosummary::
   :toctree: ../generated/

   WflowModel

Build components
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
   WflowModel.setup_riverwidth
   WflowModel.setup_gauges
   WflowModel.setup_precip_forcing
   WflowModel.setup_temp_pet_forcing
   WflowModel.setup_constant_pars

Model specific attributes
-------------------------

.. autosummary::
   :toctree: ../generated/

   WflowModel.flwdir
   WflowModel.basins
   WflowModel.rivers

Model specific methods
----------------------

.. autosummary::
   :toctree: ../generated/

   WflowModel.set_flwdir
   WflowModel.clip_staticmaps
   WflowModel.clip_forcing