.. _model_components:
.. currentmodule:: hydromt_wflow.wflow

Model components
================

For wflow, the different components available for building or updating are:

.. autosummary::
   :toctree: generated/
   :nosignatures:

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


.. warning::

    In wflow, the order in which the components are listed in the ini file is important:  setup_rivers should be run 
    right after setup_basemaps as it influences several other setup components (lakes, reservoirs, riverwidth, gauges); 
    setup_riverwidth should be listed after setup_lakes and setup_reservoirs.