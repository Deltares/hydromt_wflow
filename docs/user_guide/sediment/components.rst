.. _model_components_sed:
.. currentmodule:: hydromt_wflow.wflow_sediment

Model components
================

For wflow_sediment, the different components available for building or updating are:

.. autosummary::
   :toctree: ../../generated/
   :nosignatures:

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
   WflowSedimentModel.setup_constant_pars


.. warning::

    As for wflow, in wflow_sediment the order in which the components are listed in the ini file is important:  setup_rivers should be run 
    right after setup_basemaps as it influences several other setup components (lakes, reservoirs, riverwidth, riverbedsed, gauges); 
    setup_riverwidth should be listed after setup_lakes and setup_reservoirs.