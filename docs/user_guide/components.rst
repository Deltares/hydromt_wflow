.. _model_components:
.. currentmodule:: hydromt_plugin.plugin

Model components
================

.. note::

  Document here the different model components available and if they are links between the components (one component 
  should be run before others or if updating this component then others should be updated as well).

For plugin, the different components available for building or updating are:

.. autosummary::
   :toctree: ../generated/
   :nosignatures:

   PluginModel.setup_config
   PluginModel.setup_basemaps
   PluginModel.setup_gauges


.. warning::

    In plugin, the order in which the components are listed in the ini file is important: setup_rivers should 
    be run before setup_lakes.