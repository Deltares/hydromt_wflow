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
----------------

.. autosummary::
   :toctree: ../_generated

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


.. autoclass:: hydromt_wflow.WflowSedimentModel
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:
