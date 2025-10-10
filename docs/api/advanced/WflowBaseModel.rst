.. currentmodule:: hydromt_wflow

.. _api_wflowbasemodel:

==============
WflowBaseModel
==============

The ``WflowBaseModel`` class provides core methods and components shared by all Wflow models.

.. autosummary::
   :toctree: ../../_generated

   WflowBaseModel

High level and I/O methods
--------------------------

.. autosummary::
   :toctree: ../../_generated

   WflowBaseModel.build
   WflowBaseModel.update

   WflowBaseModel.read
   WflowBaseModel.write

   WflowBaseModel.read_config
   WflowBaseModel.write_config

   WflowBaseModel.read_grid
   WflowBaseModel.write_grid

   WflowBaseModel.read_geoms
   WflowBaseModel.write_geoms

   WflowBaseModel.read_forcing
   WflowBaseModel.write_forcing

   WflowBaseModel.read_states
   WflowBaseModel.write_states

   WflowBaseModel.read_outputs

Attributes
----------

.. autosummary::
   :toctree: ../../_generated

   WflowBaseModel.crs
   WflowBaseModel.root
   WflowBaseModel.flwdir
   WflowBaseModel.basins
   WflowBaseModel.rivers

Other general methods
---------------------

.. autosummary::
   :toctree: ../../_generated

   WflowBaseModel.set_config
   WflowBaseModel.get_config
   WflowBaseModel.set_forcing
   WflowBaseModel.set_grid
   WflowBaseModel.set_geoms
   WflowBaseModel.set_states

Setup methods
-------------

.. autosummary::
   :toctree: ../../_generated

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

.. autoclass:: hydromt_wflow.WflowBaseModel
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

.. rubric:: Components

.. currentmodule:: hydromt_wflow.components
.. include:: advanced/components.rst
   :start-line: 8
