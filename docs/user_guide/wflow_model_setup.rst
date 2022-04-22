.. currentmodule:: hydromt_wflow.wflow
.. _model_set_up:

======================
Wflow model components
======================

The HydroMT Wflow plugin helps you preparing or updating several components of a Wflow model such as topography information, landuse, soil or forcing.
The main interactions are available from the HydroMT Command Line Interface and allow you to configure
HydroMT in order to build or update or clip Wflow models.

When building or updating a model from command line a
`model region <https://deltares.github.io/hydromt/preview/user_guide/build_region.html?highlight=region>`_; a model setup
:ref:`configuration <model_config>` (.ini file) with model components and options and, optionally,
a `data sources <https://deltares.github.io/hydromt/preview/user_guide/get_data.html>`_ (.yml) file should be prepared.

.. _model_components:

Wflow model setup components
============================

An overview of the available Wflow model setup components
is provided in the table below. When using HydroMT from the command line only the
setup components are exposed. Click on
a specific method see its documentation.

.. autosummary::
   :toctree: ../_generated/
   :nosignatures:

   ~WflowModel.setup_config
   ~WflowModel.setup_basemaps
   ~WflowModel.setup_rivers
   ~WflowModel.setup_lakes
   ~WflowModel.setup_reservoirs
   ~WflowModel.setup_glaciers
   ~WflowModel.setup_lulcmaps
   ~WflowModel.setup_laimaps
   ~WflowModel.setup_soilmaps
   ~WflowModel.setup_hydrodem
   ~WflowModel.setup_gauges
   ~WflowModel.setup_areamap

.. _model_files>:

Wflow datamodel files
=====================

The following table provides an overview of which :py:class:`~hydromt_wflow.WflowModel`
attribute contains which Wflow in- and output files. The files are read and written with the associated
read- and write- methods, i.e. :py:func:`~hydromt_wflow.wflow.WflowModel.read_config`
and :py:func:`~hydromt_wflow.wflow.WflowModel.write_config` for the
:py:attr:`~hydromt_wflow.wflow.WflowModel.config`  attribute.


.. list-table:: Wflow model data
   :widths: 30 70
   :header-rows: 1

   * - :py:class:`~hydromt_wflow.WflowModel` attribute
     - Wflow files
   * - :py:attr:`~hydromt_wflow.WflowModel.config`
     - wflow_sbm.toml
   * - :py:attr:`~hydromt_wflow.WflowModel.staticmaps`
     - staticmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowModel.staticgeoms`
     - geometries from the staticgeoms folder (basins.geojson, rivers.geojson etc.)
   * - :py:attr:`~hydromt_wflow.WflowModel.forcing`
     - inmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowModel.states`
     - instates.nc
   * - :py:attr:`~hydromt_wflow.WflowModel.results`
     - output.nc, output_scalar.nc, output.csv




