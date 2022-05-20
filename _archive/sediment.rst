.. currentmodule:: hydromt_wflow.wflow_sediment

==============
Wflow Sediment
==============
With the hydromt_wflow plugin, you can easily work with wflow sediment models. This plugin helps you preparing or updating 
several :ref:`components <model_components_sed>` of a wflow sediment model such as topography information, landuse or soil.
The :ref:`main interactions <model_config_sed>` are available from the HydroMT Command Line Interface and allow you to configure 
HydroMT in order to build or update or clip wflow sediment models.

When building or updating a model from command line a model region_; a model setup 
configuration (.ini file) with model components and options and, optionally, 
a data_ sources (.yml) file should be prepared.

Note that the order in which the components are listed in the ini file is important: 

- `setup_basemaps` should always be run first to determine the model domain
- `setup_rivers` should be run right after `setup_basemaps` as it influences several other setup components (lakes, reservoirs, riverbedsed, floodplains, gauges)

For python users all WflowSediment attributes and methods are available, see :ref:`api_model_sediment`

.. _model_components_sed:

WflowSediment model components
==============================

An overview of the available WflowSedimentModel setup components
is provided in the table below. When using hydromt from the command line only the
setup components are exposed. Click on
a specific method see its documentation.

.. autosummary::
   :toctree: ../_generated/
   :nosignatures:

   ~WflowSedimentModel.setup_config
   ~WflowSedimentModel.setup_basemaps
   ~WflowSedimentModel.setup_rivers
   ~WflowSedimentModel.setup_lakes
   ~WflowSedimentModel.setup_reservoirs
   ~WflowSedimentModel.setup_lulcmaps
   ~WflowSedimentModel.setup_laimaps
   ~WflowSedimentModel.setup_canopymaps
   ~WflowSedimentModel.setup_soilmaps
   ~WflowSedimentModel.setup_riverbedsed
   ~WflowSedimentModel.setup_gauges
   ~WflowModel.setup_areamap
   ~WflowSedimentModel.setup_constant_pars

WflowSediment datamodel
=======================

The following table provides an overview of which :py:class:`~hydromt_wflow.WflowSedimentModel` 
attribute contains which WflowSediment in- and output files. The files are read and written with the associated 
read- and write- methods, i.e. :py:func:`~hydromt_wflow.WflowSedimentModel.read_config` 
and :py:func:`~hydromt_wflow.WflowSedimentModel.write_config` for the 
:py:attr:`~hydromt_wflow.WflowSedimentModel.config`  attribute. 


.. list-table:: WflowSedimentModel data
   :widths: 30 70
   :header-rows: 1

   * - :py:class:`~hydromt_wflow.WflowSedimentModel` attribute
     - Wflow sediment files
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.config`
     - wflow_sediment.toml
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.staticmaps`
     - staticmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.staticgeoms`
     - geometries from the staticgeoms folder (basins.geojson, rivers.geojson etc.)
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.forcing`
     - inmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.states`
     - instates.nc
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.results`
     - output.nc, output_scalar.nc, output.csv

.. _model_config_sed:

WflowSediment configuration
===========================

This HydroMT plugin provides an implementation for the wflow_sediment model in order to build, update or clip from 
command line. Specific details on the HydroMT CLI methods can be found in 
https://deltares.github.io/hydromt/latest/user_guide/cli.html 

Configuration file
------------------
Settings to build or update a wflow model are managed in a configuration file. In this file, 
every option from each :ref:`model component <model_components_sed>` can be changed by the user 
in its corresponding section.

Below is an example of ini file that can be used to build a complete wflow_sediment model
:download:`.ini file <../_examples/wflow_sediment_build.ini>`. Each section corresponds
to a model component with the same name.

.. literalinclude:: ../_examples/wflow_sediment_build.ini
   :language: Ini

Selecting data
--------------
Data sources in HydroMT are provided in one of several yaml libraries. These libraries contain required 
information on the different data sources so that HydroMT can process them for the different models. There 
are three ways for the user to select which data libraries to use:

- If no yaml file is selected, HydroMT will use the data stored in the 
  `hydromt-artifacts <https://github.com/DirkEilander/hydromt-artifacts>`_ 
  which contains an extract of global data for a small region around the Piave river in Northern Italy.
- Another options for Deltares users is to select the deltares-data library (requires access to the Deltares 
  P-drive). In the command lines examples below, this is done by adding either **-dd** or **--deltares-data** 
  to the build / update command line.
- Finally, the user can prepare its own yaml libary (or libraries) (see 
  `HydroMT documentation <https://deltares.github.io/hydromt/latest/user_guide/data.html>`_ to check the guidelines). 
  These user libraries can be added either in the command line using the **-d** option and path/to/yaml or in the **ini file** 
  with the **data_libs** option in the [global] sections.

Building a model
----------------
This plugin allows to build a complete model from available data. Once the configuration and 
data libraries are set, you can build a model by using:

.. code-block:: console

    activate hydromt-wflow
    hydromt build wflow_sediment path/to/built_model "{'basin': [x, y]}" -i wflow_sediment_build.ini -d data_sources.yml -vvv

The recommended `region options <https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options>`_ 
for a proper implementation of the wflow_sediment model are:

- basin
- subbasin


Updating a model
----------------
This plugin allows to update any components from a wflow_sediment model. To do so, list the components to update in a configuration file,
if needed edit your data library with new data sources required for the update and use the command:

.. code-block:: console

    activate hydromt-wflow
    hydromt update wflow_sediment path/to/model_to_update -o path/to/updated_model -i wflow_sediment_update.ini -d data_sources.yml -vvv


Clipping a model
----------------
This plugin allows to clip the following parts of an existing model for a smaller region from command line:

- staticmaps
- forcing

To clip a smaller model from an existing one use:

.. code-block:: console

    activate hydromt-wflow
    hydromt clip wflow_sediment path/to/model_to_clip path/to/clipped_model "{'basin' [1001]}" -vvv

As for building, the recommended `region options <https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options>`_ 
for a proper implementation of the clipped model are:

- basin
- subbasin

Extending a wflow model with a wflow_sediment model
---------------------------------------------------
If you already have a wflow model and you want to extend it in order to include sediment as well, then you do not need to build the 
wflow_sediment from scratch. You can instead ``update`` the wflow model with the additional components needed by wflow_sediment.
These components are available in a template :download:`.ini file <../_examples/wflow_extend_sediment.ini>` and shown below. The corresponding
command line would be:

.. code-block:: console

    activate hydromt-wflow
    hydromt update wflow_sediment path/to/wflow_model_to_extend -i wflow_extend_sediment.ini -d data_sources.yml -vvv

.. literalinclude:: ../_examples/wflow_extend_sediment.ini
   :language: Ini

.. _data: https://deltares.github.io/hydromt/latest/user_guide/data.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options