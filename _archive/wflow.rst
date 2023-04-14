.. currentmodule:: hydromt_wflow.wflow

=====
Wflow
=====
With the hydromt_wflow plugin, you can easily work with wflow (SBM) models. This plugin helps you preparing or updating 
several :ref:`components <model_components>` of a wflow model such as topography information, landuse, soil or forcing.
The :ref:`main interactions <model_config>` are available from the HydroMT Command Line Interface and allow you to configure 
HydroMT in order to build or update or clip wflow models.

When building or updating a model from command line a model region_; a model setup 
configuration (.ini file) with model components and options and, optionally, 
a data_ sources (.yml) file should be prepared.

Note that the order in which the components are listed in the ini file is important: 

- `setup_basemaps` should always be run first to determine the model domain
- `setup_rivers` should be run right after `setup_basemaps` as it influences several other setup components (lakes, reservoirs, riverwidth, gauges)

For python users all Wflow attributes and methods are available, see :ref:`api_model`

.. _model_components:

Wflow model components
======================

An overview of the available WflowModel setup components
is provided in the table below. When using hydromt from the command line only the
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

Wflow datamodel
===============

The following table provides an overview of which :py:class:`~hydromt_wflow.WflowModel` 
attribute contains which Wflow in- and output files. The files are read and written with the associated 
read- and write- methods, i.e. :py:func:`~hydromt_wflow.wflow.WflowModel.read_config` 
and :py:func:`~hydromt_wflow.wflow.WflowModel.write_config` for the 
:py:attr:`~hydromt_wflow.wflow.WflowModel.config`  attribute. 


.. list-table:: WflowModel data
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

.. _model_config:

Wflow configuration
===================
This HydroMT plugin provides an implementation for the wflow model in order to build, update or clip from 
command line. Specific details on the HydroMT CLI methods can be found in 
https://deltares.github.io/hydromt/latest/user_guide/cli.html 

Configuration file
------------------
Settings to build or update a wflow model are managed in a configuration file. In this file, 
every option from each :ref:`model component <model_components>` can be changed by the user 
in its corresponding section.

Below is an example of ini file that can be used to build a complete wflow model
:download:`.ini file <../_examples/wflow_build.ini>`. Each section corresponds
to a model component with the same name.

.. literalinclude:: ../_examples/wflow_build.ini
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
    hydromt build wflow path/to/built_model "{'basin': [x, y]}" -i wflow_build.ini -d data_sources.yml -vvv

The recommended `region options <https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options>`_ 
for a proper implementation of this model are:

- basin
- subbasin

Updating a model
----------------
This plugin allows to update any components from a wflow model. To do so, list the components to update in a configuration file,
if needed edit your data library with new data sources required for the update and use the command:

.. code-block:: console

    activate hydromt-wflow
    hydromt update wflow path/to/model_to_update -o path/to/updated_model -i wflow_update.ini -d data_sources.yml -vvv


Clipping a model
----------------
This plugin allows to clip the following parts of an existing model for a smaller region from command line:

- staticmaps
- forcing

To clip a smaller model from an existing one use:

.. code-block:: console

    activate hydromt-wflow
    hydromt clip wflow path/to/model_to_clip path/to/clipped_model "{'basin' [1001]}" -vvv

As for building, the recommended `region options <https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options>`_ 
for a proper implementation of the clipped model are:

- basin
- subbasin
   ~WflowModel.setup_precip_forcing
   ~WflowModel.setup_temp_pet_forcing
   ~WflowModel.setup_constant_pars


.. _data: https://deltares.github.io/hydromt/latest/user_guide/data.html
.. _region: https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options