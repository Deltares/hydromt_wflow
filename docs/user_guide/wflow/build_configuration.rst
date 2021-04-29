.. _model_config:

Model configuration
===================

This HydroMT plugin provides an implementation for the wflow model in order to build, update or clip from 
command line. Specific details on the HydroMT CLI methods can be found in 
https://hydromt.readthedocs.io/en/latest/user_guide/cli.html 

Configuration file
------------------
Settings to build or update a wflow model are managed in a configuration file. In this file, 
every option from each :ref:`model component <model_components>` can be changed by the user 
in its corresponding section.

Below is an example of ini file that can be used to build a complete wflow model
:download:`.ini file <../../examples/examples/wflow_build.ini>`. Each section corresponds 
to a model component with the same name.

.. literalinclude:: ../../examples/examples/wflow_build.ini
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
  `HydroMT documentation <https://hydromt.readthedocs.io/en/latest/user_guide/data.html>`_ to check the guidelines). 
  These user libraries can be added either in the command line using the **-d** option and path/to/yaml or in the **ini file** 
  with the **data_libs** option in the [global] sections.

Building a model
----------------
This plugin allows to build a complete model from available data. Once the configuration and 
data libraries are set, you can build a model by using:

.. code-block:: console

    activate hydromt-wflow
    hydromt build wflow path/to/built_model "{'basin': [x, y]}" -i wflow_build.ini -d data_sources.yml -vvv

The recommended `region options <https://hydromt.readthedocs.io/en/latest/user_guide/cli.html#region-options>`_ 
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

As for building, the recommended `region options <https://hydromt.readthedocs.io/en/latest/user_guide/cli.html#region-options>`_ 
for a proper implementation of the clipped model are:

- basin
- subbasin