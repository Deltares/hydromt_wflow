.. _sediment_build:

Building a model
================
This plugin allows to build a complete model from available data. Once the configuration and 
data libraries are set, you can build a model by using:

.. code-block:: console

    activate hydromt-wflow
    hydromt build wflow_sediment path/to/built_model "{'basin': [x, y]}" -i wflow_sediment_build.ini -d data_sources.yml -vvv

The recommended `region options <https://deltares.github.io/hydromt/latest/user_guide/cli.html#region-options>`_ 
for a proper implementation of the wflow_sediment model are:

- basin
- subbasin

.. _model_config_sed:

WflowSediment configuration (.ini file)
=======================================

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

.. toctree::
    :hidden:

    Example: Build Wflow sediment model <../_examples/build_sediment>