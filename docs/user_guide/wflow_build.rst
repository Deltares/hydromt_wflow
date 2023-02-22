.. _wflow_build:

Building a model
================

This plugin allows to build a complete model from available data. Once the configuration and 
data libraries are set, you can build a model by using:

.. code-block:: console

    hydromt build wflow path/to/built_model -r "{'basin': [x, y]}" -i wflow_build.ini -d data_sources.yml -vvv


.. Note::
  From HydroMT version 0.7.0 onwards the region argument is optional and should be preceded by a -r or --region flag.
  The resolution (previously -r) argument has been moved to the setup_basemaps section in the .ini configuration file.

The recommended `region options <https://deltares.github.io/hydromt/latest/user_guide/model_region>`_
for a proper implementation of this model are:

- basin
- subbasin

.. _model_config:

Configuration file
------------------
Settings to build or update a Wflow model are managed in a configuration file. In this file,
every option from each :ref:`model method <model_methods>` can be changed by the user
in its corresponding section.

Note that the order in which the components are listed in the ini file is important:

- `setup_basemaps` should always be run first to determine the model domain
- `setup_rivers` should be run right after `setup_basemaps` as it influences several other setup components (lakes, reservoirs, riverwidth, gauges)

Below is an example ini file that can be used to build a complete Wflow model
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
  P-drive). In the command lines examples below, this is done by adding either **--dd** or **--deltares-data**
  to the build / update command line.
- Finally, the user can prepare its own yaml libary (or libraries) (see
  `HydroMT documentation <https://deltares.github.io/hydromt/latest/index>`_ to check the guidelines).
  These user libraries can be added either in the command line using the **-d** option and path/to/yaml or in the **ini file**
  with the **data_libs** option in the [global] sections.

.. toctree::
    :hidden:
    
    Example: Build Wflow model <../_examples/build_model.ipynb>
