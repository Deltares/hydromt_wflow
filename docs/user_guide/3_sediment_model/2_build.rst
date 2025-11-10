.. _sediment_build:

Building a model
================

This plugin allows to build a complete Wflow Sediment model from available data. Once the configuration and
data libraries are set, you can build a model by using:

.. code-block:: console

    activate hydromt-wflow
    hydromt build wflow_sediment path/to/built_model -i wflow_sediment_build.yml -d data_sources.yml -vvv

.. Note::
  From HydroMT version 0.7.0 onwards the region argument is optional and should be preceded by a -r or \-\-region flag.
  The resolution (previously -r) argument has been moved to the setup_basemaps section in the .yml configuration file.
  From HydroMT version 1.0 onwards, the region argument has been moved to ``setup_basemaps`` function arguments and is no longer available via cli.


The recommended `region options <https://deltares.github.io/hydromt/latest/user_guide/models/model_region.html>`_
for a proper implementation of the Wflow Sediment model are:

- basin
- subbasin

.. _model_config_sed:

Configuration file
------------------
Settings to build or update a Wflow Sediment model are managed in a configuration file. In this file,
every option from each :ref:`model component <model_methods_sed>` can be changed by the user
in its corresponding section.

Note that the order in which the components are listed in the configuration file is important:

- `setup_basemaps` should always be run first to determine the model domain
- `setup_rivers` should be run right after `setup_basemaps` as it influences several other setup components (lakes, reservoirs, riverbedsed, floodplains, gauges)

Below is an example configuration file that can be used to build a complete Wflow Sediment model
:download:`.yml file </_examples/wflow_sediment_build.yml>`. Each section corresponds
to a model component with the same name.

.. literalinclude:: /_examples/wflow_sediment_build.yml
   :language: yaml

Selecting data
--------------
Data sources in HydroMT are provided in one of several yaml libraries. These libraries contain required
information on the different data sources so that HydroMT can process them for the different models. There
are three ways for the user to select which data libraries to use:

- For testing and examples purposes, HydroMT can use the data stored in the
  `hydromt-artifacts <https://github.com/DirkEilander/hydromt-artifacts>`_
  which contains an extract of global data for a small region around the Piave river in Northern Italy. to
  use this predefined catalog, the user can add **-d artifact_data** to the build / update command line.
- Another options for Deltares users is to select the deltares_data library (requires access to the Deltares
  P-drive). In the command lines examples below, this is done by adding **-d deltares_data** predefined catalog
  to the build / update command line.
- Finally, the user can prepare its own yaml catalog (see
  `HydroMT documentation <https://deltares.github.io/hydromt/latest/index>`_ to check the guidelines).
  These user libraries can be added either in the command line using the **-d** option and path/to/yaml or in the **ini file**
  with the **data_libs** option in the [global] sections.

Extending a Wflow (SBM) model with a Wflow Sediment model
---------------------------------------------------------
If you already have a Wflow model and you want to extend it in order to include sediment as well, then you do not need to build the
Wflow Sediment model from scratch. You can instead ``update`` the Wflow model with the additional components needed by Wflow Sediment.
These components are available in a template :download:`.yml file </_examples/wflow_extend_sediment.yml>` and shown below. The corresponding
command line would be:

.. code-block:: console

    activate hydromt-wflow
    hydromt update wflow_sediment path/to/wflow_model_to_extend -o path/to/wflow_sediment_model -i wflow_extend_sediment.yml -d data_sources.yml -vvv

.. literalinclude:: /_examples/wflow_extend_sediment.yml
   :language: yaml

.. _data: https://deltares.github.io/hydromt/latest/user_guide/data_catalog/data_overview.html

Examples
--------
To know more about building a Wflow-Sediment model from scratch, check the following examples:

- :ref:`Building a Wflow Sediment model from command line <example-build_sediment>`
