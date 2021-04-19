.. _model_config:

Model configuration
===================

.. note::

  Document here the steps to build/update your model and the corresponding ini file.


This HydroMT plugin provides an implementation for the wflow model in order to build, update or clip from 
command line. Specific details on the HydroMT CLI methods can be found in 
https://hydromt.readthedocs.io/en/latest/user_guide/cli.html 

Configuration file
------------------
Settings to build or update a wflow model are managed in a configuration file. In this file, 
every option from each :ref:`model component <model_components>` can be changed by the user 
in its corresponding section.

Below is an example of ini file that can be used to build a complete wflow model
:download:`.ini file </../examples/model_build.ini>`. Each section corresponds 
to a model component with the same name.

.. note::

  If applicable document available [global] options.

.. literalinclude:: /../examples/model_build.ini
   :language: Ini


Building a model
----------------
This plugin allows to build a complete model from available data. Once the configuration and 
data libraries are set, you can build a model by using:

.. code-block:: console

    activate hydromt
    hydromt build wflow path/to/built_model "{'basin': [x, y]}" -i model_build.ini -d data_sources.yml -vvv

The recommended `region options <https://hydromt.readthedocs.io/en/latest/user_guide/cli.html#region-options>`_ 
for a proper implementation of this model are:

- basin
- outlet
- subbasin

Updating a model
----------------
This plugin allows to update any components from a wflow model. To do so, list the components to update in a configuration file,
if needed edit your data library with new data sources required for the update and use the command:

.. code-block:: console

    activate hydromt
    hydromt update wflow path/to/model_to_update -o path/to/updated_model -i model_update.ini -d data_sources.yml -vvv


Clipping a model
----------------
This plugin allows to clip the following parts of an existing model for a smaller region from command line:

- staticmaps
- forcing

To clip a smaller model from an existing one use:

.. code-block:: console

    activate hydromt
    hydromt clip wflow path/to/model_to_clip path/to/clipped_model "{'basin' [1001]}" -vvv

As for building, the recommended `region options <https://hydromt.readthedocs.io/en/latest/user_guide/cli.html#region-options>`_ 
for a proper implementation of the clipped model are:

- basin
- outlet
- subbasin