.. _sediment_update:

Updating a model
----------------
This plugin allows to update any components from a wflow_sediment model. To do so, list the components to update in a configuration file,
if needed edit your data library with new data sources required for the update and use the command:

.. code-block:: console

    activate hydromt-wflow
    hydromt update wflow_sediment path/to/model_to_update -o path/to/updated_model -i wflow_sediment_update.ini -d data_sources.yml -vvv
