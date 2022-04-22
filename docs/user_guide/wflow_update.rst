.. _wflow_update:

Updating a model
----------------
This plugin allows to update any components from a wflow model. To do so, list the components to update in a configuration file,
if needed edit your data library with new data sources required for the update and use the command:

.. code-block:: console

    activate hydromt-wflow
    hydromt update wflow path/to/model_to_update -o path/to/updated_model -i wflow_update.ini -d data_sources.yml -vvv

. toctree::
    :hidden:

    Example: Update Wflow model (landuse) <../_examples/update_model_landuse>
    Example: Update Wflow model (forcing) <../_examples/update_model_forcing>
    Example: Update Wflow model (gauges) <../_examples/update_model_gauges>