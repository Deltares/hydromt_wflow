.. _wflow_update:

Updating a model
----------------
To add or change one or more components of an existing Wflow model the ``update`` method can be used.

**Steps in brief:**

1) You have an **existing model** schematization. This model does not have to be complete.
2) Prepare or use a pre-defined **data catalog** with all the required data sources, see `working with data <https://deltares.github.io/hydromt/latest/user_guide/data_main.html>`_.
3) Prepare a **model configuration** with the methods that you want to use to add or change components of your model: see `model configuration <https://deltares.github.io/hydromt/latest/user_guide/model_config.html>`_.
4) **Update** your model using the CLI or Python interface.

.. code-block:: console

    activate hydromt-wflow
    hydromt update wflow path/to/model_to_update -o path/to/updated_model -i wflow_update.ini -d data_sources.yml -vvv

.. NOTE::

    By default, the updated model will overwrite your existing one. To save the updated model in a different
    folder, use the -o path/to/updated_model option of the CLI.

.. TIP::

    By default all model data is written at the end of the update method. If your update however
    only affects a certain model data (e.g. staticmaps or forcing) you can add a write_* method
    (e.g. `write_staticmaps`, `write_forcing`) to the .ini file and only these data will be written.

    Note that the model config is often changed as part of the a model method and `write_config`
    should thus be added to the .ini file to keep the model data and config consistent.

.. toctree::
    :hidden:

    Example: Update Wflow model (landuse) <../_examples/update_model_landuse.ipynb>
    Example: Update Wflow model (forcing) <../_examples/update_model_forcing.ipynb>
    Example: Update Wflow model (gauges) <../_examples/update_model_gauges.ipynb>
