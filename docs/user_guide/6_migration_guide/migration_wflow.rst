.. _migration_wflow_v1:

Migrating to Wflow.jl v1.0.0
=============================

The Wflow.jl v1 update mostly introduces new organisation of the model configuration (TOML), renamed or new (for sediment only) parameters and merging of lakes and reservoirs.
For a complete overview of the changes, refer to the official `Wflow.jl changelog <https://deltares.github.io/Wflow.jl/dev/changelog.html>`_.


To convert an existing v0.x wflow sbm model with hydromt, you can use the cli command:

.. code-block:: bash

   hydromt update <model_type> <model_root_v0> -o <model_root_v1> -i <upgrade_v1.yml>  -v

Where

- ``<model_type>`` is ``wflow_sbm`` or ``wflow_sediment``
- ``<model_root>`` is the folder containing your model
- ``<upgrade_v1.yml>`` is a configuration file specifying how to handle the migration.
- ``<model_root_v1>`` is the output folder for the migrated model.
- For sediment: ``-d data_catalog.yml`` to specify a data catalog for preparing the extra parameters of wflow sediment.

Template upgrade configuration files:

- :download:`Wflow-SBM Upgrade yml </_examples/wflow_update_v1_sbm.yml>`
- :download:`Wflow-Sediment Upgrade yml </_examples/wflow_update_v1_sediment.yml>`

An example migration workflow notebook, is available :ref:`here <example-upgrade_to_wflow_v1>`.
