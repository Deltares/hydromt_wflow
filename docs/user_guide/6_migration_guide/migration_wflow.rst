.. _migration_wflow_v1:

Migrating to Wflow.jl v1.0.0
=============================

The Wflow.jl v1 update mostly introduces new organisation of the model configuration (TOML), renamed or new (for sediment only) parameters and merging of lakes and reservoirs.
For a complete overview of the changes, refer to the official `Wflow.jl changelog <https://deltares.github.io/Wflow.jl/dev/changelog.html>`_.

HydroMT-Wflow provides a dedicated CLI command to upgrade your model:

.. code-block:: bash

   hydromt_wflow upgrade <model_root> -o <output_dir> [--model-type wflow_sbm] [-i config.yml] [-d data_catalog] [-v]

Where

- ``<model_root>`` is the folder containing your existing model.
- ``<output_dir>`` is the output folder for the upgraded model (must not already exist).
- ``--model-type`` is ``wflow_sbm`` (default) or ``wflow_sediment``.
- ``-i config.yml`` is an optional upgrade configuration file (see below).
- ``-d data_catalog`` specifies a data catalog, required for sediment model upgrades.
- ``-v`` enables verbose logging.

Run ``hydromt_wflow upgrade --help`` for a detailed overview of the command and options.

The command copies your model to the output directory and applies all necessary
upgrade steps automatically. Your original model files are never modified.

Upgrade configuration
---------------------

For SBM models with a default config filename (``wflow_sbm.toml``), no upgrade
configuration file is needed.

If your config file has a non-default name, or if you need to pass options for
specific upgrade steps (e.g. sediment), create a YAML file:

.. code-block:: yaml

   config_filename: wflow_sediment.toml

   0.8_1.0:
     soil_fn: soilgrids

- ``config_filename`` overrides the config filename to look for in the model root.
- Version-keyed sections (e.g. ``0.8_1.0``) pass options to the corresponding
  upgrade step.

A template for sediment models is available at:
:download:`Wflow-Sediment Upgrade yml </_examples/data/wflow_upgrade/wflow_upgrade_sediment.yml>`

Examples
--------

Upgrade an SBM model:

.. code-block:: bash

   hydromt_wflow upgrade ./my_model -o ./my_model_upgraded -v

Upgrade a sediment model with options:

.. code-block:: bash

   hydromt_wflow upgrade ./my_model -o ./my_model_upgraded --model-type wflow_sediment -i upgrade_config.yml -d artifact_data -v

An example migration workflow notebook is available :ref:`here <example-upgrade_to_wflow_v1>`.
