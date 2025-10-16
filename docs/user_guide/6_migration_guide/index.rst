v1 Migration Guide
====================

This section describes how to migrate HydroMT-Wflow models and configurations to newer versions of the HydroMT core and Wflow framework. It includes detailed steps, references to updated data structures, and example migration workflows.

.. _migration_hydromt_v1:

With the release of HydroMT Core v1 and Wflow v1, several major changes have been introduced that affect how models, data catalogs, and configuration files are defined.
This section provides an overview of the required migration steps for existing HydroMT-Wflow users.
For detailed guidance, refer to the official `HydroMT migration guide <https://deltares.github.io/hydromt/stable/guides/plugin_dev/migrating_to_v1.html>`_.

Migration to HydroMT Core v1
============================

HydroMT v1 introduces a component-based architecture to replace the previous inheritance model.
Instead of all model functionality being defined in a single `Model` class, a model is now composed of modular `ModelComponent` classes such as `GridComponent`, `VectorComponent`, or `ConfigComponent`.
This structure makes models more flexible, extensible, and easier to maintain.

YAML Configuration Changes
--------------------------

The HydroMT model configuration format has been overhauled.
The root YAML file now includes three main keys: `modeltype`, `global`, and `steps`.

- `modeltype`: Defines which model plugin is being used (e.g. `wflow`).
- `global`: Defines model-wide configuration, including components and data libraries.
- `steps`: Replaces the old numbered dictionary format with a sequential list of function calls.

Functions are now explicitly mapped to model or component methods using the `<component>.<method>` syntax.

For a complete example of the new configuration format, see the Wflow v1 YAML template: :download:`wflow_build.yml <../../_examples/wflow_build.yml>`.

For a more information on the format changes, see this section in the HydroMT migration guide: `Changes to the yaml HydroMT configuration file format <https://deltares.github.io/hydromt/stable/guides/plugin_dev/migrating_to_v1.html#changes-to-the-yaml-hydromt-configuration-file-format>`_.


Data Catalog Format Changes
---------------------------

The data catalog structure has been refactored to introduce a more modular design and clearer separation of responsibilities across several new classes (`DataSource`, `Driver`, `URIResolver`, and `DataAdapter`).

Key format changes:

- `path` → renamed to `uri`
- `filesystem` moved under `driver`
- `unit_add`, `unit_mult`, `rename`, etc. moved under `data_adapter`
- A single catalog entry can now reference multiple data variants or versions

For detailed information about the format changes, see this section in the HydroMT migration guide: `Changes to the data catalog yaml file format <https://deltares.github.io/hydromt/stable/guides/plugin_dev/migrating_to_v1.html#datacatalog>`_

Removing Dictionary-like Catalog Access
---------------------------------------

Catalog access has changed from dictionary-style lookups to an explicit API.
Update your code accordingly:

+------------------------------+------------------------------------------+
| v0.x                         | v1                                       |
+==============================+==========================================+
| `catalog["name"]`            | `catalog.get_source("name")`             |
+------------------------------+------------------------------------------+
| `'name' in catalog`          | `catalog.contains_source("name")`        |
+------------------------------+------------------------------------------+
| `catalog.keys()`             | `catalog.get_source_names()`             |
+------------------------------+------------------------------------------+
| `catalog["name"] = data`     | `catalog.set_source("name", data)`       |
+------------------------------+------------------------------------------+


Migration to Wflow v1
=====================

The Wflow v1 update introduces new model configuration parameters, renamed parameters and upgraded file structures consistent with HydroMT Core v1.

To convert an existing v0.x hydromt wflow sbm model, you can use the cli command:

.. code-block:: bash

   hydromt update <model_type> <model_root> -i <upgrade_v1.yml> -v

Where

- `<model_type>` is `wflow_sbm` or `wflow_sediment`
- `<model_root>` is the folder containing your model
- `<upgrade_v1.yml>` is a configuration file specifying how to handle the migration.

Template upgrade configuration files:

- :download:`Wflow-SBM Upgrade yml <../../_examples/wflow_update_v1_sbm.yml>`
- :download:`Wflow-Sediment Upgrade yml <../../_examples/wflow_update_v1_sediment.yml>`


Summarized Migration Steps
===========================

Users migrating from earlier versions of HydroMT-Wflow should:

1. Update their HydroMT YAML configuration file to match the v1 schema.
2. Migrate their data catalog following the updated v1 format.
3. Convert model configuration and parameter files to YAML if they were previously in `.ini` or `.toml` format.
4. Run the migration command to update the model structure and files.

An example migration workflow notebook, is available `here <../../_examples/upgrade_to_wflow_v1.ipynb>`_

.. toctree::
   :titlesonly:
   :hidden:

   ../../_examples/upgrade_to_wflow_v1
