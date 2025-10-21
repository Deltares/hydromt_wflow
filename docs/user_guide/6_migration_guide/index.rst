v1 Migration Guide
====================

This section describes how to migrate HydroMT-Wflow models and configurations to newer versions of the HydroMT core and Wflow framework. It includes detailed steps, references to updated data structures, and example migration workflows.
With the release of HydroMT Core v1 and Wflow v1, several major changes have been introduced that affect how models, data catalogs, and configuration files are defined.
This section provides an overview of the required migration steps for existing HydroMT-Wflow users.

.. _migration_hydromt_v1:

Migration to HydroMT Core v1
============================

HydroMT v1 introduces a component-based architecture to replace the previous inheritance model.
Instead of all model functionality being defined in a single `Model` class, a model is now composed of modular `ModelComponent` classes such as `GridComponent`, `VectorComponent`, or `ConfigComponent`.
This structure makes models more flexible, extensible, and easier to maintain.
For detailed guidance, refer to the official `HydroMT migration guide <https://deltares.github.io/hydromt/stable/guides/plugin_dev/migrating_to_v1.html>`_.

YAML Configuration Changes
--------------------------

The HydroMT model configuration format has been overhauled.
The root YAML file now includes three main keys: `modeltype`, `global`, and `steps`.

- `modeltype` (optional): Defines which model plugin is being used (e.g. `wflow_sbm` or `wflow_sediment`).
- `global`: Defines model-wide configuration, including data
- `steps`: Replaces the old numbered dictionary format with a sequential list of function calls.

Functions are now explicitly mapped to model or component methods using the `<component>.<method>` syntax.

For a complete example of the new configuration format, see the Wflow v1 YAML template: :download:`wflow_build.yml <../../_examples/wflow_build.yml>`.

For more information on the format changes, see this section in the HydroMT migration guide: `Changes to the yaml HydroMT configuration file format <https://deltares.github.io/hydromt/stable/guides/plugin_dev/migrating_to_v1.html#changes-to-the-yaml-hydromt-configuration-file-format>`_.


Data Catalog Format Changes
---------------------------

The data catalog structure has been refactored to introduce a more modular design and clearer separation of responsibilities across several new classes (`DataSource`, `Driver`, `URIResolver`, and `DataAdapter`).

Key format changes:

- `path` → renamed to `uri`
- `filesystem` moved under `driver`
- `unit_add`, `unit_mult`, `rename`, etc. moved under `data_adapter`
- A single catalog entry can now reference multiple data variants or versions

For detailed information about the format changes, see this section in the HydroMT migration guide: `Changes to the data catalog yaml file format <https://deltares.github.io/hydromt/stable/guides/plugin_dev/migrating_to_v1.html#datacatalog>`_

Main Changes for Python Users
=============================

In HydroMT-Wflow v1, the internal data structure and API were redesigned to improve consistency and maintainability.
Most changes affect how model components (such as ``staticmaps`` and ``forcing``) are accessed and how model data is read and written.

Component Changes
-----------------

The model components are now **dedicated classes** rather than raw data objects (e.g., ``xarray``, ``dict``, or ``geopandas``).
Each component can be accessed via the ``model`` instance and exposes its underlying data through the ``.data`` property.

+--------------------------------+--------------------------------+
| v0.x                           | v1                             |
+================================+================================+
| ``model.grid``                 | ``model.staticmaps``           |
+--------------------------------+--------------------------------+
| ``model.results``              | ``model.output_grid``          |
|                                | ``model.output_scalar``        |
|                                | ``model.output_csv``           |
+--------------------------------+--------------------------------+
| ``model.<component>``          | ``model.<component>.data``     |
+--------------------------------+--------------------------------+
| ``model.write_<component>()``  | ``model.<component>.write()``  |
+--------------------------------+--------------------------------+
| ``model.read_<component>()``   | ``model.<component>.read()``   |
+--------------------------------+--------------------------------+
| ``model.set_<component>()``    | ``model.<component>.set()``    |
+--------------------------------+--------------------------------+

Example: Accessing Component Data
---------------------------------

Each component provides structured access to its data via the ``.data`` property.

.. code-block:: python

    from hydromt_wflow import WflowSbmModel

    model = WflowSbmModel(root="path/to/model", mode="r")

    # Access xarray.Dataset of static maps
    staticmaps = model.staticmaps.data

    # Access geometries (GeoDataFrames)
    geoms = model.geoms.data

    # Access forcing data (xarray.Dataset)
    forcing = model.forcing.data

Example: Writing Components
---------------------------

Read and write operations are now handled at the **component level**.

.. code-block:: python

    # Write configuration file
    model.config.write()

    # Write updated staticmaps to disk
    model.staticmaps.write()

    # Read forcing component explicitly
    model.forcing.read()

These changes provide a clearer and more modular interface, making it easier to manipulate model components independently.

Migration to Wflow v1
=====================

The Wflow v1 update mostly introduces new organisation of the model configuration (TOML), renamed or new (for sediment only) parameters and merging of lakes and reservoirs.

To convert an existing v0.x wflow sbm model with hydromt, you can use the cli command:

.. code-block:: bash

   hydromt update <model_type> <model_root_v0> -i <upgrade_v1.yml> -o option <model_root_v1> -v

Where

- `<model_type>` is `wflow_sbm` or `wflow_sediment`
- `<model_root>` is the folder containing your model
- `<upgrade_v1.yml>` is a configuration file specifying how to handle the migration.
- `<model_root_v1>` is the output folder for the migrated model.

Template upgrade configuration files:

- :download:`Wflow-SBM Upgrade yml <../../_examples/wflow_update_v1_sbm.yml>`
- :download:`Wflow-Sediment Upgrade yml <../../_examples/wflow_update_v1_sediment.yml>`

An example migration workflow notebook, is available `here <../../_examples/upgrade_to_wflow_v1.ipynb>`_


Summarized Migration Steps
===========================

Users migrating from earlier versions of HydroMT-Wflow should:

1. Update their HydroMT YAML configuration file to match the v1 schema. (This includes converting `.ini` and `.toml` files to YAML format.)
2. Migrate their data catalog following the updated v1 format.
3. Run the migration command to update the model structure and files.


.. toctree::
   :titlesonly:
   :hidden:

   ../../_examples/upgrade_to_wflow_v1
