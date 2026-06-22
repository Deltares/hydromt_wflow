Migration Guide
===============

Both HydroMT and Wflow.jl are now at version 1.0.0 :octicon:`sparkle-fill;1.5em`

These versions introduce several significant changes to the model structure, configuration files, and data handling.
This section describes how to migrate HydroMT-Wflow models and configurations to newer versions of the HydroMT core and Wflow framework. It includes detailed steps, references to updated data structures, and example migration workflows.
With the release of HydroMT Core v1 and Wflow v1, several major changes have been introduced that affect how models, data catalogs, and configuration files are defined.
This section provides an overview of the required migration steps for existing HydroMT-Wflow users.

It is divided into two main parts:

.. grid:: 2
    :gutter: 2

    .. grid-item-card::
        :text-align: center
        :link: migration_hydromt
        :link-type: doc

        :octicon:`file-moved;5em;sd-text-icon blue-icon`
        +++
        Migrating to HydroMT v1

    .. grid-item-card::
        :text-align: center
        :link: migration_wflow
        :link-type: doc

        :octicon:`file-moved;5em;sd-text-icon blue-icon`
        +++
        Migrating to Wflow.jl v1.0.0

Users migrating from earlier versions of HydroMT-Wflow or Wflow.jl should follow these general steps:

1. Update their HydroMT YAML configuration file to match the v1 schema. (This includes converting `.ini` and `.toml` files to YAML format.)
2. Migrate their data catalog following the updated v1 format.
3. Run the migration command to update the model structure and files.

.. toctree::
    :maxdepth: 2
    :hidden:
    :titlesonly:

    migration_hydromt
    migration_wflow
