.. _migration_hydromt_v1:

Migrating to HydroMT v1
=======================

HydroMT v1 introduces a component-based architecture to replace the previous inheritance model.
Instead of all model functionality being defined in a single ``Model`` class, a model is now composed of modular ``ModelComponent``
classes such as ``GridComponent``, ``VectorComponent``, or ``ConfigComponent``.
This structure makes models more flexible, extensible, and easier to maintain.
For detailed guidance, refer to the official `HydroMT migration guide <https://deltares.github.io/hydromt/stable/guides/plugin_dev/migrating_to_v1.html>`_.

For HydroMT-Wflow, the names and arguments of a few setup methods have changed. The names of the maps in staticmaps have been updated to reflect better
which process they belong to and maps that are not needed by Wflow.jl now start with ``meta_`` to indicate they are metadata only. The names of maps
in staticmaps is now also customizable to avoid having to duplicate fully staticmaps.nc for small changes or scenario analysis.

**For HydroMT-Wflow, one of the major change is that** ``wflow`` **and** ``WflowModel`` **are now** ``wflow_model`` **and** ``WflowSbmModel`` **.**

YAML Configuration Changes
--------------------------

Information
^^^^^^^^^^^
The HydroMT model configuration format has been overhauled and the ini format is not supported anymore.
The root YAML file now includes three main keys: ``modeltype``, ``global``, and ``steps``.

- ``modeltype`` (optional): Defines which model plugin is being used (e.g. ``wflow_sbm`` or ``wflow_sediment``).
- ``global``: Defines model-wide configuration, including data catalog(s), name of the model configuration toml file etc.
- ``steps``: Replaces the old numbered dictionary format with a sequential list of function calls.

Some of the functions (component specific read and write) are now explicitly mapped to model or component methods using the `<component>.<method>` syntax.

For a complete example of the new configuration format, see the Wflow v1 YAML template: :download:`wflow_build.yml </_examples/wflow_build.yml>`.

For more information on the format changes, see this section in the HydroMT migration guide: `Changes to the yaml HydroMT configuration file format <https://deltares.github.io/hydromt/stable/guides/plugin_dev/migrating_to_v1.html#changes-to-the-yaml-hydromt-configuration-file-format>`_.

How to upgrade
^^^^^^^^^^^^^^
In general, we advise to switch to the new YAML format by re-using and adapting the new template files provided in the HydroMT-Wflow examples folder:

- :download:`Wflow-SBM Build yml </_examples/wflow_build.yml>`
- :download:`Wflow-Sediment Build yml </_examples/wflow_sediment_build.yml>`

If you do wish to update an existing file, you will have to do this manually. Here are some things you should be aware of:

- The ``global`` section remains the same. ``config_fn`` is renamed to ``config_filename``.
- Add the ``steps`` section and convert the numbered dictionary to a list of function calls. Be careful with indents.
- Specific ``read`` and ``write`` functions are now at the component level. For example, ``write_forcing`` becomes ``forcing.write``.
- ``setup_config`` now needs a ``data`` argument rather than directly the Wflow model options.
- The names of the setup methods have changed only for the reservoirs and lakes components:
  - ``setup_lakes`` becomes ``setup_reservoirs_no_control`` (some of the arguments have then also changed)
  - ``setup_reservoirs`` becomes ``setup_reservoirs_simple_control``
- ``setup_basemaps`` must include the ``region`` (no longer available from CLI)
- ``setup_rivers`` method is now split into two methods:
  - ``setup_rivers``: sets up the river network and cross-sections
  - ``setup_river_roughness``: sets up manning roughness of the river
- ``setup_lulcmaps`` and equivalent: the parameters have changed. User-defined land use mapping tables will need to update the name of the columns.
- ``setup_constant_pars``: the paramaters are now the Wflow.jl parameter names rather than a name that HydroMT adds to the staticmaps.
- ``setup_other_demand`` and ``setup_lulcmaps_with_paddy``: some of the variables have been renamed (e.g. ``industry_net``, ``demand_paddy_h_min``).

Data Catalog Format Changes
---------------------------

Information
^^^^^^^^^^^
The data catalog structure has been refactored to introduce a more modular design and clearer separation of responsibilities across several new classes (`DataSource`, `Driver`, `URIResolver`, and `DataAdapter`).

Key format changes:

- ``path`` renamed to ``uri``
- ``filesystem`` or ``driver_kwargs`` moved under ``driver``
- ``unit_add``, ``unit_mult``, ``rename``, etc. moved under ``data_adapter``
- ``crs`` and ``nodata`` moved under ``metadata`` (renamed from ``meta``)
- A single catalog entry can now reference multiple data variants or versions

For detailed information about the format changes, see this section in the HydroMT migration guide: `Changes to the data catalog yaml file format <https://deltares.github.io/hydromt/stable/guides/plugin_dev/migrating_to_v1.html#datacatalog>`_

How to upgrade
^^^^^^^^^^^^^^
All existing pre-defined catalogs have been updated to the new format. For your own catalogs, you can upgrade
easily with the HydroMT ``check`` command:

.. code-block:: bash

   hydromt check -d /path/to/data_catalog.yml --format v0 --upgrade -v

Main Changes for Python Users
-----------------------------

In HydroMT-Wflow v1, the internal data structure and API were redesigned to improve consistency and maintainability.
Most changes affect how model components (such as ``staticmaps`` and ``forcing``) are accessed and how model data is read and written.
Another change is that to better differentiate between wflow SBM and flow Sediment, the ``WflowModel`` class is now ``WflowSbmModel``.

Component Changes
^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read and write operations are now handled at the **component level**.

.. code-block:: python

    # Write configuration file
    model.config.write()

    # Write updated staticmaps to disk
    model.staticmaps.write()

    # Read forcing component explicitly
    model.forcing.read()

These changes provide a clearer and more modular interface, making it easier to manipulate model components independently.
