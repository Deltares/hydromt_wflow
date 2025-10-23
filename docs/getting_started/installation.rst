.. _installation_guide:

==================
Installation Guide
==================

Getting started
===============

HydroMT-Wflow is a model plugin for `HydroMT <https://deltares.github.io/hydromt>`_, extending its core functionalities with Wflow-specific components and workflows.
It can be installed as a standalone package or alongside other HydroMT model plugins (e.g. HydroMT-SFINCS, HydroMT-Fiat).
We recommend installing HydroMT-Wflow in a dedicated Python environment to ensure dependency consistency.

Prerequisite: Python installation
=================================

You will need **Python 3.11 or newer** and a package/environment manager such as pip, conda, mamba or uv.
These tools simplify installing packages and managing isolated environments.

If you do not yet have one installed, we recommend either:

- `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_
- `Miniforge (Mambaforge) <https://conda-forge.org/docs/>`_
- `uv <https://docs.astral.sh/uv/>`_

Both conda variants come preconfigured with the **conda-forge** channel, which provides free and open packages used by HydroMT.

Installing HydroMT-Wflow
========================

HydroMT-Wflow is available from both **PyPI** and **conda-forge**.
The simplest and most flexible approach is to install it using **pip** inside a new environment.

Installation using uv / pip
----------------------------

We recommend creating a clean environment to avoid dependency conflicts. For example:

.. code-block:: console

    $ conda create -n hydromt-wflow uv python=3.11
    $ conda activate hydromt-wflow
    $ uv pip install hydromt_wflow

This will install HydroMT-Wflow along with HydroMT core and all required dependencies.

Installation using uv
---------------------

`uv <https://docs.astral.sh/uv/>`_ is a fast drop-in replacement for pip and virtualenv.
It creates lightweight virtual environments and installs dependencies using a lockfile-based resolver for reproducibility.

To add HydroMT-Wflow to an existing uv environment:

.. code-block:: console

    $ uv add hydromt_wflow


To verify the installation, you can list the installed HydroMT plugins:

.. code-block:: console

    $ uv run hydromt --plugins
        Model plugins:
            - model (hydromt 1.3.0)
            - wflow_sbm (hydromt_wflow 1.0.0)
            - wflow_sediment (hydromt_wflow 1.0.0)
        Component plugins:
            - ConfigComponent (hydromt 1.3.0)
            - DatasetsComponent (hydromt 1.3.0)
            - GeomsComponent (hydromt 1.3.0)
            - GridComponent (hydromt 1.3.0)
            - MeshComponent (hydromt 1.3.0)
            - SpatialDatasetsComponent (hydromt 1.3.0)
            - TablesComponent (hydromt 1.3.0)
            - VectorComponent (hydromt 1.3.0)
        Driver plugins:
            - dataset_xarray (hydromt 1.3.0)
            - geodataframe_table (hydromt 1.3.0)
            - geodataset_vector (hydromt 1.3.0)
            - geodataset_xarray (hydromt 1.3.0)
            - pandas (hydromt 1.3.0)
            - pyogrio (hydromt 1.3.0)
            - raster_xarray (hydromt 1.3.0)
            - rasterio (hydromt 1.3.0)
        Catalog plugins:
            - deltares_data (hydromt 1.3.0)
            - artifact_data (hydromt 1.3.0)
            - aws_data (hydromt 1.3.0)
            - gcs_cmip6_data (hydromt 1.3.0)
        Uri_resolver plugins:
            - convention (hydromt 1.3.0)
            - raster_tindex (hydromt 1.3.0)


Installing optional dependencies
--------------------------------

HydroMT-Wflow provides several optional dependencies that extend its capabilities, such as additional data sources or hydrological processing functions.
You can install these easily using pip's extras syntax:

.. code-block:: console

    $ uv pip install "hydromt_wflow[extra]"

Or when using uv:

.. code-block:: console

    $ uv add hydromt_wflow[extra]


This will install optional packages such as:

- **gwwapi** – provides access to Global Water Watch reservoir datasets.
- **hydroengine** – enables integration with Google Earth Engine.
- **wradlib** – provides radar rainfall processing and interpolation tools.
- **pyet** – adds evapotranspiration computation support.

For a list of all the optional dependency groups and their contents, have a look at the `pyproject.toml` file. Use `hydromt_wflow[full]` to install all optional dependencies.


Installing via conda
--------------------

HydroMT-Wflow is also available through the conda-forge channel. You can install it directly with:

.. code-block:: console

    $ conda create -n hydromt-wflow -c conda-forge hydromt_wflow
    $ conda activate hydromt-wflow

Note that some optional dependencies (e.g. ``gwwapi`` or ``hydroengine``) are only available through PyPI.
You can install them afterwards with pip inside your conda environment:

.. code-block:: console

    (hydromt-wflow) $ uv pip install "hydromt_wflow[extra]"

Developer installation
======================

If you want to contribute to HydroMT-Wflow or modify its source code, see the
:ref:`Developer installation guide <dev_env>`.

For development work, you can use either a Conda-based setup or **Pixi**, which provides a fully reproducible project environment.
Pixi should be used only in developer installations — not for general users — since it manages dependencies project-locally and is less suited for managing multiple plugins globally.
