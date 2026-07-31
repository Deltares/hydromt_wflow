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

You will need **Python 3.11 or greater** and a package manager such as uv, pixi, or others in order to use HydroMT-Wflow.
These package managers help you to install (Python) packages and manage environments such that different installations do not conflict.

If you do not yet have one installed, we recommend either:

- `uv <https://docs.astral.sh/uv/>`_: uses `pypi.org <https://pypi.org>`_ for downloading dependencies.
- `pixi <https://pixi.sh>`_: by default uses `conda-forge <https://conda-forge.org/>`_ for downloading dependencies,
but will also search `pypi.org <https://pypi.org>`_ for dependencies it cannot find on conda-forge.

It is also possible to use other package managers, such as pip or conda.
The benefits of uv and pixi over pip and conda are that they install Python directly in the project folder,
which avoids conflicts with other packages and allows you to have multiple versions of Python installed on your system.
They work differently to conda/pip, where the typical workflow is to activate a named environment first
(for example with ``conda activate <env>``), and then run installation and CLI commands from that active environment.
In uv and pixi, project environment management is tied to the project folder itself and is integrated into one tool-driven
workflow and commands are typically run through the project tool (for example ``pixi run ...`` or ``uv run ...``).


Installing HydroMT-Wflow
========================

HydroMT-Wflow is available from PyPI and conda-forge, and can be installed using uv, pixi, or others.
Here we will describe the installation using **uv** and the **pixi** package manager.

Basic Installation
------------------

We strongly recommend installing HydroMT-Wflow in a separate environment to avoid conflicts with other packages.
Therefore we use uv or pixi to create a new project folder with Python directly installed.

.. tab-set::
  :sync-group: package-manager

  .. tab-item:: uv
    :sync: uv

    .. code-block:: console

      $ uv init my_project
      $ cd my_project
      $ uv add project
      $ uv sync

    .. note::
      If you want to develop a model plugin, we recommend running :code:`uv init` with the ``--library`` option,
      which will create a library project instead of an application project.

  .. tab-item:: pixi
    :sync: pixi

    .. code-block:: console

      $ pixi init my_project
      $ cd my_project
      $ pixi add project

    .. note::
      pixi resolves packages by default via conda-forge, if there are still unresolved packages it will try to use pypi.
      It is also possible to use pypi by calling :code:`pixi add --pypi hydromt_wflow`.
      We recommend to not mix conda-forge and pypi packages in the same environment, but it is possible.

To test whether the installation was successful, run :code:`uv run hydromt --plugins` on uv, or :code:`pixi run hydromt --plugins` on pixi.
The output should look similar to the example below:

.. tab-set::
  :sync-group: package-manager

  .. tab-item:: uv
    :sync: uv

    .. code-block:: console

      $ uv run hydromt --plugins
        Model plugins:
            - model (hydromt x.y.z)
            - wflow_sbm (hydromt_wflow x.y.z)
            - wflow_sediment (hydromt_wflow x.y.z)
        Component plugins:
            - ConfigComponent (hydromt x.y.z)
            - DatasetsComponent (hydromt x.y.z)
            - GeomsComponent (hydromt x.y.z)
            - GridComponent (hydromt x.y.z)
            - MeshComponent (hydromt x.y.z)
            - SpatialDatasetsComponent (hydromt x.y.z)
            - TablesComponent (hydromt x.y.z)
            - VectorComponent (hydromt x.y.z)
        Driver plugins:
            - dataset_xarray (hydromt x.y.z)
            - geodataframe_table (hydromt x.y.z)
            - geodataset_vector (hydromt x.y.z)
            - geodataset_xarray (hydromt x.y.z)
            - pandas (hydromt x.y.z)
            - pyogrio (hydromt x.y.z)
            - raster_xarray (hydromt x.y.z)
            - rasterio (hydromt x.y.z)
        Catalog plugins:
            - deltares_data (hydromt x.y.z)
            - artifact_data (hydromt x.y.z)
            - aws_data (hydromt x.y.z)
            - gcs_cmip6_data (hydromt x.y.z)
        Uri_resolver plugins:
            - convention (hydromt x.y.z)
            - raster_tindex (hydromt x.y.z)

  .. tab-item:: pixi
    :sync: pixi

    .. code-block:: console

      $ pixi run hydromt --plugins
        Model plugins:
            - model (hydromt x.y.z)
            - wflow_sbm (hydromt_wflow x.y.z)
            - wflow_sediment (hydromt_wflow x.y.z)
        Component plugins:
            - ConfigComponent (hydromt x.y.z)
            - DatasetsComponent (hydromt x.y.z)
            - GeomsComponent (hydromt x.y.z)
            - GridComponent (hydromt x.y.z)
            - MeshComponent (hydromt x.y.z)
            - SpatialDatasetsComponent (hydromt x.y.z)
            - TablesComponent (hydromt x.y.z)
            - VectorComponent (hydromt x.y.z)
        Driver plugins:
            - dataset_xarray (hydromt x.y.z)
            - geodataframe_table (hydromt x.y.z)
            - geodataset_vector (hydromt x.y.z)
            - geodataset_xarray (hydromt x.y.z)
            - pandas (hydromt x.y.z)
            - pyogrio (hydromt x.y.z)
            - raster_xarray (hydromt x.y.z)
            - rasterio (hydromt x.y.z)
        Catalog plugins:
            - deltares_data (hydromt x.y.z)
            - artifact_data (hydromt x.y.z)
            - aws_data (hydromt x.y.z)
            - gcs_cmip6_data (hydromt x.y.z)
        Uri_resolver plugins:
            - convention (hydromt x.y.z)
            - raster_tindex (hydromt x.y.z)

Installing optional dependencies
--------------------------------

HydroMT-Wflow provides several optional dependencies that extend its capabilities,
such as additional data sources or hydrological processing functions.

Optional packages include:

- **gwwapi** - provides access to Global Water Watch reservoir datasets.
- **hydroengine** - enables integration with Google Earth Engine.
- **wradlib** - provides radar rainfall processing and interpolation tools.
- **pyet** - adds evapotranspiration computation support.

To install these optional dependencies, you can use the following uv/pip commands:

.. tab-set::
  :sync-group: package-manager

  .. tab-item:: uv
    :sync: uv

    .. code-block:: console

      $ uv add "hydromt_wflow[docs]"
      $ uv add "hydromt_wflow[examples]"
      $ uv add "hydromt_wflow[extra]"
      $ uv add "hydromt_wflow[test]"
      $ uv add "hydromt_wflow[all]"

  .. tab-item:: pixi
    :sync: pixi

    Not required, as pixi install via conda-forge, which contains all optional dependencies by default.

For a list of all the optional dependency groups and their contents, have a look at the `pyproject.toml` file. Use `hydromt_wflow[all]` to install all optional dependencies.

Developer installation
======================

If you want to contribute to HydroMT-Wflow or modify its source code, see the
:ref:`Developer installation guide <dev_env>`.

To install the latest development version of HydroMT-Wflow,
you can clone the repository and checkout the desired version tag:

.. code-block:: console

  $ git clone https://github.com/Deltares/hydromt_wflow.git
  $ git checkout v<x.y.z>
