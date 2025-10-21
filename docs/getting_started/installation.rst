.. _installation_guide:

==================
Installation Guide
==================

Installation
============

There are various ways in which one can install hydromt_wflow in a python environment, the main being:

**Choosing an installation method**

While both Conda and Pixi provide robust ways to install HydroMT-Wflow, your choice depends on your workflow and environment needs.

Conda is a well-established package manager, ideal if you want direct control over environment creation and dependency resolution.
It is widely used in scientific computing and works well when integrating HydroMT-Wflow into an existing Conda-based workflow.
Use this if you are already familiar with Conda environments or need tighter control over package versions.

Pixi is a project-based environment manager that simplifies dependency handling and environment reproducibility.
It is our recommended option for developers or if you would like to update the code because it handles both Conda and PyPI dependencies in a streamlined way and ensures consistency across installations.
Pixi is especially useful if you want to encapsulate HydroMT-Wflow and its dependencies within a project environment without manually managing Conda channels or environments.

Choose Conda if you want flexibility and integration with existing workflows, and Pixi if you prefer a streamlined project-oriented installation with minimal manual configuration.

- :ref:`Miniforge/conda <user_install_conda>`
- :ref:`Pixi <user_install_pixi>`

.. _user_install_conda:

Installation using Conda
------------------------

.. warning::

  Due to the changes Anaconda made to their `licencing agreements in 2024 <https://legal.anaconda.com/policies/en/?name=terms-of-service#anaconda-terms-of-service>`
  using any packages from the anaconda channel (which is available by default in the main `conda` and `mamba` distributions) may require a paid license.
  Therefore we highly recommend you only use the free and community maintained `conda-forge` channel. While you can configure existing `conda` / `mamba`
  installations to do this correctly, we recommend that if you do not want to use pixi, that you use a `miniforge<https://github.com/conda-forge/miniforge>` distribution which has this correctly
  configured by default.

You can install HydroMT-Wflow in a new environment called ``hydromt-wflow``:

.. code-block:: console

  $ conda create -n hydromt-wflow -c conda-forge hydromt_wflow

Then, activate the environment (as stated by mamba/conda depending on which you are using) to start making use of that environment:

.. code-block:: console

  $ conda activate hydromt-wflow

After it has been activated you can install hydromt-wflow into it using this command:

.. code-block:: console

  (hydromt-wflow) $ conda install hydromt_wflow

.. Tip::

    If you already have this environment with this name either remove it with
    `conda env remove -n hydromt-wflow` **or** set a new name for the environment
    by changing `-n <name>` to the name of your new environment.

After you have installed ``hydromt_wflow`` in your environment you will also need to add
the optional dependencies to it, if you want to make use of all the functionalities available:

.. code-block:: console

  (hydromt-wflow) $ pip install gwwapi
  (hydromt-wflow) $ pip install hydroengine
  (hydromt-wflow) $ conda install pcraster
  (hydromt-wflow) $ conda install wradlib
  (hydromt-wflow) $ conda install pyet
  (hydromt-wflow) $ conda install gdal

For a more indepth explanation on the dependencies see :ref:`this section <optional_dependencies>`.


.. _user_install_pixi:

Installation using Pixi
-----------------------

.. Tip::

    This is our recommended way of installing HydroMT-Wflow!


If you do not have a ``pyproject.toml`` yet you can make one by executing the command:

.. code-block:: console

    $ pixi init --format pyproject myproject


A new folder with a ``pyproject.toml`` called ``myproject`` will be created for you. After this, you can
navigate to this new directory and add ``hydromt_wflow`` as a dependency:

.. code-block:: console

    $ pixi add hydromt_wflow[extra] --pypi

Pixi will then add it as a dependency to the project. The ``[extra]`` instruct pixi to also
include some optional dependencies.

If you don't want these extra dependencies (or need specific versions, or only want some)

You can also add it like so:

.. code-block:: console

    $ pixi add hydromt_wflow

This will install hydromt_wflow from conda-forge without any of the optional dependencies
which you can then add yourself afterwards as you like:

.. code-block:: console

  $ pixi add gwwapi --pypi
  $ pixi add hydroengine --pypi
  $ pixi add pcraster
  $ pixi add wradlib
  $ pixi add pyet
  $ pixi add gdal

the ``--pypi`` in this case is necessary because these dependencies are only available through pypi and not conda-forge
adding this flag will tell pixi to install them from there.

For a more indepth explanation on the dependencies see :ref:`this section <optional_dependencies>`.

Once you have your new (or existing ``pyproject.toml``) file install the pixi
environment and activate it with the following commands to be able to start using it:

.. code-block:: console

    $ pixi install
    $ pixi shell activate


If you did activate the shell like above you should now be able to run any python script like usual:

.. code-block:: console

  (hydromt-wflow) $ python path/to/script.py

If you did not activate the shell you can still run the script in the environment by running it through pixi:

.. code-block:: console

  (hydromt-wflow) $ pixi run path/to/script.py

If you intend to only use ``hydromt_wflow`` via the command line interface (CLI, see also the explanation in
the `HydroMT-core docs <https://deltares.github.io/hydromt/stable/guides/user_guide/hydromt_cli.html>`_), then you can also install it globally using pixi.
This will allow you to access HydroMT functionality without having to create a `pyproject.toml` file for your project.
Building or updating a Wflow model is done by calling `hydromt wflow ...` in the CLI, which means that we need to access `hydromt_wflow` through `hydromt`.

This means that we need to globally install `hydromt` with a `hydromt_wflow` dependency:

.. code-block:: console

  $ pixi global install hydromt --expose hydromt="hydromt" --with hydromt_wflow

This will install hydromt and hydromt_wflow in an isolated environment for you and make it available to run from basically
anywhere on your system through the commandline. For more information on global tools in pixi, see also the description in the
`pixi documentation <https://pixi.sh/latest/global_tools/introduction/#basic-usage>`_.

.. Warning::

  The current version of hydromt_wflow is not compatible with hydromt version 1 and later. As a result, we need to restrict the version of hydromt
  to ensure that the latest version of hydromt_wflow is installed:

  .. code-block:: console

      $ pixi global install hydromt"==0.10.1" --expose hydromt="hydromt" --with hydromt_wflow

Install HydroMT-Wflow in an existing environment
------------------------------------------------

To install HydroMT-Wflow in an existing environment execute the command below
where you replace ``<environment_name>`` with the name of the existing environment.
Note that if some dependencies are not installed from conda-forge but from other
channels the installation may fail.

.. code-block:: console

  $ conda install -c conda-forge hydromt_wflow -n <environment_name>

.. code-block:: console

  $ conda activate <environment_name>

After you have installed ``hydromt_wflow`` in your environment you will also need to add
the optional dependencies to it, if you want to make use of all the functionalities available:

.. code-block:: console

  (<environment_name>) $ pip install gwwapi
  (<environment_name>) $ pip install hydroengine
  (<environment_name>) $ conda install pcraster
  (<environment_name>) $ conda install wradlib
  (<environment_name>) $ conda install pyet
  (<environment_name>) $ conda install gdal

For a more indepth explanation on the dependencies see :ref:`this section <optional_dependencies>`.


Developer install
==================
To be able to test and develop the HydroMT-Wflow package see instructions in the :ref:`Developer installation guide <dev_env>`.


.. _optional_dependencies:

Optional Dependencies
=====================

HydroMT-Wflow has several optional dependencies that need to be installed in your environment to enable specific
functionalities, though they are not necessary for hydromt-wflow to function as a whole. Due to limitations in the conda-forge
package specification you will have to install these yourself in addition to hydromt-wflow if you want to use them.

They are:

- `pcraster <https://pcraster.geo.uu.nl>`_ This package is used for the reading and writing of pcr maps, which is the file format used by the old Wflow software (written in Python). It is therefore required when you want to convert Wflow models from the Python version to the Julia version. Note that the pcraster package is only available on conda-forge which works if you install hydromt-wflow using Pixi or Conda. If you install hydromt-wflow through pypi you will not be able to access this functionality.
- `gwwapi <https://github.com/global-water-watch/gww-api>`_ This package is used for providing more resources about reservoirs monitored by the Global Water Watch project. This package is only available through pypi at the moment.
- `hydroengine <https://github.com/openearth/hydro-engine>`_ Similar to ``gwwapi`` the ``hydroengine`` package gives access to more data sources, and is currently only available through pypi.
- `wradlib <https://github.com/wradlib/wradlib>`_ Provides downloading and processing functionalities of radar weather data and interpolation functions. Available in both conda-forge and pypi.
- `pyet <https://github.com/pyet-org/pyet>`_ Provides processing functionalities of several methods to calculate evapotranspriation. Available in both conda-forge and pypi.
- `gdal <https://gdal.org/en/stable/>`_ Provides many drivers and GIS transformations.  Only available through conda-forge.

Since some dependencies are only available through conda-forge and some only through pypi, you will need a packange manager that can handle both.
