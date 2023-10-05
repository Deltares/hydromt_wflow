.. _installation_guide:

==================
Installation Guide
==================

Prerequisites
=============
For more information about the prerequisites for an installation of the HydroMT package
and related dependencies, please visit the documentation of
`HydroMT core <https://deltares.github.io/hydromt/latest/getting_started/installation.html#installation-guide>`_

Compared to HydroMT, HydroMT-Wflow has additional dependencies, namely:

- `toml <https://github.com/uiri/toml>`_
- `pcraster <https://pcraster.geo.uu.nl>`_ (optional)
- `gwwapi <https://github.com/global-water-watch/gww-api>`_ (optional)
- `hydroengine <https://github.com/openearth/hydro-engine>`_ (optional)

If you already have a python & conda installation but do not yet have mamba installed,
we recommend installing it into your *base* environment using:

.. code-block:: console

  $ conda install mamba -n base -c conda-forge


Installation
============

HydroMT-Wflow is available from pypi and conda-forge.
We recommend installing using mamba from conda-forge in a new environment.

.. Note::

    In the commands below you can exchange `mamba` for `conda`, see
    `here <https://deltares.github.io/hydromt/latest/getting_started/installation.html#installation-guide>`_
    for the difference between both.

Install HydroMT-Wflow in a new environment (recommended!)
---------------------------------------------------------

You can install HydroMT-Wflow in a new environment called `hydromt-wflow` together with
all optional (see above) and a few additional dependencies with:

.. code-block:: console

  $ mamba env create -f https://raw.githubusercontent.com/Deltares/hydromt_wflow/main/environment.yml

Then, activate the environment (as stated by mamba/conda) to start making use of HydroMT-Wflow:

.. code-block:: console

  conda activate hydromt-wflow

.. Tip::

    If you already have this environment with this name either remove it with
    `conda env remove -n hydromt-wflow` **or** set a new name for the environment
    by adding `-n <name>` to the line below.

Install HydroMT-Wflow in an existing environment
------------------------------------------------

To install HydroMT-Wflow in an existing environment execute the command below
where you replace `<environment_name>` with the name of the existing environment.
Note that if some dependencies are not installed from conda-forge but from other
channels the installation may fail.

.. code-block:: console

   $ mamba install -c conda-forge hydromt_wflow -n <environment_name>

.. Note::

    Please take into account that gwwapi or hydroengine packages are not available from conda and therefore have to be installed from pypi separately.

.. code-block:: console

  $ pip install gwwapi
  $ pip install hydroengine

Developer install
==================
To be able to test and develop the HydroMT-Wflow package see instructions in the :ref:`Developer installation guide <dev_env>`.
