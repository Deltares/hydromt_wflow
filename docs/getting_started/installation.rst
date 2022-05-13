.. _installation_guide:

==================
Installation Guide
==================

Prerequisites
=============
For more information about the prerequisites for an installation of the HydroMT package and related dependencies, please visit the
documentation of `HydroMT core <https://deltares.github.io/hydromt/latest/getting_started/installation.html#installation-guide>`_

Compared to HydroMT, HydroMT-Wflow has additionnal dependencies, namely:

- `toml <https://github.com/uiri/toml>`_
- `pcraster <https://pcraster.geo.uu.nl>`_ (optional)
- `hydroengine <https://github.com/openearth/hydro-engine>`_ (optional)

Installation
============

HydroMT-Wflow is available from pypi and conda-forge, but we recommend installing from conda-forge in a new conda environment.

.. Note::

    In the commands below you can exchange `mamba` for `conda`, see
    `here <https://deltares.github.io/hydromt/latest/getting_started/installation.html#installation-guide>`_ for the difference between both.

Install HydroMT-Wflow in a new environment
------------------------------------------
.. Tip::

    This is our recommended way of installing HydroMT-Wflow!

To install HydroMT-Wflow in a new environment called `hydromt-wflow` from the conda-forge channel do:

.. code-block:: console

  $ conda create -n hydromt-wflow -c conda-forge hydromt_wflow

Then, activate the environment (as stated by mamba/conda) to start making use of HydroMT-Wflow:

.. code-block:: console

  conda activate hydromt-wflow

This will install **almost** all dependencies including the core HydroMT library and the model API as well
as the model plugins **Wflow** and **Wflow Sediment**. To complete the installation, add manually the hydroengine dependency:

.. Note::

    The hydroengine package is not available from conda and therefore has to be installed from pypi separately.

.. code-block:: console

  $ pip install hydroengine

**Alternatively** to install hydromt_wflow using pip do (not recommended):

.. Note::

    Make sure this is installed in the same environment as HydroMT.

.. code-block:: console

  $ pip install hydromt_wflow

Install HydroMT-Wflow in an existing environment
------------------------------------------------

To install HydroMT-Wflow **using mamba or conda** execute the command below after activating the correct environment.
Note that if some dependencies are not installed from conda-forge the installation may fail.

.. code-block:: console

   $ conda install -c conda-forge hydromt_wflow

.. Note::

    Take also here into account that hydroengine package is not available from conda and therefore has to be installed from pypi separately.

.. code-block:: console

  $ pip install hydroengine

For **Using pip** from pypi (not recommended) see above

Developer install
==================
To be able to test and develop the HydroMT-Wflow package see instructions in the :ref:`Developer installation guide <dev_env>`.
