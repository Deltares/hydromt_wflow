.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Deltares/hydromt_wflow/main?urlpath=lab/tree/examples

This folder contains several iPython notebook examples for **HydroMT-wflow**.

These examples can be run online or on your local machine.
To run these examples online press the **binder** badge above.

Local installation
------------------

To run these examples on your local machine you need a copy of the examples folder
of the repository and an installation of HydroMT-Wflow including some additional
packages required to run the notebooks.

1 - Install HydroMT-wflow
*************************

The first step is to install HydroMT-Wflow and the other Python dependencies in a separate environment,
see the **Install HydroMT-Wflow in a new environment** section in the
`installation guide <https://deltares.github.io/hydromt_wflow/latest/getting_started/installation.html>`_

To run the notebooks, you need to install additional dependencies that are jupyterlab to
run the notebooks and cartopy for some plots. To install these packages in your existing
hydromt-wflow environment, do (these packages are also available in conda-forge):

.. code-block:: console

  $ conda activate hydromt-wflow
  $ pip install jupyterlab
  $ pip install cartopy

.. note::

  To run the example notebook using pcraster, you will also need to install it from
  conda-forge (not available on pypi):

  .. code-block:: console

    $ conda install -c conda-forge pcraster

2 - Download the content of the examples and notebooks
******************************************************
To run the examples locally, you will need to download the content of the hydromt_wflow repository.
You have two options:

  1. Download and unzip the examples manually
  2. Clone the hydromt_wflow GitHub repository

.. warning::

  Depending on your installed version of hydromt and hydromt_wflow, you will need to download the correct versions of the examples.
  To check the version of hydromt_wflow that you have installed, do:

  .. code-block:: console

    $ hydromt --models

    model plugins:
     - wflow (hydromt_wflow 0.4.1)
     - wflow_sediment (hydromt_wflow 0.4.1)
    generic models (hydromt 0.9.1)
     - grid_model
     - vector_model
     - mesh_model
     - network_model

In the examples above, we see version 0.4.1 of hydromt_wflow is installed and version 0.9.1 of hydromt.

**Option 1: manual download and unzip**

To manually download the examples on Windows, do (!replace with your own hydromt_wflow version!):

.. code-block:: console

  $ curl https://github.com/Deltares/hydromt_wflow/archive/refs/tags/v0.4.1.zip -O -L
  $ tar -xf v0.4.1.zip
  $ ren hydromt_wflow-0.4.1 hydromt_wflow

You can also download, unzip and rename manually if you prefer, rather than using the windows command prompt.

**Option 2: cloning the hydromt_wflow repository**

For git users, you can also get the examples by cloning the hydromt_wflow github repository and checkout the version
you have installed:

.. code-block:: console

  $ git clone https://github.com/Deltares/hydromt_wflow.git
  $ git checkout v0.4.1

3 - Running the examples
************************
Finally, start a jupyter lab server inside the **examples** folder
after activating the **hydromt-wflow** environment, see below.

Alternatively, you can run the notebooks from `Visual Studio Code <https://code.visualstudio.com/download>`_.

.. code-block:: console

  $ conda activate hydromt-wflow
  $ cd hydromt-wflow/examples
  $ jupyter lab
