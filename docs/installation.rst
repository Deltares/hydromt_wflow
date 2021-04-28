Installation
============

User install
------------

HydroMT is available from pypi and conda-forge, but we recommend installing with conda.

The hydromt_wflow plugin is currently only available from PyPi.

To install hydromt using pip do:

.. code-block:: console

  pip install hydromt_wflow

We recommend installing a hydromt-wflow environment including the hydromt_wflow package
based on the environment.yml file in the repository root.

.. code-block:: console

  conda env create -f environment.yml


Developper install
------------------
If you want to download the wflow plugin directly from git to easily have access to the latest developments or 
make changes to the code you can use the following steps.

First, clone hydromt's wflow plugin ``git`` repo from
`github <https://github.com/Deltares/hydromt_wflow>`_, then navigate into the 
the code folder (where the envs folder and pyproject.toml are located):

.. code-block:: console

    $ git clone https://github.com/Deltares/hydromt_wflow.git
    $ cd hydromt_wflow

Then, make and activate a new hydromt-wflow conda environment based on the envs/hydromt-wflow.yml
file contained in the repository:

.. code-block:: console

    $ conda env create -f envs/hydromt-wflow.yml
    $ conda activate hydromt-wflow

Finally, build and install hydromt_wflow using pip.

.. code-block:: console

    $ pip install .

If you wish to make changes in hydromt_wflow, then you should make an editable install of hydromt. 
This is possible using the `flit <https://flit.readthedocs.io/en/latest/>`_ package and install command.

For Windows:

.. code-block:: console

    $ flit install --pth-file

For Linux:

.. code-block:: console

    $ flit install -s

For more information about how to contribute, see `HydroMT contributing guidelines <https://hydromt.readthedocs.io/en/latest/contributing.html>`_.
