Installation
============

User install
------------

HydroMT is available from pypi and conda-forge, but we recommend installing with conda.
It is the same for the wflow plugin.

To install HydroMT wflow plugin using conda do:

.. code-block:: console

    $ conda install hydromt_wflow -c conda-forge

To install a HydroMT environment with conda installed do:

.. code-block:: console

    $ conda create hydromt_wflow -n hydromt-wflow -c conda-forge

This will automatically install both HydroMT core library and dependencies as well as the model plugin.

.. note::

  If your model plugin is not available via conda install, you can also use pip:
  
  .. code-block:: console
  
      $ pip install hydromt_wflow


Developper install
------------------
If you want to download the wflow plugin directly from git to easily have access to the latest developmemts or 
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

Finally, build and install hydromt_wflow using pip. If you wish to develop in hydromt_wflow, then 
make an editable install of hydromt by adding ``-e`` after install:

.. code-block:: console

    $ pip install .

or for developpers:

.. code-block:: console

    $ pip install -e .

For more information about how to contribute, see `HydroMT contributing guidelines <https://hydromt.readthedocs.io/en/latest/contributing.html>`_.
