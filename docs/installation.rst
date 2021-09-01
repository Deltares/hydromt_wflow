Installation
============

User install
------------
HydroMT and its plugins are python packages. If you do not have a python installation we recommend using 
conda and `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_.

The hydromt_wflow plugin is available from PyPi and conda-forge, but we recommend installing with conda in 
a specific environment. To create a new **hydromt-wflow** environment from the command or conda prompt, do:

.. code-block:: console

  conda create --name hydromt-wflow

This will create a new empty conda environment named hydromt-wflow. To install hydromt and the hydromt_wflow plugin
in this new environment using conda do:

.. code-block:: console

  conda activate hydromt-wflow
  conda install -c conda-forge hydromt_wflow

This will install **almost** all dependencies including the core hydroMT library and the model API as well 
as the model plugins **wflow** and **wflow_sediment**. To complete the installation, add manually the hydroengine dependency 
from pypi (not available from conda):

.. code-block:: console

  pip install hydroengine


**Alternatively** to install hydromt_wflow using pip do:
Note: make sure this is installed in the same environment as hydromt.

.. code-block:: console

  pip install hydromt_wflow


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

Finally, for a fixed installation, build and install hydromt_wflow using pip.

.. code-block:: console

    $ pip install .

If you wish to make changes in hydromt_wflow, you should make an editable install of hydromt. 
This is possible using the `flit <https://flit.readthedocs.io/en/latest/>`_ package and install command.

For Windows:

.. code-block:: console

    $ flit install --pth-file

For Linux:

.. code-block:: console

    $ flit install -s

For more information about how to contribute, see `HydroMT contributing guidelines <https://deltares.github.io/hydromt/latest/contributing.html>`_.
