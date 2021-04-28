Installation
============

User install
------------

HydroMT is available from pypi and conda-forge, but we recommend installing with conda.
It is the same for the plugin plugin.

To install HydroMT plugin plugin using conda do:

.. code-block:: console

    $ conda install hydromt_plugin -c conda-forge

To install a HydroMT environment with conda installed do:

.. code-block:: console

    $ conda create hydromt_plugin -n hydromt-plugin -c conda-forge

This will automatically install both HydroMT core library and dependencies as well as the model plugin.

.. note::

  If your model plugin is not available via conda install, you can also use pip:
  
  .. code-block:: console
  
      $ pip install hydromt_plugin


Developper install
------------------
If you want to download the plugin plugin directly from git to easily have access to the latest developmemts or 
make changes to the code you can use the following steps.

First, clone hydromt's plugin plugin ``git`` repo from
`github <https://github.com/Deltares/hydromt_plugin>`_, then navigate into the 
the code folder (where the envs folder and pyproject.toml are located):

.. code-block:: console

    $ git clone https://github.com/Deltares/hydromt_plugin.git
    $ cd hydromt_plugin

Then, make and activate a new hydromt-plugin conda environment based on the envs/hydromt-plugin.yml
file contained in the repository:

.. code-block:: console

    $ conda env create -f envs/hydromt-plugin.yml
    $ conda activate hydromt-plugin

Finally, build and install hydromt_plugin using pip. If you wish to develop in hydromt_plugin, then 
make an editable install of hydromt by adding ``-e`` after install:

.. code-block:: console

    $ pip install .

or for developpers:

.. code-block:: console

    $ pip install -e .

For more information about how to contribute, see `HydroMT contributing guidelines <https://deltares.github.io/hydromt_plugin/latest/contributing.html>`_.
