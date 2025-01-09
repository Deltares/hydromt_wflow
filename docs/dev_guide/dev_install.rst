.. _dev_env:

Developer's environment
=======================
If you want to download the HydroMT-Wflow plugin directly from git to easily have access to the latest developments or
make changes to the code you can use the following steps.

First, clone the HydroMT-Wflow plugin ``git`` repo from
`github <https://github.com/Deltares/hydromt_wflow>`_, then navigate into the
the code folder (where the envs folder and pyproject.toml are located):

.. code-block:: console

    $ git clone https://github.com/Deltares/hydromt_wflow.git
    $ cd hydromt_wflow

The next step is to create an environment yaml.
This is done by calling the `make_env.py` script from commandline:
(This requires the presence of the package `tomli` or Python 3.11)

.. code-block:: console

    $ python make_env.py full

The 'full' argument guarantees all the necessary tools!
Now create the environment using conda and activate it:

.. code-block:: console

    $ conda env create -f environment.yml
    $ conda activate hydromt_wflow

If you wish to make changes in HydroMT-Wflow, you should make an editable install of the plugin.
This can be done with:

.. code-block:: console

    $ pip install -e .

If you encounter issues with the installation of some packages, you might consider cleaning conda to remove unused packages and caches.
This can be done through the following command from your base environment:

.. code-block:: console

    $ conda clean -a

It may also be advisable to clear pip's cache:

.. code-block:: console

    $ pip cache purge
