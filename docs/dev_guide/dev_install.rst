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

once you are in the directory you can install the environment using pixi with

.. code-block:: console

    $ pixi install

Afterwards you can run any of the tasks we defined for doing things such as running the test suite:

.. code-block:: console

    $ pixi run test

generating the documentation:

.. code-block:: console

    $ pixi run docs-html

We have several environments to make sure we can test ``hydromt_wflow`` under, for example,
with all of the different python versions we support. The default environment should have
everything you need, but if you want to run something (for example in CI) from a specific
environment you can do so like this:

.. code-block:: console

  $ pixi -e test-full-py311 run path/to/script.py


Editors like vscode should be able to find pixi environments you make automatically,
however if you use something else like `vim` or `helix` then you can make sure they run in
the context of your environment:

.. code-block:: console

    $ pixi run hx .

this will make sure your LSP can find your environment so it can give you proper
tab completion and other warnings if necessary.
