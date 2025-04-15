.. _installation_guide:

==================
Installation Guide
==================



Prerequisites
=============
For more information about the prerequisites for an installation of the HydroMT package
and related dependencies, please visit the documentation of
`HydroMT core <https://deltares.github.io/hydromt/latest/guides/user_guide/installation.html>`_

Compared to HydroMT, HydroMT-Wflow has additional dependencies, namely:

- `toml <https://github.com/uiri/toml>`_
- `pcraster <https://pcraster.geo.uu.nl>`_ (optional)
- `gwwapi <https://github.com/global-water-watch/gww-api>`_ (optional)
- `hydroengine <https://github.com/openearth/hydro-engine>`_ (optional)
- `wraplib <https://github.com/wradlib/wradlib>`_ (optional)
- `pyet <https://github.com/pyet-org/pyet>`_ (optional)

Installation
============

There are various ways in which one can install hydrom_wflow in a python environment, the main being:

- :ref:`Pixi <user_install_pixi>`
- :ref:`Miniforge/conda <user_install_conda>`

All of these options will be discussed below but we recommend using pixi, as it is the fastest and most
robust (and reproducible) option.

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

    $ pixi add hydromt_wflow[extra, examples] --pypi

Pixi will then add it as a dependency to the project. The ``[full]`` is to make sure you include all
of the optional dependencies.

You can also add the optional dependencies to it like so:

.. code-block:: console

  $ pixi add gwwapi --pypi
  $ pixi add hydroengine --pypi
  $ pixi add pcraster
  $ pixi add --pypi wradlib
  $ pixi add --pypi pyet
  $ pixi add gdal

the `--pypi` in this case is necessary because these dependencies are only available through pypi and not conda-forge
adding this flag will tell pixi to install them from there.

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

If you intend to only use `hydromt_wflow` via the cli you can also install it globally using pixi like so:

.. code-block:: console

  $ pixi global install hydromt_wflow

This will install hydromt_wflow in an isolated environment for you and make it available to run from basically
anywhere on your system through the commandline


.. _user_install_conda:

Installation using Conda
------------------------

.. warning::

  Due to the changes Anaconda made to their `lisencing agreements in 2024 <https://legal.anaconda.com/policies/en/?name=terms-of-service#anaconda-terms-of-service>`
  using any packages from the anaconda channel (which is available by default in the main `conda` and `mamba` distributions) may require a paid license.
  Therefore we highly recommend you only use the free and community maintained `conda-forge` channel. While you can configure existing `conda` / `mamba`
  installations to do this correctly, we recommend that if you do not want to use pixi, that you use a `miniforge<https://github.com/conda-forge/miniforge>` distribution which has this correctly
  configured by default.

You can install HydroMT-Wflow in a new environment called `hydromt-wflow`:

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

.. Note::

    Please take into account that gwwapi or hydroengine packages are not available from conda and therefore have to be installed from pypi separately (in the activated environment).

    .. code-block:: console

      (hydromt-wflow) $ pip install gwwapi
      (hydromt-wflow) $ pip install hydroengine

Install HydroMT-Wflow in an existing environment
------------------------------------------------

To install HydroMT-Wflow in an existing environment execute the command below
where you replace `<environment_name>` with the name of the existing environment.
Note that if some dependencies are not installed from conda-forge but from other
channels the installation may fail.

.. code-block:: console

  $ conda install -c conda-forge hydromt_wflow -n <environment_name>

.. Note::

    Please take into account that gwwapi or hydroengine packages are not available from conda and therefore have to be installed from pypi separately.

.. code-block:: console

  $ conda activate <environment_name>
  $ pip install gwwapi
  $ pip install hydroengine

Developer install
==================
To be able to test and develop the HydroMT-Wflow package see instructions in the :ref:`Developer installation guide <dev_env>`.
