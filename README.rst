hydroMT-wflow: wflow plugin for hydroMT
#######################################

.. image:: https://codecov.io/gh/Deltares/hydromt_wflow/branch/main/graph/badge.svg?token=ss3EgmwHhH
    :target: https://codecov.io/gh/Deltares/hydromt_wflow

.. image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
    :target: https://deltares.github.io/hydromt_wflow/latest
    :alt: Latest developers docs

.. image:: https://img.shields.io/badge/docs-stable-brightgreen.svg
    :target: https://deltares.github.io/hydromt_wflow/stable
    :alt: Stable docs last release

.. image:: https://badge.fury.io/py/hydromt_wflow.svg
    :target: https://pypi.org/project/hydromt_wflow/
    :alt: Latest PyPI version

.. image:: https://anaconda.org/conda-forge/hydromt/badges/installer/conda.svg
    :target: https://anaconda.org/conda-forge/hydromt_wflow

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Deltares/hydromt_wflow/main?urlpath=lab/tree/examples

hydroMT_ is a python package, developed by Deltares, to build and analysis hydro models.
It provides a generic model api with attributes to access the model schematization,
(dynamic) forcing data, results and states. This plugin provides an implementation 
for the wflow_ model.


.. _hydromt: https://deltares.github.io/hydromt

.. _wflow: https://github.com/Deltares/Wflow.jl


Installation
------------

hydroMT_wflow is available from pypi and conda-forge, but we recommend installing with conda.

To install hydromt_wflow using conda do:

.. code-block:: console

  conda install -c conda-forge hydromt_wflow

This will install both the hydroMT core library and the hydromt_wflow plugin. To get a full installation, 
manually add the hydroengine dependency in your environment (not available from conda install):

.. code-block:: console

  pip install hydroengine

Documentation
-------------

Learn more about hydroMT in its `online documentation <http://deltares.github.io/hydromt_wflow/latest/>`_

Contributing
------------

You can find information about contributing to hydroMT at our `Contributing page <http://deltares.github.io/hydromt_wflow/latest/contributing.html>`_.

License
-------

Copyright (c) 2021, Deltares

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
