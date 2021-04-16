hydroMT-wflow: wflow plugin for hydroMT
#######################################

.. image:: https://codecov.io/gh/Deltares/hydromt_wflow/branch/main/graph/badge.svg?token=ss3EgmwHhH
    :target: https://codecov.io/gh/Deltares/hydromt_wflow

.. image:: https://readthedocs.org/projects/hydromt_wflow/badge/?version=latest
    :target: https://hydromt_wflow.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

hydroMT_ is a python package, developed by Deltares, to build and analysis hydro models.
It provides a generic model api with attributes to access the model schematization,
(dynamic) forcing data, results and states. This plugin provides an implementation 
for the wflow_ model.


.. _hydromt: https://deltares.github.io/hydromt

.. _wflow: https://github.com/Deltares/Wflow.jl


Installation
------------

hydroMT is availble from pypi and conda-forge, but we recommend installing with conda.

To install hydromt using conda do:

.. code-block:: console

  conda install hydromt_wflow -c conda-forge

To create a hydromt environment with conda installed do:

.. code-block:: console

  conda create hydromt -n hydromt_wflow -c conda-forge

Documentation
-------------

Learn more about hydroMT in its `online documentation <https://hydromt_wflow.readthedocs.io/en/latest/>`_

Contributing
------------

You can find information about contributing to hydroMT at our `Contributing page <https://hydromt_wflow.readthedocs.io/en/latest/contributing.html>`_.

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
