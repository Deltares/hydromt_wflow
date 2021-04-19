hydroMT-wflow: wflow plugin for hydroMT
#######################################

.. image:: https://codecov.io/gh/Deltares/hydromt_wflow/branch/main/graph/badge.svg?token=ss3EgmwHhH
    :target: https://codecov.io/gh/Deltares/hydromt_wflow

.. image:: https://readthedocs.org/projects/hydromt_wflow/badge/?version=latest
    :target: https://hydromt_wflow.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. note::

  This minimal branch from the hydromt_wflow plugin can be used as a **template** to easily 
  implement new plugins for hydroMT. It contains:
  
  - git installation (including pyproject.toml) and an example environment yaml in the envs folder
  - template documentation to be edited
  - empty model class (wflow.py) to be adapted for the new model
  - template coverage test in the tests folder (model_api compliance and build test)
  - example license is MIT (same as hydromt core)


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

Copyright (c) 2019, Deltares

Licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated 
documentation files (the "Software"), to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, 
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of 
the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO 
THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
