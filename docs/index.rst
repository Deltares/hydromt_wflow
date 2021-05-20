=====================
HydroMT plugin: Wflow
=====================

`HydroMT <https://github.com/Deltares/hydromt>`_ is a python package, developed by Deltares, to build 
and analyse environmental models. It provides a generic model api with attributes to access the model schematization, 
(dynamic) forcing data, results and states. 

This plugin provides an implementation for the 
`Wflow <https://github.com/Deltares/Wflow.jl>`_ hydrological modelling framework. It details the different steps and explains how to 
use HydroMT to easily get started and work on your own Wflow model.

For detailed information on HydroMT itself, you can visit the `core documentation <https://deltares.github.io/hydromt/latest/index.html>`_.

Documentation
=============

**Getting Started**

* :doc:`intro`
* :doc:`installation`
* :doc:`examples/index`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   intro
   installation
   examples/index

**User Guide**

* :doc:`user_guide/wflow/index`
* :doc:`user_guide/sediment/index`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   user_guide/wflow/index
   user_guide/sediment/index

**Advanced topics**

* :doc:`advanced/workflows`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Advanced topics

   advanced/workflows

**References & Help**

* :doc:`api/api_index`
* :doc:`contributing`
* :doc:`changelog`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: References & Help

   api/api_index
   contributing
   changelog


License
-------

Copyright (c) 2021, Deltares

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public 
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You can find the full terms of the GNU General Public License at <https://www.gnu.org/licenses/>.
