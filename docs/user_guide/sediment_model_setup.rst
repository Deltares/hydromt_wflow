===================
Model set up basics
===================

This plugin helps you preparing or updating several components of a wflow sediment model such as topography information, landuse or soil.
The main interactions are available from the HydroMT Command Line Interface and allow you to configure
HydroMT in order to build or update or clip wflow sediment models.

When building or updating a model from command line a
:ref:`model region <https://deltares.github.io/hydromt/preview/user_guide/build_region.html?highlight=region>`; a model setup
configuration (.ini file) with model components and options and, optionally, 
a :ref:`data sources <https://deltares.github.io/hydromt/preview/user_guide/get_data.html>`(.yml) file should be prepared.

Here you can find more detailed information about the basics for setting up a Wflow model with HydroMT:

* :ref:`Wflow model components <model_components_sed>`
* :ref:`Wflow configuration (.ini file) <model_config_sed>`
* :ref:`Wflow in- and output files <model_files_sed>`

Note that the order in which the components are listed in the ini file is important: 

- `setup_basemaps` should always be run first to determine the model domain
- `setup_rivers` should be run right after `setup_basemaps` as it influences several other setup components (lakes, reservoirs, riverbedsed, floodplains, gauges)

.. toctree::
    :hidden:

    sediment_components.rst
    sediment_config.rst
    sediment_files.rst