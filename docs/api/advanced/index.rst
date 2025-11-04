.. currentmodule:: hydromt_wflow

.. _api_reference_advanced:

#########################
Advanced API reference
#########################

The advanced HydroMT-Wflow API exposes the underlying classes, workflows, and utility
functions used to build, extend, and customize Wflow models. These interfaces are primarily
intended for developers and advanced users who want to script model setup workflows, create
custom preprocessing routines, or integrate HydroMT-Wflow components into other systems.

The API is organized into the following sections:

- :ref:`Base model <api_base_model>`: Core classes and methods for defining and managing Wflow models.
- :ref:`Model components <api_components>`: Classes and functions for handling specific model components like config, staticmaps, and forcing.
- :ref:`Workflows <api_workflows>`: Predefined workflows for common tasks such as setting up basemaps, reservoirs, and calibration.
- :ref:`Utilities <api_utils>`: Helper functions for data processing, file I/O, and other common operations.


.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   base-model
   model-components
   workflows/index
   utilities
