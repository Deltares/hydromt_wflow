.. _intro_user_guide:

User guide
==========

With the **Hydromt-Wflow plugin**, users can easily benefit from the rich set of tools of the
`HydroMT package <https://deltares.github.io/hydromt/latest/index.html>`_ to build and update
`Wflow <https://deltares.github.io/Wflow.jl/stable/>`_ models from available global and local data.

This plugin assists the Wflow modeller in:

- Quickly setting up a base Wflow model and default parameter values
- Making maximum use of the best available global or local data
- Adjusting and updating components of a Wflow model and their associated parameters in a consistent way
- Clipping existing Wflow models for a smaller extent
- Analysing Wflow model outputs

Two Wflow Model classes are currently available:

- ``wflow`` (WflowBaseModel): class for the wflow_sbm + kinematic and wflow_sbm + local inertial concepts
- ``wflow_sediment`` (WflowSedimentModel): class for the wflow_sediment concept

.. toctree::
   :caption: Table of Contents
   :maxdepth: 2
   :hidden:

   wflow.rst
   sediment.rst
   process_analyze.rst
