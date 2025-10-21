.. _intro_user_guide:

User guide
==========

The user guide is organised through the following sections:

.. grid:: 3
    :gutter: 1

    .. grid-item-card::
        :text-align: center
        :link: 1_getting_started_hydromt/index
        :link-type: doc

        :octicon:`rocket;10em`
        +++
        Getting started with HydroMT

    .. grid-item-card::
        :text-align: center
        :link: 2_sbm_model/index
        :link-type: doc

        :octicon:`drop;10em`
        +++
        Wflow SBM

    .. grid-item-card::
        :text-align: center
        :link: 3_sediment_model/index
        :link-type: doc

        :octicon:`mountain;10em`
        +++
        Wflow Sediment

    .. grid-item-card::
        :text-align: center
        :link: 4_pre_and_post_processing/index
        :link-type: doc

        :octicon:`graph;10em`
        +++
        Processing and Visualization

    .. grid-item-card::
        :text-align: center
        :link: 5_setup_methods/index
        :link-type: doc

        :octicon:`cpu;10em`
        +++
        Technical Description

    .. grid-item-card::
        :text-align: center
        :link: 6_migration_guide/index
        :link-type: doc

        :octicon:`arrow-switch;10em`
        +++
        Migration Guide


With the **HydroMT-Wflow plugin**, users can easily benefit from the rich set of tools of the
`HydroMT package <https://deltares.github.io/hydromt/latest/index.html>`_ to build and update
`Wflow <https://deltares.github.io/Wflow.jl/stable/>`_ models from available global and local data.

This plugin assists Wflow modellers in:

- Quickly setting up a base Wflow model and default parameter values
- Making maximum use of the best available global or local data
- Adjusting and updating components of a Wflow model and their associated parameters in a consistent way
- Clipping existing Wflow models for a smaller extent
- Analysing Wflow model outputs

Two Wflow model classes are currently available:

- ``wflow_sbm`` (:class:`~hydromt_wflow.WflowSbmModel`): for the **wflow_sbm + kinematic** and **wflow_sbm + local inertial** concepts
- ``wflow_sediment`` (:class:`~hydromt_wflow.WflowSedimentModel`): for the **wflow_sediment** concept


.. toctree::
   :caption: Table of Contents
   :maxdepth: 2
   :hidden:

   1_getting_started_hydromt/index.rst
   2_sbm_model/index.rst
   3_sediment_model/index.rst
   4_pre_and_post_processing/index.rst
   5_setup_methods/index.rst
   6_migration_guide/index.rst
