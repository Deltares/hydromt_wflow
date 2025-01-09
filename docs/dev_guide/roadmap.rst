.. _roadmap:

Roadmap
=======

Ambition
--------
This package aims to make the building process of Wflow models **fast**, **scalable**, **modular** and **reproducible**
by making the best use of `current and future methods developped in HydroMT core <https://deltares.github.io/hydromt/latest/dev/roadmap.html>`_.

Short-term plans
----------------

Support for additionnal Wflow concepts
""""""""""""""""""""""""""""""""""""""
Currently only support for the SBM (kinematic wave / local inertial) and Sediment concepts are available.
Work is now carried out to also support:

- SBM + groundwater flow concept in the WflowModel class, see https://github.com/Deltares/hydromt_wflow/pull/56
- Flextopo concept in the new WflowFlextopoMocel class, see https://github.com/Deltares/hydromt_wflow/pull/45

Connection to Delft-FEWS
""""""""""""""""""""""""
New functios are being implemented in order to be able to directly export and run a Wflow model into a
Delft-FEWS configuration. See https://github.com/Deltares/hydromt_wflow/pull/81

Projected Wflow models
""""""""""""""""""""""
Wflow models can currently only be built in the global EPSG:4326 CRS. Additionnal developments are needed to also be able to
build models in other (projected) CRS. See https://github.com/Deltares/hydromt_wflow/issues/66

Better data processing workflow
"""""""""""""""""""""""""""""""
There are a lot of ideas on how to improve the current Wflow methods to improve processing of different types of
data (gauges, lakes, reservoirs, forcing...). These ideas are listed in the `issue board <https://github.com/Deltares/hydromt_wflow/issues>`_
of HydroMT-Wflow and you can also add your own ideas there as well using the {bdg-info}`enhancement` badge when creating
your issue.
