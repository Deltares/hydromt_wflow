.. currentmodule:: hydromt_wflow.workflows

.. _api_workflows:

=========
Workflows
=========

The HydroMT-Wflow workflows module provides functions to create and manage various
kinds of data for Wflow models. The workflows are organized into the following sections:

- :ref:`Basemaps <api_workflows_basemaps>` — Prepare basemap layers such as topography and hydrography.
- :ref:`Connect <api_workflows_connect>` — Manage connectivity between wflow and other models (eg 1D river models).
- :ref:`Demand <api_workflows_demand>` — Generate domestic, irrigation, and other water demand data.
- :ref:`Forcing <api_workflows_forcing>` — Create meteorological forcing data such as precipitation and PET.
- :ref:`Gauges <api_workflows_gauges>` — Handle gauging station locations including upstream area snapping.
- :ref:`Glaciers <api_workflows_glaciers>` — Derive glacier extent, maps, and attributes.
- :ref:`Land use <api_workflows_landuse>` — Create and modify land use, land cover, and vegetation-related maps.
- :ref:`Reservoirs <api_workflows_reservoirs>` — Generate and manage reservoir geometry and parameter data.
- :ref:`River <api_workflows_river>` — Derive river network attributes such as width, bathymetry, and floodplain volume.
- :ref:`Root zone climate <api_workflows_rootzoneclim>` — Prepare climatic variables for root zone modeling.
- :ref:`Soil grids <api_workflows_soilgrids>` — Generate soil property maps from SoilGrids datasets.
- :ref:`Soil parameters <api_workflows_soilparams>` — Derive soil hydraulic conductivity and vegetation-based adjustments.
- :ref:`States <api_workflows_states>` — Prepares cold states for compatibility with Delft-FEWS.

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   basemaps
   connect
   demand
   forcing
   gauges
   glaciers
   landuse
   reservoirs
   river
   rootzoneclim
   soilgrids
   soilparams
   states
