.. currentmodule:: hydromt_wflow

.. _model_set_up:

============================
Model methods and components
============================

The HydroMT-Wflow plugin helps you preparing or updating several inputs of a Wflow model
such as topography information, landuse, soil or forcing using setup methods.
The main interactions are available from the HydroMT Command Line Interface and allow you to configure
HydroMT in order to build or update or clip Wflow models.

When building or updating a model from command line a
`model region <https://deltares.github.io/hydromt/stable/guides/user_guide/model_region.html>`_; a model setup
:ref:`configuration <model_config>` (.yml file) with model methods and options and, optionally,
a `data catalog <https://deltares.github.io/hydromt/stable/guides/user_guide/data_overview.html>`_ (.yml) file should be prepared.

.. _model_methods:

Model setup methods
===================

An overview of the available Wflow model setup methods
is provided in the tables below. When using HydroMT from the command line only the
setup methods are exposed. Click on
a specific method to see its documentation.

Configuration (TOML)
--------------------
Defines and manages model configuration, global parameters, and output settings.

.. list-table::
    :widths: 20 60 20
    :header-rows: 1
    :stub-columns: 1

    * - Method
      - Explanation
      - Required Setup Method
    * - :py:meth:`~WflowSbmModel.setup_config`
      - Update config with a dictionary
      -
    * - :py:meth:`~WflowSbmModel.setup_config_output_timeseries`
      - Add new variable/column to the netcdf/csv output section of the toml based on a selected gauge/area map.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_constant_pars`
      -  Setup constant parameter maps for all active model cells.
      -

Topography and Rivers
---------------------
Prepares elevation maps, drainage networks, and river-related features used to simulate flow routing and floodplain processes.

.. list-table::
    :widths: 20 60 20
    :header-rows: 1
    :stub-columns: 1

    * - Method
      - Explanation
      - Required Setup Method
    * - :py:meth:`~WflowSbmModel.setup_basemaps`
      - Set the region of interest and res (resolution in degrees) of the model.
      -
    * - :py:meth:`~WflowSbmModel.setup_rivers`
      - Set all river parameter maps.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_river_roughness`
      - Set river Manning roughness coefficient.
      - :py:meth:`~WflowSbmModel.setup_rivers`
    * - :py:meth:`~WflowSbmModel.setup_floodplains`
      - Add floodplain information (can be either 1D or 2D).
      - :py:meth:`~WflowSbmModel.setup_rivers`

Reservoirs and Glaciers
-----------------------
Adds reservoirs and glaciers, and defines their impact on hydrological storage and flow regulation.

.. list-table::
    :widths: 20 60 20
    :header-rows: 1
    :stub-columns: 1

    * - Method
      - Explanation
      - Required Setup Method
    * - :py:meth:`~WflowSbmModel.setup_reservoirs_no_control`
      - Generate maps of uncontrolled reservoirs (lakes, weirs) areas and outlets as well as parameters with average reservoir area, depth a discharge values.
      - :py:meth:`~WflowSbmModel.setup_rivers`
    * - :py:meth:`~WflowSbmModel.setup_reservoirs_simple_control`
      - Generate maps of controlled reservoir areas and outlets as well as parameters with average reservoir area, demand, min and max target storage capacities and discharge capacity values.
      - :py:meth:`~WflowSbmModel.setup_rivers`
    * - :py:meth:`~WflowSbmModel.setup_glaciers`
      - Generate maps of glacier areas, area fraction and volume fraction, as well as tables with temperature threshold, melting factor and snow-to-ice conversion fraction.
      - :py:meth:`~WflowSbmModel.setup_basemaps`

Land Use and Vegetation
-----------------------
Defines land use and vegetation properties, including LULC and LAI maps, which influence evapotranspiration and interception processes.

.. list-table::
    :widths: 20 60 20
    :header-rows: 1
    :stub-columns: 1

    * - Method
      - Explanation
      - Required Setup Method
    * - :py:meth:`~WflowSbmModel.setup_lulcmaps`
      - Derive several wflow maps based on landuse- landcover (LULC) raster data.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_lulcmaps_from_vector`
      - Derive several wflow maps based on landuse- landcover (LULC) vector data.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_lulcmaps_with_paddy`
      - Derive several wflow maps based on landuse- landcover (LULC) raster data with paddy rice.
      - :py:meth:`~WflowSbmModel.setup_soilmaps`
    * - :py:meth:`~WflowSbmModel.setup_laimaps`
      - Set leaf area index (LAI) climatology maps per month.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_laimaps_from_lulc_mapping`
      - Set leaf area index (LAI) climatology maps per month based on landuse mapping.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_rootzoneclim`
      - Derive an estimate of the rooting depth from hydroclimatic data (as an alternative from the look-up table). The method can be applied for current conditions and future climate change conditions.
      - :py:meth:`~WflowSbmModel.setup_soilmaps`, :py:meth:`~WflowSbmModel.setup_laimaps` or equivalent

Soil
----
Sets up soil-related data including soil maps and hydraulic properties.

.. list-table::
    :widths: 20 60 20
    :header-rows: 1
    :stub-columns: 1

    * - Method
      - Explanation
      - Required Setup Method
    * - :py:meth:`~WflowSbmModel.setup_soilmaps`
      - Derive several (layered) soil parameters based on a database with physical soil properties using available point-scale (pedo)transfer functions (PTFs) from literature with upscaling rules to ensure flux matching across scales.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_ksathorfrac`
      - Prepare the saturated hydraulic conductivity horizontal ratio from an existing map.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_ksatver_vegetation`
      - Prepare ksatver from soil and vegetation parameters.
      - :py:meth:`~WflowSbmModel.setup_soilmaps`, :py:meth:`~WflowSbmModel.setup_laimaps` or equivalent.

Water demands and Allocation
----------------------------
Defines domestic, irrigation, and other water demand maps and allocation parameters.

.. list-table::
    :widths: 20 60 20
    :header-rows: 1
    :stub-columns: 1

    * - Method
      - Explanation
      - Required Setup Method
    * - :py:meth:`~WflowSbmModel.setup_allocation_areas`
      -  Create water demand allocation areas.
      - :py:meth:`~WflowSbmModel.setup_rivers`
    * - :py:meth:`~WflowSbmModel.setup_allocation_surfacewaterfrac`
      -  Create fraction of surface water used for allocation of the water demands.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_domestic_demand`
      -  Create domestic water demand from grid.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_domestic_demand_from_population`
      -  Create domestic water demand using demand per capita and gridded population.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_other_demand`
      -  Create other water demand (eg industry, livestock).
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_irrigation`
      -  Create irrigation areas and trigger for paddy and nonpaddy crops from a raster file.
      - :py:meth:`~WflowSbmModel.setup_lulcmaps` or equivalent, :py:meth:`~WflowSbmModel.setup_laimaps` or equivalent
    * - :py:meth:`~WflowSbmModel.setup_irrigation_from_vector`
      -  Create irrigation areas and trigger for paddy and nonpaddy crops from a vector file.
      - :py:meth:`~WflowSbmModel.setup_lulcmaps` or equivalent, :py:meth:`~WflowSbmModel.setup_laimaps` or equivalent

Forcing
-------
Sets up meteorological forcing inputs such as precipitation, temperature, and potential evapotranspiration.

.. list-table::
    :widths: 20 60 20
    :header-rows: 1
    :stub-columns: 1

    * - Method
      - Explanation
      - Required Setup Method
    * - :py:meth:`~WflowSbmModel.setup_precip_forcing`
      -  Setup gridded precipitation forcing at model resolution.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_precip_from_point_timeseries`
      -  Setup precipitation forcing from station data at model resolution.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_temp_pet_forcing`
      -  Setup gridded temperature and optionally compute reference evapotranspiration forcing at model resolution.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.setup_pet_forcing`
      -  Setup gridded reference evapotranspiration forcing at model resolution.
      - :py:meth:`~WflowSbmModel.setup_basemaps`

States
------
Defines initial hydrological state variables such as soil moisture and groundwater storage.

.. list-table::
    :widths: 20 60 20
    :header-rows: 1
    :stub-columns: 1

    * - Method
      - Explanation
      - Required Setup Method
    * - :py:meth:`~WflowSbmModel.setup_cold_states`
      -  Setup wflow cold states based on data in staticmaps.
      - :py:meth:`~WflowSbmModel.setup_soilmaps`, :py:meth:`~WflowSbmModel.setup_constant_pars`, :py:meth:`~WflowSbmModel.setup_reservoirs_no_control`, :py:meth:`~WflowSbmModel.setup_reservoirs_simple_control`, :py:meth:`~WflowSbmModel.setup_glaciers`, :py:meth:`~WflowSbmModel.setup_irrigation` or equivalent

Output Locations
----------------
Defines outlets, gauges, and spatial masks used for reporting model results.

.. list-table::
    :widths: 20 60 20
    :header-rows: 1
    :stub-columns: 1

    * - Method
      - Explanation
      - Required Setup Method
    * - :py:meth:`~WflowSbmModel.setup_outlets`
      - Set the default gauge map based on basin outlets.
      - :py:meth:`~WflowSbmModel.setup_rivers`
    * - :py:meth:`~WflowSbmModel.setup_gauges`
      - Set the default gauge map based on a gauges_fn data.
      - :py:meth:`~WflowSbmModel.setup_rivers`
    * - :py:meth:`~WflowSbmModel.setup_areamap`
      -  Setup area map from vector data to save wflow outputs for specific area.
      - :py:meth:`~WflowSbmModel.setup_basemaps`

Other Setup Methods
-------------------
Additional high-level utilities to modify model geometry, link external models, or upgrade model versions.

.. list-table::
    :widths: 20 60 20
    :header-rows: 1
    :stub-columns: 1

    * - Method
      - Explanation
      - Required Setup Method
    * - :py:meth:`~WflowSbmModel.setup_1dmodel_connection`
      -  Setup subbasins and gauges to save results from wflow to be used in 1D river models.
      - :py:meth:`~WflowSbmModel.setup_rivers`
    * - :py:meth:`~WflowSbmModel.setup_grid_from_raster`
      -  Setup staticmaps from raster to add parameters from direct data.
      - :py:meth:`~WflowSbmModel.setup_basemaps`
    * - :py:meth:`~WflowSbmModel.upgrade_to_v1_wflow`
      -  Upgrade a model from a Wflow.jl 0.x to 1.0 .
      -
    * - :py:meth:`~WflowSbmModel.clip`
      -  Clip a sub-region of an existing model.
      -


.. _model_components:

Model components
================

The following table provides an overview of which :py:class:`~hydromt_wflow.WflowSbmModel`
components contains which Wflow in- and output files. The files are read and written with the associated
read- and write- methods, i.e. :py:func:`~WflowSbmModel.config.read`
and :py:func:`~WflowSbmModel.config.write` for the
:py:attr:`~WflowSbmModel.config` component.


.. list-table:: Wflow model data
   :widths: 30 70
   :header-rows: 1

   * - :py:class:`~hydromt_wflow.WflowSbmModel` model
     - Wflow files
   * - :py:attr:`~hydromt_wflow.WflowSbmModel.config`
     - wflow_sbm.toml
   * - :py:attr:`~hydromt_wflow.WflowSbmModel.staticmaps`
     - staticmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowSbmModel.geoms`
     - geometries from the staticgeoms folder (basins.geojson, rivers.geojson etc.)
   * - :py:attr:`~hydromt_wflow.WflowSbmModel.forcing`
     - inmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowSbmModel.states`
     - instates.nc
   * - :py:attr:`~hydromt_wflow.WflowSbmModel.tables`
     - tabular data (csv format, e.g. lake_hq.csv, lake_sh.csv)
   * - :py:attr:`~hydromt_wflow.WflowSbmModel.output_grid`
     - output.nc (defined in [output.netcdf_grid] TOML section)
   * - :py:attr:`~hydromt_wflow.WflowSbmModel.output_scalar`
     - output_scalar.nc (defined in [output.netcdf_scalar] TOML section)
   * - :py:attr:`~hydromt_wflow.WflowSbmModel.output_csv`
     - output.csv (defined in [output.csv] TOML section)
