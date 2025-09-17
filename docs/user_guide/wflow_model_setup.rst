.. currentmodule:: hydromt_wflow

.. _model_set_up:

============================
Model methods and components
============================

The HydroMT-Wflow plugin helps you preparing or updating several methods of a Wflow model such as topography information, landuse, soil or forcing.
The main interactions are available from the HydroMT Command Line Interface and allow you to configure
HydroMT in order to build or update or clip Wflow models.

When building or updating a model from command line a
`model region <https://deltares.github.io/hydromt/stable/guides/user_guide/model_region.html>`_; a model setup
:ref:`configuration <model_config>` (.yml file) with model methods and options and, optionally,
a `data sources <https://deltares.github.io/hydromt/stable/guides/user_guide/data_overview.html>`_ (.yml) file should be prepared.

.. _model_methods:

Model setup methods
===================

An overview of the available Wflow model setup methods
is provided in the table below. When using HydroMT from the command line only the
setup methods are exposed. Click on
a specific method see its documentation.

.. list-table::
    :widths: 20 55
    :header-rows: 1
    :stub-columns: 1

    * - Method
      - Explanation
    * - :py:func:`~WflowSbmModel.setup_config`
      - Update config with a dictionary
    * - :py:func:`~WflowSbmModel.setup_basemaps`
      - This component sets the region of interest and res (resolution in degrees) of the model.
    * - :py:func:`~WflowSbmModel.setup_rivers`
      - This component sets the all river parameter maps.
    * - :py:func:`~WflowSbmModel.setup_floodplains`
      - This component This components adds floodplain information to the model schematization (can be either 1D or 2D).
    * - :py:func:`~WflowSbmModel.setup_reservoirs_no_control`
      - This component generates maps of uncontrolled reservoirs (lakes, weirs) areas and outlets as well as parameters with average reservoir area, depth a discharge values.
    * - :py:func:`~WflowSbmModel.setup_reservoirs_simple_control`
      - This component generates maps of controlled reservoir areas and outlets as well as parameters with average reservoir area, demand, min and max target storage capacities and discharge capacity values.
    * - :py:func:`~WflowSbmModel.setup_glaciers`
      - This component generates maps of glacier areas, area fraction and volume fraction, as well as tables with temperature threshold, melting factor and snow-to-ice conversion fraction.
    * - :py:func:`~WflowSbmModel.setup_lulcmaps`
      - This component derives several wflow maps based on landuse- landcover (LULC) raster data.
    * - :py:func:`~WflowSbmModel.setup_lulcmaps_from_vector`
      - This component derives several wflow maps based on landuse- landcover (LULC) vector data.
    * - :py:func:`~WflowSbmModel.setup_lulcmaps_with_paddy`
      - This component derives several wflow maps based on landuse- landcover (LULC) raster data with paddy rice.
    * - :py:func:`~WflowSbmModel.setup_laimaps`
      - This component sets leaf area index (LAI) climatology maps per month.
    * - :py:func:`~WflowSbmModel.setup_laimaps_from_lulc_mapping`
      - This component sets leaf area index (LAI) climatology maps per month based on landuse mapping.
    * - :py:func:`~WflowSbmModel.setup_soilmaps`
      - This component derives several (layered) soil parameters based on a database with physical soil properties using available point-scale (pedo)transfer functions (PTFs) from literature with upscaling rules to ensure flux matching across scales.
    * - :py:func:`~WflowSbmModel.setup_ksathorfrac`
      - This component prepares the saturated hydraulic conductivity horizontal ratio from an existing map.
    * - :py:func:`~WflowSbmModel.setup_ksatver_vegetation`
      - This component prepares ksatver from soil and vegetation parameters.
    * - :py:func:`~WflowSbmModel.setup_rootzoneclim`
      - This component derives an estimate of the rooting depth from hydroclimatic data (as an alternative from the look-up table). The method can be applied for current conditions and future climate change conditions.
    * - :py:func:`~WflowSbmModel.setup_outlets`
      - This method sets the default gauge map based on basin outlets.
    * - :py:func:`~WflowSbmModel.setup_gauges`
      - This method sets the default gauge map based on a gauges_fn data.
    * - :py:func:`~WflowSbmModel.setup_areamap`
      -  Setup area map from vector data to save wflow outputs for specific area.
    * - :py:func:`~WflowSbmModel.setup_config_output_timeseries`
      - This method add new variable/column to the netcf/csv output section of the toml based on a selected gauge/area map.
    * - :py:func:`~WflowSbmModel.setup_precip_forcing`
      -  Setup gridded precipitation forcing at model resolution.
    * - :py:func:`~WflowSbmModel.setup_precip_from_point_timeseries`
      -  Setup precipitation forcing from station data at model resolution.
    * - :py:func:`~WflowSbmModel.setup_temp_pet_forcing`
      -  Setup gridded temperature and optionally compute reference evapotranspiration forcing at model resolution.
    * - :py:func:`~WflowSbmModel.setup_pet_forcing`
      -  Setup gridded reference evapotranspiration forcing at model resolution.
    * - :py:func:`~WflowSbmModel.setup_constant_pars`
      -  Setup constant parameter maps for all active model cells.
    * - :py:func:`~WflowSbmModel.setup_allocation_areas`
      -  Create water demand allocation areas.
    * - :py:func:`~WflowSbmModel.setup_allocation_surfacewaterfrac`
      -  Create fraction of surface water used for allocation of the water demands.
    * - :py:func:`~WflowSbmModel.setup_domestic_demand`
      -  Create domestic water demand from grid.
    * - :py:func:`~WflowSbmModel.setup_domestic_demand_from_population`
      -  Create domestic water demand using demand per capita and gridded population.
    * - :py:func:`~WflowSbmModel.setup_other_demand`
      -  Create other water demand (eg industry, livestock).
    * - :py:func:`~WflowSbmModel.setup_irrigation`
      -  Create irrigation areas and trigger for paddy and nonpaddy crops from a raster file.
    * - :py:func:`~WflowSbmModel.setup_irrigation_from_vector`
      -  Create irrigation areas and trigger for paddy and nonpaddy crops from a vector file.
    * - :py:func:`~WflowSbmModel.setup_1dmodel_connection`
      -  Setup subbasins and gauges to save results from wflow to be used in 1D river models.
    * - :py:func:`~WflowSbmModel.setup_grid_from_raster`
      -  Setup staticmaps from raster to add parameters from direct data.
    * - :py:func:`~WflowSbmModel.setup_cold_states`
      -  Setup wflow cold states based on data in staticmaps.
    * - :py:func:`~WflowSbmModel.upgrade_to_v1_wflow`
      -  Upgrade a model from a Wflow.jl 0.x to 1.0 .


.. _model_components:

Model components
================

The following table provides an overview of which :py:class:`~hydromt_wflow.WflowSbmModel`
components contains which Wflow in- and output files. The files are read and written with the associated
read- and write- methods, i.e. :py:func:`~WflowSbmModel.read_config`
and :py:func:`~WflowSbmModel.write_config` for the
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
