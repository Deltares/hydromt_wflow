============================
Model methods and components
============================

This plugin helps you preparing or updating several components of a Wflow Sediment model such as topography information, landuse or soil.
The main interactions are available from the HydroMT Command Line Interface and allow you to configure
HydroMT in order to build or update or clip Wflow Sediment models.

When building or updating a model from command line a
`model region <https://deltares.github.io/hydromt/latest/user_guide/model_region>`_; a model setup
:ref:`configuration <model_config_sed>` (.yml file) with model components and options and, optionally,
a `data sources <https://deltares.github.io/hydromt/latest/user_guide/data_main>`_ (.yml) file should be prepared.

.. currentmodule:: hydromt_wflow

.. _model_methods_sed:

Model methods
=============

An overview of the available Wflow Sediment model setup components
is provided in the table below. When using HydroMT from the command line only the
setup components are exposed. Click on
a specific method see its documentation.

.. list-table::
    :widths: 20 55
    :header-rows: 1
    :stub-columns: 1

    * - Method
      - Explanation
    * - :py:func:`~WflowSedimentModel.setup_config`
      - Update config with a dictionary
    * - :py:func:`~WflowSedimentModel.setup_basemaps`
      - This component sets the region of interest and res (resolution in degrees) of the model.
    * - :py:func:`~WflowSedimentModel.setup_rivers`
      - This component sets the all river parameter maps.
    * - :py:func:`~WflowSedimentModel.setup_natural_reservoirs`
      - This component generates maps of lake (natural reservoirs) areas and outlets as well as parameters such as average area.
    * - :py:func:`~WflowSedimentModel.setup_reservoirs`
      - This component generates maps of reservoir areas and outlets as well as parameters such as average area.
    * - :py:func:`~WflowSedimentModel.setup_lulcmaps`
      - This component derives several wflow maps based on landuse- landcover (LULC) raster data.
    * - :py:func:`~WflowBaseModel.setup_lulcmaps_from_vector`
      - This component derives several wflow maps based on landuse- landcover (LULC) vector data.
    * - :py:func:`~WflowSedimentModel.setup_laimaps`
      - This component sets leaf area index (LAI) climatology maps per month.
    * - :py:func:`~WflowSedimentModel.setup_laimaps_from_lulc_mapping`
      - This component sets leaf area index (LAI) climatology maps per month based on landuse mapping.
    * - :py:func:`~WflowSedimentModel.setup_canopymaps`
      - Setup sediments based canopy height maps.
    * - :py:func:`~WflowSedimentModel.setup_soilmaps`
      - Setup sediments based soil parameter maps.
    * - :py:func:`~WflowSedimentModel.setup_riverwidth`
      - This component sets the river width parameter based on a power-lay relationship with a predictor.
    * - :py:func:`~WflowSedimentModel.setup_riverbedsed`
      - Setup sediments based river bed characteristics maps.
    * - :py:func:`~WflowSedimentModel.setup_outlets`
      - This method sets the default gauge map based on basin outlets.
    * - :py:func:`~WflowSedimentModel.setup_gauges`
      - This method sets the default gauge map based on a gauges_fn data.
    * - :py:func:`~WflowSedimentModel.setup_areamap`
      - Setup area map from vector data to save wflow outputs for specific area.
    * - :py:func:`~WflowSedimentModel.setup_config_output_timeseries`
      - This method add a new variable/column to the netcf/csv output section of the toml based on a selected gauge/area map.
    * - :py:func:`~WflowSedimentModel.setup_constant_pars`
      - Setup constant parameter maps.
    * - :py:func:`~WflowSedimentModel.setup_grid_from_raster`
      -  Setup staticmaps from raster to add parameters from direct data.
    * - :py:func:`~WflowSedimentModel.upgrade_to_v1_wflow`
      -  Upgrade a model from a Wflow.jl 0.x to 1.0 .


.. _model_components_sed:

Model components
================

The following table provides an overview of which :py:class:`~hydromt_wflow.WflowSedimentModel`
components contains which Wflow Sediment in- and output files. The files are read and written with the associated
read- and write- methods, i.e. :py:func:`~hydromt_wflow.WflowSedimentModel.read_config`
and :py:func:`~hydromt_wflow.WflowSedimentModel.write_config` for the
:py:attr:`~hydromt_wflow.WflowSedimentModel.config`  component.


.. list-table:: Wflow Sediment model data
   :widths: 30 70
   :header-rows: 1

   * - :py:class:`~hydromt_wflow.WflowSedimentModel` model
     - Wflow sediment files
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.config`
     - wflow_sediment.toml
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.staticmaps`
     - staticmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.geoms`
     - geometries from the staticgeoms folder (basins.geojson, rivers.geojson etc.)
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.forcing`
     - inmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.states`
     - instates.nc
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.output_grid`
     - output.nc (defined in [output.netcdf_grid] TOML section)
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.output_scalar`
     - output_scalar.nc (defined in [output.netcdf_scalar] TOML section)
   * - :py:attr:`~hydromt_wflow.WflowSedimentModel.output_csv`
     - output.csv (defined in [output.csv] TOML section)
