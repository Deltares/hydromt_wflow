.. currentmodule:: hydromt_wflow

.. _model_set_up:

======================
Model components
======================

The HydroMT Wflow plugin helps you preparing or updating several components of a Wflow model such as topography information, landuse, soil or forcing.
The main interactions are available from the HydroMT Command Line Interface and allow you to configure
HydroMT in order to build or update or clip Wflow models.

When building or updating a model from command line a
`model region <https://deltares.github.io/hydromt/preview/user_guide/model_region>`_; a model setup
:ref:`configuration <model_config>` (.ini file) with model components and options and, optionally,
a `data sources <https://deltares.github.io/hydromt/preview/user_guide/data_main>`_ (.yml) file should be prepared.

.. _model_components:

Model setup components
============================

An overview of the available Wflow model setup components
is provided in the table below. When using HydroMT from the command line only the
setup components are exposed. Click on
a specific method see its documentation.

.. list-table::
    :widths: 20 55
    :header-rows: 1
    :stub-columns: 1

    * - Component
      - Explanation
    * - :py:func:`~WflowModel.setup_config`
      - Update config with a dictionary
    * - :py:func:`~WflowModel.setup_basemaps`
      - This component sets the region of interest and res (resolution in degrees) of the model.
    * - :py:func:`~WflowModel.setup_rivers`
      - This component sets the all river parameter maps.
    * - :py:func:`~WflowModel.setup_lakes`
      - This component generates maps of lake areas and outlets as well as parameters with average lake area, depth a discharge values.
    * - :py:func:`~WflowModel.setup_reservoirs`
      - This component generates maps of reservoir areas and outlets as well as parameters with average reservoir area, demand, min and max target storage capacities and discharge capacity values.
    * - :py:func:`~WflowModel.setup_glaciers`
      - This component generates maps of glacier areas, area fraction and volume fraction, as well as tables with temperature threshold, melting factor and snow-to-ice convertion fraction.
    * - :py:func:`~WflowModel.setup_lulcmaps`
      - This component derives several wflow maps are derived based on landuse- landcover (LULC) data.
    * - :py:func:`~WflowModel.setup_laimaps`
      - This component sets leaf area index (LAI) climatology maps per month.
    * - :py:func:`~WflowModel.setup_soilmaps`
      - This component derives several (layered) soil parameters based on a database with physical soil properties using available point-scale (pedo)transfer functions (PTFs) from literature with upscaling rulesto ensure flux matching across scales.
    * - :py:func:`~WflowModel.setup_hydrodem`
      - This component adds a hydrologically conditioned elevation (hydrodem) map for river and/or land local-inertial routing.
    * - :py:func:`~WflowModel.setup_gauges`
      - This components sets the default gauge map based on basin outlets and additional gauge maps based on gauges_fn data.
    * - :py:func:`~WflowModel.setup_areamap`
      -  Setup area map from vector data to save wflow outputs for specific area.


.. _model_files>:

Datamodel files
=====================

The following table provides an overview of which :py:class:`~hydromt_wflow.WflowModel`
attribute contains which Wflow in- and output files. The files are read and written with the associated
read- and write- methods, i.e. :py:func:`~WflowModel.read_config`
and :py:func:`~WflowModel.write_config` for the
:py:attr:`~WflowModel.config`  attribute.


.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - :py:class:`~hydromt_wflow.WflowModel` attribute
     - Wflow files
   * - :py:attr:`~hydromt_wflow.WflowModel.config`
     - wflow_sbm.toml
   * - :py:attr:`~hydromt_wflow.WflowModel.staticmaps`
     - staticmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowModel.staticgeoms`
     - geometries from the staticgeoms folder (basins.geojson, rivers.geojson etc.)
   * - :py:attr:`~hydromt_wflow.WflowModel.forcing`
     - inmaps.nc
   * - :py:attr:`~hydromt_wflow.WflowModel.states`
     - instates.nc
   * - :py:attr:`~hydromt_wflow.WflowModel.results`
     - output.nc, output_scalar.nc, output.csv




