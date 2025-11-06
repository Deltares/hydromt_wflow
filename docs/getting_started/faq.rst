.. currentmodule:: hydromt_wflow

.. _faq:

Frequently asked questions
==========================

This page contains some FAQ / tips and tricks to work with HydroMT-Wflow.
For more general questions on how to work with data or the HydroMT config and command line,
you can visit the `HydroMT core FAQ page <https://deltares.github.io/hydromt/stable/overview/faq.html>`_.

Building a Wflow model
----------------------

 | **Q**: Can I use other region arguments than ``basin`` or ``subbasin``?

To build a Wflow model, it is strongly recommended to use the ``basin`` or ``subbasin`` when defining your region of interest.
This ensures that all upstream contributing cells are included. Using other options might cause your Wflow model run to fail.
If you know exactly what you are doing, you can use the ``geom`` option and provide an exact shapefile of the
basin or subbasins that was prepared using the same DEM source intended for :py:func:`~WflowSbmModel.setup_basemaps`.

 | **Q**: How to make sure sub-basins will be properly derived by HydroMT?

When deriving sub-basins, HydroMT needs to know how to snap outlet points to the stream network.
You can specify a snapping argument such as stream order (*strord*) or upstream area (*uparea*) in the region definition:

- ``{'subbasin': [xmin, ymin, xmax, ymax], 'strord': 3}``
- ``{'subbasin': 'path/to/geometry.shp', 'uparea': 100000}``

This ensures sub-basin delineation is consistent with the river network derived from your DEM.

 | **Q**: Can I use different datasets for precipitation, temperature, and PET forcing data?

Yes. That's exactly why the methods were separated. You can use different sources for precipitation, temperature, and PET forcing:

- :py:func:`~WflowSbmModel.setup_precip_forcing`
- :py:func:`~WflowSbmModel.setup_temp_pet_forcing`
- :py:func:`~WflowSbmModel.setup_pet_forcing`

Each method allows specifying its own data source (e.g. *precip_fn*, *temp_pet_fn*, or *pet_fn*), and HydroMT will handle the resampling and temporal alignment automatically.

 | **Q**: Can I add several gauges at the same time?

Yes. Since HydroMT v1, you can add multiple gauges directly in the configuration file using the **new list-based YAML format**.
You no longer need to enumerate methods (e.g. *setup_gauges2*).
The example below shows how to define multiple gauge sources:

.. code-block:: yaml

    steps:
      - setup_gauges:
          gauges_fn: my_gauges1
      - setup_gauges:
          gauges_fn: my_gauges2

Updating a Wflow model
----------------------

 | **Q**: Is there an easy way to update reservoir parameters in my Wflow model?

Yes. You can directly edit the ``meta_reservoirs_simple_control.geojson`` or ``meta_reservoirs_no_control.geojson`` files saved
by HydroMT in the *staticgeoms* folder. Once you have updated the parameters, simply provide these edited GeoJSON files as your
new local input data when rebuilding or updating your model.

 | **Q**: Can I select a specific Wflow TOML config file when updating my model?

Yes. You can define this in the ``global`` section of your HydroMT configuration file using the ``config_filename`` argument:

.. code-block:: yaml

    global:
      config_filename: path/to/my_wflow_config.toml

Others
------

 | **Q**: Can I convert my old Wflow model to the new Wflow Julia version with HydroMT?

Conversion from old Python Wflow models to Wflow Julia is **no longer supported** since HydroMT-Wflow version 1.0.
You can use an older HydroMT-Wflow version for that task and follow the steps described in its documentation.

HydroMT-Wflow **does** support upgrading Wflow Julia v0.x models to Wflow Julia v1.x using:

- :py:func:`~WflowSbmModel.upgrade_to_v1_wflow` for SBM models, and
- :py:func:`~WflowSedimentModel.upgrade_to_v1_wflow` for sediment models.

See the example notebook for details: :ref:`here <example-upgrade_to_wflow_v1>`.
