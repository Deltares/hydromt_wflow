.. currentmodule:: hydromt_wflow

.. _faq:

Frequently asked questions
==========================

This page contains some FAQ / tips and tricks to work with HydroMT-Wflow.
For more general questions on how to work with data or the HydroMT config and command line,
you can visit the `HydroMT core FAQ page <https://deltares.github.io/hydromt/latest/getting_started/faq.html>`_

Building a Wflow model
----------------------

 | **Q**: Can I use other region arguments that ``basin`` or ``subbasin`` ?

To build a Wflow model, it is strongly recommended to use the ``basin`` or ``subbasin`` when defining your region of interest.
This ensures that all upstream contributing cells are well taken into account. Using other options might cause your Wflow model
run crash. If you are sure of what you are doing, you can optionnaly use the ``geom`` option and provide an exact shapefile of the
basin/subbasins you want to derive, that was prepared using the same DEM source that you intend to use in the :py:func:`~WflowModel.setup_basemaps`
method.

 | **Q**: Can I derive several sub-basins at the same time using a bounding box or a geometry instead of a list of point corrdinates?

To derive subbasin with HydroMT, you also need to provide in your region options a snapping arguments so that HydroMT knows what type of
subbasin you want to create. You can use either streamorder *strord* or upstream area *uparea* type of snapping:

- "{'subbasin': [xmin, ymin, xmax, ymax], 'strord': 3}"
- "{'subbasin': 'path/to/geomtry.shp', 'uparea': 100000}"

 | **Q**: Can I use a different dataset for precipitation and temperature forcing data?

Yes, that is one of the reason why the two were separated into two separate methods: :py:func:`~WflowModel.setup_precip_forcing` and
:py:func:`~WflowModel.setup_temp_pet_forcing`, you can just use a different data source for *precip_fn* and *temp_pet_fn* and HydroMT
will do its resampling magic!


 | **Q**: Can I add several gauges at the same time ?

Sure! HydroMT allows you to use several times the same method in your configuration file. For this, you just need to start
enumerating the methods by adding a number to the end of the method name. For example:

.. code-block:: console

    [setup_gauges]
    gauges_fn = my_gauges1

    [setup_gauges2]
    gauges_fn = my_gauges2
    derive_outlets = False # To avoid deriving basin outlets twice

Updating a Wflow model
----------------------

 | **Q**: Is there an easy way to update reservoirs or lakes parameters in my Wflow model ?

To easily update reservoirs or lakes parameters, you can directly use the *reservoirs.geojson* and *lakes.geojson* that are saved
by HydroMT into the *staticgeoms* folder. Once you have updated the parameters with the new value, just use these geojson files as
your new "local" input data!

 | **Q**: Can I select a specific Wflow TOML config file when updating my model ?

It is possible. In that case, you need to start your HydroMT configuration with a :py:func:`~WflowModel.read_config` method
where you can specify which TOML file to use using the *config_fn* argument.

Others
------

 | **Q**: Can I convert my old Wflow python model to the new Wflow Julia version with HydroMT ?

HydroMT is still able to read the python PCRaster based maps of Wflow with the method :py:func:`~WflowModel.read_staticmaps_pcr`.
So actually just a read and write of your python model will do most of the job :) The intbl are however not taken into account so if
you have single values intbl, you can use the :py:func:`~WflowModel.setup_constant_pars` method. For lakes and reservoirs parameters,
the best is to use a shapefile of yours lakes and reservoirs with the right columns for each parameters (see previous question on updating
reservoir).
