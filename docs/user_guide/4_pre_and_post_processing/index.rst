.. _process_visualize:

========================================
Pre and postprocessing and visualization
========================================

The Hydromt-Wflow plugin provides several possibilities to postprocess and visualize the model data and model results:

*   :ref:`Prepare flow directions <example-prepare_ldd>` using the `flw methods of HydroMT <https://deltares.github.io/hydromt/stable/api/gis.html#flow-direction-methods>`_
*   :ref:`Convert static maps to mapstack <example-convert_staticmaps_to_mapstack>` for further processing and analysis
*   Plot :ref:`static maps <example-plot_wflow_staticmaps>`, :ref:`forcing <example-plot_wflow_forcing>` and
    :ref:`results <example-plot_wflow_results>` by means of additional python packages
*   Use the `statistical methods of HydroMT <https://deltares.github.io/hydromt/stable/guides/advanced_user/methods_stats.html>`_ to statistically analyze the model results
*   Upgrade your old Wflow model to the Wflow.jl version 1 format using the :ref:`upgrade <example-upgrade_to_wflow_v1>` example.


.. toctree::
    :hidden:
    :titlesonly:

    /_examples/prepare_ldd.ipynb
    /_examples/convert_staticmaps_to_mapstack.ipynb
    /_examples/plot_wflow_staticmaps.ipynb
    /_examples/plot_wflow_forcing.ipynb
    /_examples/plot_wflow_results.ipynb
    /_examples/upgrade_to_wflow_v1.ipynb
