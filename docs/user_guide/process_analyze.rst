.. _process_visualize:

========================================
Pre and postprocessing and visualization
========================================

The Hydromt-Wflow plugin provides several possibilities to postprocess and visualize the model data and model results:

*   `Prepare flow directions and related maps from a DEM <../_examples/prepare_ldd.ipynb>`_ using the `flw methods of HydroMT <https://deltares.github.io/hydromt/stable/api/gis.html#flow-direction-methods>`_
*   `Convert Wflow staticmaps netcdf to raster files <../_examples/convert_staticmaps_to_mapstack.ipynb>`_ for further processing and analysis
*   Plot `staticmaps <../_examples/plot_wflow_staticmaps.ipynb>`_, `forcing data <../_examples/plot_wflow_forcing.ipynb>`_ and
    `model results <../_examples/plot_wflow_results.ipynb>`_ by means of additional python packages
*   Use the `statistical methods of HydroMT <https://deltares.github.io/hydromt/stable/guides/advanced_user/methods_stats.html>`_
    to statistically analyze the model results
*   Upgrade your old Wflow model to the Wflow.jl version 1 format using the `upgrade_to_wflow_v1 <../_examples/upgrade_to_wflow_v1.ipynb>`_ example.

.. toctree::
    :hidden:

    Example: Prepare flow directions and related maps from a DEM <../_examples/prepare_ldd.ipynb>
    Example: Convert wflow staticmaps netcdf to raster files <../_examples/convert_staticmaps_to_mapstack.ipynb>
    Example: Plot Wflow staticmaps <../_examples/plot_wflow_staticmaps.ipynb>
    Example: Plot Wflow forcing data <../_examples/plot_wflow_forcing.ipynb>
    Example: Plot Wflow results data <../_examples/plot_wflow_results.ipynb>
    Example: Upgrade to Wflow.jl version 1 <../_examples/upgrade_to_wflow_v1.ipynb>
