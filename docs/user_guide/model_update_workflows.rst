Land Surface Temperature Workflows
=================================

This guide explains how to use the landsurfacetemp workflows in hydromt_wflow to replace the patchwork approach for updating model forcing variables.

Overview
--------

The landsurfacetemp workflows provide structured methods for processing land surface temperature variables and adding them to model forcing. These workflows follow hydromt patterns and provide a more organized approach compared to manual patchwork methods.

Available Functions
-----------------

Land Surface Temperature Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``landsurfacetemp`` module provides specialized functions for land surface temperature calculations:

.. code-block:: python

    from hydromt_wflow.workflows import landsurfacetemp

    # Process wind data
    wind_processed = landsurfacetemp.wind(
        da_model=mod.grid["wflow_dem"],
        wind_u=wind_u_data,
        wind_v=wind_v_data,
        altitude=10,
        altitude_correction=True,
    )

    # Process albedo data
    albedo_processed = landsurfacetemp.albedo(
        albedo=albedo_data,
        da_model=mod.grid["wflow_dem"],
    )

    # Process emissivity data
    emissivity_processed = landsurfacetemp.emissivity(
        emissivity=emissivity_data,
        da_model=mod.grid["wflow_dem"],
    )

    # Process radiation data
    radiation_processed = landsurfacetemp.radiation(
        radiation=radiation_data,
        da_model=mod.grid["wflow_dem"],
        var_name="shortwave_down",
    )

Adding Variables to Forcing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``add_var_to_forcing`` function provides a standardized way to add variables to model forcing:

.. code-block:: python

    from hydromt_wflow.workflows import landsurfacetemp

    # Add a single variable to forcing
    mod = landsurfacetemp.add_var_to_forcing(
        mod=mod,
        ds=variable_data,
        var="variable_name",
        freq="D",  # optional time resampling
    )

Complete Example
---------------

Here's a complete example showing how to replace the patchwork approach:

.. code-block:: python

    from hydromt_wflow import WflowModel
    from hydromt.data_catalog import DataCatalog
    from hydromt_wflow.workflows import forcing, landsurfacetemp
    import xarray as xr

    # Setup
    cats = DataCatalog(["data_catalog.yml"])
    mod = WflowModel(root="model_path", config_fn="wflow_sbm.toml", mode="r+")
    mod.read_config()
    mod.read_forcing()
    mod.read_geoms()

    # Add albedo using landsurfacetemp workflow
    ds_albedo = cats.get_rasterdataset("albedo")
    mod = landsurfacetemp.add_var_to_forcing(
        mod=mod,
        ds=ds_albedo,
        var="albedo",
    )

    # Add shortwave_down using landsurfacetemp workflow
    ds_shortwave = cats.get_rasterdataset("shortwave_down")
    mod = landsurfacetemp.add_var_to_forcing(
        mod=mod,
        ds=ds_shortwave,
        var="shortwave_down",
    )

    # Process wind
    ds_wind = cats.get_rasterdataset("era5")[["wind10_u", "wind10_v"]]
    ds_wind = ds_wind.sel(time=slice("2020-01-01", "2020-12-31"))
    
    wind_processed = landsurfacetemp.wind(
        da_model=mod.grid["wflow_dem"],
        wind_u=ds_wind["wind10_u"],
        wind_v=ds_wind["wind10_v"],
        altitude=10,
    )
    
    mod = landsurfacetemp.add_var_to_forcing(
        mod=mod,
        ds=wind_processed,
        var="wind",
    )

    # Process PET (using existing forcing workflow)
    # ... PET calculation code ...
    
    mod = landsurfacetemp.add_var_to_forcing(
        mod=mod,
        ds=pet_data,
        var="pet_makkink_lsa",
    )

    # Standardize encoding using existing patterns
    encoding = mod.forcing["precip"].encoding
    fillval = encoding["_FillValue"]
    
    for key, item in mod.forcing.items():
        # enforce nodata
        if "_FillValue" not in item.attrs:
            remasked = item.where(~mask)
            remasked.raster.set_nodata(fillval)
            remasked.raster.attrs["_FillValue"] = fillval
            mod.forcing[key] = remasked
        
        # enforce time range
        mod.forcing[key] = mod.forcing[key].sel(time=slice("2020-01-01", "2020-12-31"))
        
        # copy encoding from reference variable
        for enc in list(encoding.keys())[:10]:
            da = mod.forcing[key]
            da.encoding[enc] = encoding[enc]
            mod.forcing[key] = da

    # Update model configuration using existing patterns
    mod.config["starttime"] = "2020-01-01T00:00:00"
    mod.config["endtime"] = "2020-12-31T00:00:00"
    mod.config["input"]["path_forcing"] = "inmaps/LSA_forcing_2020.nc"
    
    mod.set_root(new_root, mode="w+")
    mod.write_forcing(os.path.join(new_root, "inmaps", "LSA_forcing_2020.nc"))

Benefits
--------

Using these workflow methods provides several advantages over the patchwork approach:

1. **Standardization**: Consistent processing across all variables
2. **Error Handling**: Built-in validation and error checking
3. **Logging**: Comprehensive logging for debugging
4. **Reusability**: Functions can be used across different projects
5. **Maintainability**: Easier to maintain and update
6. **Documentation**: Clear function signatures and docstrings

Migration from Patchwork Approach
--------------------------------

To migrate from the patchwork approach:

1. Replace manual `add_var_to_forcing` calls with `landsurfacetemp.add_var_to_forcing`
2. Use `landsurfacetemp.wind()` for wind processing
3. Use `landsurfacetemp.albedo()` for albedo processing
4. Use `landsurfacetemp.emissivity()` for emissivity processing
5. Use `landsurfacetemp.radiation()` for radiation processing
6. Use existing patterns for encoding standardization and configuration updates

The landsurfacetemp workflows maintain the same functionality while providing better structure and error handling, following the existing hydromt patterns in the codebase. 