"""
Example: Using hydromt_wflow workflows for model forcing updates.

This example demonstrates how to use the landsurfacetemp workflow methods to update
model forcing variables in a structured way, replacing the patchwork approach.
"""

from hydromt_wflow import WflowModel
from hydromt.data_catalog import DataCatalog
from hydromt_wflow.workflows import forcing, landsurfacetemp
import xarray as xr
import matplotlib.pyplot as plt
import os

def plot_forcing(mod, var, time=300, vmax=1, new_root=None):
    """Plot forcing variable for visualization."""
    fig, ax = plt.subplots(figsize=(14, 10))
    bounds = mod.geoms["basins"].bounds
    mod.forcing[var].isel(time=time).raster.mask_nodata().plot(ax=ax, vmin=0, vmax=vmax)
    mod.geoms["basins"].plot(ax=ax, facecolor="none", edgecolor="magenta")
    ax.set_xlim(bounds.minx.values, bounds.maxx.values)
    ax.set_ylim(bounds.miny.values, bounds.maxy.values)
    ax.set_aspect("equal")
    ax.set_title(f"{var} t={time}")
    if new_root is not None:
        plt.savefig(os.path.join(new_root, f"{var}.png"), dpi=300)
    else:
        plt.show()

def main():
    """Main function demonstrating the landsurfacetemp workflow approach."""
    
    # setup paths and catalogs
    cats = DataCatalog(["data_catalog_old_version.yml", "deltares_data", "hyras_dc.yml"])
    root = r"p:\1000365-002-wflow\tmp\wflow-gleam\model\base_model_canopy"
    new_root = r"p:\1000365-002-wflow\tmp\wflow-gleam\model\canopy_model_lsa_forcing"
    
    # load model
    mod = WflowModel(root=root, config_fn="wflow_sbm.toml", mode="r+")
    mod.read_config()
    mod.read_forcing()
    mod.read_geoms()
    
    # add albedo using landsurfacetemp workflow
    ds_albedo = cats.get_rasterdataset("albedo")
    mod = landsurfacetemp.add_var_to_forcing(
        mod=mod,
        ds=ds_albedo,
        var="albedo",
    )
    
    # add shortwave_down using landsurfacetemp workflow
    ds_shortwave = cats.get_rasterdataset("shortwave_down")
    mod = landsurfacetemp.add_var_to_forcing(
        mod=mod,
        ds=ds_shortwave,
        var="shortwave_down",
    )
    
    # process wind using the landsurfacetemp workflow
    ds_wind = cats.get_rasterdataset("era5")[["wind10_u", "wind10_v"]]
    ds_wind = ds_wind.sel(time=slice("2020-01-01", "2020-12-31"))
    
    wind_processed = landsurfacetemp.wind(
        da_model=mod.grid["wflow_dem"],
        wind_u=ds_wind["wind10_u"],
        wind_v=ds_wind["wind10_v"],
        altitude=10,
    )
    
    # add wind to forcing using landsurfacetemp workflow
    mod = landsurfacetemp.add_var_to_forcing(
        mod=mod,
        ds=wind_processed,
        var="wind",
    )
    
    # process PET using existing forcing workflow
    dem = mod.grid["wflow_dem"]
    ds_pet = xr.Dataset(coords={
        "time": mod.forcing["shortwave_down"].time,
        "latitude": dem.latitude,
        "longitude": dem.longitude
    })
    
    ds_pet["temp"] = mod.forcing["temp"].sel(time=ds_pet.time).raster.reproject_like(dem)
    ds_pet["kin"] = mod.forcing["shortwave_down"].sel(time=ds_pet.time).raster.reproject_like(dem)
    ds_pet["press_msl"] = cats.get_rasterdataset(
        "era5",
        geom=mod.geoms["basins"].buffer(10)
    )["press_msl"].sel(time=ds_pet.time).raster.reproject_like(dem)
    
    # standardize nodata values
    var_list = ["temp", "kin", "press_msl"]
    ds_ex = mod.grid["wflow_dem"]
    fillval = ds_ex.attrs["_FillValue"]
    mask = ds_ex.values == fillval
    for var in ds_pet.data_vars:
        ds_pet[var] = ds_pet[var].where(~mask)
        ds_pet[var].raster.set_nodata(fillval)
        ds_pet[var].raster.attrs["_FillValue"] = fillval
    
    # drop unnecessary coordinates
    ds_pet = ds_pet.drop_vars([
        coord for coord in ds_pet.coords 
        if coord not in var_list + ["latitude", "longitude", "time"]
    ])
    
    # calculate PET
    ds_pet["pet_makkink_lsa"] = forcing.pet(
        ds_pet,
        temp=ds_pet.temp,
        dem_model=mod.grid["wflow_dem"],
        method="makkink",
        press_correction=True,
    )
    
    # add PET to forcing using landsurfacetemp workflow
    mod = landsurfacetemp.add_var_to_forcing(
        mod=mod,
        ds=ds_pet["pet_makkink_lsa"],
        var="pet_makkink_lsa",
    )
    
    # standardize encoding across all forcing variables using existing patterns
    encoding = mod.forcing["precip"].encoding
    fillval = encoding["_FillValue"]
    
    for key, item in mod.forcing.items():
        # enforce nodata
        if "_FillValue" not in item.attrs:
            print(f"{key}: no _FillValue")
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
    
    # rename variables if needed
    rename_keys = {"pet": "pet_makkink_hyras"}
    for k, v in rename_keys.items():
        if k in mod.forcing:
            mod.forcing[v] = mod.forcing[k]
        if k in mod.forcing:
            mod.forcing.pop(k)
    
    # update model configuration using existing patterns
    mod.config["starttime"] = "2020-01-01T00:00:00"
    mod.config["endtime"] = "2020-12-31T00:00:00"
    mod.config["input"]["path_forcing"] = "inmaps/LSA_forcing_2020.nc"
    
    # save updated model
    mod.set_root(new_root, mode="w+")
    mod.write_forcing(os.path.join(new_root, "inmaps", "LSA_forcing_2020.nc"))
    
    print("Model forcing updated successfully using landsurfacetemp workflow methods!")
    
    # optional: plot results
    plot_forcing(mod, "albedo", time=300, new_root=new_root)
    plot_forcing(mod, "shortwave_down", time=300, vmax=100, new_root=new_root)
    plot_forcing(mod, "wind", time=300, vmax=4, new_root=new_root)
    plot_forcing(mod, "pet_makkink_lsa", time=300, vmax=2, new_root=new_root)

if __name__ == "__main__":
    main() 