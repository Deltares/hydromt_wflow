"""
Example: Using setup_landsurfacetemp with net radiation calculation.

This example demonstrates how to use the setup_landsurfacetemp method
to calculate net radiation in preprocessing, which is then used by Wflow.jl
instead of calculating it internally.
"""

from hydromt_wflow import WflowModel
from hydromt.data_catalog import DataCatalog
import xarray as xr
import matplotlib.pyplot as plt
import os

def main():
    """Main function demonstrating the setup_landsurfacetemp with net radiation approach."""
    
    # setup paths and catalogs
    cats = DataCatalog(["data_catalog_old_version.yml", "deltares_data", "hyras_dc.yml"])
    root = r"p:\1000365-002-wflow\tmp\wflow-gleam\model\base_model_canopy"
    new_root = r"p:\1000365-002-wflow\tmp\wflow-gleam\model\canopy_model_lsa_forcing_net_radiation"
    
    # create new model
    mod = WflowModel(root=root, config_fn="wflow_sbm.toml", mode="r+")
    mod.read_config()
    mod.read_forcing()
    mod.read_geoms()
    
    # setup land surface temperature forcing with net radiation calculation
    mod.setup_landsurfacetemp(
        # Input data sources
        albedo_fn="albedo",  # from data catalog
        shortwave_fn="shortwave_down",  # from data catalog
        wind_u_fn="era5",  # wind U component from ERA5
        wind_v_fn="era5",  # wind V component from ERA5
        
        # Processing options
        wind_altitude=10,  # Wind measurement altitude (default: 10m)
        wind_altitude_correction=False,  # Let Wflow.jl handle altitude correction
        pet_method="makkink",
        press_correction=True,
        temp_correction=True,
        wind_correction=True,
        
        # Output names
        output_names={
            "albedo": "albedo",
            "shortwave_down": "shortwave_down", 
            "wind": "wind",
            "pet_makkink_lsa": "pet_makkink_lsa",
        }
    )
    
    # The setup_landsurfacetemp method now automatically calculates:
    # 1. net_longwave_radiation - using the compute_net_longwave_radiation function
    # 2. net_radiation - using the compute_net_radiation function (if albedo is available)
    # 3. wind_altitude configuration - added to config for Wflow.jl to use
    
    print("Available forcing variables:")
    for var in mod.forcing.keys():
        print(f"  - {var}")
    
    # Check wind altitude configuration
    print(f"\nWind altitude configuration:")
    print(f"  wind_altitude: {mod.config['input'].get('wind_altitude', 'Not set')} m")
    print(f"  Note: Wflow.jl will use these values to apply canopy corrections")
    
    # Check if net radiation was calculated
    if "net_radiation" in mod.forcing:
        print(f"\nNet radiation calculated successfully!")
        print(f"  Shape: {mod.forcing['net_radiation'].shape}")
        print(f"  Time range: {mod.forcing['net_radiation'].time.min().values} to {mod.forcing['net_radiation'].time.max().values}")
        print(f"  Value range: {mod.forcing['net_radiation'].min().values:.2f} to {mod.forcing['net_radiation'].max().values:.2f} W m-2")
    
    if "net_longwave_radiation" in mod.forcing:
        print(f"\nNet longwave radiation calculated successfully!")
        print(f"  Shape: {mod.forcing['net_longwave_radiation'].shape}")
        print(f"  Value range: {mod.forcing['net_longwave_radiation'].min().values:.2f} to {mod.forcing['net_longwave_radiation'].max().values:.2f} W m-2")
    
    # Plot example time steps
    if "net_radiation" in mod.forcing:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot net radiation at different times
        times_to_plot = [100, 300]
        for i, time_idx in enumerate(times_to_plot):
            mod.forcing["net_radiation"].isel(time=time_idx).raster.mask_nodata().plot(
                ax=axes[i, 0], vmin=-100, vmax=400, cmap="RdBu_r"
            )
            axes[i, 0].set_title(f"Net Radiation t={time_idx}")
            
            if "net_longwave_radiation" in mod.forcing:
                mod.forcing["net_longwave_radiation"].isel(time=time_idx).raster.mask_nodata().plot(
                    ax=axes[i, 1], vmin=0, vmax=100, cmap="viridis"
                )
                axes[i, 1].set_title(f"Net Longwave Radiation t={time_idx}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(new_root, "net_radiation_example.png"), dpi=300, bbox_inches="tight")
        plt.show()
    
    # Write the model with the new forcing
    mod.set_root(new_root, mode="w+")
    mod.write_forcing(os.path.join(new_root, "inmaps", "LSA_forcing_with_net_radiation.nc"))
    
    print(f"\nModel written to: {new_root}")
    print("The net_radiation variable is now available for Wflow.jl to use directly!")
    print("Wflow.jl no longer needs to calculate net radiation internally.")
    print("Wind altitude configuration is available for canopy corrections in Wflow.jl.")

if __name__ == "__main__":
    main() 