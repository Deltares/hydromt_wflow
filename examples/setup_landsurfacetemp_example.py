"""
Example: Using setup_landsurfacetemp method to replace patchwork approach.

This example demonstrates how to use the new setup_landsurfacetemp method
to replace the manual patchwork approach in update_forcing.py.
The method automatically handles both time-varying and static data.
"""

from hydromt_wflow import WflowModel
from hydromt.data_catalog import DataCatalog
import os

def main():
    """Main function demonstrating the setup_landsurfacetemp approach."""
    
    # setup paths and catalogs
    cats = DataCatalog(["data_catalog_old_version.yml", "deltares_data", "hyras_dc.yml"])
    root = r"p:\1000365-002-wflow\tmp\wflow-gleam\model\base_model_canopy"
    new_root = r"p:\1000365-002-wflow\tmp\wflow-gleam\model\canopy_model_lsa_forcing_setup"
    
    # load model
    mod = WflowModel(root=root, config_fn="wflow_sbm.toml", mode="r+")
    mod.read_config()
    mod.read_forcing()
    mod.read_geoms()
    
    # Use the new setup_landsurfacetemp method instead of patchwork
    # The method automatically detects whether data is time-varying or static
    # and places it in the appropriate file (forcing vs staticmaps)
    mod.setup_landsurfacetemp(
        # Time-varying data (will go to forcing file)
        albedo_fn="albedo",                    # Time-varying albedo from catalog
        shortwave_fn="shortwave_down",          # Time-varying shortwave radiation
        wind_u_fn="era5",                      # Time-varying ERA5 wind components
        wind_v_fn="era5",                      # Time-varying ERA5 wind components
        
        # Static data (will go to staticmaps file)
        # emissivity_fn="static_emissivity",   # Static emissivity (if available)
        
        # Configuration options
        wind_altitude=10,                      # Wind measurement altitude
        wind_altitude_correction=False,        # Don't correct to 2m
        pet_method="makkink",                  # Use Makkink method for PET
        press_correction=True,                 # Apply pressure correction
        temp_correction=True,                  # Apply temperature correction
        wind_correction=True,                  # Apply wind correction
        reproj_method="nearest_index",         # Reprojection method
        output_names={                         # Custom output names
            "albedo": "albedo",
            "emissivity": "emissivity",
            "shortwave_down": "shortwave_down", 
            "wind": "wind",
            "pet_makkink_lsa": "pet_makkink_lsa",
        }
    )
    
    # Update model configuration
    mod.config["starttime"] = "2020-01-01T00:00:00"
    mod.config["endtime"] = "2020-12-31T00:00:00"
    mod.config["input"]["path_forcing"] = "inmaps/LSA_forcing_2020.nc"
    
    # Save updated model
    mod.set_root(new_root, mode="w+")
    mod.write_forcing(os.path.join(new_root, "inmaps", "LSA_forcing_2020.nc"))
    
    print("Model forcing updated successfully using setup_landsurfacetemp method!")
    print(f"Added forcing variables: {list(mod.forcing.keys())}")
    print(f"Added grid variables: {list(mod.grid.keys())}")

def example_static_data():
    """Example showing how to use static data with setup_landsurfacetemp."""
    
    # Example with static albedo data
    mod = WflowModel(root="path/to/model", mode="r+")
    mod.read_config()
    mod.read_geoms()
    
    # Create static albedo data (no time dimension)
    import xarray as xr
    import numpy as np
    
    # Static albedo data (will go to staticmaps)
    static_albedo = xr.DataArray(
        np.random.uniform(0.1, 0.3, (100, 100)),  # No time dimension
        dims=["y", "x"],
        coords={"y": np.linspace(0, 1, 100), "x": np.linspace(0, 1, 100)},
        name="albedo"
    )
    
    # Use static data
    mod.setup_landsurfacetemp(
        albedo_fn=static_albedo,  # Static data -> goes to grid/staticmaps
        # Other parameters...
    )
    
    print("Static albedo added to grid/staticmaps")

if __name__ == "__main__":
    main() 