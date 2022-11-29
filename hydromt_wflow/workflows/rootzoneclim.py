# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
from scipy import optimize
import xarray as xr

from hydromt import flw


logger = logging.getLogger(__name__)

__all__ = ["rootzoneclim"]


#TODO: Find a way to make this also possible for variable Imax values, e.g. from
# wflow_sbm LAI.
def determine_Peffective_Interception_explicit(ds_sub, Imax, LAI = None):
    """
    Function to determine the effective precipitation, interception evaporation
    and canopy storage based on (daily) values of precipitation and potential
    evaporation.

    Parameters
    ----------
    ds_sub : xarray dataset
        xarray dataset containing precipitation and potential evaporation
        (precip_mean and pet_mean).
    Imax : float
        The maximum interception storage capacity [mm].
    LAI : xarray datarray, optional
        Xarray dataarray from staticmaps containing the LAI values for a number
        of time steps, e.g. every month. If present, this is used to determine
        Imax per time step. The default is None.

    Returns
    -------
    ds_sub : xarray datset
        same as above, but with effective precipitation, interception evaporation
        and canopy storage added.
    """

    nr_time_steps = len(ds_sub.time)

    # Add new empty variables to the output
    ds_sub = ds_sub.assign(evap_interception = lambda ds_sub: ds_sub["precip_mean"] * np.nan)
    ds_sub = ds_sub.assign(precip_effective = lambda ds_sub: ds_sub["precip_mean"] * np.nan)
    ds_sub = ds_sub.assign(canopy_storage = lambda ds_sub: ds_sub["precip_mean"] * np.nan)

    #initialize with empty canopy_storage
    ds_sub['canopy_storage'].loc[dict(time = ds_sub.time[0])] = 0

    # Loop through the time steps and determine the variables per time step.
    for i in range(0, nr_time_steps):
        #TODO: implement Imax as function of LAI
        # Imax = LAI.iloc[i]
        
        # Determine epot and precip for this time step
        Epdt = ds_sub["pet_mean"].isel(time = i)
        Pdt  = ds_sub["precip_mean"].isel(time = i)

        # Determine the variables with a simple interception reservoir approach
        ds_sub['canopy_storage'].loc[dict(time = ds_sub.time[i])] = ds_sub['canopy_storage'].isel(time = i) + Pdt
        ds_sub['precip_effective'].loc[dict(time = ds_sub.time[i])] = np.maximum(0, ds_sub['canopy_storage'].isel(time = i) - Imax)
        ds_sub['canopy_storage'].loc[dict(time = ds_sub.time[i])] = ds_sub['canopy_storage'].isel(time = i) - ds_sub['precip_effective'].isel(time = i)
        ds_sub['evap_interception'].loc[dict(time = ds_sub.time[i])] = np.minimum(Epdt, ds_sub['canopy_storage'].isel(time = i))
        ds_sub['canopy_storage'].loc[dict(time = ds_sub.time[i])] = ds_sub['canopy_storage'].isel(time = i) - ds_sub['evap_interception'].isel(time = i)
        
        # Add the new canopy storage to the next time step
        if i < nr_time_steps - 1:
            ds_sub['canopy_storage'].loc[dict(time = ds_sub.time[i+1])] = ds_sub['canopy_storage'].isel(time = i)

    return ds_sub

# def gumbel_su_calc_xr(ds, name_col = 'Sr_def', coords_var1 = 'runs', coords_var2 = 'catchments', time = 'time'):
#     """
#     ds is a dataset with the minimum deficit per year as a positive value
#     name_col is the name of the variable containing the deficit
#     coords_var 1 is the first coordinate alon which gumbel needs to be calculated i.e. runs
#     coords_var 2 is the first coordinate alon which gumbel needs to be calculated i.e. different catchments
#     time is the name of the time variable

#     ds outcome includes Sr,wb for different return periods
#     """
#     #gumbel calculation
# #    import pdb; pdb.set_trace()
#     annual_mean = ds[name_col].mean(time)#.values #step2: mean
#     annual_std = ds[name_col].std(time)#.values #step3: st dev

#     #calc alpha and mu
#     alpha = (sp.sqrt(6) * annual_std) / sp.pi #step4: alpha
#     u = annual_mean - 0.5772 * alpha #step5: u

#     #determine Su for different return periods using alpha and mu
#     RP = [2,3,5,10,15,20,25,50,60,100]
#     ds['RP'] = RP

#     yt = -np.log(-np.log(1-(1/ds['RP'])))
#     ds['yt'] = yt

#     ds['Sr_gumbel'] = ((coords_var1, coords_var2, 'RP'), np.zeros((len(ds[coords_var1]), len(ds[coords_var2]), len(ds.RP))))
#     ds['Sr_gumbel'] = u + alpha * ds['yt']


def Zhang(omega, Ep_over_P, Ea_over_P):
    """
    This is the Zhang equation with omega as in Teuling et al., 2019.
    This function is used to get omega for historical situations when 
    Ep_over_P and Ea_over_P are known (assuming Ea_over_P = 1 - Q_over_P).
    This equation is solved for Zhang eq = 0.

    Parameters
    ----------
    omega : TYPE
        DESCRIPTION.
    Ep_over_P : TYPE
        The aridity index.
    Ea_over_P : TYPE
        The evaporative index.

    Returns
    -------
    Value 0.
        
    References
    ----------
    #TODO: Add Fu et al., Zhang and Teuling et al. references here.
    """
    return 1 + Ep_over_P - (1 + Ep_over_P**omega)**(1/omega) - Ea_over_P

# def Zhang_future(omega, Ep_over_P_future):
#     """
#     once omega has been derived for historical situations, it can be used to derive the new runoff coeff
#     when future Ep_over_P is known
#     it will allow the RC to shift over the same line as the historical situation.
#     """
#     RC_future = - (Ep_over_P_future - (1 + Ep_over_P_future**omega)**(1/omega))
#     return RC_future


def determine_budyko_curve_terms(ds_sub_annual, ds_sub_annual_count, threshold):
    """
    Parameters
    ----------
    ds_sub_annual : xarray dataset
        xarray dataset containing per subcatchment the annual precipitation, 
        potential evaporation and specific discharge sums.
    ds_sub_annual_count: xarray dataset
        xarray dataset containing per subcatchment the number of days, per year,
        that contain data.
    threshold: int
        Required minimum number of days in a year containing data to take the
        year into account for the calculation.

    Returns
    -------
    ds_sub_annual : xarray dataset
        Similar to input, but containing the discharge coefficient, aridity 
        index and the evaporative index as long term averages.

    """
    #TODO: Threshold is only used on Q availability, should it also be done on 
    # the other variables?
    ds_sub_annual['discharge_coeff'] = (ds_sub_annual['specific_Q'].where(ds_sub_annual_count['specific_Q'] > threshold) / ds_sub_annual['precip_mean'].where(ds_sub_annual_count['specific_Q'] > threshold)).mean('time', skipna=True)
    ds_sub_annual['aridity_index'] = (ds_sub_annual['pet_mean'] / ds_sub_annual['precip_mean']).mean('time', skipna=True)
    ds_sub_annual['evap_index'] = 1 - ds_sub_annual['discharge_coeff']
    
    # Make sure Ea = P - Q < Ep. If not, we will not use that subcatchment for
    # the calculations.
    #TODO: What do we do when the largest "subcatchment", which covers all other subcatchments, has Ea > Ep?
    ds_sub_annual['discharge_coeff'] = ds_sub_annual['discharge_coeff'].where(ds_sub_annual["evap_index"] < ds_sub_annual["aridity_index"])
    ds_sub_annual['aridity_index'] = ds_sub_annual['aridity_index'].where(ds_sub_annual["evap_index"] < ds_sub_annual["aridity_index"])
    ds_sub_annual['evap_index'] = ds_sub_annual['evap_index'].where(ds_sub_annual["evap_index"] < ds_sub_annual["aridity_index"])
    
    return ds_sub_annual


def rootzoneclim(ds, dsrun, ds_like, flwdir, Imax=2.0, logger=logger):
    """
    Returns root zone storage parameter for current observed and (optionally 
    for) future climate-based streamflow data. 
    The root zone storage parameter is calculated per subcatchment and is 
    converted to a gridded map at model resolution.

    The following maps are calculated:\
    - `rootzone_storage`                  : maximum root zone storage capacity [mm]\
    - `rootzone_depth_climate`            : maximum root zone storage depth [mm]\
                    
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with forcing data.
    dsrun : str
        Geodataset with streamflow locations and timeseries.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    flwdir : FlwDirRaster
        flwdir object
    Imax : float, optional
        The maximum interception storage capacity [mm]. The default is 2.0 mm.

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing root zone storage capacity.
        
    References
    ----------
    TODO: add paper reference
    """
    #TODO: Add the future climate implementations
    
    # Set the output dataset at model resolution
    ds_out = xr.Dataset(coords=ds_like.raster.coords)

    # Make a basin map containing all subcatchments from geodataset of observed 
    # streamflow
    x, y = dsrun.x.values, dsrun.y.values
    # ds_basin = flw.basin_map(ds_like, flwdir, ids=dsrun.index.values, xy=(x, y))[0] #TODO: remove?
    gdf_basins = pd.DataFrame()
    # Loop over basins and get per gauge location a polygon of the upstream
    # area.
    for i, id in enumerate(dsrun.index.values):
        ds_basin_single = flw.basin_map(
            ds_like,
            flwdir,
            ids=dsrun.index.values[i],
            xy=(x[i], y[i]),
            stream=ds_like["wflow_river"],
        )[0]
        ds_basin_single.name = int(dsrun.index.values[i])
        ds_basin_single.raster.set_crs(ds_like.raster.crs)
        gdf_basin_single = ds_basin_single.raster.vectorize()
        gdf_basins = pd.concat([gdf_basins, gdf_basin_single])
    # Set index to catchment id
    gdf_basins.index = gdf_basins.value.astype("int")
    
    # #TODO: remove this, but for now a way to save the geodataset as shapefile
    # # for inspection in GIS programs    
    # import os   
    # import geopandas
    # outfile = "c:\\Users\\imhof_rn\\OneDrive - Stichting Deltares\\Documents\\SITO\\Root_zone_storage\\geodataset_inspection\\basins_check.shp" 
    # # Export the data
    # geopandas.GeoDataFrame(geometry=gdf_basins['geometry']).to_file(outfile,driver='ESRI Shapefile') 
    
    # Add the catchment area to gdf_basins and sort in a descending order
    # Note that we first have to reproject to a cylindircal equal area in
    # order to preserve the area measure. 
    gdf_basins_copy = gdf_basins.copy()
    gdf_basins['area'] = gdf_basins_copy['geometry'].to_crs({'proj':'cea'}).area
    gdf_basins = gdf_basins.sort_values(by="area", ascending=False)
    gdf_basins_copy = None 

    # calculate mean areal precip and pot evap for the full upstream area of each gauge.
    ds_sub = ds.raster.zonal_stats(gdf_basins, stats=["mean"])
    
    # Add the discharge per location-index value to ds_sub
    # First, sort dsrun based on descending subcatchment area (which is already
    # done in ds_sub)
    dsrun = dsrun.sel(index=ds_sub.index.values)
    # Get the specific discharge (mm/timestep) per location in order to have
    # everything in mm/timestep
    #TODO: add check for time interval - time interval ds_sub and dsrun should
    # be the same, otherwise we should adjust it.
    time_interval = (ds_sub.time[1].values - ds_sub.time[0].values).astype('timedelta64[s]').astype(np.int32)
    #TODO: should we fix the variable name to "run" or should we make this
    # adjustable? In the first case, we should raise an error if that is missing.
    dsrun = dsrun.assign(
        specific_Q=dsrun["run"]/np.array(gdf_basins["area"]) * time_interval * 1000.0
        )
    # Add dsrun to ds_sub
    if dsrun["specific_Q"].dims == ("time", "index"):
        ds_sub = ds_sub.assign(specific_Q = dsrun["specific_Q"].transpose())
    elif dsrun["specific_Q"].dims == ("index", "time"):
        ds_sub = ds_sub.assign(specific_Q = dsrun["specific_Q"])
    else:
        raise ValueError(
            "run_fn, the timeseries with discharge per x,y location, has not the right dimensions. Dimensions (time, index) or (index, time) expected"
            )
    
    # Get year sums of ds_sub
    #TODO: what do we do with October as start of the hydrolgoical year?
    ds_sub_annual = ds_sub.resample(time = 'AS-Oct').sum('time', skipna=True)
    # A counter will be used and a threshold, to only use years with sufficient
    # days containing data in the subsequent calculations
    ds_sub_annual_count = ds_sub.resample(time = 'AS-Oct').count('time')
    missing_threshold = 330
    
    # Determine discharge coefficient, the aridity index and the evaporative
    # index
    ds_sub_annual = determine_budyko_curve_terms(ds_sub_annual, 
                                                 ds_sub_annual_count, 
                                                 threshold=missing_threshold,
                                                 )
    
    # Determine omega
    logger.info("Calculating the omega values, this can take a while") #TODO: Can we speed this up?
    ds_sub_annual = ds_sub_annual.assign(omega = lambda ds_sub_annual: ds_sub_annual['discharge_coeff'] * np.nan)
    counter=0
    tot_count = len(ds_sub_annual["index"].values)
    for subcatch_index in ds_sub_annual["index"].values:
        counter += 1
        print(f"Calculating omega for subcatchment {counter} out of {tot_count}")
        omega = optimize.brentq(
            Zhang, 
            1, 
            8, 
            args=(ds_sub_annual.sel(index=subcatch_index)["aridity_index"].values, 
                  ds_sub_annual.sel(index=subcatch_index)["evap_index"].values)
            )
        ds_sub_annual["omega"].loc[dict(index = subcatch_index)] = omega 
    
    #TODO: Add future climate values here
    
    # Determine effective precipitation, interception evporation and canopy 
    # storage
    #TODO: Add time variable Imax (based on LAI)
    ds_sub = determine_Peffective_Interception_explicit(ds_sub, Imax=Imax)
    
    # Determine long-term tranpsiration
    
    # Determine storage deficit
    
    # From the storage deficit, determine the root-zone storage capacity using
    # a Gumbel distribution.
    
    logger.info("calculate rootzone storage capacity")
    #TODO: future climate
    
    # import pdb; pdb.set_trace()

    # da_basins = flw.basin_map(
    #                         self.staticmaps, self.flwdir, idxs=idxs, ids=ids
    #                     )[0]
    #                     mapname = self._MAPS["basins"] + "_" + basename
    #                     self.set_staticmaps(da_basins, name=mapname)
    #                     gdf_basins = self.staticmaps[mapname].raster.vectorize()
    #                     self.set_staticgeoms(gdf_basins, name=mapname.replace("wflow_", ""))
   
    # Scale the Srmax to the model resolution
    # srmax = srmax.raster.reproject_like(ds_like, method="average")
    
    #TODO: Do something with the nans here when adding all subcatchments
    
    # Store the Srmax fur the current and future climate in ds_out
    # ds_out["rootzone_storage"] = srmax.astype(np.float32)
    #TODO: make optional
    # ds_out["rootzone_storage_climate"] = srmax.astype(np.float32)

    return ds_out
