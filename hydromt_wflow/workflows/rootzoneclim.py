# -*- coding: utf-8 -*-

import logging
import numpy as np
import pandas as pd
from scipy import optimize
import xarray as xr

from hydromt import flw
import pyflwdir


logger = logging.getLogger(__name__)

__all__ = ["rootzoneclim"]


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
    # Determine the terms (note that the discharge coefficient and evaporative
    # index for the future climate, if present, are not correct yet and will
    # be adjusted at a later stage)
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


def determine_omega(ds_sub_annual):
    """
    This function uses the Zhang function to determine the omega parameter for
    ds_sub_annual.

    Parameters
    ----------
    ds_sub_annual : xarray dataset
        Xarray dataset containing at least the discharge coefficient, aridity
        index and evaporative index for the different forcing types.

    Returns
    -------
    ds_sub_annual : xarray dataset
        Same as above, but with the omega parameter added. The omega parameter
        is the same for all forcing types and based on the observations.

    """
    # Constract the output variable in the dataset
    ds_sub_annual = ds_sub_annual.assign(
        omega = lambda ds_sub_annual: ds_sub_annual['discharge_coeff'] * np.nan
        )
    # Load the aridity index and evaporative index as np arrays (this saves
    # calculation time in the loop below)
    aridity_index = ds_sub_annual.sel(forcing_type="obs")["aridity_index"].values
    evap_index = ds_sub_annual.sel(forcing_type="obs")["evap_index"].values
    # Set the temporary omega variable
    omega = np.zeros(
        (len(ds_sub_annual.index), 
         len(ds_sub_annual.forcing_type))
        ) 
       
    for subcatch_index_nr in range(len(ds_sub_annual.index)):
        try:
            omega_temp = optimize.brentq(
                Zhang, 
                1, 
                8, 
                args=(aridity_index[subcatch_index_nr], 
                      evap_index[subcatch_index_nr])
                )
            omega[subcatch_index_nr, :] = np.repeat(omega_temp, len(ds_sub_annual.forcing_type)) 
        #TODO: check this (possible error that occurs: "ValueError: f(a) and f(b) must have different signs")
        except ValueError:
            omega[subcatch_index_nr, :] = np.repeat(np.NaN, len(ds_sub_annual.forcing_type)) 
    
    # Add omega to the xr dataset
    ds_sub_annual["omega"] = (("index", "forcing_type"), omega)
        
    return ds_sub_annual


#TODO: Find a way to make this also possible for variable Imax values, e.g. from
# wflow_sbm LAI.
def determine_Peffective_Interception_explicit(ds_sub, Imax, LAI = None):
    """
    Function to determine the effective precipitation, interception evaporation
    and canopy storage based on (daily) values of precipitation and potential
    evaporation.

    Parameters
    ----------
    ds_sub : xarray datase
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
    # Make the output dataset ready for the output
    ds_sub["evap_interception"] = (
        ("index", "forcing_type", "time"),
        np.zeros(
            (len(ds_sub["index"]), len(ds_sub["forcing_type"]), len(ds_sub["time"]))
            )
        )
    ds_sub["precip_effective"] = (
        ("index", "forcing_type", "time"),
        np.zeros(
            (len(ds_sub["index"]), len(ds_sub["forcing_type"]), len(ds_sub["time"]))
            )
        )
    ds_sub["canopy_storage"] = (
        ("index", "forcing_type", "time"),
        np.zeros(
            (len(ds_sub["index"]), len(ds_sub["forcing_type"]), len(ds_sub["time"]))
            )
        )    
    # Calculate it per forcing type
    for forcing_type in ds_sub["forcing_type"].values:
        nr_time_steps = len(ds_sub.sel(forcing_type=forcing_type).time)
    
        # Add new empty variables that will be filled in the loop
        evap_interception = np.zeros(
            (len(ds_sub.sel(forcing_type=forcing_type)["index"]), 
             len(ds_sub.sel(forcing_type=forcing_type)["time"]))
            ) 
        precip_effective = np.zeros(
            (len(ds_sub.sel(forcing_type=forcing_type)["index"]), 
             len(ds_sub.sel(forcing_type=forcing_type)["time"]))
            ) 
        canopy_storage = np.zeros(
            (len(ds_sub.sel(forcing_type=forcing_type)["index"]), 
             len(ds_sub.sel(forcing_type=forcing_type)["time"]))
            )
       
        # Load the potential evaporation and precipitation into memory to speed up
        # the subsequent for loop
        Epdt = ds_sub.sel(forcing_type=forcing_type)["pet_mean"].values
        Pdt = ds_sub.sel(forcing_type=forcing_type)["precip_mean"].values
        # Loop through the time steps and determine the variables per time step.
        for i in range(0, nr_time_steps):
            #TODO: implement Imax as function of LAI
            # Imax = LAI.iloc[i]
            # Determine the variables with a simple interception reservoir approach
            canopy_storage[:,i] = canopy_storage[:,i] + Pdt[:,i]
            precip_effective[:,i] = np.maximum(0, canopy_storage[:,i] - Imax)
            canopy_storage[:,i] = canopy_storage[:,i] - precip_effective[:,i]
            evap_interception[:,i] = np.minimum(Epdt[:,i], canopy_storage[:,i])
            canopy_storage[:,i] = canopy_storage[:,i] - evap_interception[:,i]
            # Update Si for the next time step
            if i < nr_time_steps - 1:
                canopy_storage[:,i+1] = canopy_storage[:,i]
        
        ds_sub["evap_interception"].loc[dict(forcing_type=forcing_type)] = evap_interception
        ds_sub["precip_effective"].loc[dict(forcing_type=forcing_type)] = precip_effective
        ds_sub["canopy_storage"].loc[dict(forcing_type=forcing_type)] = canopy_storage    
    
    return ds_sub


def determine_storage_deficit(ds_sub):
    """
    Function to determine the storage deficit for every time step, subcatchment
    location and datset in ds_sub.

    Parameters
    ----------
    ds_sub : xarray dataset
        xarray dataset containing the daily or higher-resolution data.

    Returns
    -------
    ds_sub : xarray dataset
        Same as above, but containing the storage deficits per time step for all
        forcing types.
    """      
    
    # transpiration = ds_sub["transpiration"]
    # precip_effective = ds_sub["precip_effective"]
    # storage_deficit = ds_sub["storage_deficit"]
    # transpiration = transpiration.to_dataframe()
    # precip_effective = precip_effective.to_dataframe()
    # storage_deficit = storage_deficit.to_dataframe()
    
    
    transpiration = ds_sub["transpiration"].values
    precip_effective = ds_sub["precip_effective"].values
    storage_deficit = np.zeros((
        len(ds_sub['index']), 
        len(ds_sub['forcing_type']), 
        len(ds_sub['time'])
        ))
    
    # Determine the storage deficit per time step
    for i in range(1,len(ds_sub.time)):
        # ds_sub["storage_deficit"].loc[dict(time = ds_sub.time[i])] = np.minimum(
        #     0, 
        #     ds_sub["storage_deficit"].isel(time = i-1) + (ds_sub["precip_effective"].isel(time = i) - ds_sub["transpiration"]).isel(time = i)
        #     )
        storage_deficit[:, :, i] = np.minimum(
            0,
            storage_deficit[:, :, i-1] + (precip_effective[:, :, i] - transpiration[:, :, i])
            )
    
    # Create the storage deficit data array
    ds_sub["storage_deficit"] = (('index', 'forcing_type', 'time'), storage_deficit)
        
        
    # If there are climate projections present, adjust the storage deficit for
    # the future projections based on Table S1 in Bouaziz et al., 2002, HESS.
    # cc_hist remains as is (see Table S1)
    if len(ds_sub.forcing_type) > 1:
        ds_sub["storage_deficit"].loc[dict(forcing_type="cc_fut")] = ds_sub["storage_deficit"].sel(forcing_type="obs") + np.minimum(
            0.0,
            ds_sub["storage_deficit"].sel(forcing_type="cc_fut") - ds_sub["storage_deficit"].sel(forcing_type="cc_hist")
            )        
    
    return ds_sub


def fut_discharge_coeff(ds_sub_annual):
    """
    Function to determine the future discharge coefficient, based on a given 
    omega value (generally same as in the current-climate observations), the
    aridity index of the future climate and the difference in Q and PET between
    the simulated historical and simulated future climate.

    Parameters
    ----------
    ds_sub_annual : xarray dataset
        Xarray dataset containing at least the future omega and aridity index.

    Returns
    -------
    ds_sub_annual : xarray dataset
        Similar to previous, but containing the new future discharge coefficient
        and evaporative index.
    """
    # Determine the delP and delEP for cc_fut
    Ep = ds_sub_annual['pet_mean'].sel(forcing_type = "obs").mean("time")
    P = ds_sub_annual['precip_mean'].sel(forcing_type = "obs").mean("time")
    delP = (ds_sub_annual['precip_mean'].sel(forcing_type = ['cc_hist', 'cc_fut']).mean('time').diff('runs')).sel(forcing_type = 'cc_fut')
    delEp = (ds_sub_annual['pet_mean'].sel(forcing_type = ['cc_hist', 'cc_fut']).mean('time').diff('runs')).sel(forcing_type = 'cc_fut')
    
    # Determine the difference in discharge between the observations and the 
    # future climate simulations
    Q_obs_mean = ds_sub_annual['specific_Q'].mean("time")
    omega = ds_sub_annual["omega"].sel(forcing_type = "cc_fut")
    aridity_index_fut = (Ep+delEp)/(P + delP)
    change_Q_total = Zhang_future(omega, aridity_index_fut) * (P + delP) - Q_obs_mean
        
    # Determine the discharge coefficient for the future climate
    ds_sub_annual['discharge_coeff'].loc[dict(forcing_type = 'cc_fut')] = (Q_obs_mean + change_Q_total) / (P + delP)
    
    return ds_sub_annual


def gumbel_su_calc_xr(storage_deficit_annual, 
                      storage_deficit_count, 
                      return_period, 
                      threshold):
    """
    Function to determine the Gumbel distribution for a set of return periods.

    Parameters
    ----------
    storage_deficit_annual : xarray dataset
        xarray dataset with the minimum deficit per year as a positive value.
    storage_deficit_count : xarray dataset
        xarray dataset containing the number of days with data per year. This
        indicates on how many days a minimum in year_min_storage_deficit is
        based.
    return_period : list
        List with one or more values indiciating the return period(s) (in 
        years) for wich the rootzone storage depth should be calculated.
    threshold : int
        Required minimum number of days in a year containing data to take the
        year into account for the calculation..

    Returns
    -------
    gumbel : xarray dataset
        xarray dataset, similar to year_min_storage_deficit, but containing the
        root-zone storage capacity per return period.

    """
    # Only take the years into account that contain more than [threshold] days
    # of data.
    storage_deficit_annual = storage_deficit_annual.where(storage_deficit_count > threshold) 
        
    # Calculate the mean and standard deviation for the Gumbel distribution
    annual_mean = storage_deficit_annual.mean('time', skipna=True)
    annual_std = storage_deficit_annual.std('time', skipna=True)
    
    # Calculate alpha and mu
    alpha = (np.sqrt(6.0) * annual_std) / np.pi
    mu = annual_mean - 0.5772 * alpha

    # Create the output dataset
    gumbel = storage_deficit_annual.to_dataset(name="storage_deficit")

    # Set the return periods
    RP = return_period
    gumbel['RP'] = RP
    gumbel["rootzone_storage"] = (
        ('index', 'forcing_type', 'RP'), 
        np.zeros((len(gumbel["index"]), len(gumbel["forcing_type"]), len(gumbel["RP"])))
        )
    
    # Determine the root zone storage for different return periods using alpha 
    #and mu
    gumbel['yt'] = -np.log(-np.log(1 - (1 / gumbel['RP'])))
    gumbel['rootzone_storage'] = mu + alpha * gumbel['yt']

    return gumbel


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


def Zhang_future(omega, aridity_index):
    """
    Once omega has been derived for historical situations, it can be used to 
    derive the new discharge coefficient when the future aridity index (Ep/P) 
    is known. This will allow the discharge coeffcient to shift over the same 
    line as the historical situation.

    Parameters
    ----------
    omega : xarray datarray
        Xarray datarray containing the future omega values per sub catchment.
    aridity_index : xarray datarray
        Xarray datarray containing the future aridity index values per sub 
        catchment.

    Returns
    -------
    xarray datarray containing the future discharge coefficient.
    """  
    return - (aridity_index - (1 + aridity_index**omega)**(1/omega))


def check_inputs(start_hydro_year,
                 start_field_capacity,
                 dsrun,
                 ds_obs,
                 ds_cc_hist,
                 ds_cc_fut):
    
    # Start with some initial checks
    list_of_months = ["Jan", 
                      "Feb", 
                      "Mar", 
                      "Apr", 
                      "May", 
                      "Jun", 
                      "Jul", 
                      "Aug", 
                      "Sep", 
                      "Oct", 
                      "Nov", 
                      "Dec"]
    if start_hydro_year not in list_of_months:
        raise ValueError(
            f"start_hydro_year not in {list_of_months}: provide a valid month"
            )
    if start_field_capacity not in list_of_months:
        raise ValueError(
            f"start_field_capacity not in {list_of_months}: provide a valid month"
            )
    
    if "run" not in list(dsrun.keys()):
        raise ValueError(
            "Variable run not in run_fn"
            )
    
    if "precip" not in list(ds_obs.keys()):
        raise ValueError(
            "Variable precip not in forcing_obs_fn"
            )

    if "pet" not in list(ds_obs.keys()):
        raise ValueError(
            "Variable pet not in forcing_obs_fn"
            )
    if ds_cc_hist != None:
        if "precip" not in list(ds_cc_hist.keys()):
            raise ValueError(
                "Variable precip not in forcing_cc_hist_fn"
                )
    
        if "pet" not in list(ds_cc_hist.keys()):
            raise ValueError(
                "Variable pet not in forcing_cc_hist_fn"
                )
    if ds_cc_fut != None:
        if "precip" not in list(ds_cc_fut.keys()):
            raise ValueError(
                "Variable precip not in forcing_cc_fut_fn"
                )
    
        if "pet" not in list(ds_cc_fut.keys()):
            raise ValueError(
                "Variable pet not in forcing_cc_fut_fn"
                )
            
    return None


def rootzoneclim(ds_obs,
                 ds_cc_hist,
                 ds_cc_fut,
                 dsrun, 
                 ds_like, 
                 flwdir, 
                 return_period,
                 Imax, 
                 start_hydro_year,
                 start_field_capacity,
                 LAI,
                 rooting_depth,
                 chunksize,
                 logger=logger):
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
    ds_obs : xarray.Dataset
        Dataset with the observed forcing data.
    ds_cc_hist : xarray.Dataset
        Dataset with the simulated historical forcing data, based on a climate
        model.
    ds_cc_fut : xarray.Dataset
        Dataset with the simulated future climate forcing data, based on a 
        climate model.
    dsrun : str
        Geodataset with streamflow locations and timeseries.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    flwdir : FlwDirRaster
        flwdir object
    return_period : list
        List with one or more values indiciating the return period(s) (in 
        years) for wich the rootzone storage depth should be calculated.
    Imax : float
        The maximum interception storage capacity [mm].
    start_hydro_year : str
        The start month (abreviated to the first three letters of the month,
        starting with a capital letter) of the hydrological year. 
    start_field_capacity : str
        The end of the wet season / commencement of dry season. This is the
        moment when the soil is at field capacity, i.e. there is no storage
        deficit yet.
    rooting_depth : bool
        Boolean indicating whether also the rooting depth (rootzone storage / 
        (theta_s - theta_r)) should be stored.
    LAI : bool
        Determine whether the LAI will be used to determine Imax.
    chunksize : int
        Chunksize on time dimension for processing data (not for saving to 
        disk!). If None, a chunksize of 1000 is used on the time dimension.
        
    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing root zone storage capacity.
        
    References
    ----------
    Bouaziz, L. J. E., Aalbers, E. E., Weerts, A. H., Hegnauer, M., Buiteveld, 
    H., Lammersen, R., Stam, J., Sprokkereef, E., Savenije, H. H. G. and 
    Hrachowitz, M. (2022). Ecosystem adaptation to climate change: the 
    sensitivity of hydrological predictions to time-dynamic model parameters, 
    Hydrology and Earth System Sciences, 26(5), 1295-1318. DOI: 
    10.5194/hess-26-1295-2022.
    """
    # Start with some initial checks
    check_inputs(start_hydro_year, 
                 start_field_capacity,
                 dsrun,
                 ds_obs,
                 ds_cc_hist,
                 ds_cc_fut)
    
    # Concatenate all forcing types (obs, cc_hist, cc_fut) into on xr dataset
    if ds_cc_hist != None and ds_cc_fut != None:
        ds_concat = xr.concat(
            [ds_obs, ds_cc_hist, ds_cc_fut], 
            pd.Index(["obs", "cc_hist", "cc_fut"], name="forcing_type")
            )
    else:
        ds_concat = xr.concat(
            [ds_obs], 
            pd.Index(["obs"], name="forcing_type")
            )
    
    # if LAI == True:
    #     ds_concat["LAI"] = ds_like["LAI"]
    #     ds_concat["swood"] = ds_like["swood"]
    #     ds_concat["sl"] = ds_like["sl"]
    
    # Set the output dataset at model resolution
    ds_out = xr.Dataset(coords=ds_like.raster.coords)

    # Make a basin map containing all subcatchments from geodataset of observed 
    # streamflow
    x, y = dsrun.x.values, dsrun.y.values
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

    # Add the catchment area to gdf_basins and sort in a descending order
    # # Note that we first have to reproject to a cylindircal equal area in #TODO: new method added, which one do we keep?
    # # order to preserve the area measure. 
    # gdf_basins_copy = gdf_basins.copy()
    # gdf_basins['area'] = gdf_basins_copy['geometry'].to_crs({'proj':'cea'}).area
    # gdf_basins = gdf_basins.sort_values(by="area", ascending=False)
    # gdf_basins_copy = None 
    # Calculate the catchment area based on the upstream area in the
    # staticmaps
    areas = []
    for index in gdf_basins.index:
        areas.append(
            ds_like["wflow_uparea"].sel(
                lat=dsrun.sel(index=index)["y"].values, 
                lon=dsrun.sel(index=index)["x"].values, 
                method="nearest"
                ).values * 1e6
            )
    gdf_basins["area"] = areas
    gdf_basins = gdf_basins.sort_values(by="area", ascending=False)
    
    # calculate mean areal precip and pot evap for the full upstream area of each gauge.
    ds_sub = ds_concat.raster.zonal_stats(gdf_basins, stats=["mean"])
    logger.info("Computing zonal statistics, this can take a while")
    ds_sub = ds_sub.compute()
    
    # # If LAI = True, determine the Imax for every time step in the LAI data
    # if LAI == True:
    #     ds_sub["Imax"] = ds_sub["LAI"] * ds_sub["swood"] + ds_sub["LAI"] * ds_sub["sl"]
    
    # Get the time step of the datasets and make sure they all have a daily
    # time step. If not, resample.
    time_step = 86400 # in seconds
    if (ds_sub.time[1].values - ds_sub.time[0].values).astype('timedelta64[s]').astype(np.int32) != time_step:
        ds_sub = ds_sub.resample(time = "1D").sum('time', skipna=True)
    if (dsrun.time[1].values - dsrun.time[0].values).astype('timedelta64[s]').astype(np.int32) != time_step:
        dsrun = dsrun.resample(time = "1D").mean('time', skipna=True)
        
    # Add specific discharge per location-index value to ds_sub
    # First, sort dsrun based on descending subcatchment area (which is already
    # done in ds_sub)
    dsrun = dsrun.sel(index=ds_sub.index.values)
    # Get the specific discharge (mm/timestep) per location in order to have
    # everything in mm/timestep
    dsrun = dsrun.assign(
        specific_Q=dsrun["run"]/np.array(gdf_basins["area"]) * time_step * 1000.0
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
       
    # Determine effective precipitation, interception evporation and canopy 
    # storage
    #TODO: Add time variable Imax (based on LAI)
    logger.info("Determine effective precipitation, interception evporation and canopy storage")
    ds_sub = determine_Peffective_Interception_explicit(ds_sub, Imax=Imax)
 
    # Rechunk data
    if chunksize == None:
        chunksize = 100
    ds_sub = ds_sub.chunk(chunks={'index': len(ds_sub.index), 'forcing_type': 1, 'time': chunksize})
 
    # Get year sums of ds_sub
    #TODO: check if this still works when different time periods are present (i.e. current and future climate)
    ds_sub_annual = ds_sub.resample(time = f'AS-{start_hydro_year}').sum('time', skipna=True)
    # A counter will be used and a threshold, to only use years with sufficient
    # days containing data in the subsequent calculations
    ds_sub_annual_count = ds_sub.resample(time = f'AS-{start_hydro_year}').count('time')
    missing_threshold = 330
    
    # Determine discharge coefficient, the aridity index and the evaporative
    # index
    ds_sub_annual = determine_budyko_curve_terms(ds_sub_annual, 
                                                 ds_sub_annual_count, 
                                                 threshold=missing_threshold)
    ds_sub_annual_count = None
    
    # Determine omega
    logger.info("Calculating the omega values, this can take a while")
    ds_sub_annual = determine_omega(ds_sub_annual)
    
    # Determine future discharge ratio for cc_fut if ds_cc_fut exists
    if ds_cc_fut != None:
        ds_sub_annual = fut_discharge_coeff(ds_sub_annual)
    
    # Determine long-term interception, potential evaporation and tranpsiration; 
    # use runoff coefficient instead of Qobs to calculate actual evaporation for 
    # each climate projection
    transpiration_long_term = ds_sub_annual['precip_effective'].mean('time') -  ds_sub_annual['precip_mean'].mean('time') * ds_sub_annual['discharge_coeff'] #ds_annual['Qobs_mm'].mean('time')
    interception_long_term = ds_sub_annual['evap_interception'].mean('time') 
    pet_long_term = ds_sub_annual['pet_mean'].mean('time') 
    
    # Determine the transpiration on the finer time step (e.g. daily)
    ds_sub['transpiration'] = (ds_sub['pet_mean'] - ds_sub['evap_interception']) * transpiration_long_term / (pet_long_term - interception_long_term)
    transpiration_long_term = None
    interception_long_term = None
    pet_long_term = None
    
    # Determine storage deficit
    logger.info("Determining the storage deficit")
    ds_sub = determine_storage_deficit(ds_sub)
    
    # From the storage deficit, determine the rootzone storage capacity using
    # a Gumbel distribution.
    logger.info("Calculating the Gumbel distribution and rootzone storage capacity")
    # Determine the yearly minima in storage deficit (and make sure the values
    # are positive)
    storage_deficit_annual = - (ds_sub["storage_deficit"].resample(time=f'AS-{start_field_capacity}').min('time', skipna=True))

    # A counter will be used to only use years with sufficient days containing 
    # data for the Gumbel distribution
    storage_deficit_count = ds_sub["storage_deficit"].resample(time=f'AS-{start_field_capacity}').count('time')
    
    # Subsequently, determine the Gumbel distribution
    gumbel = gumbel_su_calc_xr(storage_deficit_annual, 
                               storage_deficit_count.sel(time=storage_deficit_count.time[1:]), 
                               threshold=missing_threshold,
                               )
    
    #TODO: pick the rootzone storage based on the requested return period
    
    # Create a new geopandas dataframe, which will be used to rasterize the
    # results
    ds_basins_all = flw.basin_map(
        ds_like,
        flwdir,
        ids=dsrun.index.values,
        xy=(x, y),
        stream=ds_like["wflow_river"],
    )[0]
    # ds_basin_all.name = int(dsrun.index.values)
    ds_basins_all.raster.set_crs(ds_like.raster.crs)
    gdf_basins_all = ds_basins_all.raster.vectorize()
    # Add the area and sort by area (from large to small)
    areas = []
    for index in gdf_basins_all.value:
        areas.append(
            ds_like["wflow_uparea"].sel(
                lat=dsrun.sel(index=index)["y"].values, 
                lon=dsrun.sel(index=index)["x"].values, 
                method="nearest"
                ).values * 1e6
            )
    gdf_basins_all["area"] = areas
    gdf_basins_all = gdf_basins_all.sort_values(by="area", ascending=False)
    
    # Add the rootzone storage, per forcing type and return period, to
    # gdf_basin_all
    for return_period in gumbel.RP.values:
        for forcing_type in gumbel.forcing_type.values:
            gdf_basins_all[f"{forcing_type}_{str(return_period)}"] = gumbel["rootzone_storage"].sel(RP=return_period, forcing_type=forcing_type)
            # Make sure to give the NaNs a value, otherwise they will become 0.0
            gdf_basins_all[f"{forcing_type}_{str(return_period)}"] = gdf_basins_all[f"{forcing_type}_{str(return_period)}"].fillna(-999)
   
    # Rasterize this (from large subcatchments to small ones)
    for return_period in gumbel.RP.values:
        for forcing_type in gumbel.forcing_type.values:    
            da_area = ds_like.raster.rasterize(
                gdf=gdf_basins_all,
                col_name=f"{forcing_type}_{str(return_period)}",
                nodata=-999,
                all_touched=True,
                ).to_dataset(name="rasterized_temp")
            # Fill up the not a numbers with the data from a downstream point 
            out_raster = pyflwdir.FlwdirRaster.fillnodata(
                flwdir,
                data=da_area["rasterized_temp"], 
                nodata=-999, 
                direction='up', 
                how='max',
                )
            # Make sure to fill up full domain with value of most downstream point
            fill_value = None
            for value in gdf_basins_all[f"{forcing_type}_{str(return_period)}"]:
                if value > 0.0:
                    if fill_value == None:
                        fill_value = value
            out_raster = np.where(out_raster == -999.0, fill_value, out_raster)
            # Store the result in ds_out
            #TODO: perhaps we should use a mask here.
            ds_out[f"rootzone_storage_{forcing_type}_{str(return_period)}"] = (
                ("lat", "lon"), 
                out_raster
                )
            # Also store the RootingDepth if requested
            if rooting_depth == True:
                ds_out[f"RootingDepth_{forcing_type}_{str(return_period)}"] = (
                    ("lat", "lon"), 
                    out_raster / (ds_like["thetaS"].values - ds_like["thetaR"].values)
                    )            
    
    return ds_out


