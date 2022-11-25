# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import pandas as pd
import logging
from hydromt import flw


logger = logging.getLogger(__name__)

__all__ = ["rootzoneclim"]


# def runModel_Minterception_explicit(ds, Imax, P_var = 'prec_subcatch', Ep_var = 'epot_subcatch', delT = 1,  LAI = None):

#     nT = len(ds.time)

#     #add new variables that are image of Precpitation varibale
#     ds = ds.assign(Ei = lambda ds: ds[P_var] * np.nan)
#     ds = ds.assign(Pe = lambda ds: ds[P_var] * np.nan)
#     ds = ds.assign(Si = lambda ds: ds[P_var] * np.nan)
# #    ds = ds.assign(Si = lambda ds: ds[P_var] * 0) #initialize with zero all time steps if missing at start?

#     #initialize with empty Si
#     ds['Si'].loc[dict(time = ds.time[0])] = 0

#     for i in range(0,nT):
#         #could implement Imax as function of LAI
#     #        Imax = LAI.iloc[i]

#         Epdt = ds[Ep_var].isel(time = i) * delT
#         Pdt  = ds[P_var].isel(time = i) * delT

#         # Interception Reservoir - most simple
#         ds['Si'].loc[dict(time = ds.time[i])] = ds['Si'].isel(time = i) + Pdt
#         ds['Pe'].loc[dict(time = ds.time[i])] = np.maximum(0, ds['Si'].isel(time = i) - Imax)
#         ds['Si'].loc[dict(time = ds.time[i])] = ds['Si'].isel(time = i) - ds['Pe'].isel(time = i)
#         ds['Ei'].loc[dict(time = ds.time[i])] = np.minimum(Epdt, ds['Si'].isel(time = i))
#         ds['Si'].loc[dict(time = ds.time[i])] = ds['Si'].isel(time = i) - ds['Ei'].isel(time = i)

#         if i<nT-1:
#             ds['Si'].loc[dict(time = ds.time[i+1])] = ds['Si'].isel(time = i)

#     return ds

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

# def Zhang(omega, Ep_over_P, Ea_over_P):
#     """
#     This is the Zhang formula with omega as in Teuling et al 2019
#     this function is used to get omega for historical situations when Ep_over_P and Ea_over_P are known (we assume Ea_over_P = 1 - Q_over_P)
#     Zhang eq = 0
#     """
#     return 1 + Ep_over_P - (1 + Ep_over_P**omega)**(1/omega) - Ea_over_P

# def Zhang_future(omega, Ep_over_P_future):
#     """
#     once omega has been derived for historical situations, it can be used to derive the new runoff coeff
#     when future Ep_over_P is known
#     it will allow the RC to shift over the same line as the historical situation.
#     """
#     RC_future = - (Ep_over_P_future - (1 + Ep_over_P_future**omega)**(1/omega))
#     return RC_future


def rootzoneclim(ds, dsrun, ds_like, flwdir, logger=logger):
    """
    Returns root zone storage parameter based on climate and observed streamflow data. Calculated per subcatchment and converted to gridded map at model resolution.
    add paper ref

    The following maps are calculated:\
    - `rootzone_storage`                  : maximum root zone storage capacity [mm]\
    - `rootzone_depth_climate`            : maximum root zone storage depth [mm]\
                    
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with forcing data.
    ds_like : xarray.DataArray
        Dataset at model resolution.
    dsrun : str
        Geodataset with streamflow locations and timeseries.
    flwdir : FlwDirRaster
        flwdir object

    Returns
    -------
    ds_out : xarray.Dataset
        Dataset containing root zone storage capacity.
    """

    ds_out = xr.Dataset(coords=ds_like.raster.coords)

    # make basin map from geodataset of observed streamflow
    x, y = dsrun.x.values, dsrun.y.values
    ds_basin = flw.basin_map(ds_like, flwdir, ids=dsrun.index.values, xy=(x, y))[0]
    gdf_basins = pd.DataFrame()
    # loop over basins
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
    # set index to catchment id
    gdf_basins.index = gdf_basins.value

    # calculate mean areal precip and pot evap for the full upstream area of each gauge.
    ds_sub = ds.raster.zonal_stats(gdf_basins, stats=["mean"])

    ds_sub

    # import pdb; pdb.set_trace()

    # da_basins = flw.basin_map(
    #                         self.staticmaps, self.flwdir, idxs=idxs, ids=ids
    #                     )[0]
    #                     mapname = self._MAPS["basins"] + "_" + basename
    #                     self.set_staticmaps(da_basins, name=mapname)
    #                     gdf_basins = self.staticmaps[mapname].raster.vectorize()
    #                     self.set_staticgeoms(gdf_basins, name=mapname.replace("wflow_", ""))

    logger.info("calculate rootzone storage capacity")

    # srmax = srmax.raster.reproject_like(ds_like, method="average")
    # ds_out["rootzone_storage"] = srmax.astype(np.float32)

    return ds_out
