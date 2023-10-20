"""Rootzoneclim workflows for Wflow plugin."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import pyflwdir
import xarray as xr
from hydromt import flw
from scipy import optimize

logger = logging.getLogger(__name__)

__all__ = ["rootzoneclim"]


def determine_budyko_curve_terms(
    ds_sub_annual,
):
    """
    Determine the Budyko terms.

    (Note that the discharge coefficient and evaporative
    index for the future climate, if present, are not correct yet and will
    be adjusted at a later stage).

    Parameters
    ----------
    ds_sub_annual : xr.Dataset
        Dataset containing per subcatchment the annual precipitation,
        potential evaporation and specific discharge sums.

    Returns
    -------
    ds_sub_annual : xr.Dataset
        Similar to input, but containing the discharge coefficient, aridity
        index and the evaporative index as long term averages.

    """
    ds_sub_annual["discharge_coeff"] = (
        ds_sub_annual["specific_Q"] / ds_sub_annual["precip_mean"]
    ).mean("time", skipna=True)
    ds_sub_annual["aridity_index"] = (
        ds_sub_annual["pet_mean"] / ds_sub_annual["precip_mean"]
    ).mean("time", skipna=True)
    ds_sub_annual["evap_index"] = 1 - ds_sub_annual["discharge_coeff"]

    # Make sure Ea = P - Q < Ep. If not, we will not use that subcatchment for
    # the calculations.
    ds_sub_annual["discharge_coeff"] = ds_sub_annual["discharge_coeff"].where(
        ds_sub_annual["evap_index"] < ds_sub_annual["aridity_index"]
    )
    ds_sub_annual["aridity_index"] = ds_sub_annual["aridity_index"].where(
        ds_sub_annual["evap_index"] < ds_sub_annual["aridity_index"]
    )
    ds_sub_annual["evap_index"] = ds_sub_annual["evap_index"].where(
        ds_sub_annual["evap_index"] < ds_sub_annual["aridity_index"]
    )

    # Final check, if a coefficient < 0.0 or >>1.0, set to nan
    ds_sub_annual["discharge_coeff"] = (
        ds_sub_annual["discharge_coeff"]
        .where(ds_sub_annual["discharge_coeff"] < 10.0)
        .where(ds_sub_annual["discharge_coeff"] > 0.0)
    )
    ds_sub_annual["aridity_index"] = (
        ds_sub_annual["aridity_index"]
        .where(ds_sub_annual["aridity_index"] < 10.0)
        .where(ds_sub_annual["aridity_index"] > 0.0)
    )
    ds_sub_annual["evap_index"] = (
        ds_sub_annual["evap_index"]
        .where(ds_sub_annual["evap_index"] < 10.0)
        .where(ds_sub_annual["evap_index"] > 0.0)
    )

    return ds_sub_annual


def determine_omega(ds_sub_annual):
    """
    Determine the omega parameter.

    This function uses the Zhang function (as defined in Teuling et al., 2019)
    to determine the omega parameter for ds_sub_annual.

    Teuling, A. J., de Badts, E. A. G., Jansen, F. A., Fuchs, R., Buitink, J.,
    Hoek van Dijke, A. J., and Sterling, S. M.: Climate change, reforestation/
    afforestation, and urbanization impacts on evapotranspiration and streamflow
    in Europe, Hydrol. Earth Syst. Sci., 23, 3631–3652,
    https://doi.org/10.5194/hess-23-3631-2019, 2019.

    Parameters
    ----------
    ds_sub_annual : xr.Dataset
        Dataset containing at least the discharge coefficient, aridity
        index and evaporative index for the different forcing types.

    Returns
    -------
    ds_sub_annual : xr.Dataset
        Same as above, but with the omega parameter added. The omega parameter
        is the same for all forcing types and based on the observations.

    """
    # Constract the output variable in the dataset
    ds_sub_annual = ds_sub_annual.assign(
        omega=lambda ds_sub_annual: ds_sub_annual["discharge_coeff"] * np.nan
    )
    # Load the aridity index and evaporative index as np arrays (this saves
    # calculation time in the loop below)
    # calculate omega for "obs" and "cc_hist":
    if "cc_hist" in ds_sub_annual.forcing_type:
        forcing_types = ["obs", "cc_hist"]
    else:
        forcing_types = ["obs"]

    for forcing_type in forcing_types:
        aridity_index = ds_sub_annual.sel(forcing_type=forcing_type)[
            "aridity_index"
        ].values
        evap_index = ds_sub_annual.sel(forcing_type=forcing_type)["evap_index"].values

        # Set the temporary omega variable
        omega = np.zeros((len(ds_sub_annual.index),))

        for subcatch_index_nr in range(len(ds_sub_annual.index)):
            if evap_index[subcatch_index_nr] > 0:  # make sure evap index is not nan.
                try:
                    omega_temp = optimize.brentq(
                        Zhang,
                        0.000000000001,
                        100,
                        args=(
                            aridity_index[subcatch_index_nr],
                            evap_index[subcatch_index_nr],
                        ),
                    )
                    omega[subcatch_index_nr] = omega_temp
                # possible error that occurs: "ValueError: f(a) and f(b) must
                # have different signs") -- increase a and b range solves the issue.
                except ValueError:
                    logger.warning("No value for omega could be derived.")
                    omega[subcatch_index_nr] = np.nan
            else:
                omega[subcatch_index_nr] = np.nan

        # Add omega to the xr dataset
        ds_sub_annual["omega"].loc[dict(forcing_type=forcing_type)] = omega

    return ds_sub_annual


def determine_Peffective_Interception_explicit(ds_sub, Imax, intercep_vars_sub=None):
    """
    Determine the effective precipitation, interception evaporation and canopy storage.

    Based on (daily) values of precipitation and potential evaporation.

    Parameters
    ----------
    ds_sub : xr.Dataset
        Dataset containing precipitation and potential evaporation
        (precip_mean and pet_mean).
    Imax : float
        The maximum interception storage capacity [mm].
    intercep_vars_sub : xr.Dataarray
        Dataarray from staticmaps containing the Imax values for a number
        of time steps, e.g. every month. If present, this is used to determine
        Imax per time step. The default is None.

    Returns
    -------
    ds_sub : xr.Dataset
        same as above, but with effective precipitation, interception evaporation
        and canopy storage added.
    """
    # Make the output dataset ready for the output
    ds_sub["evap_interception"] = xr.full_like(
        ds_sub["precip_mean"], fill_value=ds_sub["precip_mean"].vector.nodata
    )
    ds_sub["evap_interception"].load()
    ds_sub["precip_effective"] = xr.full_like(
        ds_sub["precip_mean"],
        fill_value=ds_sub["precip_mean"].vector.nodata,
    )
    ds_sub["precip_effective"].load()
    ds_sub["canopy_storage"] = xr.full_like(
        ds_sub["precip_mean"],
        fill_value=ds_sub["precip_mean"].vector.nodata,
    )
    ds_sub["canopy_storage"].load()

    # Calculate it per forcing type
    for forcing_type in ds_sub["forcing_type"].values:
        nr_time_steps = len(
            ds_sub[["precip_mean", "pet_mean"]]
            .sel(forcing_type=forcing_type)
            .dropna("time")
            .time
        )

        # Add new empty variables that will be filled in the loop
        evap_interception = np.zeros(
            (len(ds_sub.sel(forcing_type=forcing_type)["index"]), nr_time_steps)
        )
        precip_effective = np.zeros(
            (len(ds_sub.sel(forcing_type=forcing_type)["index"]), nr_time_steps)
        )
        canopy_storage = np.zeros(
            (len(ds_sub.sel(forcing_type=forcing_type)["index"]), nr_time_steps)
        )

        # Load the potential evaporation and precipitation into memory to speed up
        # the subsequent for loop.
        # make sure order of coord is the same.
        Epdt = (
            ds_sub.sel(forcing_type=forcing_type)["pet_mean"]
            .dropna("time")
            .transpose("index", "time")
            .values
        )
        Pdt = (
            ds_sub.sel(forcing_type=forcing_type)["precip_mean"]
            .dropna("time")
            .transpose("index", "time")
            .values
        )
        # Loop through the time steps and determine the variables per time step.
        for i in range(0, nr_time_steps):
            if intercep_vars_sub is not None:
                # TODO: for now assumed that LAI contains monthly data,
                # change this for future
                month = pd.to_datetime(ds_sub.time[i].values).month
                Imax = intercep_vars_sub["Imax"].sel(time=month)
            # Determine the variables with a simple interception reservoir approach
            canopy_storage[:, i] = canopy_storage[:, i] + Pdt[:, i]
            precip_effective[:, i] = np.maximum(0, canopy_storage[:, i] - Imax)
            canopy_storage[:, i] = canopy_storage[:, i] - precip_effective[:, i]
            evap_interception[:, i] = np.minimum(Epdt[:, i], canopy_storage[:, i])
            canopy_storage[:, i] = canopy_storage[:, i] - evap_interception[:, i]

            # Update Si for the next time step
            if i < nr_time_steps - 1:
                canopy_storage[:, i + 1] = canopy_storage[:, i]

        # insert in ds for the time that is available in each forcing type for
        # precip and pet
        time_forcing_type = (
            ds_sub[["precip_mean", "pet_mean"]]
            .sel(forcing_type=forcing_type)
            .dropna("time")
            .time
        )
        ds_sub["evap_interception"].loc[
            dict(forcing_type=forcing_type, time=time_forcing_type)
        ] = evap_interception
        ds_sub["precip_effective"].loc[
            dict(forcing_type=forcing_type, time=time_forcing_type)
        ] = precip_effective
        ds_sub["canopy_storage"].loc[
            dict(forcing_type=forcing_type, time=time_forcing_type)
        ] = canopy_storage

    return ds_sub


def determine_storage_deficit(ds_sub, correct_cc_deficit):
    """
    Determine the storage deficit for every time step.

    Also for the subcatchment location and dataset in ds_sub.

    Parameters
    ----------
    ds_sub : xr.Dataset
        Dataset containing the daily or higher-resolution data.

    Returns
    -------
    ds_sub : xr.Dataset
        Same as above, but containing the storage deficits per time step for all
        forcing types.
    """
    # make sure the order of the coordinates is always the same.
    # Calculate it per forcing type

    ds_sub["storage_deficit"] = xr.full_like(
        ds_sub["precip_mean"],
        fill_value=ds_sub["precip_mean"].vector.nodata,
    )
    ds_sub["storage_deficit"].load()

    for forcing_type in ds_sub["forcing_type"].values:
        time_forcing_type = (
            ds_sub["precip_effective"]
            .sel(forcing_type=forcing_type)
            .dropna("time")
            .time.values
        )
        transpiration = (
            ds_sub["transpiration"]
            .sel(forcing_type=forcing_type, time=time_forcing_type)
            .transpose("index", "time")
            .values
        )
        precip_effective = (
            ds_sub["precip_effective"]
            .sel(forcing_type=forcing_type, time=time_forcing_type)
            .transpose("index", "time")
            .values
        )

        storage_deficit = np.zeros((len(ds_sub["index"]), len(time_forcing_type)))

        # Determine the storage deficit per time step
        for i in range(1, len(time_forcing_type)):
            storage_deficit[:, i] = np.minimum(
                0,
                storage_deficit[:, i - 1]
                + (precip_effective[:, i] - transpiration[:, i]),
            )

        # Create the storage deficit data array
        ds_sub["storage_deficit"].loc[
            dict(forcing_type=forcing_type, time=time_forcing_type)
        ] = storage_deficit

    # If there are climate projections present, adjust the storage deficit for
    # the future projections based on Table S1 in Bouaziz et al., 2002, HESS.
    # cc_hist remains as is (see Table S1)
    if (len(ds_sub.forcing_type) > 1) & (correct_cc_deficit == True):
        if (
            len(
                ds_sub["precip_mean"]
                .sel(forcing_type=["cc_fut", "cc_hist"])
                .dropna("time")
                .time
            )
            > 0
        ):
            ds_sub["storage_deficit"].loc[dict(forcing_type="cc_fut")] = ds_sub[
                "storage_deficit"
            ].sel(forcing_type="obs") + np.minimum(
                0.0,
                ds_sub["storage_deficit"].sel(forcing_type="cc_fut")
                - ds_sub["storage_deficit"].sel(forcing_type="cc_hist"),
            )
        else:
            logger.warning(
                "Time period of cc_hist and cc_fut does not overlap. \
Correct_cc_deficit not applied."
            )

    return ds_sub


def fut_discharge_coeff(ds_sub_annual, correct_cc_deficit):
    """
    Determine the future discharge coefficient.

    Based on a given omega value (generally same as in the
    current-climate observations), the aridity index of the future climate and the
    difference in Q and PET between the simulated historical and
    simulated future climate.

    Parameters
    ----------
    ds_sub_annual : xr.Dataset
        Dataset containing at least the future omega and aridity index.

    Returns
    -------
    ds_sub_annual : xr.Dataset
        Similar to previous, but containing the new future discharge coefficient
        and evaporative index.
    """
    # Determine the delP and delEP for cc_fut
    Ep = ds_sub_annual["pet_mean"].sel(forcing_type="obs").mean("time")
    P = ds_sub_annual["precip_mean"].sel(forcing_type="obs").mean("time")
    delP = (
        ds_sub_annual["precip_mean"]
        .sel(forcing_type=["cc_hist", "cc_fut"])
        .mean("time")
        .diff("forcing_type")
        .sel(forcing_type="cc_fut")
    )
    delEp = (
        ds_sub_annual["pet_mean"]
        .sel(forcing_type=["cc_hist", "cc_fut"])
        .mean("time")
        .diff("forcing_type")
        .sel(forcing_type="cc_fut")
    )

    # Determine the difference in discharge between the observations and the
    # future climate simulations
    Q_obs_mean = ds_sub_annual["specific_Q"].mean("time")
    # if correct_cc_deficit=False -- omega is from cc_hist, else omega is from obs
    if correct_cc_deficit == True:
        omega = ds_sub_annual["omega"].sel(forcing_type="obs")
    else:
        omega = ds_sub_annual["omega"].sel(forcing_type="cc_hist")
    aridity_index_fut = (Ep + delEp) / (P + delP)
    change_Q_total = Zhang_future(omega, aridity_index_fut) * (P + delP) - Q_obs_mean

    # Determine the discharge coefficient for the future climate
    ds_sub_annual["discharge_coeff"].loc[dict(forcing_type="cc_fut")] = (
        Q_obs_mean + change_Q_total
    ) / (P + delP)

    return ds_sub_annual


def gumbel_su_calc_xr(
    storage_deficit_annual, storage_deficit_count, return_period, threshold
):
    """
    Determine the Gumbel distribution of the annual maximum storage deficits.

    For a set of return periods.

    Parameters
    ----------
    storage_deficit_annual : xr.Dataset
        Dataset with the minimum deficit per year as a positive value.
    storage_deficit_count : xr.Dataset
        Dataset containing the number of days with data per year. This
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
    gumbel : xr.Dataset
        Dataset, similar to year_min_storage_deficit, but containing the
        root-zone storage capacity per return period.

    """
    # Only take the years into account that contain more than [threshold] days
    # of data.
    storage_deficit_annual = storage_deficit_annual.where(
        storage_deficit_count > threshold
    )

    # Calculate the mean and standard deviation for the Gumbel distribution
    annual_mean = storage_deficit_annual.mean("time", skipna=True)
    annual_std = storage_deficit_annual.std("time", skipna=True)

    # Calculate alpha and mu
    alpha = (np.sqrt(6.0) * annual_std) / np.pi
    mu = annual_mean - 0.5772 * alpha

    # Create the output dataset
    gumbel = storage_deficit_annual.to_dataset(name="storage_deficit")

    # Set the return periods
    RP = return_period
    gumbel["RP"] = RP
    gumbel["rootzone_storage"] = (
        ("index", "forcing_type", "RP"),
        np.zeros(
            (len(gumbel["index"]), len(gumbel["forcing_type"]), len(gumbel["RP"]))
        ),
    )

    # Determine the root zone storage for different return periods using alpha
    # and mu
    gumbel["yt"] = -np.log(-np.log(1 - (1 / gumbel["RP"])))
    gumbel["rootzone_storage"] = mu + alpha * gumbel["yt"]

    return gumbel


def Zhang(omega, aridity_index, evap_index):
    """
    Calculate omega according to Zhang.

    This is the Zhang equation with omega as in Teuling et al., 2019.
    This function is used to get omega for historical situations when
    aridity_index and evap_index are known (assuming evap_index = 1 - discharge_coeff).
    This equation is solved for Zhang eq = 0.

    Parameters
    ----------
    omega : float
        Parameter of the Budyko curve.
    aridity_index : float
        The aridity index.
    evap_index : float
        The evaporative index.

    Returns
    -------
    Value 0.

    References
    ----------
    Fu, B.: On the calculation of the evaporation from land surface,
    Scientia Atmospherica Sinica, 5, 23–31, 1981 (in Chinese).

    Teuling, A. J., de Badts, E. A. G., Jansen, F. A., Fuchs, R., Buitink, J.,
    Hoek van Dijke, A. J., and Sterling, S. M.: Climate change, reforestation/
    afforestation, and urbanization impacts on evapotranspiration and streamflow
    in Europe, Hydrol. Earth Syst. Sci., 23, 3631–3652,
    https://doi.org/10.5194/hess-23-3631-2019, 2019.

    Zhang, L., Hickel, K., Dawes, W. R., Chiew, F. H., Western, A. W., and
    Briggs, P. R.: A rational function approach for estimating mean annual
    evapotranspiration, Water Resour. Res., 40, 1–14,
    https://doi.org/10.1029/2003WR002710, 2004.
    """
    return 1 + aridity_index - (1 + aridity_index**omega) ** (1 / omega) - evap_index


def Zhang_future(omega, aridity_index):
    """
    Calculate discharge coefficients according to Zhang.

    Once omega has been derived for historical situations, it can be used to
    derive the new discharge coefficient when the future aridity index (Ep/P)
    is known. This will allow the discharge coeffcient to shift over the same
    line as the historical situation.

    Parameters
    ----------
    omega : xr.Datarray
        Datarray containing the future omega values per sub catchment.
    aridity_index : xr.Datarray
        Datarray containing the future aridity index values per sub
        catchment.

    Returns
    -------
    Datarray containing the future discharge coefficient.
    """
    return -(aridity_index - (1 + aridity_index**omega) ** (1 / omega))


def check_inputs(
    start_hydro_year, start_field_capacity, dsrun, ds_obs, ds_cc_hist, ds_cc_fut
):
    # Start with some initial checks
    list_of_months = [
        "Jan",
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
        "Dec",
    ]
    if start_hydro_year not in list_of_months:
        raise ValueError(
            f"start_hydro_year not in {list_of_months}: provide a valid month"
        )
    if start_field_capacity not in list_of_months:
        raise ValueError(
            f"start_field_capacity not in {list_of_months}: provide a valid month"
        )

    if "discharge" not in list(dsrun.keys()):
        raise ValueError("Variable discharge not in run_fn")

    if "precip" not in list(ds_obs.keys()):
        raise ValueError("Variable precip not in forcing_obs_fn")

    if "pet" not in list(ds_obs.keys()):
        raise ValueError("Variable pet not in forcing_obs_fn")
    if ds_cc_hist is not None:
        if "precip" not in list(ds_cc_hist.keys()):
            raise ValueError("Variable precip not in forcing_cc_hist_fn")

        if "pet" not in list(ds_cc_hist.keys()):
            raise ValueError("Variable pet not in forcing_cc_hist_fn")
    if ds_cc_fut is not None:
        if "precip" not in list(ds_cc_fut.keys()):
            raise ValueError("Variable precip not in forcing_cc_fut_fn")

        if "pet" not in list(ds_cc_fut.keys()):
            raise ValueError("Variable pet not in forcing_cc_fut_fn")

    return None


def rootzoneclim(
    dsrun: xr.Dataset,
    ds_obs: xr.Dataset,
    ds_like: xr.Dataset,
    flwdir: xr.DataArray,
    ds_cc_hist: Optional[xr.Dataset] = None,
    ds_cc_fut: Optional[xr.Dataset] = None,
    return_period: list = [2, 3, 5, 10, 15, 20, 25, 50, 60, 100],
    Imax: float = 2.0,
    start_hydro_year: str = "Sep",
    start_field_capacity: str = "Apr",
    LAI: bool = False,
    rootzone_storage: bool = False,
    correct_cc_deficit: bool = False,
    chunksize: int = 100,
    missing_days_threshold: int = 330,
    logger=logger,
):
    """
    Estimates the root zone storage parameter.

    for current observed and (optionally) for future climate-based streamflow data.

    The root zone storage capacity parameter is calculated per subcatchment and is
    converted to a gridded map at model resolution. Optionally, this function
    can return the wflow_sbm parameter RootingDepth by dividing the root zone
    storage parameter by (theta_s - theta_r).

    The method is based on the estimation of maximum annual storage deficits based on
    precipitation and estimated actual evaporation time series, which in turn are
    estimated from observed streamflow data and long-term precipitation and
    potential evap. data, as explained in Bouaziz et al. (2022).
    The main assumption is that vegetation adapts its rootzone storage capacity to
    overcome dry spells with a certain return period
    (typically 20 years for forest ecosystems).
    In response to a changing climtate, it is likely that vegetation also adapts its
    rootzone storage capacity, thereby changing model parameters for future conditions.
    This method also allows to estimate the change in rootzone storage capacity in
    response to a changing climate.

    Parameters
    ----------
    dsrun : xr.Dataset
        Geodataset with streamflow locations and timeseries, named "discharge" (m3/s).
        The geodataset expects the coordinate names "index" (for each station id).
    ds_obs : xr.Dataset
        Dataset with the observed forcing data (precip and pet) [mm/timestep].
    ds_like : xr.Dataset
        Dataset with staticmaps at model resolution.
    flwdir : FlwDirRaster
        flwdir object
    ds_cc_hist : xr.Dataset
        Dataset with the simulated historical forcing data (precip and pet) \
[mm/timestep],
        based on a climate model.
        The default is None.
    ds_cc_fut : xr.Dataset
        Dataset with the simulated future climate forcing data (precip and pet) \
[mm/timestep],
        based on a climate model.
        The default is None.
    return_period : list
        List with one or more values indiciating the return period(s) (in
        years) for wich the rootzone storage depth should be calculated.
        The default is [2,3,5,10,15,20,25,50,60,100]
    Imax : float
        The maximum interception storage capacity [mm].
        The default is 2 mm.
    start_hydro_year : str
        The start month (abreviated to the first three letters of the month,
        starting with a capital letter) of the hydrological year.
        The default is "Sep".
    start_field_capacity : str
        The end of the wet season / commencement of dry season. This is the
        moment when the soil is at field capacity, i.e. there is no storage
        deficit yet.
        The default is "Apr".
    rootzone_storage : bool
        Boolean to indicate whether the rootzone storage maps
        should be stored in the staticmaps or not. The default is False.
    LAI : bool
        Determine whether the LAI will be used to determine Imax.
        Requires to have run setup_laimaps.
        The default is False.
    chunksize : int
        Chunksize on time dimension for processing data (not for saving to
        disk!). A default value of 100 is used on the time dimension.
    correct_cc_deficit : bool
        Determines whether a bias-correction of the future deficit should be
        applied. If the climate change scenario and hist period are bias-corrected,
        this should probably be set to False.
    missing_days_threshold: int, optional
            Minimum number of days within a year for that year to be counted in the
            long-term Budyko analysis.

    Returns
    -------
    ds_out : xr.Dataset
        Dataset containing root zone storage capacity (optional) and RootingDepth for
        several forcing and return periods.
    gdf_basins_all : GeoDataFrame
        Geodataframe containing the root zone storage capacity values for
        each basin before filling NaN.


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
    check_inputs(
        start_hydro_year, start_field_capacity, dsrun, ds_obs, ds_cc_hist, ds_cc_fut
    )

    # If LAI = True, create a new xr dataset containing the interception pars
    if LAI == True:
        intercep_vars = ds_like.LAI.to_dataset(name="LAI")
        intercep_vars["Swood"] = ds_like["Swood"]
        intercep_vars["Sl"] = ds_like["Sl"]

    # Set the output dataset at model resolution
    ds_out = xr.Dataset(coords=ds_like.raster.coords)
    x_dim = ds_out.raster.x_dim
    y_dim = ds_out.raster.y_dim

    # Make a basin map containing all subcatchments from geodataset of observed
    # streamflow
    x_dim_dsrun = dsrun.vector.x_name
    y_dim_dsrun = dsrun.vector.y_name
    x, y, ids = dsrun[x_dim_dsrun].values, dsrun[y_dim_dsrun].values, dsrun.index.values
    gdf_basins = pd.DataFrame()

    # Loop over basins and get per gauge location a polygon of the upstream
    # area.
    for i, id in enumerate(dsrun.index.values):
        ds_basin_single = flw.basin_map(
            ds_like,
            flwdir,
            ids=ids[i],
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
    # make sure to also snap to river when retrieving areas
    idxs_gauges = flwdir.snap(xy=(x, y), mask=ds_like["wflow_river"].values)[0]
    areas_uparea = ds_like["wflow_uparea"].values.flat[idxs_gauges]
    df_areas = pd.DataFrame(index=ids, data=areas_uparea * 1e6, columns=["area"])
    gdf_basins = pd.concat([gdf_basins, df_areas], axis=1)
    gdf_basins = gdf_basins.sort_values(by="area", ascending=False)
    # drop basins where area is NaN.
    gdf_basins = gdf_basins[(gdf_basins["area"] >= 0)]

    # calculate mean areal precip and pot evap for the full upstream area of each gauge.
    # loop over ds_obs, ds_cc_hist and ds_cc_fut as they might have
    # different coordinate systems and then merge.
    ds_sub_obs = ds_obs.raster.zonal_stats(gdf_basins, stats=["mean"])
    logger.info("Computing zonal statistics for obs, this can take a while")
    ds_sub_obs = ds_sub_obs.compute()

    if ds_cc_hist is not None:
        ds_sub_cc_hist = ds_cc_hist.raster.zonal_stats(gdf_basins, stats=["mean"])
        logger.info("Computing zonal statistics for cc_hist, this can take a while")
        ds_sub_cc_hist = ds_sub_cc_hist.compute()

    if ds_cc_fut is not None:
        ds_sub_cc_fut = ds_cc_fut.raster.zonal_stats(gdf_basins, stats=["mean"])
        logger.info("Computing zonal statistics for cc_fut, this can take a while")
        ds_sub_cc_fut = ds_sub_cc_fut.compute()

    # Concatenate all forcing types (obs, cc_hist, cc_fut) into a
    # xr dataset after zonal stats
    if ds_cc_hist is not None and ds_cc_fut is not None:
        ds_sub = xr.concat(
            [ds_sub_obs, ds_sub_cc_hist, ds_sub_cc_fut],
            pd.Index(["obs", "cc_hist", "cc_fut"], name="forcing_type"),
        )
    else:
        ds_sub = xr.concat([ds_sub_obs], pd.Index(["obs"], name="forcing_type"))

    # Also get the zonal statistics of the intercep_vars
    if LAI == True:
        intercep_vars_sub = intercep_vars.raster.zonal_stats(gdf_basins, stats=["mean"])
        intercep_vars_sub = intercep_vars_sub.compute()
        # Determine the Imax for every time step in the LAI data
        intercep_vars_sub["Imax"] = (
            intercep_vars_sub["Swood_mean"]
            + intercep_vars_sub["LAI_mean"] * intercep_vars_sub["Sl_mean"]
        )
    else:
        intercep_vars_sub = None

    # Get the time step of the datasets and make sure they all have a daily
    # time step. If not, resample.
    time_step = 86400  # in seconds
    if (ds_sub.time[1].values - ds_sub.time[0].values).astype("timedelta64[s]").astype(
        np.int32
    ) != time_step:
        ds_sub = ds_sub.resample(time="1D").sum("time", skipna=True)
    if (dsrun.time[1].values - dsrun.time[0].values).astype("timedelta64[s]").astype(
        np.int32
    ) != time_step:
        dsrun = dsrun.resample(time="1D").mean("time", skipna=True)

    # Determine effective precipitation, interception evporation and canopy
    # storage
    logger.info(
        "Determine effective precipitation, interception evporation and canopy storage"
    )
    ds_sub = determine_Peffective_Interception_explicit(
        ds_sub, Imax=Imax, intercep_vars_sub=intercep_vars_sub
    )

    # Add specific discharge per location-index value to ds_sub
    # First, sort dsrun based on descending subcatchment area (which is already
    # done in ds_sub)
    dsrun = dsrun.sel(index=ds_sub.index.values)
    # Get the specific discharge (mm/timestep) per location in order to have
    # everything in mm/timestep
    dsrun = dsrun.assign(
        specific_Q=dsrun["discharge"].transpose("time", "index")
        / np.array(gdf_basins["area"])
        * time_step
        * 1000.0
    )
    # Add specific discharge to ds_sub
    ds_sub = xr.merge([ds_sub, dsrun["specific_Q"].to_dataset()], compat="override")

    # Rechunk data
    ds_sub = ds_sub.chunk(
        chunks={"index": len(ds_sub.index), "forcing_type": 1, "time": chunksize}
    )

    # Get year sums of ds_sub
    # a threshold is used to use only years with sufficient data
    ds_sub_annual = ds_sub.resample(time=f"AS-{start_hydro_year}").sum(
        "time", skipna=True, min_count=missing_days_threshold
    )

    # Determine discharge coefficient, the aridity index and the evaporative
    # index
    ds_sub_annual = determine_budyko_curve_terms(
        ds_sub_annual,
    )
    # set runoff coefficient of cc_hist equal to runoff coeff of obs
    if correct_cc_deficit == True:
        ds_sub_annual["discharge_coeff"].loc[
            dict(forcing_type="cc_hist")
        ] = ds_sub_annual["discharge_coeff"].sel(forcing_type="obs")

    # Determine omega
    logger.info("Calculating the omega values, this can take a while")
    ds_sub_annual = determine_omega(ds_sub_annual)

    # Determine future discharge ratio for cc_fut if ds_cc_fut exists
    if ds_cc_fut is not None:
        ds_sub_annual = fut_discharge_coeff(ds_sub_annual, correct_cc_deficit)

    # Determine long-term interception, potential evaporation and tranpsiration;
    # use runoff coefficient instead of Qobs to calculate actual evaporation for
    # each climate projection
    transpiration_long_term = (
        ds_sub_annual["precip_effective"].mean("time")
        - ds_sub_annual["precip_mean"].mean("time") * ds_sub_annual["discharge_coeff"]
    )  # ds_annual['Qobs_mm'].mean('time')
    interception_long_term = ds_sub_annual["evap_interception"].mean("time")
    pet_long_term = ds_sub_annual["pet_mean"].mean("time")

    # Determine the transpiration on the finer time step (e.g. daily)
    ds_sub["transpiration"] = (
        (ds_sub["pet_mean"] - ds_sub["evap_interception"])
        * transpiration_long_term
        / (pet_long_term - interception_long_term)
    )
    transpiration_long_term = None
    interception_long_term = None
    pet_long_term = None

    # Determine storage deficit
    logger.info("Determining the storage deficit")
    ds_sub = determine_storage_deficit(ds_sub, correct_cc_deficit)

    # From the storage deficit, determine the rootzone storage capacity using
    # a Gumbel distribution.
    logger.info("Calculating the Gumbel distribution and rootzone storage capacity")
    # Determine the yearly minima in storage deficit (and make sure the values
    # are positive)
    storage_deficit_annual = -(
        ds_sub["storage_deficit"]
        .resample(time=f"AS-{start_field_capacity}")
        .min("time", skipna=True)
    )

    # A counter will be used to only use years with sufficient days containing
    # data for the Gumbel distribution
    storage_deficit_count = (
        ds_sub["storage_deficit"]
        .resample(time=f"AS-{start_field_capacity}")
        .count("time")
    )

    # Subsequently, determine the Gumbel distribution
    gumbel = gumbel_su_calc_xr(
        storage_deficit_annual,
        storage_deficit_count.sel(time=storage_deficit_count.time[1:]),
        return_period=return_period,
        threshold=missing_days_threshold,
    )

    # Create a new geopandas dataframe, which will be used to rasterize the
    # results
    ds_basins_all = flw.basin_map(
        ds_like,
        flwdir,
        ids=ids,
        xy=(x, y),
        stream=ds_like["wflow_river"],
    )[0]
    ds_basins_all.raster.set_crs(ds_like.raster.crs)
    gdf_basins_all = ds_basins_all.raster.vectorize()
    gdf_basins_all.index = gdf_basins_all.value.astype("int")

    # Add the area and sort by area (from large to small)
    # use previous df
    gdf_basins_all = pd.concat([gdf_basins_all, df_areas], axis=1)
    gdf_basins_all = gdf_basins_all.sort_values(by="area", ascending=False)
    # again drop basins where area is NaN
    gdf_basins_all = gdf_basins_all[(gdf_basins_all["area"] >= 0)]

    # Add the rootzone storage to gdf_basins_all, per forcing type and return
    # period
    for return_period in gumbel.RP.values:
        for forcing_type in gumbel.forcing_type.values:
            gdf_basins_all[
                f"rootzone_storage_{forcing_type}_{str(return_period)}"
            ] = gumbel["rootzone_storage"].sel(
                RP=return_period, forcing_type=forcing_type
            )
            # Make sure to give the NaNs a value, otherwise they will become 0.0
            gdf_basins_all[
                f"rootzone_storage_{forcing_type}_{str(return_period)}"
            ] = gdf_basins_all[
                f"rootzone_storage_{forcing_type}_{str(return_period)}"
            ].fillna(
                -999
            )

    # Rasterize this (from large subcatchments to small ones)
    for return_period in gumbel.RP.values:
        for forcing_type in gumbel.forcing_type.values:
            da_area = ds_like.raster.rasterize(
                gdf=gdf_basins_all,
                col_name=f"rootzone_storage_{forcing_type}_{str(return_period)}",
                nodata=-999,
                all_touched=True,
            ).to_dataset(name="rasterized_temp")
            # Fill up the not a numbers with the data from a downstream point
            out_raster = pyflwdir.FlwdirRaster.fillnodata(
                flwdir,
                data=da_area["rasterized_temp"],
                nodata=-999,
                direction="up",
                how="max",
            )
            # Make sure to fill up full domain with value of most downstream point
            # that contains values
            fill_value = None
            for value in gdf_basins_all[
                f"rootzone_storage_{forcing_type}_{str(return_period)}"
            ]:
                if value > 0.0:
                    if fill_value is None:
                        fill_value = value
            out_raster = np.where(out_raster == -999.0, fill_value, out_raster)
            # Store the rootzone_storage in ds_out is rootzone_storage flag is
            # set to True.
            if rootzone_storage == True:
                ds_out[f"rootzone_storage_{forcing_type}_{str(return_period)}"] = (
                    (y_dim, x_dim),
                    out_raster,
                )
            # Store the RootingDepth in ds_out
            ds_out[f"RootingDepth_{forcing_type}_{str(return_period)}"] = (
                (y_dim, x_dim),
                out_raster / (ds_like["thetaS"].values - ds_like["thetaR"].values),
            )

    return ds_out, gdf_basins_all
