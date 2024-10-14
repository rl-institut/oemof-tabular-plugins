"""
This script holds functions used for different facades based on literature.
In addition, there are functions defined by logic based on these literature functions.
Please note the abbreviations in the function descriptions.

Name/Content                                Abbrev.     Source
---------------------------------------------------------------------------------------------------
A SIMPLE crop model                         SIMPLE      https://doi.org/10.1016/j.eja.2019.01.009
Crop evapotranspiration                     FAO56       ISBN: 978-92-5-104219-9
Agricultural Reference Index for Drought    ARID        https://doi.org/10.2134/agronj2011.0286
PV power output incl. temperature           PV          https://doi.org/10.1016/j.solener.2015.03.004
"""

import numpy as np
import pandas as pd

"""
Conversion factors as global variables
--------------------------------------
These conversion factors should be defined and applied consistently.

Assuming all values to be rational numbers,
multiply with the conversion factor if the unit is linked to the numerator or
divide by the conversion factor if the unit is linked to the denominator.
(numerator/denominator)

Example 1: Convert W over an hourly time step to MJ per day
    1 W over an hourly time step = 1 Wh/h
        1 Wh = 1 J/s * 1 h = 1 J/s * 3600 s = 3600 J    -> multiply by C_WH_TO_J because it's a numerator
        1 J = 1/1000000 MJ                              -> multiply by C_J_TO_MJ because it's a numerator
        1/h = 1/d / (1/24) = 24/d                       -> divide by C_H_TO_D because it's a denominator

Example 2: Convert h/d to d/d                           -> multiply by C_H_TO_D because now it's a numerator
"""
C_D_TO_H = 24
C_H_TO_D = 1 / C_D_TO_H
C_WH_TO_J = 3600
C_J_TO_WH = 1 / C_WH_TO_J
C_MJ_TO_J = 1000000
C_J_TO_MJ = 1 / C_MJ_TO_J
C_KG_TO_G = 1000
C_G_TO_KG = 1 / C_KG_TO_G
C_KW_TO_W = 1000
C_W_TO_KW = 1 / C_KW_TO_W


def delta_tt(t, t_base):
    """
    SIMPLE:
    Temperature above base temperature to be added to cumulative temperature every hour.
    Cumulative temperature is the measure for plant development, also known as thermal time.
    The original model is based on daily values, this work assumes
    time series with hourly values and converts delta_tt accordingly.
    Eq. (1)

    Parameters
    ----------
    t: numeric
        air temperature [°C]
    t_base: numeric
        base temp. for plant growth [°C]

    Returns
    ---------
    delta_tt: numeric
        additional thermal time to be added at one time step (hourly) [K]
    """
    if t > t_base:
        delta_tt = (t - t_base) / C_D_TO_H
    else:
        delta_tt = 0
    return delta_tt


def f_temp(t, t_opt, t_base):
    """
    SIMPLE:
    Temperature effect on plant growth for every time step
    Eq. (7), Fig. 1(a)

    Parameters
    ----------
    t: numeric
        air temperature [°C]
    t_opt: numeric
        optimal temp. for plant growth [°C]
    t_base: numeric
        base temp. for plant growth [°C]

    Returns
    ----------
    f_temp: numeric, in [0,1]
        temperature factor for biomass generation [-]
    """
    if t < t_base:
        f_temp = 0
    elif t_base <= t < t_opt:
        f_temp = (t - t_base) / (t_opt - t_base)
    else:
        f_temp = 1
    return f_temp


def f_heat(t, t_max, t_ext, **kwargs):
    """
    SIMPLE:
    Heat stress effect on plant growth
    Eq. (8), Fig. 1(b)

    Parameters
    ----------
    t: numeric
        air temperature [°C]
    t_max: numeric
        temperature where plant growth starts to be impaired [°C]
    t_ext: numeric
        temperature where plant growth stops due to extreme heat [°C]

    Returns
    ----------
    f_heat: numeric, in [0,1]
        heat stress factor for biomass generation [-]

    Notes
    ----------
    SIMPLE is a daily model and uses daily maximum temperature as t,
    this model is hourly and uses hourly air temperature as t.
    """
    if t <= t_max:
        f_heat = 1
    elif t_max < t <= t_ext:
        f_heat = 1 - (t - t_max) / (t_ext - t_max)
    else:
        f_heat = 0
    return f_heat


def f_water(et_p, et_a, s_water, **kwargs):
    """
    SIMPLE:
    Water stress effect on plant growth
    Eq. (11) and (12); Fig. 1(d)

    Parameters
    ----------
    et_p: numeric
        potential evapotranspiration of water [mm]
    et_a: numeric
        actual evapotranspiration of water [mm]
    s_water: numeric
        plant sensitivity to water stress [-]

    Returns
    ----------
    f_water: numeric, in [0,1]
        water stress factor for biomass generation [-]

    Notes
    ----------
    et_p is set as equal to be the reference evapotranspiration et_0,
    et_a is calculated as min(et_p, 0.096*PAW) in the function
    'soil_water_balance' according to ARID.
    """
    arid = 1 - et_a / et_p
    f_water = 1 - s_water * arid
    return f_water


def delta_i50b(cum_temp, f_heat, f_water, i50maxh, i50maxw, **kwargs):
    """
    SIMPLE:
    Heat and water stress effect on solar interception due to faster canopy senescence
    Eq. (9) and (13) combined

    Parameters
    ----------
    cum_temp: numeric
        cumulative temperature already experienced by the plant [K]
    f_heat: numeric, in [0,1]
        heat stress factor for biomass generation [-]
    f_water: numeric, in [0,1]
        water stress factor for biomass generation [-]
    i50maxh: numeric
        coefficient for heat tress impact on i50b [-]
    i50maxw: numeric
        coefficient for water stress impact on i50b [-]

    Returns
    ----------
    delta_i50b: numeric
        hourly increase in i50b [K]

    Notes
    ----------
    The higher i50b, the sooner the plant degenerates (faster canopy senescence).
    Delta_i50b will be added to i50b in the function 'f_solar'
    """
    if cum_temp < 1:
        delta_i50b = 0
    else:
        delta_i50b = (i50maxh * (1 - f_heat) + i50maxw * (1 - f_water)) / C_D_TO_H
    return delta_i50b


def f_solar(cum_temp, delta_i50b, t_sum, i50a, i50b, f_solar_max, **kwargs):
    """
    SIMPLE:
    Interception of incoming radiation by plant canopy according to
    plant development stage (frac of sunlight that is used by the plant)
    Eq. (6); Fig. 1(d)
    Eq. (9) and (13)

    Parameters
    ----------
    cum_temp: numeric
        cumulative temperature already experienced by the plant [K]
    delta_i50b: numeric
        hourly increase in i50b
    t_sum: numeric
        cumulative temperature until maturity [K]
    i50a: numeric
        cumulative temp. from sowing on to reach 50% of
        solar radiation interception during growth [K]
    i50b: numeric
        cumulative temp. until maturity to fall back to 50% of
        solar rad. intercept. during decline [K]
    f_solar_max:
        maximum solar radiation interception [-]

    Returns
    ----------
    f_solar: numeric, in [0,f_solar_max]
        solar radiation interception factor for biomass generation [-]

    Notes
    ----------
    SIMPLE presents two equations for f_solar = f(cum_temp), one for leaf growth and
    one for leaf senescence, but fails to specify the transition point (a cum_temp value).
    To get a smoove transition, the transition point was defined as
    'cum_temp_to_reach_f_solar_max' and calculated by setting the first equation for f_solar
    equal to 0.999f_solar_max and solving for cum_temp.

    The delta_i50b values (function 'delta_i50b') are cumulated and the cumulative value
    is added to i50b for every time step so that i50b is increasing over time as intended.
    """
    cum_temp_to_reach_f_solar_max = i50a - 100 * np.log(1 / 999)
    if cum_temp < 1:
        f_solar = 0
    # First equation: Leaf growth period
    elif cum_temp < cum_temp_to_reach_f_solar_max:
        i50b += delta_i50b
        f_solar = f_solar_max / (1 + np.exp(-0.01 * (cum_temp - i50a)))
    # Second equation: Leaf senescence period
    elif cum_temp < t_sum:
        i50b += delta_i50b
        f_solar = f_solar_max / (1 + np.exp(0.01 * (cum_temp - (t_sum - i50b))))
    else:
        f_solar = 0
    return f_solar


def soil_heat_flux(ghi, r_n):
    """
    FAO56:
    Soil heat flux for hourly calculations
    Eq. (45) and (46)

    Parameters
    ----------
    ghi: numeric
        gloabl horizontal irradiation [W/m²]
    r_n: numeric
        net radiation [W/m²]

    Returns
    ----------
    g: numeric
        soil heat flux at that time step (hourly) [W/m²]

    Notes
    ----------
    Night time is simply specified at the time when ghi is below 1.
    R_n should be positive during the day and negative during the night.
    However, errors will be insignificant.
    """
    ghi_at_night = 1
    if ghi > ghi_at_night:
        g = 0.1 * r_n
    else:
        g = 0.5 * r_n
    return g


def vap_pressure(t):
    """
    FAO56:
    Water vapour saturation pressure at specific temperature
    Eq. (11)

    Parameters
    ----------
    t: numeric
        tempearture [°C]

    Returns
    ----------
    e: numeric
        saturation vapour pressure [kPa]
    """
    e = 0.6108 * np.exp(17.27 * t / (t + 237.3))
    return e


def potential_evapotranspiration(z, t_air, t_dp, w10, r_n, g, **kwargs):
    """
    FAO56:
    Reference evapotranspiration according to FAO Penman-Monteith method
    Eq. (6), (7), (8), (13)

    Parameters
    ----------
    z: numeric
        elevation [m]
    t_air: numeric
        air temperature [°C]
    t_dp: numeric
        dew point temperature [°C]
    w10: numeric
        wind speed in 10m height
    r_n: numeric
        net radiation [W/m²]
    g: numeric
        soil heat flux [W/m²]

    Returns
    ----------
    et_p: numeric
        potential evapotranspiration
        (simplified as reference evapotranspiration,
        not crop specific)

    Notes
    ----------
    Hourly time series are used as if it was for the whole day,
    et_p is then returned converted back to hourly.
    """
    cp_air = 1.013e-3  # specific heat at constant pressure [MJ/(kg °C)]
    epsilon = 0.622  # ratio molecular weight of water vapour/dry air [-]
    h_vap = 2.45  # latent heat of vaporization [MJ/kg]
    r_n *= C_WH_TO_J * C_J_TO_MJ / C_H_TO_D  # W/m² over an hour to MJ/(m²*day)
    g *= C_WH_TO_J * C_J_TO_MJ / C_H_TO_D  # W/m² over an hour to MJ/(m²*day)
    # wind speed at 2m above ground [m/s] (Eq. 47)
    w2 = w10 * 4.87 / np.log(672.58)
    # atmospheric pressure at elevation z [kPa] (Eq. 7)
    p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26
    # psychrometric constant (Eq. 8)
    gamma = cp_air * p / (h_vap * epsilon)
    # slope of saturation vapour pressure curve (Eq. (13)
    delta = (
        4098 * (0.6108 * np.exp(17.27 * t_air / (t_air + 237.3))) / (t_air + 237.3) ** 2
    )
    # saturation vapour pressure [kPa]
    e_s = vap_pressure(t_air)
    # actual vapour pressure (sat vap press at dewpoint temp) [kPa]
    e_a = vap_pressure(t_dp)
    # reference evapotranspiration [mm/m²*day] (Eq. 6)
    et_0 = (
        0.408 * delta * (r_n - g) + gamma * 900 / (t_air + 273) * w2 * (e_s - e_a)
    ) / (delta + gamma * 1.34 * w2)
    et_p = et_0 / C_D_TO_H  # [mm/m²*h]
    return et_p


def runoff(p, rcn, **kwargs):
    """
    ARID:
    Surface runoff of precipitation
    Eq. (10)

    Parameters
    ----------
    p: numeric
        precipitation [mm]
    rcn: numeric
        runoff curve number [-]

    Returns
    ----------
    r: numeric
        surface runoff water [mm]
    """
    s = 25400 / rcn - 254  # potential maximum retention
    i_a = 0.2 * s  # initial abstraction
    if p > i_a:
        r = (p - i_a) ** 2 / (p + i_a - s) / C_D_TO_H
    else:
        r = 0.0
    return r


def deep_drainage(ddc, rzd, awc, swc_bd, **kwargs):
    """
    ARID:
    Deep drainage of soil water
    Eq. (9)

    Parameters
    ----------
    ddc: numeric
        deep drainage coefficient [-]
    rzd: numeric
        rootzone depth [m]
    awc: numeric
        average water content [-]
    swc_bd: numeric
        soil water content before drainage [-]

    Returns
    ----------
    d: numeric
        deep drainage water [mm]
    """
    if swc_bd > awc:
        d = ddc * rzd * (swc_bd - awc) / C_D_TO_H
    else:
        d = 0
    return d


def soil_water_balance(df, has_irrigation, ddc, rzd, awc, **kwargs):
    """
    ARID
    Soil water balance
    Based on (Eq. 8)

    Parameters
    ----------
    df: DataFrame object
        required time series columns are ["tp_ground", "runoff"]
    has_irrigation: Bool
        decides if irrigation will be applied,
        if True irrigation will be equal to actual evapotranspiration
    ddc: numeric
        deep drainage coefficient
    rzd: numeric
        rootzone depth [m]
    awc: numeric
        average water content

    Returns
    -------
    df: DataFrame object
        new time series columns are
        ["deep_drain", "et_a", "irrigation", "swc"]
    """
    wuc = 0.096  # water uptake constant
    df["deep_drain"] = 0.0
    df["et_a"] = 0.0
    df["irrigation"] = 0.0
    df["swc"] = 0.0

    # Set awc as a starting value for swc
    swc_cache = awc
    df.loc[df.index[0], "swc"] = swc_cache
    # Loop through df and update swc based on the previous timestep
    for index, row in df.iloc[1:].iterrows():
        df.loc[index, "deep_drain"] = deep_drainage(
            ddc=ddc, rzd=rzd, awc=awc, swc_bd=swc_cache
        )
        df.loc[index, "et_a"] = min(
            wuc * rzd * swc_cache / C_D_TO_H, row["et_p"]
        )  # PAW = rzd * swc

        if has_irrigation:
            water_deficit = df.loc[index, "et_a"] - (row["tp_ground"] - row["runoff"])
            df.at[index, "irrigation"] = max(water_deficit, 0)

        delta_swc = (
            row["tp_ground"]
            - row["runoff"]
            - df.loc[index, "deep_drain"]
            - df.loc[index, "et_a"]
            + df.loc[index, "irrigation"]
        ) / rzd

        df.loc[index, "swc"] = swc_cache + delta_swc
        swc_cache = df.loc[index, "swc"]
    return df


def power(rad, t_air, p_rated, rad_ref, t_ref, noct, **kwargs):
    """
    PV:
    Hourly PV power output in relation to incoming radiation
    Eq. (1) and (2)

    Parameters
    ----------
    rad: numeric
        incoming radiation [W]
    t_air: numeric
        air temperature [°C]
    p_rated: numeric
        rated power output [W]
    rad_ref: numeric
        reference radiation for rated power output [W]
    t_ref: numeric
        reference temperature for rated power output [°C]
    noct: numeric
        normal operating cell temperature [°C]

    Returns
    -------
    p: numeric
        power output [W]
    """
    f_temp = 1 - 3.7e-3 * (t_air + ((noct - 20) / 800) * rad - t_ref)
    p = p_rated * rad / rad_ref * f_temp
    return p


def tt_base(date, t_air, sowing_date, t_base, **kwargs):
    """
    Thermal time experienced by plant as measure for plant development
    from sowing_date until end of the time series (base year)

    Parameters
    ----------
    date: Timestamp object
        date and time of the regarded time step
    t_air: numeric
        air temperature [°C]
    sowing_date: Timestamp object
        date and time when cultivation period starts
    t_base: numeric
        base temp. for plant growth [°C]

    Returns
    -------
    tt: numeric
        additional thermal time for all time steps (hourly)
        from 'sowing_date' until end of the time series (end of year)

    Notes
    -----
    'date' and 'sowing_date' need to match in type and format


    """
    if date < sowing_date:
        tt = 0
    else:
        tt = delta_tt(t=t_air, t_base=t_base)
    return tt


def tt_extension(date, t_air, sowing_date, harvest_date, t_base, **kwargs):
    """
    Additional thermal time experienced by plant as measure for plant development
    from beginning of the time series until harvest date.

    Parameters
    ----------
    date: Timestamp object
        date and time of the regarded time step
    t_air: numeric
        air temperature [°C]
    sowing_date: Timestamp object
        date and time when cultivation period starts
    harvest_date: Timestamp object
        date and time when cultivation period ends
    t_base: numeric
        base temp. for plant growth [°C]

    Returns
    -------
    tt: numeric
        additional thermal time for all time steps (hourly)
        from beginning of the time series (year) until 'harvest_date'
        if 'harvest_date' is in the following year.

    Notes
    -----
    This is based on the assumption, that the time series is one year and growth may extend
    to the following year (if so, harvest_date < sowing_date in MM-DD format).
    As there is only one year simulated, the beginning of that year is used to mimic the next year.
    This allows for plant growth simulation up to 12 months, starting at any time step.
    """
    if date < harvest_date < sowing_date:
        tt = delta_tt(t=t_air, t_base=t_base)
    else:
        tt = 0
    return tt


def tt_cache(
    date,
    cum_temp_base_cache,
    cum_temp_ext_cache,
    harvest_date,
    sowing_date,
    **kwargs,
):
    """
    Cumulated therma time experienced in the base year cached for extension to the following year,
    cumulative temperature experienced in the following year until harvest_date removed afterwards,
    so that it is not added up on the base year.


    Parameters
    ----------
    date: Timestamp object
        date and time of the regarded time step
    cum_temp_base_cache: numeric
        cumulative temperature or thermal time experienced in the base year [K]
    cum_temp_ext_cache: numeric
        cumulative temperature or thermal time experienced in the following year [K]
    sowing_date: Timestamp object
        date and time when cultivation period starts
    harvest_date: Timestamp object
        date and time when cultivation period ends

    Returns
    -------
    tt: numeric
        additional thermal time for all time steps (hourly)
        to be added to year two or removed from year one
    """
    if date <= harvest_date < sowing_date:
        tt = cum_temp_base_cache
    else:
        tt = -cum_temp_ext_cache
    return tt


def specify_cultivation_parameters(dates, sowing_date, harvest_date, **kwargs):
    """
    Adapt sowing_date and harvest_date to input time series (year and timezone),
    convert them to Timestamp objects, specify them if not provided

    Parameters
    ----------
    dates: DatetimeIndex
        DataFrame index created from list of datetime.datetime objects
    sowing_date: str
        date in MM-DD format
    harvest_date: str
        date in MM-DD format

    Returns
    -------
    Dict(
        sowing_date: Timestamp object
            YYYY-MM-DD HH:MM:SS with right timezone (if available), matching parameter 'dates'
        harvest_date: Timestamp object
            YYYY-MM-DD HH:MM:SS with right timezone (if available), matching parameter 'dates'
        has_custom_harvest: Bool
            if True, plant maturity will be calculated according to 'harvest_date'
            (function 'custom_cultivation_period')
        )
    """
    year = str(dates[0].year)
    timezone = dates[0].tz
    # Opt. 1: sowing_date and harvest date given, plant maturity (t_sum) will be updated (custom_harvest=True)
    if sowing_date and harvest_date:
        sowing_date = pd.Timestamp(year + "-" + sowing_date).tz_localize(timezone)
        harvest_date = pd.Timestamp(year + "-" + harvest_date).tz_localize(timezone)
        has_custom_harvest = True
        # If harvest and sowing date are the same, move harvest date one time step back to avoid problems
        if sowing_date == harvest_date:
            harvest_date = dates[dates.index(harvest_date) - 1]
    # Opt. 2: only sowing date, harvest_date (end of cultivation period) is one day before (following year),
    # maturity according to SIMPLE
    elif sowing_date and not harvest_date:
        sowing_date = pd.Timestamp(year + "-" + sowing_date).tz_localize(timezone)
        harvest_date = dates[dates.index(sowing_date) - 1]
        has_custom_harvest = False
    # Opt. 3: no dates, growth from start of the year till maturity (from SIMPLE)
    else:
        sowing_date = dates[0]
        harvest_date = dates[-1]
        has_custom_harvest = False

    return {
        "sowing_date": sowing_date,
        "harvest_date": harvest_date,
        "has_custom_harvest": has_custom_harvest,
    }


def calc_cumulative_temperature(df, sowing_date, harvest_date, t_base, **kwargs):
    """
    Parameters
    ----------
    df: DataFrame object
        required time series columns are ["t_air"]
        requires DatetimeIndex (index of Timestamp objects)
    sowing_date: Timestamp object
        has to match index in type and format
    harvest_date: Timestamp object
        has to match index in type and format
    t_base: numeric
        base temp. for plant growth [°C]

    Returns
    -------
    df: DataFrame object
        additional columns are ["cum_temp"]
    """
    df["cum_temp_base_year"] = df.apply(
        lambda row: tt_base(
            date=row.name, t_air=row["t_air"], sowing_date=sowing_date, t_base=t_base
        ),
        axis=1,
    ).cumsum()

    df["cum_temp_extension"] = df.apply(
        lambda row: tt_extension(
            date=row.name,
            t_air=row["t_air"],
            sowing_date=sowing_date,
            harvest_date=harvest_date,
            t_base=t_base,
        ),
        axis=1,
    ).cumsum()

    cum_temp_base_cache = df["cum_temp_base_year"].iat[-1]
    cum_temp_ext_cache = df["cum_temp_extension"].iat[-1]

    df["cum_temp_cache"] = df.apply(
        lambda row: tt_cache(
            date=row.name,
            cum_temp_base_cache=cum_temp_base_cache,
            cum_temp_ext_cache=cum_temp_ext_cache,
            harvest_date=harvest_date,
            sowing_date=sowing_date,
        ),
        axis=1,
    )

    df["cum_temp"] = (
        df["cum_temp_base_year"] + df["cum_temp_extension"] + df["cum_temp_cache"]
    )
    df.drop(columns=["cum_temp_base_year", "cum_temp_extension", "cum_temp_cache"])
    return df


def calc_f_temp(df, t_opt, t_base, **kwargs):
    """
    Parameters
    ----------
    df: DataFrame object
        required time series columns are ["t_air"]
    t_opt: numeric
        optimal temp. for plant growth [°C]
    t_base: numeric
        base temp. for plant growth [°C]

    Returns
    -------
    df: DataFrame object
        additional columns are ["f_temp"]
    """
    df["f_temp"] = df["t_air"].apply(
        lambda t_air: f_temp(t=t_air, t_opt=t_opt, t_base=t_base)
    )
    return df


def calc_f_heat(df, t_max, t_ext, **kwargs):
    """
    Parameters
    ----------
    df: DataFrame object
        required time series columns are ["t_air"]
    t_max: numeric
        temperature where plant growth starts to be impaired [°C]
    t_ext: numeric
        temperature where plant growth stops due to extreme heat [°C]

    Returns
    -------
    df: DataFrame object
        additional columns are ["f_heat"]
    """
    df["f_heat"] = df["t_air"].apply(
        lambda t_air: f_heat(t=t_air, t_max=t_max, t_ext=t_ext)
    )
    return df


def custom_cultivation_period(df, harvest_date, has_custom_harvest, t_sum, **kwargs):
    """
    Adapts plant growth curve (f_solar) to custom harvest_date

    Parameters
    ----------
    df: DataFrame object
        required time series columns are ["cum_temp"]
        requires DatetimeIndex (index of Timestamp objects)
    harvest_date: Timestamp object
        has to match index in type and format
    has_custom_harvest: Bool
        if True, new t_sum is calculated
    t_sum: numeric
        cumulative temperature until maturity [K]

    Returns
    -------
    Dict(t_sum: numeric)
        new cumulative temperature for plant maturity [K]
    """
    if has_custom_harvest and harvest_date in df.index:
        return {"t_sum": df.loc[harvest_date, "cum_temp"]}
    else:
        return {"t_sum": t_sum}


def calc_f_water(
    df,
    elevation,
    has_rainwater_harvesting,
    has_irrigation,
    frt,
    gcr,
    rcn,
    awc,
    ddc,
    rzd,
    s_water,
    **kwargs,
):
    """
    Parameters
    ----------
    df: DataFrame object
        required columns are ["ghi", "t_air", "t_dp", "tp", "windspeed"]
    elevation: numeric
        elevation of the location where APV is modelled
    has_rainwater_harvesting: Bool
        if False, rainwater_harvest time series will be 0
    has_irrigation:
        if False, irrigation time series will be 0
    frt: numeric
        radiation transmission factor [-]
    gcr: numeric
        ground coverage ratio [-]
    rcn: numeric
        runoff curve number [-]
    awc: numeric
        average water content of the soil [-]
    ddc: numeric
        deep drainage coefficient [-]
    rzd: numeric
        root zone depth of the crops [m]
    s_water: numeric
        crop sensitivity to water stress [-]

    Returns
    -------
    df: DataFrame object
        additional columns are ["f_water", "irrigation"]
        removed columns are ["t_air", "t_dp", "windspeed"]
    """
    albedo = 0.23  # assuming ground albedo=0.23 for g and et_p (FAO56)
    df["rad_net"] = df["ghi"] * (1 - albedo) * frt

    df["g"] = df.apply(
        lambda row: soil_heat_flux(ghi=row["ghi"], r_n=row["rad_net"]), axis=1
    )

    df["et_p"] = df.apply(
        lambda row: potential_evapotranspiration(
            z=elevation,
            t_air=row["t_air"],
            t_dp=row["t_dp"],
            w10=row["windspeed"],
            r_n=row["rad_net"],
            g=row["g"],
        ),
        axis=1,
    )
    df.drop(columns=["t_air", "t_dp", "windspeed", "rad_net", "g"])

    df["tp_ground"] = df["tp"] * (1 - gcr) if has_rainwater_harvesting else df["tp"]

    df["runoff"] = df["tp_ground"].apply(lambda tp: runoff(p=tp, rcn=rcn))

    # soil water balance calculates deep drainage, actual evapotranspiration and irrigation
    df = soil_water_balance(
        df=df, has_irrigation=has_irrigation, awc=awc, ddc=ddc, rzd=rzd
    )
    df.drop(columns=["tp_ground", "runoff"])

    df["f_water"] = df.apply(
        lambda row: f_water(et_p=row["et_p"], et_a=row["et_a"], s_water=s_water), axis=1
    )
    df.drop(columns=["et_p", "et_a"])
    return df


def calc_f_solar(df, i50maxh, i50maxw, t_sum, i50a, i50b, f_solar_max, **kwargs):
    """
    Parameters
    ----------
    df: DataFrame object
        required columns are ["cum_temp", "f_heat", "f_water"]
    i50maxh: numeric
        coefficient for heat tress impact on i50b [-]
    i50maxw: numeric
        coefficient for water stress impact on i50b [-]
    t_sum: numeric
        cumulative temperature until maturity [K]
    i50a: numeric
        cumulative temp. from sowing on to reach 50% of
        solar radiation interception during growth [K]
    i50b: numeric
        cumulative temp. until maturity to fall back to 50% of
        solar rad. intercept. during decline [K]
    f_solar_max:
        maximum solar radiation interception [-]


    Returns
    -------
    df: DataFrame object
        additional columns are ["f_solar"]
        removed columns are ["cum_temp"]
    """
    df["delta_i50b"] = df.apply(
        lambda row: delta_i50b(
            cum_temp=row["cum_temp"],
            f_heat=row["f_heat"],
            f_water=row["f_water"],
            i50maxh=i50maxh,
            i50maxw=i50maxw,
        ),
        axis=1,
    ).cumsum()

    df["f_solar"] = df.apply(
        lambda row: f_solar(
            cum_temp=row["cum_temp"],
            delta_i50b=row["delta_i50b"],
            t_sum=t_sum,
            i50a=i50a,
            i50b=i50b,
            f_solar_max=f_solar_max,
        ),
        axis=1,
    )
    df.drop(columns=["cum_temp", "delta_i50b"])
    return df


def calc_biomass(df, frt, rue, **kwargs):
    """
    Parameters
    ----------
    df: DataFrame object
        required columns are ["ghi", "f_solar", "f_temp", "f_heat", "f_water"]
    frt: numeric
        radiation transmission factor [-]
    rue: numeric
        radiation use efficiency [-]


    Returns
    -------
    df: DataFrame object
        additional columns are ["total_biomass"]
        removed columns are ["f_solar", "f_temp", "f_heat", "f_water"]
    """
    df["total_biomass"] = (
        rue
        * df["ghi"]
        * C_WH_TO_J
        * C_J_TO_MJ
        * frt
        * df["f_solar"]
        * df["f_temp"]
        * df[["f_heat", "f_water"]].min(axis=1)
        * C_G_TO_KG
    )
    df.drop(columns=["f_solar", "f_temp", "f_heat", "f_water"])
    return df["total_biomass"]


def adapt_irrigation(df):
    """
    Sets irrigation = 0 outside the cultivation period (if f_solar == 0)
    """
    # Set irrigation to 0 outside the cultivation period (f_solar == 0)
    df.loc[df["f_solar"] == 0, "irrigation"] = 0
    return df
