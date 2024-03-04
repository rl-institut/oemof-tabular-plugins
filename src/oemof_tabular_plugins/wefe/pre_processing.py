import pandas as pd

# ------- Simulating the electricity production by photovoltaic panels

def calc_pv_tf(t_air, ghi, p_rpv, r_ref, n_t, t_c_ref, noct):

    r"""
    Calculates the temperature factor influencing solar energy production
    ----
    Parameters
    ----------
    t_air: ambient air temperature as pd.series or list
    p_rpv: rated Power of photovoltaic panel
    r_ref: solar radiation at reference conditions
    n_t: temperature coefficient of photovoltaic panel
    t_c_ref: cell temperature at reference conditions
    noct: normal operating cell temperature
    ghi: global horizontal irradiance
    Returns
    -------
    pv_te : list of numerical values:
         temperature coefficients for calculating biomass rate

    """
    # Check if input arguments have proper type and length
    if not isinstance(t_air, (list, pd.Series)):
        raise TypeError("Argument 't_air' is not of type list or pd.Series!")
    # pv_te = []    # creating a list
    t_c = t_air + ((noct-20)/800)*ghi
    pv_te = p_rpv*(1/r_ref)*(1+n_t*(t_c-t_c_ref))
    return pv_tf