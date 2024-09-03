from dataclasses import field
from typing import Sequence, Union

import numpy as np
import pandas as pd

from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

import dataclasses
from oemof.tabular._facade import dataclass_facade, Facade
from oemof.tabular.facades import Conversion
from oemof_tabular_plugins.wefe.global_specs.crops import crop_dict


# SOME PLOTS AND PRINTS FOR DEBUGGING PURPOSES
import matplotlib.pyplot as plt

@dataclass_facade
# @dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class SimpleCrop(Converter, Facade):
    r"""Crop Model Converter with one input and one output. The crop growth factor
    is calculated out drought, heat, temperature, and water availabilty impact
     and considered for biomass production calculation.

    Parameters
    ----------
    crop_type: str
        the name of crop as defined in global_specs/crops.py
    from_bus: oemof.solph.Bus
        An oemof bus instance where the PV panel unit is connected to with
        its input.
    to_bus: oemof.solph.Bus
        An oemof bus instance where the PV panel unit is connected to with
        its output.
    capacity: numeric
        The capacity of crop. It is expressed in cultivated area [m²]
    marginal_cost: numeric
        Marginal cost for one unit of produced output. Default: 0
    carrier_cost: numeric
        Carrier cost for one unit of used input. Default: 0
    capacity_cost: numeric
        Investment costs per unit of output capacity.
        If capacity is not set, this value will be used for optimizing the
        conversion output capacity.
    expandable: boolean or numeric (binary)
        True, if capacity can be expanded within optimization. Default: False.
    capacity_potential: numeric
        Maximum invest capacity in unit of output capacity.
    capacity_minimum: numeric
        Minimum invest capacity in unit of output capacity.
    input_parameters: dict (optional)
        Set parameters on the input edge of the conversion unit
        (see oemof.solph for more information on possible parameters)
    output_parameters: dict (optional)
        Set parameters on the output edge of the conversion unit
         (see oemof.solph for more information on possible parameters)

    ghi: time series
        Global horizontal irradiance
    et_0: time series
        potential evapotranspiration [m³]

    t_air: time series
        Ambient air temperature

    vwc: time series
        the voluemtric water content (vwc) in the root zone depth in m³; metric to express soil moisture


    SHOULD INCLUDE FUNCTIONS AND EXAMPLE HERE

    # TODO these parameters below are most likely provided by crop_dict

    light_saturation_point: numeric
        [lux]
    HI: numeric
        Harvest Index; share of biomass which is harvested as actual fruit
    I50A: numeric
        Irradiation fitting parameter
    I50B: numeric
        Irradation fitting parameter
    t_opt: numeric
         optimal temperature for biomass growth
    t_base: numeric
        base temperature for phenological development and growth
    rue: numeric
        Radiation Use efficiency (above ground only and without respiration) (g/MJm²)
    t_heat: numeric
        t_heat is the threshold temperature when biomass growth rate starts to be reduced by heat stress
    t_extreme: numeric
        t_ext is the extreme temperature threshold when the biomass growth rate reaches 0 due to heat stress
    t_sum: numeric
        cumulative temperature until harvest
    s_water: numeric
        sensitivity of RUE to the ARID index for specific plant (simple crop model)
    rzd: numeric
        root zone depth [m]

    """

    from_bus: Bus

    to_bus: Bus

    carrier: str

    tech: str

    capacity: float = None

    marginal_cost: float = 0

    carrier_cost: float = 0

    capacity_cost: float = None

    expandable: bool = False

    capacity_potential: float = float("+inf")

    capacity_minimum: float = None

    input_parameters: dict = field(default_factory=dict)

    output_parameters: dict = field(default_factory=dict)

    crop_type: str = ""

    time_index: Union[float, Sequence[float]] = None

    t_air: Union[float, Sequence[float]] = None

    ghi: Union[float, Sequence[float]] = None

    et_0: Union[float, Sequence[float]] = None

    vwc: Union[float, Sequence[float]] = None

    # should be MM-DD with the year taken from the timeindex column of the input time series
    sowing_date: str = ""  # TODO check for dates formats YYYY-MM-DD

    harvest_date: str = ""

    # def __init__(self,crop_type, t_air, ghi, et_0, vwc, sowing_date,harvest_date,*args, **kwargs):
    #
    #     super().__init__(*args,**kwargs)
    #     self.crop_type=crop_type
    #     self.t_air =t_air
    #     self.ghi = ghi
    #     self.et_0 = et_0
    #     self.vwc = vwc
    #     self.sowing_date=sowing_date
    #     self.harvest_date=harvest_date

    @classmethod
    def processing_raw_inputs(self, resource, results_df):
        # function to apply on df from above

        return results_df

    # @classmethod
    # def validate_datapackage(self, resource):
    #     # modify the resource (datapackage.resource)
    #     # should it return the resource?
    #     pass
    def get_crop_param(self, param_name):
        return crop_dict[self.crop_type][param_name]

    # efficiency equals biomass production factor calculate in the facade crop.py; caapcity equals area [m²]

    def calc_Fsolar(self, time_index, sowing_date, harvest_date, t_air, t_base, t_sum, i50a, i50b, f_solar_max, **kwargs):
        """ Calculates f_solar, the factor for interception of incoming solar radiation by plant canopy
            according to development stage [-].
            ----
        Parameters
        ----------
        time_index: time as pd.series or list
        sowing_date: time of sowing/start of cultivation period as str in "MM-DD" format
        harvest_date: time of harvest/end of cultivation period as str in "MM-DD" format
        t_air: ambient temperature as pd.series or list
        t_base: base temperature for biomass growth
        t_sum: cumulative temperature required from sowing to maturity
        i50a: cumulative temperature required from sowing on to reach 50% of solar radiation interception during growth
        i50b: cumulative temperature required until maturity to fall back to 50% of solar interception during decline
        f_solar_max: maximum factor for solar interception

        Returns
        -------
        f_solar_list : list of numerical values:
             solar interception coefficients for calculating biomass rate

        Notes
        -------
        This resembles the plant growth curve displayed in Fig 1.(e) of https://doi.org/10.1016/j.eja.2019.01.009

         """
        # Check input time series compatibility
        if not isinstance(t_air, (list, pd.Series)):
            print("Argument 'temp' is not of type list or pd.Series!")
        if not isinstance(time_index, (list, pd.Series)):
            print("Argument 'time index' is not of type list or pd.Series!")
        if len(t_air) != len(time_index):
            raise ValueError("Length mismatch between t_air and time_index profiles.")

        # Convert dates to Timestamp objects matching the time index
        dates = list(pd.to_datetime(time_index))
        year = str(dates[0].year)
        timezone = dates[0].tz
        # Opt. 1: sowing_date and harvest date given, plant maturity (t_sum) will be updated (custom_harvest=True)
        if sowing_date and harvest_date:
            sowing_date = pd.Timestamp(year + '-' + sowing_date).tz_localize(timezone)
            harvest_date = pd.Timestamp(year + '-' + harvest_date).tz_localize(timezone)
            custom_harvest = True
            # If harvest and sowing date are the same, move harvest date one time step back to avoid problems
            if sowing_date == harvest_date:
                harvest_date = dates[dates.index(harvest_date) - 1]
        # Opt. 2: only sowing date, harvest_date (end of cultivation period) is one day before (following year),
        # maturity according to SIMPLE
        elif sowing_date and not harvest_date:
            sowing_date = pd.Timestamp(year + '-' + sowing_date).tz_localize(timezone)
            harvest_date = dates[dates.index(sowing_date) - 1]
            custom_harvest = False
        # Opt. 3: no dates, growth from start of the year till maturity (from SIMPLE)
        else:
            sowing_date = time_index[0]
            harvest_date = time_index[-1]
            custom_harvest = False

        print(f"sowing date: {sowing_date}\nharvest date: {harvest_date}\ncustom cultivation period: {custom_harvest}")

        def development_base_year(time_index, sowing_date, t_air, t_base):
            """
            Cumulative temperature experienced by plant as measure for plant development
            from sowing_date until end of the same year (base year)
            """

            delta_tt_list = []  # creating a list
            for temp, date in zip(t_air, time_index):
                if date < sowing_date:
                    delta_tt = 0
                else:
                    if temp > t_base:
                        delta_tt = (temp - t_base) / 24  # SIMPLE crop model has daily temp values, convert to hourly
                    else:
                        delta_tt = 0
                delta_tt_list.append(delta_tt)
            cumulative_temp = np.cumsum(delta_tt_list)
            return cumulative_temp

        def development_extension(time_index, sowing_date, harvest_date, t_air, t_base):
            """
            Additional cumulative temperature experienced by plant as measure for plant development
            if growth extends to following year (until harvest_date if such is given).
            Note that harvest_date (MM-DD) has to be before sowing_date (MM-DD), growth cannot extend beyond one year in total.
            """
            delta_tt_list = []  # creating a list
            for temp, date in zip(t_air, time_index):
                if date < harvest_date < sowing_date:
                    if temp > t_base:
                        delta_tt = (temp - t_base) / 24  # SIMPLE crop model has daily temp values, convert to hourly
                    else:
                        delta_tt = 0
                else:
                    delta_tt = 0
                delta_tt_list.append(delta_tt)
            cumulative_temp = np.cumsum(delta_tt_list)
            return cumulative_temp

        def development_cache(time_index, sowing_date, harvest_date, cum_temp_base_cache, cum_temp_ext_cache):
            """
            Cumulative temperature experienced in the base year cached for extension in the following year,
            cumulative temperature experienced in the following year until harvest_date removed afterwards
            so it does not interfer with base year [K]
            """
            delta_tt_list = []  # creating a list
            for date in time_index:
                if date <= harvest_date < sowing_date:
                    delta_tt = cum_temp_base_cache
                else:
                    delta_tt = -cum_temp_ext_cache
                delta_tt_list.append(delta_tt)
            return delta_tt_list

        # Create 3 lists for cumulative_temperature: base year, following year (extension), cached values
        cumulative_temp_1 = development_base_year(dates, sowing_date, t_air, t_base)
        cum_temp_base_cache = cumulative_temp_1[-1]

        cumulative_temp_2 = development_extension(dates, sowing_date, harvest_date, t_air, t_base)
        cum_temp_ext_cache = cumulative_temp_2[-1]

        cumulative_temp_3 = development_cache(dates, sowing_date, harvest_date, cum_temp_base_cache, cum_temp_ext_cache)

        # Add the three lists together to get total cumulative temperature
        cumulative_temp = [ct1 + ct2 + ct3 for ct1, ct2, ct3 in zip(cumulative_temp_1, cumulative_temp_2, cumulative_temp_3)]

        # Update t_sum if custom_harvest = True (custom harvest date provided instead of maturity according to SIMPLE)
        t_sum = cumulative_temp[dates.index(harvest_date)] if custom_harvest else t_sum
        print(f"t_sum: {t_sum}")

        # Assumption for transition point from growth period to senescence (decline) period: f_solar = 0.999f_solar_max
        # with f_solar being a function of cum_temp
        cum_temp_to_reach_f_solar_max = i50a - 100 * np.log(1 / 999)

        # f_solar(cum_temp) according to SIMPLE
        f_solar_list = []
        for cum_temp in cumulative_temp:
            if cum_temp < 1:
                f_solar = 0
            elif cum_temp < cum_temp_to_reach_f_solar_max:
                f_solar = f_solar_max / (1 + np.exp(-0.01 * (cum_temp - i50a)))
            elif cum_temp < t_sum:
                f_solar = f_solar_max / (1 + np.exp(0.01 * (cum_temp - (t_sum - i50b))))
            else:
                f_solar = 0
            f_solar_list.append(f_solar)

        # plt.plot(dates, np.array(cumulative_temp_1) / 1000, color='orange', label='cum_temp_base')
        # plt.plot(dates, np.array(cumulative_temp_2) / 1000, color='yellow', label='cum_temp_ext')
        # plt.plot(dates, np.array(cumulative_temp_3) / 1000, color='purple', label='cum_temp_cache')
        # plt.plot(dates, np.array(cumulative_temp) / 1000, '--', color='red', label='cum_temp')
        # plt.plot(dates, f_solar_list, '--', color='green', label='fSOLAR')
        # plt.grid(True)
        # plt.legend()
        #
        # plt.show()

        return np.array(f_solar_list)

    def calc_Ftemp(self, t_air, t_opt, t_base, **kwargs):
        r"""
        Calculates the temperature on the biomass rate
        ----
        Parameters
        ----------
        t_air: ambient temperature as pd.series or list
        t_opt: optimum temperature for biomass growth
        t_base: base temperature for biomass growth

        Returns
        -------
        te : list of numerical values:
             temperature coefficients for calculating biomass rate

        Notes
        -----
        Corresponds to Fig 1.(a) of https://doi.org/10.1016/j.eja.2019.01.009


        """

        # Check if input arguments have proper type and length
        if not isinstance(t_air, (list, pd.Series)):
            print("Argument 'temp' is not of type list or pd.Series!")
        te = []  # creating a list
        # Calculate te
        for t in t_air:
            if t < t_base:
                x = 0
                te.append(x)

            elif t_base <= t <= t_opt:
                x = (t - t_base) / (t_opt - t_base)
                te.append(x)

            elif t > t_opt:
                x = 1
                te.append(x)

        return np.array(te)

    def calc_Fwater(self, et_o, vwc, s_water, **kwargs):
        r"""
        Calculates the soil water availability impact on the biomass rate
        arid factor derived from simple crop model and Woli et al. (2012), divided by 24 to transform to hourly values
        ----
        Parameters
        ----------
        et_o: potential evapotranspiration
        vwc: volumetric water content
        s_water: Sensitivity of RUE (or harvest index) to drought stress (ARID index)

        Returns
        -------
        arid : list of numerical values:
             aridity factor affecting the biomass rate

        Notes
        -----
        Corresponds to Fig 1.(d) of https://doi.org/10.1016/j.eja.2019.01.009


        """

        def arid(et_o, vwc):
            et_o = np.array(et_o)
            vwc = np.array(vwc)

            # Calculate arid
            wi = 1 - np.minimum(abs(et_o), 0.096 * vwc) / abs(et_o)
            return wi

        f_water_list = 1 - s_water * arid(et_o, vwc)
        return np.array(f_water_list)

    def calc_Fheat(self, t_air, t_max, t_ext, **kwargs):
        r"""
        Calculates the heat impact on the biomass rate
        ----
        Parameters
        ----------
        t_air: Hourly timeseries of air temperature as pd.series or list
        t_max: Threshold temperature to start accelerating senescence from heat stress (°C)
        t_ext: Threshold temperature when biomass growth rate starts to be reduced by heat stress (°C)
        Returns
        -------
        hi : list of numerical values:
             temperature coefficients for calculating biomass rate

        Notes
        -----
        Corresponds to Fig 1.(b) of https://doi.org/10.1016/j.eja.2019.01.009

        """

        def paper_function(t_max, t_heat, t_ext):
            """function witin https://doi.org/10.1016/j.eja.2019.01.009"""
            if t_max <= t_heat:
                x = 1
            elif t_heat < t_max <= t_ext:
                x = 1 - (t_max - t_heat) / (t_ext - t_heat)
            elif t_max > t_ext:
                x = 0
            return x

        # Check if input arguments have proper type and length
        if not isinstance(t_air, (list, pd.Series, np.ndarray)):
            raise TypeError("Argument t_air is not of type list or pd.Series!")

        n_timesteps = len(t_air)

        if n_timesteps == 0:
            raise ValueError("Argument t_air is empty")

        n_hours_in_day = 24

        n_days, n_hours = np.divmod(n_timesteps, n_hours_in_day)

        # in the paper they provide T_heat as T_max in the table 1a
        t_heat = t_max
        hi = []  # creating a list

        if n_days == 0:
            # there is less than a day, returing same value n_timesteps times
            n_timesteps = n_hours
            t_max = np.max(t_air)
            hi.append(np.ones(n_timesteps) * paper_function(t_max, t_heat, t_ext))
        else:
            # there is one day or more
            if n_hours != 0:
                N = n_days + 1
            else:
                N = n_days
            for i in range(0, N):
                if i == n_days:
                    if n_hours != 0:
                        # last day is not a full day
                        n_timesteps = n_hours
                    else:
                        # last day is a full day
                        n_timesteps = n_hours_in_day
                else:
                    # it is a full day
                    n_timesteps = n_hours_in_day
                t_max = np.max(
                    t_air[(i * n_hours_in_day) : (i * n_hours_in_day + n_timesteps)]
                )
                hi.append(np.ones(n_timesteps) * paper_function(t_max, t_heat, t_ext))

        return np.hstack(hi)

    @property
    def efficiency(self):

        # Conversion factor Watt Hours (WH) to Mega Joules (MJ)
        c_wh_to_mj = 3.6e-3
        crop_params = crop_dict[self.crop_type]
        fTEMP = self.calc_Ftemp(self.t_air, **crop_params)
        fWATER = self.calc_Fwater(self.et_0, self.vwc, **crop_params)
        fHEAT = self.calc_Fheat(t_air=self.t_air, **crop_params)
        fSOLAR = self.calc_Fsolar(time_index=self.time_index, t_air=self.t_air,
                                  sowing_date=self.sowing_date, harvest_date=self.harvest_date, **crop_params)

        rue = self.get_crop_param("rue")
        hi = self.get_crop_param("hi")

        # plt.plot(self.time_index, fTEMP, label='f_temp')
        # plt.plot(self.time_index, fWATER, label='f_water')
        # plt.plot(self.time_index, fHEAT, label='f_heat')
        # plt.plot(self.time_index, fSOLAR, label='f_solar')
        # plt.grid(True)
        # plt.legend()
        #
        # plt.show()

        return (
            hi * rue * fSOLAR * fTEMP * np.minimum(fWATER, fHEAT) * c_wh_to_mj
        )

    def build_solph_components(self):
        """ """

        self.conversion_factors.update(
            {
                self.from_bus: sequence(1),
                self.to_bus: sequence(abs(self.efficiency)),
            }
        )

        self.inputs.update(
            {
                self.from_bus: Flow(
                    variable_costs=self.carrier_cost, **self.input_parameters
                )
            }
        )

        self.outputs.update(
            {
                self.to_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost,
                    investment=self._investment(),
                    **self.output_parameters,
                )
            }
        )
