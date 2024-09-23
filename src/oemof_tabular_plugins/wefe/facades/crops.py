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
from oemof_tabular_plugins.wefe.facades import MIMO
from oemof_tabular_plugins.wefe.global_specs import crop_dict, soil_dict


@dataclass_facade
# @dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class SimpleCrop(Converter, Facade):
    r"""Crop Model Converter with one input and one output. The crop growth factor
    is calculated out drought, heat, temperature, and water availabilty impact
     and considered for biomass production calculation.

    Parameters
    ----------
    crop_type: str
        the name of crop as defined in global_specs/crop_specs.py
    solar_bus: oemof.solph.Bus
        An oemof bus instance where the SimpleCrop unit is connected to with
        its input, it is expected to provide W/m² irradiance.
    harvest_bus: oemof.solph.Bus
        An oemof bus instance where the PV panel unit is connected to with
        its crop-harvest output. The unit is kg
    biomass_bus: oemof.solph.Bus
        An oemof bus instance where the SimpleCrop is connected with its
        non-edible biomass output. The unit is kg
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

    solar_bus: Bus

    harvest_bus: Bus

    carrier: str

    tech: str

    biomass_bus: Bus = None

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

    sowing_date: str = ""  # TODO: MM-DD format

    harvest_date: str = ""  # TODO: MM-DD format

    @classmethod
    def validate_datapackage(self, resource):
        # check that the crop_type matches the one provided in the crop_dict
        data = pd.DataFrame.from_records(resource.read(keyed=True))

        for i, row in data[["crop_type", "name", "type"]].iterrows():
            if row.crop_type not in crop_dict:

                raise KeyError(
                    f"On line {i+1} of resource '{resource.descriptor['path']}', the crop_type attribute '{row.crop_type}' of the component of type '{row.type}', named '{row['name']}', is not available in the crop specs dict (wefe/gloabal_specs/crop_specs.py). Available crop types are: {', '.join([k for k in crop_dict.keys()])}"
                )
        return resource

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

    # efficiency equals biomass production factor calculate in the facade crop_specs.py; caapcity equals area [m²]

    def calc_Fsolar(
        self,
        time_index,
        sowing_date,
        harvest_date,
        t_air,
        t_base,
        t_sum,
        i50a,
        i50b,
        f_solar_max,
        **kwargs,
    ):
        """Calculates f_solar, the factor for interception of incoming solar radiation by plant canopy
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
            sowing_date = pd.Timestamp(year + "-" + sowing_date).tz_localize(timezone)
            harvest_date = pd.Timestamp(year + "-" + harvest_date).tz_localize(timezone)
            custom_harvest = True
            # If harvest and sowing date are the same, move harvest date one time step back to avoid problems
            if sowing_date == harvest_date:
                harvest_date = dates[dates.index(harvest_date) - 1]
        # Opt. 2: only sowing date, harvest_date (end of cultivation period) is one day before (following year),
        # maturity according to SIMPLE
        elif sowing_date and not harvest_date:
            sowing_date = pd.Timestamp(year + "-" + sowing_date).tz_localize(timezone)
            harvest_date = dates[dates.index(sowing_date) - 1]
            custom_harvest = False
        # Opt. 3: no dates, growth from start of the year till maturity (from SIMPLE)
        else:
            sowing_date = dates[0]
            harvest_date = dates[-1]
            custom_harvest = False

        print(
            f"sowing date: {sowing_date}\nharvest date: {harvest_date}\ncustom cultivation period: {custom_harvest}"
        )

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
                        delta_tt = (
                            temp - t_base
                        ) / 24  # SIMPLE crop model has daily temp values, convert to hourly
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
                        delta_tt = (
                            temp - t_base
                        ) / 24  # SIMPLE crop model has daily temp values, convert to hourly
                    else:
                        delta_tt = 0
                else:
                    delta_tt = 0
                delta_tt_list.append(delta_tt)
            cumulative_temp = np.cumsum(delta_tt_list)
            return cumulative_temp

        def development_cache(
            time_index,
            sowing_date,
            harvest_date,
            cum_temp_base_cache,
            cum_temp_ext_cache,
        ):
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
            return np.array(delta_tt_list)

        # Create 3 lists for cumulative_temperature: base year, following year (extension), cached values
        cumulative_temp_1 = development_base_year(dates, sowing_date, t_air, t_base)
        cum_temp_base_cache = cumulative_temp_1[-1]

        cumulative_temp_2 = development_extension(
            dates, sowing_date, harvest_date, t_air, t_base
        )
        cum_temp_ext_cache = cumulative_temp_2[-1]

        cumulative_temp_3 = development_cache(
            dates, sowing_date, harvest_date, cum_temp_base_cache, cum_temp_ext_cache
        )

        # Add the three lists together to get total cumulative temperature
        cumulative_temp = cumulative_temp_1 + cumulative_temp_2 + cumulative_temp_3

        # Update t_sum if custom_harvest = True (custom harvest date provided instead of maturity according to SIMPLE)
        t_sum = cumulative_temp[dates.index(harvest_date)] if custom_harvest else t_sum
        # print(f"t_sum: {t_sum}")

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
        c_kg_to_g = 1e-3
        crop_params = crop_dict[self.crop_type]
        fTEMP = self.calc_Ftemp(self.t_air, **crop_params)
        fWATER = self.calc_Fwater(self.et_0, self.vwc, **crop_params)
        fHEAT = self.calc_Fheat(t_air=self.t_air, **crop_params)
        fSOLAR = self.calc_Fsolar(
            time_index=self.time_index,
            t_air=self.t_air,
            sowing_date=self.sowing_date,
            harvest_date=self.harvest_date,
            **crop_params,
        )

        rue = self.get_crop_param("rue")

        return rue * fSOLAR * fTEMP * np.minimum(fWATER, fHEAT) * c_wh_to_mj * c_kg_to_g

    def build_solph_components(self):
        """ """
        hi = self.get_crop_param("hi")
        self.conversion_factors.update(
            {
                self.solar_bus: sequence(1),
                self.harvest_bus: sequence(hi * abs(self.efficiency)),
            }
        )

        self.inputs.update(
            {
                self.solar_bus: Flow(
                    variable_costs=self.carrier_cost, **self.input_parameters
                )
            }
        )

        self.outputs.update(
            {
                self.harvest_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost,
                    investment=self._investment(),
                    **self.output_parameters,
                )
            }
        )

        if self.biomass_bus is not None:
            self.conversion_factors.update(
                {self.biomass_bus: sequence((1 - hi) * abs(self.efficiency))}
            )
            self.outputs.update(
                {
                    self.biomass_bus: Flow(),
                }
            )


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class MimoCrop(MIMO):
    r"""
    Crop Model Converter with 3 inputs (irradiation, precipitation, irrigation)
    and 2 outputs (crop harvest, remaining biomass), based on MultipleInputMultipleOutputConverter.

    Functions designed according to 3 publications, please note the abbreviations in the function descriptions:
    FAO56       ISBN: 978-92-5-104219-9
    SIMPLE      https://doi.org/10.1016/j.eja.2019.01.009
    ARID        https://doi.org/10.2134/agronj2011.0286

    Parameters
    ----------
    type: str
        will link MimoCrop to TYPEMAP, currently named 'mimo-crop'
    name: str
        a remarkable name for your component
    tech: str
        whatever, I just specified it as 'mimo'
    carrier: str
        carrier of the primary bus, should be overwritten by carriers directly assigned to buses
    primary: str
        primary bus object to which investments etc. are assigned
    expandable: boolean or numeric (binary)
        True, if capacity can be expanded within optimization. Default: False.
    capacity: numeric
        The capacity of crop. It is expressed in cultivated area [m²]
    capacity_minimum: numeric
        Minimum invest capacity in unit of output capacity.
    capacity_potential: numeric   (seems not to work as it should)
        Maximum invest capacity in unit of output capacity.
    capacity_cost: numeric
        Investment costs per unit of output capacity. If capacity is not set,
        this value will be used for optimizing the conversion output capacity.
    crop_type: str
        the name of crop as defined in global_specs/crop_specs.py
    solar_energy_bus: oemof.solph.Bus
        An oemof bus instance where the MimoCrop unit is connected to with
        its input, it is expected to provide W/m² irradiance.
    precipitation_bus: oemof.solph.Bus
        An oemof bus instance where the MimoCrop unit is connected to with
        its input, it is expected to provide mm precipitation.
    irrigation_bus: oemof.solph.Bus
        An oemof bus instance where the MimoCrop unit is connected to with
        its input, it is expected to provide mm irrigation.
    crop_bus: oemof.solph.Bus
        An oemof bus instance where the MimoCrop unit is connected to with
        its crop-harvest output. The unit is kg.
    biomass_bus: oemof.solph.Bus
        An oemof bus instance where the MimoCrop is connected with its
        non-edible, remaining (residual) biomass output. The unit is kg.
    time_profile: time series
        time profile separate from time index
    ghi_profile: time series
        global horizontal solar irradiance in W/m²
    tp_profile: time series
        total precipitation in mm
    t_air_profile: time series
        ambient air tempearture in °C
    t_dp_profile: time series
        dew point temperature in °C
    windspeed_profile: time series
        windspeed in m/s
    elevation: float
        elevation above sea level of the location
    crop_type: str
        the name of crop as defined in global_specs/crop_specs.py
    sowing_date: str
        date when cultivation period starts, MM-DD format
    harvest_date:
        date when cultivation period ends, MM-DD format
    has_irrigation: bool
        if set to False, irrigation profile will be set to 0

    Other optional parameters that could be added:
    ---------------------------------------------
    marginal_cost: numeric
        Marginal cost for one unit of produced output. Default: 0
    carrier_cost: numeric
        Carrier cost for one unit of used input. Default: 0
    input_parameters: dict (optional)
        Set parameters on the input edge of the conversion unit
        (see oemof.solph for more information on possible parameters)
    output_parameters: dict (optional)
        Set parameters on the output edge of the conversion unit
        (see oemof.solph for more information on possible parameters)

    Notes:
    ---------------------------------------------
    One input (eg. ghi) has to be volatile with a fixed capacity, other 2 dispatchable.
    MimoCrop capacity_potential seems to not work correctly.
    Two options to successfully run MimoCrop:
    1) Expandable = True and ignore capacity, use capacity of volatile source instead.
    2) Expandable = False, set capacity equal to that of the volatile source.

    In case you experience a weird error, for example "TypeError: can't multiply sequence by non-int of type 'float'",
    this is likely due to empty rows at the end of your input csv files.
    Please remove them and try again.
    """

    type: str = "mimo-crop"

    name: str = ""

    tech: str = "mimo"

    carrier: str = ""

    primary: str = ""

    expandable: bool = False

    capacity: float = 0

    capacity_minimum: float = None

    capacity_potential: float = None

    capacity_cost: float = 0

    solar_energy_bus: Bus = None

    precipitation_bus: Bus = None

    irrigation_bus: Bus = None

    crop_bus: Bus = None

    biomass_bus: Bus = None

    time_profile: Union[float, Sequence[float]] = None

    ghi_profile: Union[float, Sequence[float]] = None

    tp_profile: Union[float, Sequence[float]] = None

    t_air_profile: Union[float, Sequence[float]] = None

    t_dp_profile: Union[float, Sequence[float]] = None

    windspeed_profile: Union[float, Sequence[float]] = None

    elevation: float = 0

    crop_type: str = ""  # according to crop_dict

    sowing_date: str = ""  # MM-DD format

    harvest_date: str = ""  # MM-DD format

    has_irrigation: bool = False

    def __init__(self, **kwargs):
        """
        Preprocessing of crop input data to calculate conversion_factors
        and assign these together with the correct busses to MIMO parent class
        """

        def specify_cultivation_parameters(dates, sowing_date, harvest_date):
            """
            Convert all time-based elements to Timestamp objects,
            adapt sowing_date and harvest_date to input time series (year and timezone),
            specify sowing_date and harvest_date if not provided
            """
            year = str(dates[0].year)
            timezone = dates[0].tz
            # Opt. 1: sowing_date and harvest date given, plant maturity (t_sum) will be updated (custom_harvest=True)
            if sowing_date and harvest_date:
                sowing_date = pd.Timestamp(year + "-" + sowing_date).tz_localize(
                    timezone
                )
                harvest_date = pd.Timestamp(year + "-" + harvest_date).tz_localize(
                    timezone
                )
                has_custom_harvest = True
                # If harvest and sowing date are the same, move harvest date one time step back to avoid problems
                if sowing_date == harvest_date:
                    harvest_date = dates[dates.index(harvest_date) - 1]
            # Opt. 2: only sowing date, harvest_date (end of cultivation period) is one day before (following year),
            # maturity according to SIMPLE
            elif sowing_date and not harvest_date:
                sowing_date = pd.Timestamp(year + "-" + sowing_date).tz_localize(
                    timezone
                )
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

        def development_base_year(date, t_air, sowing_date, t_base, **kwargs):
            """
            Cumulative temperature experienced by plant as measure for plant development
            from sowing_date until end of the same year (base year) (SIMPLE) [K]
            """
            if date < sowing_date:
                delta_tt = 0
            else:
                if t_air > t_base:
                    delta_tt = (
                        t_air - t_base
                    ) / 24  # SIMPLE crop model has daily temp values, convert to hourly
                else:
                    delta_tt = 0
            return delta_tt

        def development_extension(
            date, t_air, sowing_date, harvest_date, t_base, **kwargs
        ):
            """
            Additional cumulative temperature experienced by plant as measure for plant development
            if growth extends to following year
            (harvest_date < sowing_date in MM-DD format because it would be in the next year) (SIMPLE) [K]
            """
            if date < harvest_date < sowing_date:
                if t_air > t_base:
                    delta_tt = (
                        t_air - t_base
                    ) / 24  # SIMPLE crop model has daily temp values, convert to hourly
                else:
                    delta_tt = 0
            else:
                delta_tt = 0
            return delta_tt

        def development_cache(
            date,
            cum_temp_base_cache,
            cum_temp_ext_cache,
            harvest_date,
            sowing_date,
            **kwargs,
        ):
            """
            Cumulative temperature experienced in the base year cached for extension in the following year,
            cumulative temperature experienced in the following year until harvest_date removed from base year
            (SIMPLE) [K]
            """
            if date <= harvest_date < sowing_date:
                delta_tt = cum_temp_base_cache
            else:
                delta_tt = -cum_temp_ext_cache
            return delta_tt

        def custom_cultivation_period(
            df, harvest_date, has_custom_harvest, t_sum, **kwargs
        ):
            """Calculates new t_sum for plant growth curve if custom harvest_date is given [K]"""
            if has_custom_harvest and harvest_date in df.index:
                return {"t_sum": df.loc[harvest_date, "cum_temp"]}
            else:
                return {"t_sum": t_sum}

        def temp(t_air, t_opt, t_base, **kwargs):
            """Temperature effect on plant growth (SIMPLE) [-]"""
            if t_air < t_base:
                f_temp = 0
            elif t_base <= t_air < t_opt:
                f_temp = (t_air - t_base) / (t_opt - t_base)
            else:
                f_temp = 1
            return f_temp

        def heat(t_air, t_max, t_ext, **kwargs):
            """Heat stress effect on plant growth (SIMPLE) [-]"""
            if t_air <= t_max:
                f_heat = 1
            elif t_max < t_air <= t_ext:
                f_heat = 1 - (t_air - t_max) / (t_ext - t_max)
            else:
                f_heat = 0
            return f_heat

        def soil_heat_flux(ghi, irr_w, **kwargs):
            """Soil heat flux as fraction of incoming radiation (FAO56) [W/m²]"""
            if ghi > 1e-5:
                g = 0.1 * irr_w
            else:
                g = 0.5 * irr_w
            return g

        def potential_evapotranspiration(z, t_air, t_dp, w10, irr_w, g_w, **kwargs):
            """Potential evapotranspiration for reference grass (FAO56) [mm]"""
            cp_air = 1.013e-3  # specific heat at constant pressure [MJ/(kg °C)]
            epsilon = 0.622  # ratio molecular weight of water vapour/dry air [-]
            h_vap = 2.45  # latent heat of vaporization [MJ/kg]
            rho_w = 1000  # density of water [kg/m³]
            k_c = 1  # crop coefficient (FAO) [-]
            irr = 86.4e-3 * irr_w  # W/m² to MJ/m²*day
            g = 3.6e-3 * g_w  # W/m² to MJ/m²*day
            w2 = (
                w10 * 4.87 / np.log(672.58)
            )  # wind speed at 2m above ground (FAO56) [m/s]
            p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26  # atmospheric pressure [kPa]
            gamma = cp_air * p / (h_vap * epsilon)  # psychrometric constant
            delta = (
                4098
                * (0.6108 * np.exp(17.27 * t_air / (t_air + 237.3)))
                / (t_air + 237.3) ** 2
            )  # slope of sat vap press curve

            def vap_pressure(t):
                """Water vapor saturation pressure at specific temperature [kPa]"""
                e = 0.6108 * np.exp(17.27 * t / (t + 237.3))
                return e

            e_s = vap_pressure(t_air)  # saturation vapour pressure [kPa]
            e_a = vap_pressure(
                t_dp
            )  # actual vapour pressure (sat vap press at dewpoint temp) [kPa]

            et_0 = (
                0.408 * delta * (irr - g)
                + gamma * 900 / (t_air + 273) * w2 * (e_s - e_a)
            ) / (
                delta + gamma * 1.34 * w2
            )  # [mm/m²*day]
            et_p = k_c * et_0 / 24  # [mm/m²*h]
            # q_et = et * rho_w * h_vap / 1000  # [W/m²]
            return et_p

        def runoff(p, rcn, **kwargs):
            """Surface runoff of precipitation (ARID) [mm]"""
            s = 25400 / rcn - 254
            i_a = 0.2 * s
            if p > i_a:
                r = (p - i_a) ** 2 / (p + i_a - s) / 24
            else:
                r = 0.0
            return r

        def soil_water_balance(df, has_irrigation, awc, ddc, rzd, **kwargs):
            """
            Soil water content (swc) [-] and irrigation [mm] (if True) for timestep i based on
            precipitation (tp) [mm], surface runoff [mm] for timestep i as well as
            soil water content, deep drainage [mm] and actual evapotranspiration (et_a) [mm] for timestep i-1
            (ARID)
            """
            df["deep_drain"] = 0.0
            df["et_a"] = 0.0
            df["irrigation"] = 0.0
            df["swc"] = 0.0

            def deep_drainage(ddc, rzd, water_cap, water_con_bd, **kwargs):
                """Deep drainage of soil water (ARID) [mm]"""
                if water_con_bd > water_cap:
                    d = ddc * rzd * (water_con_bd - water_cap) / 24
                else:
                    d = 0
                return d

            swc_cache = awc
            df.loc[df.index[0], "swc"] = swc_cache
            for index, row in df.iloc[1:].iterrows():
                df.loc[index, "deep_drain"] = deep_drainage(
                    ddc=ddc, rzd=rzd, water_cap=awc, water_con_bd=swc_cache
                )
                df.loc[index, "et_a"] = min(
                    0.096 * rzd * swc_cache / 24, row["et_p"]
                )  # 0.096 water uptake constant (SIMPLE)

                if has_irrigation:
                    water_deficit = df.loc[index, "et_a"] - (row["tp"] - row["runoff"])
                    df.at[index, "irrigation"] = max(water_deficit, 0)

                delta_swc = (
                    row["tp"]
                    - row["runoff"]
                    - df.loc[index, "deep_drain"]
                    - df.loc[index, "et_a"]
                    + df.loc[index, "irrigation"]
                ) / rzd

                df.loc[index, "swc"] = swc_cache + delta_swc
                swc_cache = df.loc[index, "swc"]

        def drought(et_p, et_a, s_water, **kwargs):
            """Drought stress effect on plant growth (SIMPLE) [-]"""
            arid = 1 - min(et_p, et_a) / et_p
            f_drought = 1 - s_water * arid
            return f_drought

        def faster_senescence(cum_temp, f_heat, f_drought, i50maxh, i50maxw, **kwargs):
            """Heat and drought stress effect on solar interception due to faster canopy senescence (SIMPLE) [-]"""
            if cum_temp < 1:
                delta_i50b = 0
            else:
                delta_i50b = (i50maxh * (1 - f_heat) + i50maxw * (1 - f_drought)) / 24
            return delta_i50b

        def solar_interception(
            cum_temp, delta_i50b, t_sum, i50a, i50b, f_solar_max, **kwargs
        ):
            """
            Interception of incoming radiation by plant canopy
            according to cumulative temperature experienced (SIMPLE) [-]
            """
            cum_temp_to_reach_f_solar_max = i50a - 100 * np.log(
                1 / 999
            )  # derived from f_solar = 0.999f_solar_max
            if cum_temp < 1:
                f_solar = 0
            elif cum_temp < cum_temp_to_reach_f_solar_max:
                i50b += delta_i50b
                f_solar = f_solar_max / (1 + np.exp(-0.01 * (cum_temp - i50a)))
            elif cum_temp < t_sum:
                i50b += delta_i50b
                f_solar = f_solar_max / (1 + np.exp(0.01 * (cum_temp - (t_sum - i50b))))
            else:
                f_solar = 0
            return f_solar

        # Get attributes for biomass modelling from kwargs
        self.elevation = kwargs.pop("elevation")
        self.crop_type = kwargs.pop("crop_type")
        self.sowing_date = kwargs.pop("sowing_date")
        self.harvest_date = kwargs.pop("harvest_date")
        self.has_irrigation = kwargs.pop("has_irrigation")

        # Create DataFrame out of input profiles (time series), time_profile will be set as DatetimeIndex
        profiles_dict = {
            key.replace("_profile", ""): value
            for key, value in kwargs.items()
            if key.endswith("profile")
        }
        time_index = profiles_dict.pop("time")
        profiles_df = pd.DataFrame(data=profiles_dict, index=time_index)

        # Get crop and soil parameters from database, calculate cultivation parameters
        crop_params = crop_dict[self.crop_type]
        soil_params = soil_dict[self.crop_type]
        cultivation_params = specify_cultivation_parameters(
            profiles_df.index, self.sowing_date, self.harvest_date
        )

        # Apply functions for biomass modelling on DataFrame
        profiles_df["cum_temp_base_year"] = profiles_df.apply(
            lambda row: development_base_year(
                row.name, row["t_air"], **crop_params, **cultivation_params
            ),
            axis=1,
        ).cumsum()

        profiles_df["cum_temp_extension"] = profiles_df.apply(
            lambda row: development_extension(
                row.name, row["t_air"], **crop_params, **cultivation_params
            ),
            axis=1,
        ).cumsum()

        cum_temp_base_cache = profiles_df["cum_temp_base_year"].iat[-1]
        cum_temp_ext_cache = profiles_df["cum_temp_extension"].iat[-1]

        profiles_df["cum_temp_cache"] = profiles_df.apply(
            lambda row: development_cache(
                row.name, cum_temp_base_cache, cum_temp_ext_cache, **cultivation_params
            ),
            axis=1,
        )

        profiles_df["cum_temp"] = (
            profiles_df["cum_temp_base_year"]
            + profiles_df["cum_temp_extension"]
            + profiles_df["cum_temp_cache"]
        )

        crop_params.update(
            custom_cultivation_period(profiles_df, **cultivation_params, **crop_params)
        )

        profiles_df["f_temp"] = profiles_df["t_air"].apply(
            lambda t_air: temp(t_air, **crop_params)
        )

        profiles_df["f_heat"] = profiles_df["t_air"].apply(
            lambda t_air: heat(t_air, **crop_params)
        )

        profiles_df["rad_net"] = (
            profiles_df["ghi"] * 0.67
        )  # assuming ground albedo of 0.23 for g and et_p (FAO56)

        profiles_df["g"] = profiles_df.apply(
            lambda row: soil_heat_flux(row["ghi"], row["rad_net"]), axis=1
        )

        profiles_df["et_p"] = profiles_df.apply(
            lambda row: potential_evapotranspiration(
                self.elevation,
                row["t_air"],
                row["t_dp"],
                row["windspeed"],
                row["rad_net"],
                row["g"],
            ),
            axis=1,
        )

        profiles_df["runoff"] = profiles_df["tp"].apply(
            lambda tp: runoff(tp, **soil_params)
        )

        soil_water_balance(profiles_df, self.has_irrigation, **soil_params)

        profiles_df["f_drought"] = profiles_df.apply(
            lambda row: drought(row["et_p"], row["et_a"], **crop_params), axis=1
        )

        profiles_df["delta_i50b"] = profiles_df.apply(
            lambda row: faster_senescence(
                row["cum_temp"], row["f_heat"], row["f_drought"], **crop_params
            ),
            axis=1,
        ).cumsum()

        profiles_df["f_solar"] = profiles_df.apply(
            lambda row: solar_interception(
                row["cum_temp"], row["delta_i50b"], **crop_params
            ),
            axis=1,
        )

        # Set irrigation to 0 outside the cultivation period (f_solar == 0)
        profiles_df.loc[profiles_df["f_solar"] == 0, "irrigation"] = 0

        # Calculate total biomass rate
        profiles_df["total_biomass"] = (
            crop_params["rue"]
            * profiles_df["ghi"]
            * 3.6e-3  # Wh to MJ
            * profiles_df["f_solar"]
            * profiles_df["f_temp"]
            * profiles_df[["f_heat", "f_drought"]].min(axis=1)
            * 1e-3  # g to kg
        )

        # Split into crop harvest and remaining (residual) biomass
        profiles_df["crop_harvest"] = crop_params["hi"] * profiles_df["total_biomass"]
        profiles_df["remaining_biomass"] = (
            profiles_df["total_biomass"] - profiles_df["crop_harvest"]
        )

        # Replace all values smaller than 1e-10 with 1e-10 (incl. negative values), MIMO does not handle values '0' well
        profiles_df = profiles_df.mask((profiles_df <= 1e-5), 1e-10)

        # round all floats to 10 decimal places
        profiles_df = profiles_df.round(10)

        # Update conversion factors for the different inputs and outputs
        conversion_factors = {}
        for key, value in kwargs.items():
            if hasattr(value, "type") and value.type == "bus":
                if key.startswith("solar_energy_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(1)
                if key.startswith("precipitation_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["tp"] / profiles_df["ghi"]  # input data
                    )
                if key.startswith("irrigation_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["irrigation"]
                        / profiles_df["ghi"]  # pre-processed input data
                    )
                if key.startswith("crop_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["crop_harvest"]
                        / profiles_df["ghi"]  # pre-processed input data
                    )
                if key.startswith("biomass_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["remaining_biomass"]
                        / profiles_df["ghi"]  # pre-processed input data
                    )
        kwargs.update(conversion_factors)

        self.solar_energy_bus = kwargs.pop("solar_energy_bus")
        self.precipitation_bus = kwargs.pop("precipitation_bus")
        self.irrigation_bus = kwargs.pop("irrigation_bus")
        self.crop_bus = kwargs.pop("crop_bus")
        self.biomass_bus = kwargs.pop("biomass_bus")

        # Initializes MIMO with the crop-specific buses and conversion_factors
        super().__init__(
            from_bus_0=self.solar_energy_bus,
            from_bus_1=self.precipitation_bus,
            from_bus_2=self.irrigation_bus,
            to_bus_0=self.crop_bus,
            to_bus_1=self.biomass_bus,
            **kwargs,
        )

        # Other mandatory arguments
        self.type = kwargs.pop("type", None)
        self.name = kwargs.pop("name", None)
        self.carrier = kwargs.pop("carrier", None)
        self.tech = kwargs.pop("tech", None)
        self.primary = kwargs.pop("primary", None)
        self.expandable = kwargs.pop("expandable", None)
        self.capacity = kwargs.pop("capacity", None)
        self.capacity_minimum = kwargs.pop("capacity_minimum", None)
        self.capacity_potential = kwargs.pop("capacity_potential", None)
        self.capacity_cost = kwargs.pop("capacity_cost", None)
