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
from oemof_tabular_plugins.wefe.facades import functions as f
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
    sowing_date: str
        sowing date in MM-DD format
    harvest_date: str
        harvest date in MM-DD format
    ghi: time series
        Global horizontal irradiance
    et_0: time series
        potential evapotranspiration [m³]
    t_air: time series
        Ambient air temperature
    vwc: time series
        the volumetric water content (vwc) in the root zone depth in m³; metric to express soil moisture
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

    sowing_date: str = ""  # MM-DD

    harvest_date: str = ""  # MM-DD

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
        timeindex,
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
        timeindex: time as pd.series or list
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
        if not isinstance(timeindex, (list, pd.Series)):
            print("Argument 'timeindex' is not of type list or pd.Series!")
        if len(t_air) != len(timeindex):
            raise ValueError("Length mismatch between t_air and timeindex profiles.")

        # Convert dates to Timestamp objects matching the time index
        dates = list(pd.to_datetime(time_index))
        # Adapt sowing and harvest date to time index, specify them if not provided
        cultivation_params = f.specify_cultivation_parameters(
            dates=dates, sowing_date=sowing_date, harvest_date=harvest_date
        )
        sowing_date = cultivation_params.pop("sowing_date")
        harvest_date = cultivation_params.pop("harvest_date")

        # Create three seperate lists for cumulative temperature
        delta_tt_base_list = []  # creating a list
        for temp, date in zip(t_air, time_index):
            delta_tt_base = f.tt_base(
                date=date, t_air=temp, sowing_date=sowing_date, t_base=t_base
            )
            delta_tt_base_list.append(delta_tt_base)
        tt_base_list = np.cumsum(delta_tt_base_list)

        delta_tt_ext_list = []  # creating a list
        for temp, date in zip(t_air, time_index):
            delta_tt_ext = f.tt_extension(
                date=date,
                t_air=temp,
                sowing_date=sowing_date,
                harvest_date=harvest_date,
                t_base=t_base,
            )
            delta_tt_ext_list.append(delta_tt_ext)
        tt_ext_list = np.cumsum(delta_tt_ext_list)

        tt_base_cache = tt_base_list[-1]
        tt_ext_cache = tt_ext_list[-1]

        tt_cache_list = []  # creating a list
        for date in time_index:
            delta_tt_cache = f.tt_cache(
                date=date,
                cum_temp_base_cache=tt_base_cache,
                cum_temp_ext_cache=tt_ext_cache,
                harvest_date=harvest_date,
                sowing_date=sowing_date,
            )
            tt_cache_list.append(delta_tt_cache)

        # Add the three lists together to get total cumulative temperature
        cumulative_temp = (
            np.array(tt_base_list) + np.array(tt_ext_list) + np.array(tt_cache_list)
        )

        # Update t_sum if custom_harvest = True (custom harvest date provided instead of maturity according to SIMPLE)
        has_custom_harvest = cultivation_params.pop("has_custom_harvest")
        t_sum = (
            cumulative_temp[dates.index(harvest_date)] if has_custom_harvest else t_sum
        )

        # f_solar(cum_temp) according to SIMPLE
        f_solar_list = []
        for cum_temp in cumulative_temp:
            f_solar = f.f_solar(
                cum_temp=cum_temp,
                delta_i50b=0,
                t_sum=t_sum,
                i50a=i50a,
                i50b=i50b,
                f_solar_max=f_solar_max,
            )
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
        f_temp_list = []  # creating a list
        # Calculate te
        for t in t_air:
            f_temp_list.append(f.f_temp(t=t, t_opt=t_opt, t_base=t_base))

        return np.array(f_temp_list)

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
        f_heat_list = []  # creating a list

        if n_days == 0:
            # there is less than a day, returing same value n_timesteps times
            n_timesteps = n_hours
            t_max = np.max(t_air)
            f_heat_list.append(
                np.ones(n_timesteps) * f.f_heat(t=t_max, t_max=t_heat, t_ext=t_ext)
            )
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
                f_heat_list.append(
                    np.ones(n_timesteps) * f.f_heat(t=t_max, t_max=t_heat, t_ext=t_ext)
                )

        return np.hstack(f_heat_list)

    @property
    def efficiency(self):
        """ """
        crop_params = crop_dict[self.crop_type]
        fTEMP = self.calc_Ftemp(self.t_air, **crop_params)
        fWATER = self.calc_Fwater(self.et_0, self.vwc, **crop_params)
        fHEAT = self.calc_Fheat(t_air=self.t_air, **crop_params)
        fSOLAR = self.calc_Fsolar(
            timeindex=self.time_index,
            t_air=self.t_air,
            sowing_date=self.sowing_date,
            harvest_date=self.harvest_date,
            **crop_params,
        )

        rue = self.get_crop_param("rue")

        return (
            rue
            * fSOLAR
            * fTEMP
            * np.minimum(fWATER, fHEAT)
            * f.C_WH_TO_J
            * f.C_J_TO_MJ
            * f.C_G_TO_KG
        )

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
    """
    Crop model converter with 3 inputs (irradiation, precipitation, irrigation)
    and 2 outputs (crop harvest, remaining biomass), based on MultipleInputMultipleOutputConverter.

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

    def __init__(self, **attributes):
        """
        Preprocessing of crop input data to calculate conversion_factors
        and assign these together with the correct busses to MIMO parent class
        """
        # Create DataFrame out of input profiles (time series), time_profile will be set as DatetimeIndex
        profiles_dict = {
            key.replace("_profile", ""): value
            for key, value in attributes.items()
            if key.endswith("profile")
        }
        time_index = profiles_dict.pop("time")
        profiles_df = pd.DataFrame(data=profiles_dict, index=time_index)

        # Get crop and soil parameters from database, calculate cultivation parameters
        self.crop_type = attributes.pop("crop_type")
        crop_params = crop_dict[self.crop_type]
        soil_params = soil_dict[self.crop_type]
        cultivation_params = f.specify_cultivation_parameters(
            profiles_df.index, **attributes
        )

        # Apply geometry-independent crop model functions to profiles
        f.calc_cumulative_temperature(profiles_df, **crop_params, **cultivation_params)
        f.calc_f_temp(profiles_df, **crop_params)
        f.calc_f_heat(profiles_df, **crop_params)

        # Update t_sum of crop_params if custom_harvest is True
        crop_params.update(
            f.custom_cultivation_period(
                profiles_df, **cultivation_params, **crop_params
            )
        )

        # Calculate irrigation and total biomass yield
        f.calc_f_water(
            profiles_df,
            has_rainwater_harvesting=False,
            frt=1,
            gcr=0,
            **attributes,
            **soil_params,
            **crop_params,
        )
        f.calc_f_solar(profiles_df, **crop_params)
        f.adapt_irrigation(profiles_df)
        f.calc_biomass(profiles_df, frt=1, **crop_params)

        # Format the profiles (no 0 allowed, no negative values, round to 10 decimal places)
        profiles_df = profiles_df.mask((profiles_df <= 1e-5), 1e-10)
        profiles_df = profiles_df.round(10)

        # Update conversion factors for the different inputs and outputs
        conversion_factors = {}
        for key, value in attributes.items():
            if hasattr(value, "type") and value.type == "bus":
                if key.startswith("solar_energy_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["ghi"]
                    )
                if key.startswith("precipitation_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["tp"]
                    )
                if key.startswith("irrigation_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["irrigation"]
                    )
                if key.startswith("crop_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        crop_params["hi"] * profiles_df["total_biomass"]  # crop harvest
                    )
                if key.startswith("biomass_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        (1 - crop_params["hi"])
                        * profiles_df["total_biomass"]  # residual biomass
                    )
        attributes.update(conversion_factors)

        self.solar_energy_bus = attributes.pop("solar_energy_bus")
        self.precipitation_bus = attributes.pop("precipitation_bus")
        self.irrigation_bus = attributes.pop("irrigation_bus")
        self.crop_bus = attributes.pop("crop_bus")
        self.biomass_bus = attributes.pop("biomass_bus")

        # Initializes MIMO with the crop-specific buses and conversion_factors
        super().__init__(
            from_bus_0=self.solar_energy_bus,
            from_bus_1=self.precipitation_bus,
            from_bus_2=self.irrigation_bus,
            to_bus_0=self.crop_bus,
            to_bus_1=self.biomass_bus,
            **attributes,
        )

        # Other mandatory arguments
        self.type = attributes.pop("type", None)
        self.name = attributes.pop("name", None)
        self.carrier = attributes.pop("carrier", None)
        self.tech = attributes.pop("tech", None)
        self.primary = attributes.pop("primary", None)
        self.expandable = attributes.pop("expandable", None)
        self.capacity = attributes.pop("capacity", None)
        self.capacity_minimum = attributes.pop("capacity_minimum", None)
        self.capacity_potential = attributes.pop("capacity_potential", None)
        self.capacity_cost = attributes.pop("capacity_cost", None)
