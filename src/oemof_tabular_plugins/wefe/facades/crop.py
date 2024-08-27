from dataclasses import field
from typing import Sequence, Union

from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import Facade, dataclass_facade


@dataclass_facade
class Crop(Converter, Facade):
    r"""Crop Model Converter with one input and one output. The crop growth factor
    is calculated out drought, heat, temperature, and water availabilty impact
     and considered for biomass production calculation.

    Parameters
    ----------
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
    t_air: time series
        Ambient air temperature
    vwc: time series
        the voluemtric water content (vwc) in the root zone depth in m³; metric to express soil moisture
    ghi: time series
        Global horizontal irradiance
    et_0: time series
        potential evapotranspiration [m³]

    SHOULD INCLUDE FUNCTIONS AND EXAMPLE HERE

    """
    def _crop_production(crop_dict, crop_df):
        """
        Calculate geometry parameters, performance indicators and full-year hourly conversion factor time series
        """
    crop_type = crop_dict['crop_type']
    minimum_relative_yield = crop_dict['minimum_relative_yield']

    # ----- Crop specific growth parameters from crop_dict -----
    t_base = crop_dict[crop_type]['t_base']  # minimum temperature for growth, impaired growth
    t_opt = crop_dict[crop_type]['t_opt']  # optimal temperature for growth
    t_heat = crop_dict[crop_type]['t_max']  # heat stress begins, impaired growth
    t_extreme = crop_dict[crop_type]['t_ext']  # extreme heat stress, no growth
    t_sum = crop_dict[crop_type]['t_sum']  # cumulative temperature until harvest
    i50a = crop_dict[crop_type]['i50a']
    i50b = crop_dict[crop_type]['i50b']
    i50maxh = crop_dict[crop_type]['i50maxh']
    i50maxw = crop_dict[crop_type]['i50maxw']
    s_water = crop_dict[crop_type]['s_water']
    f_solar_max = crop_dict[crop_type]['f_solar_max']
    rue = crop_dict[crop_type]['rue']
    # Convert Radiation Use Efficiency from g/(MJ*m²*h) to g/(W*m²)
    rue *= 3.6e-3

    # ----- Crop specific soil parameters from soil_dict -----
    wuc = 0.096  # alpha      ## water uptake constant
    rzd = soil_dict[crop_type]['rzd']  # zeta       ## root zone depth [mm]
    awc = soil_dict[crop_type]['rzd']  # theta_m    ## water holding capacity
    rcn = soil_dict[crop_type]['rzd']  # eta        ## runoff curve number
    ddc = soil_dict[crop_type]['rzd']  # beta       ## deep drainage coefficient

# TODO @PF do we need this kind of data type determination here?
    def development_base_year(date, sowing_date, t_air, t_base):
        """
        Cumulative temperature experienced by plant as measure for plant development (SIMPLE) [K]
        from sowing_date until end of the same year (base year)
        """
        if date < sowing_date:
            delta_cum_temp = 0
        elif t_air > t_base:
            delta_cum_temp = (t_air - t_base) / 24  # SIMPLE crop model has daily temp values, convert to hourly
        else:
            delta_cum_temp = 0
        return delta_cum_temp

    def development_extension(date, harvest_date, t_air, t_base):
        """
        Additional cumulative temperature experienced by plant as measure for plant development
        if growth extends to following year (harvest_date < sowing_date if year ignored) [K]
        """
        if date > harvest_date:
            delta_cum_temp = 0
        elif t_air > t_base:
            delta_cum_temp = (t_air - t_base) / 24  # SIMPLE crop model has daily temp values, convert to hourly
        else:
            delta_cum_temp = 0
        return delta_cum_temp

    def development_cache(date, harvest_date, cum_temp_base_cache, cum_temp_ext_cache):
        """
        Cumulative temperature experienced in the base year cached for extension in the following year,
        cumulative temperature experienced in the following year until harvest_date removed afterwards [K]
        """
        if date <= harvest_date:
            delta_cum_temp = cum_temp_base_cache
        else:
            delta_cum_temp = -cum_temp_ext_cache
        return delta_cum_temp

    def custom_cultivation_period(df, harvest_date):
        """ Calculates new t_sum for plant growth curve if custom harvest_date is given [K] """
        if harvest_date in crop_df.index:
            return df.loc[harvest_date, 'cum_temp']
    def temp(t_air, t_opt, t_base):
        """ Temperature effect on plant growth (SIMPLE) [-] """
        if t_air < t_base:
            f_temp = 0
        elif t_base <= t_air < t_opt:
            f_temp = (t_air - t_base) / (t_opt - t_base)
        else:
            f_temp = 1
        return f_temp

    def heat(t_air, t_heat, t_extreme):
        """ Heat stress effect on plant growth (SIMPLE) [-] """
        if t_air <= t_heat:
            f_heat = 1
        elif t_heat < t_air <= t_extreme:
            f_heat = 1 - (t_air - t_heat) / (t_extreme - t_heat)
        else:
            f_heat = 0
        return f_heat


    def build_solph_components(self):
        """ """
        # assign the air temperature and solar irradiance
        t_air_values = self.t_air
        ghi_values = self.ghi



        # calculates the crop growth factors
        pv_tf_values = []
        for t_air, ghi in zip(t_air_values, ghi_values):
            t_c = t_air + ((self.noct - 20) / 800) * ghi
            pv_tf = (
                self.p_rpv * (1 / self.r_ref) * (1 + self.n_t * (t_c - self.t_c_ref))
            )
            pv_tf_values.append(pv_tf)

        self.conversion_factors.update(
            {
                self.from_bus: sequence(1),
                self.to_bus: sequence(pv_tf_values),
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

        def processing_raw_inputs(self, resource, results_df):
            # function to apply on df from above

            return results_df

        def validate_datapackage(self, resource):
            # modify the resource (datapackage.resource)
            # should it return the resource?
            pass
