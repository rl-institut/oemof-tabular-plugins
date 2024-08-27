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

    t_air: Union[float, Sequence[float]] = None

    ghi: Union[float, Sequence[float]] = None

    et_0: Union[float, Sequence[float]] = None

    vwc: Union[float, Sequence[float]] = None

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

    def calc_te(self, t_air, t_opt, t_base, rue, **kwargs):
        r"""
        Calculates the temperature on the biomass rate
        ----
        Parameters
        ----------
        t_air: ambient temperature as pd.series or list
        t_opt: optimum temperature for biomass growth
        t_base: base temperature for biomass growth
        rue: Radiation use efficiency (above ground only and without respiration
        c_wh_to_mj: conversion factor Watt Hours (WH) to Mega Joules (MJ)
        Returns
        -------
        te : list of numerical values:
             temperature coefficients for calculating biomass rate

        """
        # Introduce conversion factor Watt Hours (WH) to Mega Joules (MJ)
        c_wh_to_mj = 3.6 * 10 ** (-3)
        # Check if input arguments have proper type and length
        if not isinstance(t_air, (list, pd.Series)):
            t_air = [t_air]
            print("Argument 'temp' is not of type list or pd.Series!")
        te = []  # creating a list
        # Calculate te
        for t in t_air:
            if t < t_base:
                x = 0
                te.append(x)

            elif t_base <= t <= t_opt:
                x = (t - t_base) / (t_opt - t_base) * rue * c_wh_to_mj
                te.append(x)

            elif t > t_opt:
                x = 1 * rue * c_wh_to_mj
                te.append(x)
        return np.array(te)

    def calc_arid(self, et_o, vwc, s_water, rzd, **kwargs):
        r"""
        Calculates the soil water availability impact on the biomass rate
        arid factor derived from simple crop model and Woli et al. (2012), divided by 24 to transform to hourly values
        ----
        Parameters
        ----------
        et_o: potential evapotranspiration
        vwc: volumetric water content
        rzd: root zone depth

        Returns
        -------
        arid : list of numerical values:
             aridity factor affecting the biomass rate

        """
        et_o = np.array(et_o)
        vwc = np.array(vwc)

        # Calculate arid

        # TODO why the loop as wi is constantly overwritten ...?
        wi = []
        for e in et_o:
            for m in vwc:
                wi = 1 - s_water * (1 - min(abs(e), (0.096 / 24) * m * rzd) / abs(e))
        # TODO I think this should read like that
        wi = 1 - s_water * (
            1 - np.minimum(abs(et_o), (0.096 / 24) * vwc * rzd) / abs(et_o)
        )
        return wi

    def calc_hi(self, t_max, t_opt, t_ext, **kwargs):
        r"""
        Calculates the heat impact on the biomass rate
        ----
        Parameters
        ----------
        t_max: daily maximum temperature as pd.series or list
        t_ext: threshold temperature when biomass growth rate starts to be reduced by heat stress
        t_opt: optimum temperature for biomass growth
        Returns
        -------
        hi : list of numerical values:
             temperature coefficients for calculating biomass rate

        """

        # Check if input arguments have proper type and length
        if not isinstance(t_max, (list, pd.Series)):
            t_max = [t_max]
            # raise TypeError("Argument t_max is not of type list or pd.Series!")
        hi = []  # creating a list
        # Calculate te
        for t in t_max:
            if t <= t_opt:
                x = 1
                hi.append(x)

            elif t_opt < t <= t_ext:
                x = (t - t_opt) / (t_ext - t_opt)
                hi.append(x)

            elif t > t_ext:
                x = 0
                hi.append(x)

        return np.array(hi)

    @property
    def efficiency(self):

        crop_params = crop_dict[self.crop_type]
        te = self.calc_te(self.t_air, **crop_params)
        arid = self.calc_arid(self.et_0, self.vwc, **crop_params)
        hi = self.calc_hi(**crop_params)
        return te * arid * hi

    def build_solph_components(self):
        """ """

        self.conversion_factors.update(
            {
                self.from_bus: sequence(1),
                self.to_bus: sequence(self.efficiency),
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



