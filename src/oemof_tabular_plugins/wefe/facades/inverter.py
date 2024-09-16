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
class Inverter(Converter, Facade):
    r"""Crop Model Converter with one input and one output. The crop growth factor
    is calculated out drought, heat, temperature, and water availabilty impact
     and considered for biomass production calculation.

    Parameters
    ----------
    crop_type: str
        the name of crop as defined in global_specs/crops.py
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

    dc_electricity_bus: Bus

    ac_electricity_bus: Bus

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

    efficiency: float = 0.95

    load_profile: Union[float, Sequence[float]] = None

    @classmethod
    def validate_datapackage(self, resource):
        # function to apply on datapackage

        return resource

    @classmethod
    def processing_raw_inputs(self, resource, results_df):
        # function to apply on df from above

        return results_df

    @property
    def capacity(self):
        capacity = np.max(self.load_profile) / self.efficiency

        return capacity
        # not sure how to properly set the self.capacity attribute

    def build_solph_components(self):
        """ """
        self.conversion_factors.update(
            {
                self.dc_electricity_bus: sequence(1),
                self.ac_electricity_bus: sequence(abs(self.efficiency)),
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


