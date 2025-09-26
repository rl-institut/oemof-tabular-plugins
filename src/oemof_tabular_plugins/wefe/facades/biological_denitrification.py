from dataclasses import field
from typing import Sequence, Union

import numpy as np
from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade

@dataclass_facade
class BiologicalDenitrification(Converter, Facade):
    """Biological denitrification water treatment unit with two inputs and two outputs.
    Reduces nitrate and nitrogen oxide ions in the water and converts them into nitrogen gas
    with the help of a suitable carbon source. The outputs are treated water and nitrogen gas.

    Parameters
    ----------
    electricity_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its electricity input.
    water_in_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its untreated water input.
    water_out_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its treated water output.
    N2_gas_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its nitrogen gas output.
    Cin: float
        Input pollutant concentration in g per m³ of untreated water (mg/L = g/m³). Default: 30.0
    removal_efficiency: float
        Fraction of nitrate and nitrogen oxide components removed from the untreated water.
        Value between 0 and 1. Default: 0.90
    specific_energy_consumption: float
        Specific electricity demand/consumption in kWh per m³ treated water. Default: 0.005
    carbon_source_dose: float
        Carbon source consumption in g per m³ of treated water (mg/L = g/m³). Default: 90
    carbon_source_cost: float
        Carbon source cost in USD per kg. Default: 0.40
    capacity: numeric
        The water treatment capacity (output side) of the unit.
    carrier_cost: numeric
        Carrier cost for one unit of used input. Default: 0
    capacity_cost: numeric
        Investment costs per unit of output capacity.
        If capacity is not set, this value will be used for optimizing the
        conversion output capacity.
    expandable: boolean or numeric (binary)
        True, if capacity can be expanded within optimization. Default: False.
    lifetime: int (optional)
        Lifetime of the component in years. Necessary for multi-period
        investment optimization.
        Note: Only applicable for a multi-period model. Default: None.
    age : int (optional)
        The initial age of a component (usually given in years);
        once it reaches its lifetime (considering also
        an initial age), the component is forced to 0 (retire/replace).
        Note: Only applicable for a multi-period model. Default: 0.
    fixed_costs : numeric (iterable or scalar) (optional)
        The fixed operational costs associated with a component.
        Note: Only applicable for a multi-period model. Default: None.
    capacity_potential: numeric
        Maximum invest capacity in unit of output capacity. Default: +inf.
    input_parameters: dict (optional)
        Set parameters on the input edge of the conversion unit
         (see oemof.solph for more information on possible parameters)
    output_parameters: dict (optional)
        Set parameters on the output edge of the conversion unit
         (see oemof.solph for more information on possible parameters)
    """

    electricity_bus: Bus

    water_in_bus: Bus

    water_out_bus: Bus

    N2_gas_bus: Bus

    specific_energy_consumption: float = 0.005 # kWh/m³

    carbon_source_dose: float = 90 # mg/L = g/m³

    carbon_source_cost: float = 0.40 # USD/kg

    removal_efficiency: float = 0.90

    Cin: float = 30.0 # mg/L, user specifies input concentration

    capacity: float = None

    marginal_cost: float = 0

    carrier_cost: float = 0

    capacity_cost: float = None

    expandable: bool = False

    lifetime: int = None

    age: int = 0

    fixed_costs: Union[float, Sequence[float]] = None

    capacity_potential: float = float("+inf")

    input_parameters: dict = field(default_factory=dict)

    output_parameters: dict = field(default_factory=dict)

    def build_solph_components(self):

        # Carbon source dose per m³ of treated water:
        # carbon_source_dose [mg/L] * 1e-6 [kg/m³ per mg/L] * carbon_source_cost [USD/kg]
        carbon_source_cost_per_m3 = (self.carbon_source_dose * 1e-6 * self.carbon_source_cost)

        self.conversion_factors.update(
            {
                self.electricity_bus: sequence(self.specific_energy_consumption),
                self.water_in_bus: sequence(1),
                self.water_out_bus: sequence(1),
                self.N2_gas_bus: sequence(self.Cin * self.removal_efficiency), #  g N2/m³ of treated water
            }
        )

        self.inputs.update(
            {
                self.electricity_bus: Flow(
                    variable_costs = self.carrier_cost, **self.input_parameters
                ),
                self.water_in_bus: Flow(),
            }
        )

        # Calculate output concentration based on input and efficiency
        Cout = self.Cin * (1 - self.removal_efficiency)

        self.outputs.update(
            {
                self.water_out_bus: Flow(
                    nominal_value = self._nominal_value(),
                    variable_costs = carbon_source_cost_per_m3 + self.marginal_cost,
                    investment = self._investment(),
                    Cout = Cout,
                    carbon_source_dose = self.carbon_source_dose,
                    **self.output_parameters,
                ),
                self.nitrogen_gas_bus: Flow(),
            }
        )