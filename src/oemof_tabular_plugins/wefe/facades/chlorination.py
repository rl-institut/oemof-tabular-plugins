from dataclasses import field
from typing import Sequence, Union

import numpy as np
from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade

@dataclass_facade #v1.0   #please check default values once more #improve more pending
class Chlorination(Converter, Facade):
    r""" Chlorination water treatment unit with two inputs and one output.
    Chlorine dosing is attached as a parameter to the output water stream.

    Parameters
    ----------
    electricity_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its electricity input.
    water_in_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its untreated water input.
    water_out_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its treated water output.
    specific_energy_consumption: float
        Specific electricity demand/consumption in kWh per m³ treated water. Default: 0.05
    chlorine_dose: float
        Chlorine dosage in mg/L of treated water (= g/m³). Default: 1.0
    chlorine_cost: float
        Chlorine cost in USD per kg. Default: 0.5
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

    tech: str

    carrier: str = ""

    specific_energy_consumption: float = 0.05  # kWh/m³

    chlorine_dose: float = 1.0  # mg/L ≡ g/m³

    chlorine_cost: float = 0.5  # USD/kg

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

        # Chlorine cost per m³ of treated water:
        # chlorine_dose [mg/L] * 1e-6 [kg/m³ per mg/L] * chlorine_cost [USD/kg]
        chlorine_cost_per_m3 = (self.chlorine_dose * 1e-6 * self.chlorine_cost)

        self.conversion_factors.update(
            {
                self.electricity_bus: sequence(self.specific_energy_consumption),
                self.water_in_bus: sequence(1),
                self.water_out_bus: sequence(1),
            }
        )

        self.inputs.update(
            {
                self.electricity_bus: Flow(
                    variable_costs=self.carrier_cost, **self.input_parameters
                ),
                self.water_in_bus: Flow(),
            }
        )

        self.outputs.update(
            {
                self.water_out_bus: Flow(
                    nominal_value = self._nominal_value(),
                    variable_costs = chlorine_cost_per_m3 + self.marginal_cost,
                    investment = self._investment(),
                    **self.output_parameters,
                ),
            }
        )

        # Add custom attribute separately
        self.outputs[self.water_out_bus].custom_attributes = {"chlorine_dose": self.chlorine_dose}