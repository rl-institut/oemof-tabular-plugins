from dataclasses import field
from typing import Sequence, Union

import numpy as np
from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade

@dataclass_facade #v1.0   #please check default values once more #improve more pending
class Ozonation(Converter, Facade):
    r""" Ozonation water treatment unit with two inputs and one output.
    Ozone dosing is attached as a parameter to the output water stream.

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
    ozone_dose: float
        Ozone dosage in mg/L of treated water (= g/m³). Default: 1.0
    specific_energy_per_kg_ozone: float
        Electricity consumption per kg of ozone generated (kWh/kg). Default: 12.0
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

    ozone_dose: float = 1.0  # mg/L ≡ g/m³

    specific_energy_per_kg_ozone: float = 12.0  # kWh/kg ozone generated

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

        # Electricity demand per m³ of treated water
        # ozone_dose [mg/L] * 1e-6 [kg/m³ per mg/L] * specific_energy_per_kg_ozone [kWh/kg]
        electricity_demand_per_m3 = (self.ozone_dose * 1e-6 * self.specific_energy_per_kg_ozone)

        self.conversion_factors.update(
            {
                self.electricity_bus: sequence(electricity_demand_per_m3),
                self.water_in_bus: sequence(1),
                self.water_out_bus: sequence(1),
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

        self.outputs.update(
            {
                self.water_out_bus: Flow(
                    nominal_value = self._nominal_value(),
                    variable_costs = self.marginal_cost,
                    investment = self._investment(),
                    ozone_dose = self.ozone_dose,
                    **self.output_parameters,
                ),
            }
        )