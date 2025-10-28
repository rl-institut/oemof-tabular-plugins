from dataclasses import field
from typing import Sequence, Union

import numpy as np
from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade

@dataclass_facade #v1.0   #please check default values once more #improve more pending
class PhotocatalyticUnit(Converter, Facade):
    r""" Photocatalytic water treatment unit with two inputs and one output.
    Catalyst dosing is attached as a parameter to the output water stream.

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
    specific_energy_consumption: float
        Specific electricity demand/consumption in kWh per m³ treated water. Default: 0.12
    Cin: float
        Input pollutant concentration in g per m³ of untreated water (mg/L = g/m³). Default: 10.0
    removal_efficiency: float
        Fraction of pollutants removed from the untreated water.
        Value between 0 and 1. Default: 0.8
    catalyst_dose: float
        Catalyst consumption in g per m³ of treated water (mg/L = g/m³). Default: 10.0
    catalyst_cost: float
        Catalyst cost in USD per kg. Default: 50.0
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

    specific_energy_consumption: float = 0.12  # kWh/m³

    catalyst_dose: float = 10.0  # mg/L = g/m³

    catalyst_cost: float = 50.0  # USD/kg

    removal_efficiency: float = 0.8

    Cin: float = 10.0 # mg/L, user specifies input concentration

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

        # Catalyst cost per m³ of treated water:
        # catalyst_dose [mg/L] * 1e-6 [kg/m³ per mg/L] * catalyst_cost [USD/kg]
        catalyst_cost_per_m3 = (self.catalyst_dose * 1e-6 * self.catalyst_cost)

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
                    variable_costs = catalyst_cost_per_m3 + self.marginal_cost,
                    investment = self._investment(),
                    **self.output_parameters,
                ),
            }
        )

        # Add custom attribute separately
        self.outputs[self.water_out_bus].custom_attributes = {"Cout": Cout,
                                                              "catalyst_dose": self.catalyst_dose}