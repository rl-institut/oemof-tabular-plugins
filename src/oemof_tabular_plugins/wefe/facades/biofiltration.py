from dataclasses import field
from typing import Sequence, Union

import numpy as np
from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade

@dataclass_facade #v1.0   #please check default values once more #improve more pending
class BioFiltration(Converter, Facade):
    r"""Biofiltration water treatment unit with two inputs and two outputs.
    Add parameters to the waste biomass output stream as and when required later.
    Update self.conversion_factor.update() as and when required if considered MIMO.

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
    waste_biomass_out_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its waste/spent biomass output.
    specific_energy_consumption: float
        Specific electricity demand/consumption in kWh per m³ treated water. Default: 0.12.
    efficiency: float
        Share of untreated water that becomes treated water.
        Value between 0 and 1. Default: 0.85
    biomass_waste_fraction: float
        Share of treated water that becomes waste biomass.
        value between 0 and 1. Default: 0.005
    nutrient_dose: float
        Nutrient consumption in g per m³ of treated water (mg/L = g/m³). Default: 1.0
    nutrient_cost: float
        Nutrient cost in USD per kg. Default: 1.0
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

    waste_biomass_out_bus: Bus

    tech: str

    carrier: str = ""

    specific_energy_consumption: float = 0.12  # kWh/m³

    efficiency: float = 0.85

    nutrient_dose: float = 1.0  # mg/L = g/m³

    nutrient_cost: float = 1.0  # USD/kg

    biomass_waste_fraction: float = 0.005

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

        # Nutrient cost per m³ of treated water:
        # nutrient_dose [mg/L] * 1e-6 [kg/m³ per mg/L] * nutrient_cost [USD/kg]
        nutrient_cost_per_m3 = (self.nutrient_dose * 1e-6 * self.nutrient_cost)

        self.conversion_factors.update(
            {
                # Electricity input per unit treated water output
                self.electricity_bus: sequence(self.specific_energy_consumption),
                # Raw water input per unit treated water output (inverse of efficiency)
                self.water_in_bus: sequence(1/self.efficiency),
                # Treated water output normalized to 1
                self.water_out_bus: sequence(1),
                self.waste_biomass_out_bus: sequence(self.biomass_waste_fraction),
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
                    variable_costs = nutrient_cost_per_m3 + self.marginal_cost,
                    investment = self._investment(),
                    **self.output_parameters,
                ),
                self.waste_biomass_out_bus: Flow(),
            }
        )

        # Add custom attribute separately
        self.outputs[self.water_out_bus].custom_attributes = {"nutrient_dose": self.nutrient_dose}