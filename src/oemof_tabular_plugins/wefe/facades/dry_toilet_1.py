from dataclasses import field
from typing import Sequence, Union

import numpy as np
from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade

@dataclass_facade  # v1.0   #please check default values once more #improve more pending
class Dry_Toilet(Converter, Facade):
    r""" Dry toilet unit with human feces and human urine as input, bulking agent as a
    dosing parameter and, dry feces and leachate as the output.
    The fractions and relations need to be explicitly provided for more accuracy.

    Parameters
    ----------
    human_feces_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its human feces input.
    human_urine_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its human urine input.
    dry_feces_out_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its dry feces output.
    water_out_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its separated water/leachate output.
    bulking_agent_dose: float
        Bulking agent dosage in kg/kg of dry feces produced. Default: 0.25
    bulking_agent_cost: float
        Bulking agent cost in USD per kg. Default: 0.1
    urine_dry_feces_fraction: float
        Human urine that is required in m³ per kg of dry feces. Default: 2.0
    wet_feces_dry_feces_fraction: float
        Human feces that is required in kg per kg of dry feces. Default: 3.3
    leachate_dry_feces_relation: float
        Leachate produced in m³ along with 1 kg of dry feces. Default: 0.1
    capacity: numeric
        The capacity (output side) of the unit.
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

    human_feces_bus: Bus

    human_urine_bus: Bus

    water_out_bus: Bus

    dry_feces_out_bus: Bus

    tech: str

    carrier: str = ""

    bulking_agent_dose: float = 0.25 # mg/L = g/m³

    bulking_agent_cost: float = 0.1 # USD/kg

    urine_dry_feces_fraction: float = 2.0

    wet_feces_dry_feces_fraction: float = 3.3

    leachate_dry_feces_relation: float = 0.1

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

        # Bulking agent cost per kg of compost produced:
        # bulking_agent_dose [kg/kg] * bulking_agent_cost [USD/kg]
        bulking_agent_cost_per_kg_compost = self.bulking_agent_dose * self.bulking_agent_cost

        self.conversion_factors.update(
            {
                self.human_feces_bus: sequence(self.wet_feces_dry_feces_fraction),
                self.human_urine_bus: sequence(self.urine_dry_feces_fraction),
                self.dry_feces_out_bus: sequence(1),
                self.water_out_bus: sequence(self.leachate_dry_feces_relation),
            }
        )

        self.inputs.update(
            {
                self.human_feces_bus: Flow(),
                self.human_urine_bus: Flow(),
            }
        )

        self.outputs.update(
            {
                self.water_out_bus: Flow(
                    nominal_value = self._nominal_value(),
                    variable_costs = self.marginal_cost,
                    investment = self._investment(),
                    **self.output_parameters,
                ),
                self.dry_feces_out_bus: Flow(
                    nominal_value = self._nominal_value(),
                    variable_costs = bulking_agent_cost_per_kg_compost + self.marginal_cost,
                    investment = self._investment(),
                    **self.output_parameters,
                ),
            }
        )

        # Add custom attribute separately
        self.outputs[self.dry_feces_out_bus].custom_attributes = {"bulking_agent_dose": self.bulking_agent_dose}
