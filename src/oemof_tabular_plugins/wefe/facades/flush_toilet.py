from dataclasses import field
from typing import Sequence, Union

import numpy as np
from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade

@dataclass_facade #v1.0   #please check default values once more #improve more pending
class FlushToilet(Converter, Facade):
    r""" Flush toilet unit with human feces, human urine and service water as the input
    and a combined blackwater as the output. No electricity as it is assumed to be gravity fed.

    Parameters
    ----------
    water_in_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its service water input.
    water_out_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its black water output.
    human_feces_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its human feces input.
    human_urine_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its human urine input.
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

    water_in_bus: Bus

    human_feces_bus: Bus

    human_urine_bus: Bus

    water_out_bus: Bus

    tech: str

    carrier: str = ""

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

        # Assume volume conservation: sum of feces + urine + service water = blackwater output volume

        self.conversion_factors.update(
            {
                self.human_feces_bus: sequence(1),
                self.human_urine_bus: sequence(1),
                self.water_in_bus: sequence(1),
                self.water_out_bus: sequence(3),
            }
        )

        self.inputs.update(
            {
                self.water_in_bus: Flow(),
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
            }
        )