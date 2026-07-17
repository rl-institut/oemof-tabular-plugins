from dataclasses import field
from typing import Sequence, Union

import numpy as np
from typing import Union, Sequence
from dataclasses import field
from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade

@dataclass_facade
class Digester(Converter, Facade):
    r""" A digester unit with 1 input and 2 outputs. The input is biomass
    while the outputs are rawbiogas and digestate.

    Parameters
    ----------
    biomass_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its biomass input.
    rawbiogas_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its rawbiogas output.
    efficiency: float
        Conversion efficiency from biomass to rawbiogas (e.g., energy or mass ratio).
        Default: 0.25
    capacity: numeric
        The rawbiogas production capacity (primary output side) of the unit.
    carrier_cost: numeric
        Carrier cost for one unit of used input (biomass). Default: 0
    capacity_cost: numeric
        Investment costs per unit of rawbiogas output capacity.
        If capacity is not set, this value will be used for optimizing the
        conversion output capacity.
    expandable: boolean or numeric (binary)
        True, if capacity can be expanded within optimization. Default: False.
    lifetime: int (optional)
        Lifetime of the component in years. Necessary for multi-period
        investment optimization. Default: None.
    age : int (optional)
        The initial age of a component (usually given in years). Default: 0.
    fixed_costs : numeric (iterable or scalar) (optional)
        The fixed operational costs associated with a component. Default: None.
    capacity_potential: numeric
        Maximum invest capacity in unit of output capacity. Default: +inf.
    input_parameters: dict (optional)
        Set parameters on the input edge of the conversion unit.
    output_parameters: dict (optional)
        Set parameters on the output edge of the conversion unit.
    """

    biomass_bus: Bus

    rawbiogas_bus: Bus

    digestate_bus: Bus

    tech: str

    carrier: str = ""

    efficiency: float = 0.25 # rawbiogas efficiency

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

        # In oemof, conversion factors are defined relative to a nominal reference flow.
        # Since the nominal capacity is bound to the primary output (rawbiogas),
        # we define the ratios relative to the biomass input or scale them to the main output.
        # Here, we treat biomass as the base (1), and outputs as fractions of that input.
        self.conversion_factors.update(
            {
                self.biomass_bus: sequence(1),
                self.rawbiogas_bus: sequence(self.efficiency),
                self.digestate_bus: sequence(1-(self.efficiency)),
            }
        )

        self.inputs.update(
            {
                self.biomass_bus: Flow(
                    variable_costs = self.carrier_cost, **self.input_parameters
                ),
            }
        )

        self.outputs.update(
            {
                self.rawbiogas_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost,
                    investment=self._investment(),
                    **self.output_parameters,
                ),
                self.digestate_bus: Flow(),
            }
        )