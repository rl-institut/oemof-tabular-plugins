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
class BiogasUpgrading(Converter, Facade):
    r""" A biogas upgrading unit with 1 input and 2 outputs. The input is rawbiogas from the
    digester while the outputs are captured co2 and biomethane.

    Parameters
    ----------
    rawbiogas_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its rawbiogas input.
    captured_co2_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its captured co2 output.
    biomethane_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its biomethane output.
    methane_fraction: float
        recovered biomethane(e.g., energy or mass ratio).
        Default: 0.60
    co2 captured: float
        co2 captured and losses (e.g., energy or mass ratio).
        Default: 1 - 0.60
    capacity: numeric
        The captured biomethane production capacity (primary output side) of the unit.
    carrier_cost: numeric
        Carrier cost for one unit of used input (rawbiogas). Default: 0
    capacity_cost: numeric
        Investment costs per unit of biomethane capacity.
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

    rawbiogas_bus: Bus

    captured_co2_bus: Bus

    biomethane_bus: Bus

    tech: str

    carrier: str = ""

    methane_fraction: float = 0.60 # capture efficiency

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
        # The rawbiogas stream enters the biogas upgrading unit.
        # A fraction is separated as biomethane,
        # while the remainder leaves as co2 gas.
        self.conversion_factors.update(
            {
                self.rawbiogas_bus: sequence(1),
                self.biomethane_bus: sequence(
                    self.methane_fraction
                ),
                self.captured_co2_bus: sequence(
                    1 - self.methane_fraction
                ),
            }
        )

        self.inputs.update(
            {
                self.rawbiogas_bus: Flow(
                    variable_costs=self.carrier_cost,
                    **self.input_parameters,
                ),
            }
        )

        self.outputs.update(
            {
                self.biomethane_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost,
                    investment=self._investment(),
                    **self.output_parameters,
                ),
                self.captured_co2_bus: Flow(),
            }
        )