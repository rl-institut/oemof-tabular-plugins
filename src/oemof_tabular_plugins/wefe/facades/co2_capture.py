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
class Co2Capture(Converter, Facade):
    r""" A fluegas capture unit with 1 input and 2 outputs. The input is fluegas from the
    chp while the outputs are captured co2 and residual gas.

    Parameters
    ----------
    fluegas_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its fluegas input.
    captured_co2_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its captured co2 output.
    residual_gas_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its rawgas output.
    efficiency: float
        Captured co2 efficiency (e.g., energy or mass ratio).
        Default: 0.8
    efficiency: float
        Residual gas efficiency (e.g., energy or mass ratio).
        Default: 0.2
    capacity: numeric
        The captured co2 production capacity (primary output side) of the unit.
    carrier_cost: numeric
        Carrier cost for one unit of used input (rawbiogas). Default: 0
    capacity_cost: numeric
        Investment costs per unit of co2 output capacity.
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

    fluegas_bus: Bus

    captured_co2_bus: Bus

    residual_fluegas_bus: Bus

    tech: str

    carrier: str = ""

    capture_efficiency: float = 0.9 # capture efficiency

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
        # The flue gas stream enters the capture unit.
        # A fraction is separated as captured CO2,
        # while the remainder leaves as residual flue gas.
        self.conversion_factors.update(
            {
                self.fluegas_bus: sequence(1),
                self.captured_co2_bus: sequence(
                    self.capture_efficiency
                ),
                self.residual_fluegas_bus: sequence(
                    1 - self.capture_efficiency
                ),
            }
        )

        self.inputs.update(
            {
                self.fluegas_bus: Flow(
                    variable_costs=self.carrier_cost,
                    **self.input_parameters,
                )
            }
        )

        self.outputs.update(
            {
                self.captured_co2_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost,
                    investment=self._investment(),
                    **self.output_parameters,
                ),
                self.residual_fluegas_bus: Flow(),
            }
        )