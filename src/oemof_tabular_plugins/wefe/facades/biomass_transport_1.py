from dataclasses import field
from typing import Sequence, Union

import numpy as np
from typing import Union, Sequence
from dataclasses import field

from networkx.algorithms import distance_regular
from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade

@dataclass_facade
class BiomassTransport(Converter, Facade):
    r""" A unit with 1 input and 1 output that transports biomass from a source
    location to the digester location. The input is biomass from where it is
    located while the output is biomass that will then be fed into the digester
    through the biomass-digester_bus.

    Parameters
    ----------
    biomass_source_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its biomass input.
    biomass_digester_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its biomass output.
    efficiency: float
        Electrical conversion efficiency (e.g., energy or mass ratio).
        Default: 0.50
    efficiency: float
        biomass conversion efficiency (e.g., energy or mass ratio).
        Default: 1
    capacity: numeric
        The biomass capacity (primary output side) of the unit.
    carrier_cost: numeric
        Carrier cost for one unit of used input (biomass). Default: 0
    capacity_cost: numeric
        Investment costs per unit of biomass output capacity.
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

    biomass_source_bus: Bus

    biomass_digester_bus: Bus

    tech: str

    carrier: str = ""

    distance: float = 0 # km

    transport_cost_per_tkm: float = 0  # Eur /t.km

    transport_efficiency: float = 1.0  # fraction arriving at destination

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
        # Since the nominal capacity is bound to the primary output (biomass),
        # we define the ratios relative to the biomass input or scale them to the main output.
        # Here, we treat biomass as the base (1), and outputs as fractions of that input.
        self.conversion_factors.update(
            {
                self.biomass_source_bus: sequence(1),
                self.biomass_digester_bus: sequence(
                    self.transport_efficiency
                ),
            }
        )
        transport_cost = (
            self.distance
            * self.transport_cost_per_tkm
        )

        self.inputs.update(
            {
                self.biomass_source_bus: Flow(
                    **self.input_parameters,
                ),
            }
        )

        self.outputs.update(
            {
                self.biomass_digester_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost + transport_cost,
                    investment=self._investment(),
                    **self.output_parameters,
                ),
            }
        )

