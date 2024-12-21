from dataclasses import field
from typing import Sequence, Union

import numpy as np
from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade


@dataclass_facade
class RRHydropower(Converter, Facade):
    r"""Run-of-River (RR) Hydropower Plant unit with one input and one output.

    Parameters
    ----------

    water_in_bus: oemof.solph.Bus
    electricity_out_bus: oemof.solph.Bus
        An oemof bus instance where component is connected to its electricity output.
    profile: sequence expressing the hourly river flow in m³/h; typically provided as sequence in volatile_profile.csv
    capacity: numeric
        The power capacity (peak power) of the unit.
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
    head: float
        height difference between the water source and the water's outflow point.
        This height difference determines the potential energy available for generating electricity.
    efficiency: numeric (iterable or scalar) (optional)
        The efficiency of the hydropower turbine
    age : int (optional)
        The initial age of a flow (usually given in years);
        once it reaches its lifetime (considering also
        an initial age), the flow is forced to 0.
        Note: Only applicable for a multi-period model. Default: 0.
    fixed_costs : numeric (iterable or scalar) (optional)
        The fixed costs associated with a flow.
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

    electricity_out_bus: Bus

    profile: Union[float, Sequence[float]]
    tech: str

    head: float = 0

    efficiency: float = 0.8

    carrier: str = ""

    capacity: float = None

#  conversion_factor: float = None

    marginal_cost: Union[float, Sequence[float]] = 0

    carrier_cost: float = 0

    resource_cost: float = 0

    capacity_cost: float = None

    expandable: bool = False

    lifetime: int = None

    age: int = 0

    fixed_costs: Union[float, Sequence[float]] = None

    capacity_potential: float = float("+inf")

    input_parameters: dict = field(default_factory=dict)

    output_parameters: dict = field(default_factory=dict)

    # PYCHARM itself suggested and created this init function; Maybe I will omitt again later
    # def __init__(
    #         self,
    #         label=None,
    #         inputs=None,
    #         outputs=None,
    #         conversion_factors=None,
    #         custom_attributes=None,
    # ):
    #     super().__init__(label, inputs, outputs, conversion_factors, custom_attributes)
    #     self.conversion_factor = None

    @property
    def g(self):
        """Gravitational Acceleration"""
        return 9.81  # m/s²

    @property
    def rho_w(self):
        """Water Density"""
        return 1000  # kg/m³

    def build_solph_components(self):
        """Build solph components for RRHydropower"""
        self.conversion_factor = self.g * self.rho_w * self.head * self.efficiency

        self.conversion_factors.update(
            {
                self.electricity_out_bus: sequence(self.conversion_factor),
            }
        )

        self.inputs.update(
            {
                self.water_in_bus: Flow(
                    variable_costs=self.carrier_cost, **self.input_parameters
                )
            }
        )

        self.outputs.update(
            {
                self.electricity_out_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost,
                    investment=self._investment(),
                )
            }
        )
