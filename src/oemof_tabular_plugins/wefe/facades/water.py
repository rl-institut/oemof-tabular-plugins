from dataclasses import field
from typing import Sequence, Union

import numpy as np
from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade


@dataclass_facade
class WaterFiltration(Converter, Facade):
    r"""WaterFiltration unit with two inputs and one output.

    Parameters
    ----------
    electricity_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its low temperature input.
    water_in_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its water input.
    water_out_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its water output.
    capacity: numeric
        The thermal capacity (high temperature output side) of the unit.
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

    electricity_bus: Bus

    water_in_bus: Bus

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
        """TODO change efficiencies here"""
        self.conversion_factors.update(
            {
                self.electricity_bus: sequence(1),
                self.water_in_bus: sequence(1),
                self.water_out_bus: sequence(1),
            }
        )

        self.inputs.update(
            {
                self.electricity_bus: Flow(
                    variable_costs=self.carrier_cost, **self.input_parameters
                ),
                self.water_in_bus: Flow(),
            }
        )

        self.outputs.update(
            {
                self.water_out_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost,
                    investment=self._investment(),
                )
            }
        )


@dataclass_facade
class WaterPump(Converter, Facade):
    r"""WaterPump unit with two inputs and one output.

    Parameters
    ----------
    electricity_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its low temperature input.
    water_in_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its water input.
    water_out_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its water output.
    pump_height: numeric
        The height in meters the pump must overcome.
    efficiency: numeric (iterable or scalar) (optional)
        The efficiency of the pump
    capacity: numeric (optional)
        The thermal capacity (high temperature output side) of the unit.
    carrier_cost: numeric (optional)
        Carrier cost for one unit of used input. Default: 0
    capacity_cost: numeric (optional)
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

    electricity_bus: Bus

    water_in_bus: Bus

    water_out_bus: Bus

    tech: str

    pump_height: float

    carrier: str = ""

    efficiency: Union[float, Sequence[float]] = 1

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

    @property
    def g(self):
        """Gravitational acceleration"""
        return 9.81  # m²/s

    @property
    def rho_w(self):
        """Water density"""
        return 1000  # kg/m³

    def build_solph_components(self):
        """TODO change efficiencies here"""
        # TODO ask vivek for references for water pumps
        conversion_W_to_kW = 1e-3
        conversion_m3_per_hour_to_m3_per_s = 1.0 / 3600

        if isinstance(self.efficiency, list):
            eta = np.array(self.efficiency)
        else:
            eta = self.efficiency

        self.conversion_factors.update(
            {
                self.electricity_bus: sequence(
                    self.g
                    * self.rho_w
                    * conversion_W_to_kW
                    * self.pump_height
                    * conversion_m3_per_hour_to_m3_per_s
                    / eta
                ),  # in kWh
                self.water_in_bus: sequence(1),  # in m³/h
                self.water_out_bus: sequence(1),  # in m³/h
            }
        )

        self.inputs.update(
            {
                self.electricity_bus: Flow(
                    variable_costs=self.carrier_cost, **self.input_parameters
                ),
                self.water_in_bus: Flow(),
            }
        )

        self.outputs.update(
            {
                self.water_out_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost,
                    investment=self._investment(),
                )
            }
        )
