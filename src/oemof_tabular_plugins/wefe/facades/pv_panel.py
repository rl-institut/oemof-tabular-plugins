from dataclasses import field
from typing import Sequence, Union

from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import Facade, dataclass_facade


@dataclass_facade
class PVPanel(Converter, Facade):
    r"""PV panel unit with one input and one output. The temperature factor
    is calculated and considered within the electricity generation.

    Parameters
    ----------
    from_bus: oemof.solph.Bus
        An oemof bus instance where the PV panel unit is connected to with
        its input.
    to_bus: oemof.solph.Bus
        An oemof bus instance where the PV panel unit is connected to with
        its output.
    capacity: numeric
        The conversion capacity (output side) of the unit.
    marginal_cost: numeric
        Marginal cost for one unit of produced output. Default: 0
    carrier_cost: numeric
        Carrier cost for one unit of used input. Default: 0
    capacity_cost: numeric
        Investment costs per unit of output capacity.
        If capacity is not set, this value will be used for optimizing the
        conversion output capacity.
    expandable: boolean or numeric (binary)
        True, if capacity can be expanded within optimization. Default: False.
    capacity_potential: numeric
        Maximum invest capacity in unit of output capacity.
    capacity_minimum: numeric
        Minimum invest capacity in unit of output capacity.
    input_parameters: dict (optional)
        Set parameters on the input edge of the conversion unit
        (see oemof.solph for more information on possible parameters)
    output_parameters: dict (optional)
        Set parameters on the output edge of the conversion unit
         (see oemof.solph for more information on possible parameters)
    t_air: array-like
        Ambient air temperature
    ghi: array-like
        Global horizontal irradiance
    p_rpv: numeric
        Rated power of photovoltaic panel. Default: 270
    r_ref: numeric
        Solar radiation at reference conditions. Default: 1000
    n_t: numeric
        Temperature coefficient of PV panel. Default: -0.0037
    t_c_ref: numeric
        Cell temperature at reference conditions. Default: 25
    noct: numeric
        Normal operating cell temperature. Default: 48

    SHOULD INCLUDE FUNCTIONS AND EXAMPLE HERE

    """

    from_bus: Bus

    to_bus: Bus

    carrier: str

    tech: str

    t_air: Union[float, Sequence[float]]

    ghi: Union[float, Sequence[float]]

    capacity: float = None

    marginal_cost: float = 0

    carrier_cost: float = 0

    capacity_cost: float = None

    expandable: bool = False

    capacity_potential: float = float("+inf")

    capacity_minimum: float = None

    input_parameters: dict = field(default_factory=dict)

    output_parameters: dict = field(default_factory=dict)

    p_rpv: float = 270

    r_ref: float = 1000

    n_t: float = -0.0037

    t_c_ref: float = 25

    noct: float = 48

    def build_solph_components(self):
        """ """
        if self.t_air is None or self.ghi is None:
            # handle the case when t_air or ghi is None
            print("Error: t_air or ghi is None. Cannot perform calculations.")
            return
        # assign the air temperature and solar irradiance
        t_air_values = self.t_air
        ghi_values = self.ghi
        # raise error if air temperature list and solar irradiance list are different lengths
        if len(t_air_values) != len(ghi_values):
            raise ValueError("Length mismatch between t_air and ghi profiles.")
        # calculates the temperature factor values
        pv_tf_values = []
        for t_air, ghi in zip(t_air_values, ghi_values):
            t_c = t_air + ((self.noct - 20) / 800) * ghi
            pv_tf = (
                self.p_rpv * (1 / self.r_ref) * (1 + self.n_t * (t_c - self.t_c_ref))
            )
            pv_tf_values.append(pv_tf)

        self.conversion_factors.update(
            {
                self.from_bus: sequence(1),
                self.to_bus: sequence(pv_tf_values),
            }
        )

        self.inputs.update(
            {
                self.from_bus: Flow(
                    variable_costs=self.carrier_cost, **self.input_parameters
                )
            }
        )

        self.outputs.update(
            {
                self.to_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost,
                    investment=self._investment(),
                    **self.output_parameters,
                )
            }
        )

        def processing_raw_inputs(self, resource, results_df):
            # function to apply on df from above

            return results_df

        def validate_datapackage(self, resource):
            # modify the resource (datapackage.resource)
            # should it return the resource?
            pass
