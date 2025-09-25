import numpy as np

from dataclasses import field, dataclass
from typing import Sequence, Union

from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.flows import Flow

from oemof.tabular.facades import Volatile

from oemof_tabular_plugins.wefe.facades import functions as f
from oemof_tabular_plugins.wefe.global_specs import pv_dict


@dataclass(unsafe_hash=False, frozen=False, eq=False)
class PVPanel(Volatile):
    r"""PV panel unit with one input and one output. The temperature factor
    is calculated and considered within the electricity generation.
    Note: This facade has the PV power [kW] as its capacity and can be used to model PV in general.

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

    bus: Bus

    carrier: str

    tech: str

    t_air: Union[float, Sequence[float]] = None

    ghi: Union[float, Sequence[float]] = None

    capacity: float = None

    marginal_cost: float = 0

    carrier_cost: float = 0

    capacity_cost: float = None

    expandable: bool = False

    capacity_potential: float = float("+inf")

    capacity_minimum: float = None

    output_parameters: dict = field(default_factory=dict)

    pv_type: str = ""

    latitude: float = 0


    def __init__(self, **attributes):
        """ """
        t_air = attributes.pop("t_air")
        ghi = attributes.pop("ghi")
        # TODO: these checks about t_air and ghi should be obsolete as the profiles will be checked somewhere else...
        if t_air is None or ghi is None:
            # handle the case when t_air or ghi is None
            print("Error: t_air or ghi of pv-panel component is None. Cannot perform calculations.")
            return
        # raise error if air temperature list and solar irradiance list are different lengths
        if len(ghi) != len(t_air):
            raise ValueError("Length mismatch between t_air and ghi profiles of pv-panel component.")

        # get pv params from database
        pv_params = pv_dict[attributes.pop("pv_type")]
        # tilt and pitch based on latitude
        geo_params = f.pv_geometry(latitude=attributes.pop("latitude"), **pv_params)

        # irradiation perpendicular to PV panel (global normal irradiance)
        ghi_to_gni = np.cos(np.radians(geo_params["tilt"]))
        gni = np.array(ghi) * ghi_to_gni

        # pv power (per module in W)
        pv_power = np.array(
            [
                f.power(rad=gni, t_air=t_air, **pv_params)
                for t_air, gni in zip(t_air, gni)
            ]
        )

        # capacity power (in <unit> per <unit> of installed PV capacity, for example kW)
        capacity_power = pv_power / pv_params["p_rated"]

        super().__init__(
            profile=capacity_power,
            **attributes
        )

