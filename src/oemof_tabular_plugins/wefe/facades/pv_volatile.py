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
    r"""PV panel unit with one output. The temperature factor
    is calculated and considered within the electricity generation.
    This facade initializes a facade of type "Volatile" with the PV electricity generation as it's profile.

    Parameters
    ----------
    bus: oemof.solph.Bus
        An oemof bus instance where the PV panel unit is connected to with
        its output.
    carrier: string
        Energy carrier of the output, should be 'electricity'.
        Check-out https://oemof-tabular.readthedocs.io/en/stable/usage.html for
        oemof carriers or define your own.
    tech: string
        Technology of the component, should be pv.
        Check-out https://oemof-tabular.readthedocs.io/en/stable/usage.html for
        oemof tech types or define your own.
    capacity: numeric
        The capacity (output side) of the unit, for example in kW.
        Output flow then will also be in kW.
    marginal_cost: numeric
        Marginal cost for one unit of produced output. Default: 0
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
    output_parameters: dict (optional)
        Set parameters on the output edge of the conversion unit
         (see oemof.solph for more information on possible parameters)
    t_air: array-like
        Ambient air temperature
    ghi: array-like
        Global horizontal irradiance
    pv_type: string
        Name of the pv module used to get module parameters.
        Has to match key of 'pv_dict'.
    latitude: float
        Latitude of the location where the PV panel is located. Used for calculating panel tilt.

    """

    bus: Bus

    carrier: str

    tech: str

    capacity: float = None

    marginal_cost: float = 0

    carrier_cost: float = 0

    capacity_cost: float = None

    expandable: bool = False

    capacity_potential: float = float("+inf")

    capacity_minimum: float = None

    output_parameters: dict = field(default_factory=dict)

    t_air: Union[float, Sequence[float]] = None

    ghi: Union[float, Sequence[float]] = None

    pv_type: str = "boviet_450"

    latitude: float = 0


    def __init__(self, **attributes):
        """ """
        t_air = attributes.pop("t_air")
        ghi = attributes.pop("ghi")
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

