import os.path

import numpy as np

from dataclasses import field, dataclass
from typing import Sequence, Union

from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.flows import Flow

from oemof.tabular.facades import Volatile

import oemof_tabular_plugins.wefe.global_specs.wind_turbines as wind_turbine_specs


@dataclass(unsafe_hash=False, frozen=False, eq=False)
class WindTurbine(Volatile):
    r"""Wind turbine unit with one output.

    Windspeed at hub height based on windspeed at reference height and surface roughness, source:

    Manwell, J. F., McGowan, J. G., Rogers, A. L. (2010). Wind Energy Explained:
        Theory, Design and Application. 2nd Edition. Wiley.
    Chapter 2, eq. (2.34)
    https://doi.org/10.1002/9781119994367.ch2


    Power output based on windspeed and turbine parameters, source:

    Mohammadi et al. (2012): Optimization of hybrid solar energy sources/wind turbine
        systems integrated to utility grids as microgrid (MG) under pool/bilateral/hybrid
        electricity market using PSO.
    Chapter 2, eq. (2) and (3)
    https://doi.org/10.1016/j.solener.2011.09.011


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
    windspeed: array-like
        windspeed profile in [m/s]
    surface_roughness: array-like
        surface roughness profile in [m]
    ref_height: numeric
        reference height of the windspeed profile in [m],
        e.g. windspeed profile at 100m
    turbine_type: str
        Name of the wind turbine used to get turbine parameters.
        Has to match key of 'wind_turbine_dict'.

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

    windspeed: Union[float, Sequence[float]] = None

    surface_roughness: Union[float, Sequence[float]] = None

    ref_height: float = 100.0

    turbine_type: str = "aeolos-h_50kw"


    def __init__(self, **attributes):
        """ """
        windspeed = attributes.pop("windspeed")
        surface_roughness = attributes.pop("surface_roughness")
        ref_height = attributes.pop("ref_height")

        if windspeed is None or surface_roughness is None:
            # handle the case when t_air or ghi is None
            print("Error: windspeed or surface roughness of wind turbine component is None. Cannot perform calculations.")
            return
        # raise error if air temperature list and solar irradiance list are different lengths
        if len(windspeed) != len(surface_roughness):
            raise ValueError("Length mismatch between windspeed and surface roughness profiles of wind turbine component.")

        # Get wind turbine params from database
        turbine = attributes.pop("turbine_type")
        turbines = wind_turbine_specs.wind_turbine_dict
        if turbine in turbines.keys():
            turbine_params = turbines[turbine]
        else:
            ofname = os.path.abspath(wind_turbine_specs.__file__)
            print(f"Error: Wind turbine '{turbine}' not available in {ofname}. Using default turbine instead.")
            turbine_params = turbines["aeolos-h_50kw"]

        def wind_power(v_ref, ref_height, sr, v_min, v_rated, v_max, p_rated, hub_height, **kwargs):
            """
            Calculate power output based on windspeed at reference height,
            surface roughness and turbine parameters.
            """
            v = v_ref * np.log(hub_height / sr) / np.log(ref_height / sr)

            if v < v_min:
                p = 0
            elif v_min <= v < v_rated:
                a = p_rated / (v_rated**3 - v_min**3)
                b = v_min**3 / (v_rated**3 - v_min**3)
                p = a * v**3 - b * p_rated
            elif v_rated <= v < v_max:
                p = p_rated
            else:
                p = 0

            return p

        # Get wind power output of one turbine
        power = np.array(
            [
                wind_power(v_ref=ws, ref_height=ref_height, sr=sr, **turbine_params)
                for ws, sr in zip(windspeed, surface_roughness)
            ]
        )

        # Output relative to capacity (in <unit> per <unit> of installed wind turbine capacity, for example kW)
        relative_output = power / turbine_params["p_rated"]

        super().__init__(
            profile=relative_output,
            **attributes
        )

