from dataclasses import field
from typing import Sequence, Union

import numpy as np
from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade

@dataclass_facade #v1.0   #please check default values once more #improve more pending
class SimpleOxidation(Converter, Facade):
    r""" Simple Oxidation water treatment unit with two inputs and one output.
    Oxidant dosing is attached as a parameter to the output water stream.

    Parameters
    ----------
     electricity_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its electricity input.
    water_in_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its untreated water input.
    water_out_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its treated water output.
    specific_energy_consumption: float
        Specific electricity demand/consumption in kWh per m³ treated water. Default: 0.07
    oxidant_type: str
        Type/name of oxidant. Options: 'chlorine', 'chlorine_dioxide', 'hydrogen_peroxide',
        'potassium_permanganate', or custom name. Default: 'hydrogen_peroxide'.
    oxidant_dose: float
        Oxidant dosage in mg/L of treated water (= g/m³). If None, defaults used based on oxidant_type.
    oxidant_cost: float
        Oxidant cost in USD per kg. If None, defaults used per oxidant_type.
    capacity: numeric
        The water treatment capacity (output side) of the unit.
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
        The initial age of a component (usually given in years);
        once it reaches its lifetime (considering also
        an initial age), the component is forced to 0 (retire/replace).
        Note: Only applicable for a multi-period model. Default: 0.
    fixed_costs : numeric (iterable or scalar) (optional)
        The fixed operational costs associated with a component.
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

    specific_energy_consumption: float = 0.07  # kWh/m³

    oxidant_type: str = "hydrogen_peroxide"

    oxidant_dose: Union[float, None] = None  # mg/L = g/m³

    oxidant_cost: Union[float, None] = None  # USD/kg

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

        # Default dose and cost values for common oxidants (mg/L and USD/kg)
        defaults = {
            "chlorine": {"dose": 1.0, "cost": 0.5},
            "chlorine_dioxide": {"dose": 0.8, "cost": 1.2},
            "hydrogen_peroxide": {"dose": 5.0, "cost": 2.0},
            "potassium_permanganate": {"dose": 3.0, "cost": 1.5},
        }

        if self.oxidant_type.lower() in defaults:
            # If user has provided dose, use it; otherwise use default dose
            if self.oxidant_dose is not None:
                dose = self.oxidant_dose
            else:
                dose = defaults[self.oxidant_type.lower()]["dose"]

            # If user has provided cost, use it; otherwise use default cost
            if self.oxidant_cost is not None:
                cost = self.oxidant_cost
            else:
                cost = defaults[self.oxidant_type.lower()]["cost"]

        else:
            # For custom oxidants, both dose and cost must be provided by user
            if self.oxidant_dose is None or self.oxidant_cost is None:
                raise ValueError(
                    f"Dose and cost must be provided for custom oxidant '{self.oxidant_type}'"
                )
            dose = self.oxidant_dose
            cost = self.oxidant_cost

        # Oxidant cost per m³ of treated water:
        # oxidant_dose [mg/L] * 1e-6 [kg/m³ per mg/L] * oxidant_cost [USD/kg]
        oxidant_cost_per_m3 = (dose * 1e-6 * cost)

        self.conversion_factors.update(
            {
                self.electricity_bus: sequence(self.specific_energy_consumption),
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
                    nominal_value = self._nominal_value(),
                    variable_costs = oxidant_cost_per_m3 + self.marginal_cost,
                    investment = self._investment(),
                    **self.output_parameters,
                ),
            }
        )

        # Add custom attribute separately
        self.outputs[self.water_out_bus].custom_attributes = {"oxidant_type": self.oxidant_type,
                                                              "oxidant_dose": dose}