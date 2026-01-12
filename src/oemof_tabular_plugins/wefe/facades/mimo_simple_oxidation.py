import dataclasses
from typing import Sequence, Union

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class MIMO_SimpleOxidation(MIMO):
    """
    Specialized MIMO-B Simple Oxidation facade with code-driven physics.

    Inputs:
    - electricity_bus
    - water_in_bus

    Outputs:
    - water_out_bus (PRIMARY)

    Conversion factors are derived internally from:
    - specific_energy_consumption
    """

    # ---- tabular identity ----
    type: str = "mimo_simple_oxidation"
    name: str = ""
    tech: str = "mimo"
    carrier: str = ""
    primary: str = ""

    # ---- capacity / investment ----
    expandable: bool = False
    capacity: float = None
    capacity_cost: float = None
    capacity_minimum: float = None
    capacity_potential: float = None

    # ---- buses ----
    electricity_bus: Bus = None
    water_in_bus: Bus = None
    water_out_bus: Bus = None

    # ---- physics ----
    specific_energy_consumption: float = 0.07  # kWh / m³ treated water

    # ---- oxidant parameters ----
    oxidant_type: str = "hydrogen_peroxide"
    oxidant_dose: float = None   # mg/L = g/m³
    oxidant_cost: float = None   # USD/kg

    # ---- economics ----
    marginal_cost: float = 0.0                  # €/m³ treated water
    carrier_cost: float = 0.0                   # €/kWh electricity

    # ---- multiperiod ----
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    def __init__(self, **attributes):
        """
        Specialized MIMO-B initialization:
        - validate physics
        - compute conversion factors
        - inject into MIMO
        """

        # ---------------------------
        # identity
        # ---------------------------
        self.type = attributes.pop("type", self.type)
        self.tech = attributes.pop("tech", self.tech)
        self.carrier = attributes.pop("carrier", self.carrier)
        self.primary = attributes.pop("primary", self.primary)

        # ---------------------------
        # buses
        # ---------------------------
        self.electricity_bus = attributes.pop("electricity_bus")
        self.water_in_bus = attributes.pop("water_in_bus")
        self.water_out_bus = attributes.pop("water_out_bus")

        # ---------------------------
        # physics
        # ---------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )

        # ---------------------------
        # oxidant
        # ---------------------------
        self.oxidant_type = attributes.pop("oxidant_type", self.oxidant_type)
        self.oxidant_dose = attributes.pop("oxidant_dose", self.oxidant_dose)
        self.oxidant_cost = attributes.pop("oxidant_cost", self.oxidant_cost)

        defaults = {
            "chlorine": {"dose": 1.0, "cost": 0.5},
            "chlorine_dioxide": {"dose": 0.8, "cost": 1.2},
            "hydrogen_peroxide": {"dose": 5.0, "cost": 2.0},
            "potassium_permanganate": {"dose": 3.0, "cost": 1.5},
        }

        key = self.oxidant_type.lower()

        if key in defaults:
            dose = self.oxidant_dose if self.oxidant_dose is not None else defaults[key]["dose"]
            cost = self.oxidant_cost if self.oxidant_cost is not None else defaults[key]["cost"]
        else:
            if self.oxidant_dose is None or self.oxidant_cost is None:
                raise ValueError(
                    f"Dose and cost must be provided for custom oxidant '{self.oxidant_type}'"
                )
            dose = self.oxidant_dose
            cost = self.oxidant_cost

        # oxidant cost per m³ treated water
        oxidant_cost_per_m3 = (
                dose * 1e-6 * cost
        )

        # ---------------------------
        # economics
        # ---------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)

        # ---------------------------
        # capacity / investment
        # ---------------------------
        self.expandable = attributes.pop("expandable", self.expandable)
        self.capacity = attributes.pop("capacity", self.capacity)
        self.capacity_cost = attributes.pop("capacity_cost", self.capacity_cost)
        self.capacity_minimum = attributes.pop(
            "capacity_minimum", self.capacity_minimum
        )
        self.capacity_potential = attributes.pop(
            "capacity_potential", self.capacity_potential
        )

        # ---------------------------
        # multiperiod
        # ---------------------------
        self.lifetime = attributes.pop("lifetime", self.lifetime)
        self.age = attributes.pop("age", self.age)
        self.fixed_costs = attributes.pop("fixed_costs", self.fixed_costs)

        # ==================================================
        # PHYSICS → CONVERSION FACTORS (normalized to treated water output = 1)
        # ==================================================

        electricity_per_output = self.specific_energy_consumption

        attributes.update(
            {
                # inputs
                f"conversion_factor_{self.electricity_bus.label}":
                    sequence(electricity_per_output),
                f"conversion_factor_{self.water_in_bus.label}":
                    sequence(1.0),

                # outputs
                f"conversion_factor_{self.water_out_bus.label}":
                    sequence(1.0),
            }
        )

        # ---------------------------
        # output costs
        # ---------------------------
        attributes.setdefault("output_parameters", {})
        attributes["output_parameters"].update(
            {
                "variable_costs": oxidant_cost_per_m3 + self.marginal_cost,
                "custom_attributes": {
                    "oxidant_type": self.oxidant_type,
                    "oxidant_dose": dose,
                },
            }
        )

        # ==================================================
        # MIMO initialization
        # ==================================================
        super().__init__(
            from_bus_0=self.electricity_bus,
            from_bus_1=self.water_in_bus,
            to_bus_0=self.water_out_bus,
            primary=self.primary,
            carrier_cost=self.carrier_cost,
            expandable=self.expandable,
            capacity=self.capacity,
            capacity_cost=self.capacity_cost,
            capacity_minimum=self.capacity_minimum,
            capacity_potential=self.capacity_potential,
            lifetime=self.lifetime,
            age=self.age,
            fixed_costs=self.fixed_costs,
            **attributes,
        )
