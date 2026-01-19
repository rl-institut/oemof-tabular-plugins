import dataclasses
from typing import Sequence, Union

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class MIMO_DryToilet(MIMO):
    """
    Specialized MIMO-B Dry Toilet facade with code-driven physics.

    Inputs:
    - human_feces_bus (PRIMARY, kg)
    - human_urine_bus (m³)

    Outputs:
    - dry_feces_out_bus (kg)
    - water_out_bus (m³)
    """

    # ---- tabular identity ----
    type: str = "mimo_dry_toilet"
    name: str = ""
    tech: str = "mimo"
    carrier: str = ""
    primary: str = "human_feces_bus"

    # ---- capacity / investment ----
    expandable: bool = False
    capacity: float = None
    capacity_cost: float = None
    capacity_minimum: float = None
    capacity_potential: float = None

    # ---- buses ----
    human_feces_bus: Bus = None
    human_urine_bus: Bus = None
    dry_feces_out_bus: Bus = None
    water_out_bus: Bus = None

    # ---- physics parameters ----
    wet_feces_dry_feces_fraction: float = 3.3 # kg feces / kg dry feces
    urine_dry_feces_fraction: float = 2.0     # m³ urine / kg dry feces
    leachate_dry_feces_relation: float = 0.1  # m³ leachate / kg dry feces
    feces_density: float = 1060.0  # kg/m³

    # ---- economics ----
    bulking_agent_dose: float = 0.25  # kg/kg dry feces
    bulking_agent_cost: float = 0.1  # USD/kg
    marginal_cost: float = 0.0

    # ---- multiperiod ----
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    def __init__(self, **attributes):
        """
        Specialized MIMO-B initialization:
        - validate physics
        - compute conversion factors (kg → m³)
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
        self.human_feces_bus = attributes.pop("human_feces_bus")
        self.human_urine_bus = attributes.pop("human_urine_bus")
        self.dry_feces_out_bus = attributes.pop("dry_feces_out_bus")
        self.water_out_bus = attributes.pop("water_out_bus")

        # ---------------------------
        # physics
        # ---------------------------
        self.urine_dry_feces_fraction = attributes.pop(
            "urine_dry_feces_fraction", self.urine_dry_feces_fraction
        )
        self.wet_feces_dry_feces_fraction = attributes.pop(
            "wet_feces_dry_feces_fraction", self.wet_feces_dry_feces_fraction
        )
        self.leachate_dry_feces_relation = attributes.pop(
            "leachate_dry_feces_relation", self.leachate_dry_feces_relation
        )
        self.feces_density = attributes.pop("feces_density", self.feces_density)

        # Compute feces volume per kg
        feces_volume = 1.0 / self.feces_density  # m³ per kg feces

        # ---------------------------
        # economics
        # ---------------------------

        self.bulking_agent_dose = attributes.pop("bulking_agent_dose", self.bulking_agent_dose)
        self.bulking_agent_cost = attributes.pop("bulking_agent_cost", self.bulking_agent_cost)
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)

        bulking_cost_per_kg = self.bulking_agent_dose * self.bulking_agent_cost

        # ---------------------------
        # capacity / investment
        # ---------------------------
        self.expandable = attributes.pop("expandable", self.expandable)
        self.capacity = attributes.pop("capacity", self.capacity)
        self.capacity_cost = attributes.pop("capacity_cost", self.capacity_cost)
        self.capacity_minimum = attributes.pop("capacity_minimum", self.capacity_minimum)
        self.capacity_potential = attributes.pop("capacity_potential", self.capacity_potential)

        # ---------------------------
        # multiperiod
        # ---------------------------
        self.lifetime = attributes.pop("lifetime", self.lifetime)
        self.age = attributes.pop("age", self.age)
        self.fixed_costs = attributes.pop("fixed_costs", self.fixed_costs)

        # ==================================================
        # PHYSICS → CONVERSION FACTORS (normalized to dry feces output = 1)
        # ==================================================

        attributes.update(
            {
                f"conversion_factor_{self.human_feces_bus.label}":
                    sequence(self.wet_feces_dry_feces_fraction),
                f"conversion_factor_{self.human_urine_bus.label}":
                    sequence(self.urine_dry_feces_fraction),
                f"conversion_factor_{self.dry_feces_out_bus.label}":
                    sequence(1.0),
                f"conversion_factor_{self.water_out_bus.label}":
                    sequence(self.leachate_dry_feces_relation),
            }
        )

        # ---------------------------
        # output costs
        # ---------------------------
        attributes.setdefault("output_parameters", {})
        attributes["output_parameters"].setdefault(self.dry_feces_out_bus, {})
        attributes["output_parameters"][self.dry_feces_out_bus].update(
            {
                "variable_costs": bulking_cost_per_kg + self.marginal_cost,
                "custom_attributes": {"bulking_agent_dose": self.bulking_agent_dose},
            }
        )

        # ==================================================
        # MIMO initialization
        # ==================================================
        super().__init__(
            from_bus_0=self.human_feces_bus,
            from_bus_1=self.human_urine_bus,
            to_bus_0=self.dry_feces_out_bus,
            to_bus_1=self.water_out_bus,
            primary=self.primary,
            marginal_cost=self.marginal_cost,
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