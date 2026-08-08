import dataclasses
from typing import Sequence, Union

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class MIMO_FlushToilet(MIMO):
    """
    Specialized MIMO-B Flush Toilet facade with code-driven physics.

    Inputs:
    - human_feces_bus (PRIMARY, kg)
    - human_urine_bus (m³)
    - service_water_bus (m³)

    Outputs:
    - blackwater_bus (m³)

    Conversion factors are derived internally from:
    - typical blackwater composition
    - feces density (kg → m³)
    """

    # ---- tabular identity ----
    type: str = "mimo_flush_toilet"
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
    service_water_bus: Bus = None
    blackwater_bus: Bus = None

    # ---- physics parameters ----
    feces_fraction: float = 0.01           # volume fraction in blackwater in m³
    urine_fraction: float = 0.11           # volume fraction in blackwater in m³
    service_water_fraction: float = 0.88   # volume fraction in blackwater in m³
    feces_density: float = 1060.0          # kg/m³

    # ---- economics ----
    marginal_cost: float = 0.0              # €/m³ blackwater
    service_water_cost: float = 0.0         # €/m³ service water

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
        self.service_water_bus = attributes.pop("service_water_bus")
        self.blackwater_bus = attributes.pop("blackwater_bus")

        # ---------------------------
        # physics
        # ---------------------------
        self.feces_fraction = attributes.pop("feces_fraction", self.feces_fraction)
        self.urine_fraction = attributes.pop("urine_fraction", self.urine_fraction)
        self.service_water_fraction = attributes.pop("service_water_fraction", self.service_water_fraction)
        self.feces_density = attributes.pop("feces_density", self.feces_density)

        # Compute feces volume per kg
        feces_volume = 1.0 / self.feces_density  # m³ per kg feces

        # Compute required urine and service water volumes per kg feces
        urine_per_feces = (self.urine_fraction / self.feces_fraction) * feces_volume
        water_per_feces = (self.service_water_fraction / self.feces_fraction) * feces_volume

        # Total blackwater volume per kg feces
        blackwater_per_feces = feces_volume + urine_per_feces + water_per_feces

        # ---------------------------
        # economics
        # ---------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.service_water_cost = attributes.pop("service_water_cost", self.service_water_cost)

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
        # PHYSICS → CONVERSION FACTORS (normalized to 1 kg feces input)
        # ==================================================
        attributes.update(
            {
                # inputs
                f"conversion_factor_{self.human_feces_bus.label}":
                    sequence(feces_volume),    # m³ feces
                f"conversion_factor_{self.human_urine_bus.label}":
                    sequence(urine_per_feces), # m³ urine
                f"conversion_factor_{self.service_water_bus.label}":
                    sequence(water_per_feces), # m³ service water

                # outputs
                f"conversion_factor_{self.blackwater_bus.label}":
                    sequence(blackwater_per_feces), # m³ blackwater
            }
        )

        # ---------------------------
        # input-specific costs
        # ---------------------------
        attributes.setdefault("input_parameters", {})
        attributes["input_parameters"].setdefault(
            self.service_water_bus, {}
        )["variable_costs"] = self.service_water_cost

        # ==================================================
        # MIMO initialization
        # ==================================================
        super().__init__(
            from_bus_0=self.human_feces_bus,
            from_bus_1=self.human_urine_bus,
            from_bus_2=self.service_water_bus,
            to_bus_0=self.blackwater_bus,
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
