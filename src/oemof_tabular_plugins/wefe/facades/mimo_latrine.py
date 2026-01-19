import dataclasses
from typing import Sequence, Union

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class MIMO_Latrine(MIMO):
    """
    Specialized MIMO-B Latrine facade with code-driven physics.

    Inputs:
    - human_feces_bus (PRIMARY, kg)
    - human_urine_bus (m³)

    Outputs:
    - biomass_waste_bus (m³) (sum of all inputs)
    """

    # ---- tabular identity ----
    type: str = "mimo_latrine"
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
    biomass_waste_bus: Bus = None

    # ---- physics parameters ----
    feces_density: float = 1060.0 # kg/m³

    # ---- economics ----
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
        self.biomass_waste_bus = attributes.pop("biomass_waste_bus")

        # ---------------------------
        # physics
        # ---------------------------

        # Compute feces volume per kg
        feces_volume = 1.0 / self.feces_density  # m³ per kg feces

        # ---------------------------
        # economics
        # ---------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)

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
        # PHYSICS → CONVERSION FACTORS
        #
        # Sum of inputs = output
        # (Correct oemof mass balance)
        # ==================================================
        attributes.update(
            {
                f"conversion_factor_{self.human_feces_bus.label}":
                    sequence(feces_volume),  # m³ feces
                f"conversion_factor_{self.human_urine_bus.label}":
                    sequence(1.0),  # m³ urine
                f"conversion_factor_{self.biomass_waste_bus.label}":
                    sequence(1.0),  # m³ biomass for latrine
            }
        )

        # ==================================================
        # MIMO initialization
        # ==================================================
        super().__init__(
            from_bus_0=self.human_feces_bus,
            from_bus_1=self.human_urine_bus,
            to_bus_0=self.biomass_waste_bus,
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