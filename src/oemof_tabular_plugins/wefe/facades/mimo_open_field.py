import dataclasses
from typing import Sequence, Union

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class MIMO_OpenField(MIMO):
    """
    Catch-all Open Field MIMO facade.

    Purpose:
    - Collects ALL remaining human & animal feces and urine
    - Acts as a fallback sink when toilets cannot handle flows
    - Preserves mass / volume exactly
    - Solver decides how much flows here

    Inputs (all optional at any timestep):
    - human_feces_bus
    - human_urine_bus
    - animal_feces_bus
    - animal_urine_bus

    Output:
    - biomass_waste_bus (sum of all inputs)

    Notes:
    - No PRIMARY (important for mimo superclass)
    - No capacity required
    - No investment logic
    """

    # ---- tabular identity ----
    type: str = "mimo_open_field"
    name: str = ""
    tech: str = "mimo"
    carrier: str = ""
    primary: str = "human_feces_bus"

    # ---- capacity / investment ----
    expandable: bool = False
    capacity: float = None
    capacity_cost: float = None
    capacity_minimum: float = None
    capacity_potential: float = float("+inf")

    # ---- buses ----
    human_feces_bus: Bus = None
    human_urine_bus: Bus = None
    animal_feces_bus: Bus = None
    animal_urine_bus: Bus = None
    biomass_waste_bus: Bus = None

    # ---- physics parameters ----
    feces_density: float = 1060.0  # kg/m³

    # ---- economics ----
    marginal_cost: float = 0.0   # can be >0 to penalize open defecation

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
        self.animal_feces_bus = attributes.pop("animal_feces_bus")
        self.animal_urine_bus = attributes.pop("animal_urine_bus")
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
                    sequence(feces_volume), # m³ feces
                f"conversion_factor_{self.human_urine_bus.label}":
                    sequence(1.0), # m³ urine
                f"conversion_factor_{self.animal_feces_bus.label}":
                    sequence(feces_volume), # m³ feces
                f"conversion_factor_{self.animal_urine_bus.label}":
                    sequence(1.0), # m³ urine
                f"conversion_factor_{self.biomass_waste_bus.label}":
                    sequence(1.0), # m³ biomass for open field
            }
        )

        # ==================================================
        # MIMO initialization
        # ==================================================
        super().__init__(
            from_bus_0=self.human_feces_bus,
            from_bus_1=self.human_urine_bus,
            from_bus_2=self.animal_feces_bus,
            from_bus_3=self.animal_urine_bus,
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
