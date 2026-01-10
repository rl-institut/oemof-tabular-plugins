import dataclasses
from typing import Sequence, Union

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class MIMO_ReverseOsmosis(MIMO):
    """
    Specialized MIMO-B Reverse Osmosis facade with code-driven physics.

    Inputs:
    - electricity_bus
    - feedwater_bus

    Outputs:
    - permeate_bus (PRIMARY)
    - brine_bus

    Conversion factors are derived internally from:
    - specific_energy_consumption
    - recovery_rate
    """

    # ---- tabular identity ----
    type: str = "mimo_ro"
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
    feedwater_bus: Bus = None
    permeate_bus: Bus = None
    brine_bus: Bus = None

    # ---- physics parameters ----
    specific_energy_consumption: float = 1.2   # kWh / m³ permeate
    recovery_rate: float = 0.55                # permeate / feedwater

    # ---- economics ----
    marginal_cost: float = 0.0                 # €/m³ permeate
    carrier_cost: float = 0.0                  # €/kWh electricity
    brine_disposal_cost: float = 0.0            # €/m³ brine

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
        self.feedwater_bus = attributes.pop("feedwater_bus")
        self.permeate_bus = attributes.pop("permeate_bus")
        self.brine_bus = attributes.pop("brine_bus")

        # ---------------------------
        # physics parameters
        # ---------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.recovery_rate = attributes.pop(
            "recovery_rate", self.recovery_rate
        )

        if not 0 < self.recovery_rate < 1:
            raise ValueError("recovery_rate must be between 0 and 1")

        # ---------------------------
        # economics
        # ---------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.brine_disposal_cost = attributes.pop(
            "brine_disposal_cost", self.brine_disposal_cost
        )

        # ---------------------------
        # capacity & investment
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

        # ==========================================================
        # PHYSICS → CONVERSION FACTORS (normalized to permeate = 1)
        # ==========================================================

        feedwater_per_permeate = 1.0 / self.recovery_rate
        brine_per_permeate = feedwater_per_permeate - 1.0
        electricity_per_permeate = self.specific_energy_consumption

        attributes.update(
            {
                # inputs
                f"conversion_factor_{self.electricity_bus.label}":
                    sequence(electricity_per_permeate),
                f"conversion_factor_{self.feedwater_bus.label}":
                    sequence(feedwater_per_permeate),

                # outputs
                f"conversion_factor_{self.permeate_bus.label}":
                    sequence(1.0),
                f"conversion_factor_{self.brine_bus.label}":
                    sequence(brine_per_permeate),
            }
        )

        # ---------------------------
        # cost assignment to outputs
        # ---------------------------
        attributes.setdefault("output_parameters", {})
        attributes.setdefault("output_parameters_1", {})

        if self.brine_disposal_cost > 0:
            attributes["output_parameters_1"].update(
                {"variable_costs": self.brine_disposal_cost}
            )

        # ==========================================================
        # MIMO initialization
        # ==========================================================
        super().__init__(
            from_bus_0=self.electricity_bus,
            from_bus_1=self.feedwater_bus,
            to_bus_0=self.permeate_bus,   # PRIMARY
            to_bus_1=self.brine_bus,
            primary=self.primary,
            marginal_cost=self.marginal_cost,
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
