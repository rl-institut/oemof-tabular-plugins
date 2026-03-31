import dataclasses
from typing import Sequence, Union

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class MIMO_SepticSystem(MIMO):
    """
    Specialized MIMO-B Septic System facade with code-driven physics.

    Inputs:
    - electricity_bus
    - water_in_bus

    Outputs:
    - water_out_bus (PRIMARY)
    - sludge_out_bus

    Conversion factors are derived internally from:
    - specific_energy_consumption
    - efficiency
    """

    # ---- tabular identity ----
    type: str = "mimo_septic"
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
    sludge_out_bus: Bus = None

    # ---- physics parameters ----
    specific_energy_consumption: float = 0.13   # kWh / m³ treated water (~0.0 ,if assumed gravity based)
    efficiency: float = 0.70                   # permeate / feedwater

    # ---- economics ----
    marginal_cost: float = 0.0                 # €/m³ treated water
    carrier_cost: float = 0.0                  # €/kWh electricity
    sludge_disposal_cost: float = 0.0          # €/m³ sludge

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

        # ---------------------------
        # buses
        # ---------------------------
        self.electricity_bus = attributes.pop("electricity_bus")
        self.water_in_bus = attributes.pop("water_in_bus")
        self.water_out_bus = attributes.pop("water_out_bus")
        self.sludge_out_bus = attributes.pop("sludge_out_bus")

        # primary bus
        self.primary = attributes.pop("primary", self.primary)
        if not self.primary:
            self.primary = self.water_out_bus.label

        # ---------------------------
        # physics
        # ---------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop(
            "efficiency", self.efficiency
        )

        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1]")

        # ---------------------------
        # economics
        # ---------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.sludge_disposal_cost = attributes.pop(
            "sludge_disposal_cost", self.sludge_disposal_cost
        )

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

        feedwater_per_output = 1.0 / self.efficiency
        sludge_per_output = feedwater_per_output - 1.0
        electricity_per_output = self.specific_energy_consumption

        attributes.update(
            {
                # inputs
                f"conversion_factor_{self.electricity_bus.label}":
                    sequence(electricity_per_output),
                f"conversion_factor_{self.water_in_bus.label}":
                    sequence(feedwater_per_output),

                # outputs
                f"conversion_factor_{self.water_out_bus.label}":
                    sequence(1.0),
                f"conversion_factor_{self.sludge_out_bus.label}":
                    sequence(sludge_per_output),
            }
        )

        # ---------------------------
        # output-specific costs
        # ---------------------------
        attributes.setdefault("output_parameters", {})
        attributes.setdefault("output_parameters_1", {})

        if self.sludge_disposal_cost > 0:
            attributes["output_parameters_1"].update(
                {"variable_costs": self.sludge_disposal_cost}
            )

        # ==================================================
        # MIMO initialization
        # ==================================================
        super().__init__(
            from_bus_0=self.electricity_bus,
            from_bus_1=self.water_in_bus,
            to_bus_0=self.water_out_bus,   # PRIMARY
            to_bus_1=self.sludge_out_bus,
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
