import dataclasses
from typing import Sequence, Union

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class MIMO_BioFiltration(MIMO):
    """
    Specialized MIMO-B Biofiltration facade with code-driven physics.

    Inputs:
    - electricity_bus
    - water_in_bus

    Outputs:
    - water_out_bus (PRIMARY)
    - waste_biomass_out_bus

    Conversion factors are derived internally from:
    - specific_energy_consumption
    - efficiency
    - biomass_waste_fraction
    """

    # ---- tabular identity ----
    type: str = "mimo_biof"
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
    waste_biomass_out_bus: Bus = None

    # ---- physics parameters ----
    specific_energy_consumption: float = 0.12   # kWh / m³ treated water
    efficiency: float = 0.85
    biomass_waste_fraction: float = 0.005       # fraction of treated water

    # ---- nutrient parameters ----
    nutrient_dose: float = 1.0                  # mg/L = g/m³
    nutrient_cost: float = 1.0                  # USD/kg

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
        self.waste_biomass_out_bus = attributes.pop("waste_biomass_out_bus")

        # ---------------------------
        # physics
        # ---------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop(
            "efficiency", self.efficiency
        )
        self.biomass_waste_fraction = attributes.pop(
            "biomass_waste_fraction", self.biomass_waste_fraction
        )

        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1]")
        if not 0 <= self.biomass_waste_fraction <= 1:
            raise ValueError("biomass_waste_fraction must be in [0, 1]")

        # ---------------------------
        # nutrients
        # ---------------------------
        self.nutrient_dose = attributes.pop("nutrient_dose", self.nutrient_dose)
        self.nutrient_cost = attributes.pop("nutrient_cost", self.nutrient_cost)

        # nutrient cost per m³ treated water
        nutrient_cost_per_m3 = (
            self.nutrient_dose * 1e-6 * self.nutrient_cost
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

        feedwater_per_output = 1.0 / self.efficiency
        electricity_per_output = self.specific_energy_consumption
        biomass_per_output = self.biomass_waste_fraction

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
                f"conversion_factor_{self.waste_biomass_out_bus.label}":
                    sequence(biomass_per_output),
            }
        )

        # ---------------------------
        # output costs
        # ---------------------------
        attributes.setdefault("output_parameters", {})
        attributes["output_parameters"].update(
            {
                "variable_costs": nutrient_cost_per_m3 + self.marginal_cost,
                "custom_attributes": {"nutrient_dose": self.nutrient_dose},
            }
        )

        # ==================================================
        # MIMO initialization
        # ==================================================
        super().__init__(
            from_bus_0=self.electricity_bus,
            from_bus_1=self.water_in_bus,
            to_bus_0=self.water_out_bus,   # PRIMARY
            to_bus_1=self.waste_biomass_out_bus,
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
