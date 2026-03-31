import dataclasses
from typing import Sequence, Union

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class MIMO_CoagulationFlocculation(MIMO):
    """
    Specialized MIMO-B Coagulation Flocculation facade with code-driven physics.

    Inputs:
    - electricity_bus
    - water_in_bus

    Outputs:
    - water_out_bus (PRIMARY)

    Conversion factors are derived internally from:
    - specific_energy_consumption
    """

    # ---- tabular identity ----
    type: str = "mimo_coag_flocc"
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

    # ---- physics parameters ----
    specific_energy_consumption: float = 0.04   # kWh / m³ treated water

    # ---- coagulant and flocculant parameters ----
    coagulant_dose: float = 20                  # mg/L = g/m³
    coagulant_cost: float = 1.5                 # USD/kg
    flocculant_dose: float = 2.55               # mg/L = g/m³
    flocculant_cost: float = 2.0                # USD/kg

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

        # ---------------------------
        # buses
        # ---------------------------
        self.electricity_bus = attributes.pop("electricity_bus")
        self.water_in_bus = attributes.pop("water_in_bus")
        self.water_out_bus = attributes.pop("water_out_bus")

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

        # ---------------------------
        # coagulant and flocculant
        # ---------------------------
        self.coagulant_dose = attributes.pop("coagulant_dose", self.coagulant_dose)
        self.coagulant_cost = attributes.pop("coagulant_cost", self.coagulant_cost)
        self.flocculant_dose = attributes.pop("flocculant_dose", self.flocculant_dose)
        self.flocculant_cost = attributes.pop("flocculant_cost", self.flocculant_cost)

        # coagulant and flocculant cost per m³ treated water
        coagulant_cost_per_m3 = (
                self.coagulant_dose * 1e-6 * self.coagulant_cost
        )
        flocculant_cost_per_m3 = (
                self.flocculant_dose * 1e-6 * self.flocculant_cost
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
                "variable_costs": flocculant_cost_per_m3 + coagulant_cost_per_m3 + self.marginal_cost,
                "custom_attributes": {"flocculant_dose": self.flocculant_dose,
                                      "coagulant_dose": self.coagulant_dose},
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
