import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class WaterPostTreatment(MIMO):
    """
    Lumped water post treatment facade based on MIMO.

    Purpose
    -------
    Lumped-parameter facade representing the final disinfection/polishing stage of a water treatment train: UV disinfection
    and/or chlorination, applied after core treatment to ensure microbiological safety before delivery. Like pre- and
    core-treatment, this stage may produce a concentrate stream and produce biomass, it treats the full incoming flow with
    negligible volume loss.

    Core references
    ---------------
    - Please refer to the individual MIMO based facades particularly intake structure, reverse osmosis, biofiltration,
      biological denitrification located in path : src/oemof_tabular_plugins/wefe/facades

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Feedwater requirement:
        V_feed(t) = V_treated(t) / efficiency
        [m³ feed/hr]                [-]

    Cumulative energy demand (screens, mixers, blowers, cartridge filter
    pumping, coag/floc mixing, etc.):
        E(t) = specific_energy_consumption × V_treated(t)
        [kWh/hr]

    Reject/brine (if brine_out_bus provided):
        V_brine(t) = (1/efficiency - 1) × V_treated(t)

    Waste biomass (if waste_biomass_out_bus provided):
        V_waste(t) = biomass_waste_fraction × V_treated(t)

    N2 gas (if N2_gas_bus provided):
        NO3N_removed(t) = Cin_no3n × target_removal_efficiency × anoxic_factor
        N2_produced(t)  = NO3N_removed(t) × n2_yield_per_no3n_removed × 1e-3
        [kg N2 / m3 treated water]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum treated water throughput of the post treatment stage.
    - Brine, biomass, or gas byproduct streams are modeled as optional buses/outputs in this train and may physically produce
      any of these.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "water_post_treatment"
    name: str = ""
    tech: str = "water-treatment"
    carrier: str = "water"
    primary: str = "water_out_bus"

    # ------------------------------------------------------------------
    # capacity / investment
    # ------------------------------------------------------------------
    expandable: bool = False
    capacity: float = None
    capacity_cost: float = None
    capacity_minimum: float = None
    capacity_potential: float = None

    # ------------------------------------------------------------------
    # mandatory buses
    # ------------------------------------------------------------------
    electricity_bus: Bus = None     # kWh
    water_in_bus: Bus = None        # m³ (feed water)
    water_out_bus: Bus = None       # m³ (treated water - PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    brine_out_bus: Optional[Bus] = None                 # m³  (brine reject)
    waste_biomass_out_bus: Optional[Bus] = None         # m³  (waste biomass)
    N2_gas_bus: Optional[Bus] = None                    # kg N2 (or kg N-equivalent)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = None              # kWh/m³ treated water
    efficiency: float = None                               # m³ treated water/m³ feed water
    biomass_waste_fraction: float = 0.005                  # m³ biomass / m³ treated water
    Cin_no3n: float = 30.0                                 # g N / m³ feed water
    target_removal_efficiency: float = 0.90                # 0..1
    anoxic_factor: float = 1.0                             # 0..1; derates removal for imperfect anoxic conditions
    n2_yield_per_no3n_removed: float = 1.0                 # kg N2 / kg NO3-N removed
    nutrient_dose_mg_per_L: float = 1.0                    # mg nutrient / L net treated water
    nutrient_cost_per_kg: float = 1.0                      # USD/kg nutrient
    carbon_source_dose: float = 90.0                       # g/m³ = mg/L net treated water
    carbon_source_cost: float = 0.40                       # USD/kg COD-equivalent

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # USD/m³ treated water
    carrier_cost: float = 0.0               # USD/m³ feed water
    brine_disposal_cost: float = 0.0        # USD/m³ brine
    biomass_disposal_cost: float = 0.0      # USD/m³ waste biomass

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints)
    # Based on the core literature references
    # ------------------------------------------------------------------
    # reserved for future extension

    def __init__(self, **attributes):
        # --------------------------------------------------------------
        # identity
        # --------------------------------------------------------------
        self.type = attributes.pop("type", self.type)
        self.name = attributes.pop("name", self.name)
        self.tech = attributes.pop("tech", self.tech)
        self.carrier = attributes.pop("carrier", self.carrier)
        self.primary = attributes.pop("primary", self.primary)

        # --------------------------------------------------------------
        # mandatory buses
        # --------------------------------------------------------------
        self.electricity_bus = attributes.pop("electricity_bus")
        self.water_in_bus = attributes.pop("water_in_bus")
        self.water_out_bus = attributes.pop("water_out_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.brine_out_bus = attributes.pop("brine_out_bus", None)
        self.waste_biomass_out_bus = attributes.pop("waste_biomass_out_bus", None)
        self.N2_gas_bus = attributes.pop("N2_gas_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop(
            "efficiency", self.efficiency
        )
        self.biomass_waste_fraction = attributes.pop(
            "biomass_waste_fraction", self.biomass_waste_fraction
        )
        self.Cin_no3n = attributes.pop(
            "Cin_no3n", self.Cin_no3n
        )
        self.target_removal_efficiency = attributes.pop(
            "target_removal_efficiency", self.target_removal_efficiency
        )
        self.anoxic_factor = attributes.pop(
            "anoxic_factor", self.anoxic_factor
        )
        self.n2_yield_per_no3n_removed = attributes.pop(
            "n2_yield_per_no3n_removed", self.n2_yield_per_no3n_removed
        )
        self.nutrient_dose_mg_per_L = attributes.pop(
            "nutrient_dose_mg_per_L", self.nutrient_dose_mg_per_L
        )
        self.nutrient_cost_per_kg = attributes.pop(
            "nutrient_cost_per_kg", self.nutrient_cost_per_kg
        )
        self.carbon_source_dose = attributes.pop(
            "carbon_source_dose", self.carbon_source_dose
        )
        self.carbon_source_cost = attributes.pop(
            "carbon_source_cost", self.carbon_source_cost
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.brine_disposal_cost = attributes.pop("brine_disposal_cost", self.brine_disposal_cost)
        self.biomass_disposal_cost = attributes.pop("biomass_disposal_cost", self.biomass_disposal_cost)
        self.expandable = attributes.pop("expandable", self.expandable)
        self.capacity = attributes.pop("capacity", self.capacity)
        self.capacity_cost = attributes.pop("capacity_cost", self.capacity_cost)
        self.capacity_minimum = attributes.pop(
            "capacity_minimum", self.capacity_minimum
        )
        self.capacity_potential = attributes.pop(
            "capacity_potential", self.capacity_potential
        )

        # --------------------------------------------------------------
        # multiperiod
        # --------------------------------------------------------------
        self.lifetime = attributes.pop("lifetime", self.lifetime)
        self.age = attributes.pop("age", self.age)
        self.fixed_costs = attributes.pop("fixed_costs", self.fixed_costs)
        self.output_parameters = attributes.pop("output_parameters", {})

        # --------------------------------------------------------------
        # documentation / calibration defaults
        # --------------------------------------------------------------
        # reserved for future extension

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._feedwater_per_treated_water = 1.0 / self.efficiency
        self._brine_per_treated_water = self._feedwater_per_treated_water - 1.0
        self._biomass_per_treated_water = self.biomass_waste_fraction
        effective_removal_efficiency = (
                self.target_removal_efficiency * self.anoxic_factor
        )
        no3n_removed_per_m3 = self.Cin_no3n * effective_removal_efficiency
        n2_production_per_m3 = no3n_removed_per_m3 * self.n2_yield_per_no3n_removed * 1e-3
        # 1 mg/L = 1 g/m³ = 0.001 kg/m³  →  factor = * 1e-3
        self.water_out_variable_costs = 0.0
        if self.waste_biomass_out_bus is not None:
            # Nutrient cost per m³ of treated water:
            # nutrient_dose [mg/L] * 1e-3 [kg/m³ per mg/L] * nutrient_cost [USD/kg]
            nutrient_cost_per_m3 = (self.nutrient_dose_mg_per_L * 1e-3 * self.nutrient_cost_per_kg)
            self.water_out_variable_costs += nutrient_cost_per_m3
        if self.N2_gas_bus is not None:
            # Carbon source dose per m³ of treated water:
            # carbon_source_dose [mg/L] * 1e-3 [kg/m³ per mg/L] * carbon_source_cost [USD/kg]
            carbon_source_cost_per_m3 = (self.carbon_source_dose * 1e-3 * self.carbon_source_cost)
            self.water_out_variable_costs += carbon_source_cost_per_m3

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_treated_water
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.brine_out_bus is not None:
            attributes[f"conversion_factor_{self.brine_out_bus.label}"] = sequence(
                self._brine_per_treated_water
            )

        if self.waste_biomass_out_bus is not None:
            attributes[f"conversion_factor_{self.waste_biomass_out_bus.label}"] = sequence(
                self._biomass_per_treated_water
            )
        if self.N2_gas_bus is not None:
            attributes[f"conversion_factor_{self.N2_gas_bus.label}"] = sequence(
                n2_production_per_m3
            )

        # --------------------------------------------------------------
        # availability as activity bound
        # --------------------------------------------------------------
        # reserved for future extension

        # --------------------------------------------------------------
        # primary bus label resolution
        # --------------------------------------------------------------
        if self.primary == "water_out_bus":
            primary_label = self.water_out_bus.label
        elif self.primary == "water_in_bus":
            primary_label = self.water_in_bus.label
        elif self.primary == "electricity_bus":
            primary_label = self.electricity_bus.label
        else:
            primary_label = self.primary

        # --------------------------------------------------------------
        # initialize base MIMO facade
        # --------------------------------------------------------------
        super().__init__(
            from_bus_0=self.electricity_bus,
            from_bus_1=self.water_in_bus,
            to_bus_0=self.water_out_bus,
            primary=primary_label,
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
            **self._optional_bus_kwargs(),
            **attributes,
        )

        # ------------------------------------------------------------
        # PATCH: MIMO's create_flow() (mimo_converter.py) never wires
        # variable_costs onto any Flow, and only ever sets nominal_value
        # on the primary bus's Flow when expandable=True. Patch the
        # already-built Flow objects directly since
        # MultiInputMultiOutputConverter/MIMO cannot be modified.
        # ------------------------------------------------------------
        self._apply_flow_parameters()

    def _apply_flow_parameters(self):

        # --------------------------------------------------------------
        # output-specific costs
        # --------------------------------------------------------------
        total_marginal_cost = self.marginal_cost + self.water_out_variable_costs

        if self.water_out_bus in self.outputs:
            out_flow = self.outputs[self.water_out_bus]
            out_flow.variable_costs = sequence(total_marginal_cost)
            if not self.expandable and self.capacity is not None:
                out_flow.nominal_value = self.capacity
            custom_attrs = (getattr(self, "output_parameters", None) or {}).get(
                "custom_attributes"
            )
            if custom_attrs:
                for attribute, value in custom_attrs.items():
                    setattr(out_flow, attribute, value)

        if self.brine_out_bus is not None and self.brine_out_bus in self.outputs:
            self.outputs[self.brine_out_bus].variable_costs = sequence(
                self.brine_disposal_cost
            )

        if self.waste_biomass_out_bus is not None and self.waste_biomass_out_bus in self.outputs:
            self.outputs[self.waste_biomass_out_bus].variable_costs = sequence(
                self.biomass_disposal_cost
            )

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 1
        # inputs
        for bus in []:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.brine_out_bus,
            self.waste_biomass_out_bus,
            self.N2_gas_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1].")
        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")
        if not 0 < self.biomass_waste_fraction <= 1:
            raise ValueError("biomass_waste_fraction must be in (0, 1].")
        if self.Cin_no3n < 0:
            raise ValueError("Cin_no3n must be >= 0.")
        if not 0 <= self.target_removal_efficiency <= 1:
            raise ValueError("target_removal_efficiency must be in [0, 1].")
        if not 0 <= self.anoxic_factor <= 1:
            raise ValueError("anoxic_factor must be in [0, 1].")

        non_negative = {
            "biomass_disposal_cost": self.biomass_disposal_cost,
            "n2_yield_per_no3n_removed": self.n2_yield_per_no3n_removed,
            "nutrient_dose_mg_per_L": self.nutrient_dose_mg_per_L,
            "nutrient_cost_per_kg": self.nutrient_cost_per_kg,
            "carbon_source_dose": self.carbon_source_dose,
            "carbon_source_cost": self.carbon_source_cost,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")
