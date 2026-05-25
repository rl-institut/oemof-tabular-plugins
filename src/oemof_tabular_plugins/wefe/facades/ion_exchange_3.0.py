import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class IonExchange(MIMO):
    """
    Literature-informed ion exchange water treatment facade based on MIMO.

    Purpose
    -------
    Generic water-treatment facade for ion exchange (IX) systems used in
    desalination pre-treatment, softening, nitrate removal, and similar
    applications. The model is designed as a bookkeeping/process-yield unit
    representing an IX unit as a water-treatment intervention. It is not a
    full breakthrough-cycle or contaminant-transport model.

    Core references
    ---------------
    1. WaterTAP Technical Brief: Ion Exchange Model Demonstration and
       Optimization (NREL/OSTI-86512, 2023): steady-state model variables,
       cost structure, resin capacity and selectivity as key sensitivities.
    2. Veolia Handbook of Industrial Water Treatment, Chapter 8 – Ion Exchange
       & Water Demineralization: regeneration steps, exhaustion behavior,
       practical operating limits, and cost relevance of regenerant disposal.
    3. Ion Exchange for Water Treatment (PDH Academy course note, 2023):
       service/breakthrough/regeneration logic and parallel-vessel
       engineering simplifications.
    4. AWWA Ion Exchange for Drinking Water Treatment: gold-standard design
       reference for operating modes, resin selection, and system design.

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Feedwater requirement (Veolia / PDH):
        feedwater_per_output = 1 / water_recovery       [m³_feed / m³_product]

    Electricity demand (WaterTAP):
        electricity_per_output = SEC                    [kWh / m³_product]

    Regenerant demand (Veolia; PDH):
        regenerant_per_output = regenerant_dose_kg_per_m3_product
                                                        [kg_chem / m³_product]

    Brine reject output (Veolia mass balance):
        brine_per_output = 1 / water_recovery - 1.0     [m³_brine / m³_product]

    Output costs (placed on flow edges, not lumped into marginal_cost):
        water_out_bus:   resin_replacement_cost_per_m3_product  [€/m³]  → output_parameters
        waste_brine_bus: waste_disposal_cost_per_m3_product     [€/m³]  → output_parameters_1

    Note: regenerant chemical cost should be included in marginal_cost at
    scenario/tabular level (marginal_cost = base_opex + regenerant_cost_per_m3).

    Availability derating (activity bound, PDH; Veolia):
        Q_net(t) <= availability * Q_cap                [m³/hr]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum
      treated-water throughput of the IX unit.
    - Electricity and feedwater inputs are normalized to one unit of treated
      water output via conversion factors (WaterTAP steady-state convention).
    - regenerant_bus is the correct optional INPUT for chemical supply
      (HCl / NaOH / NaCl). It acts as a material flow tracker in the WEFE
      graph. Chemical cost should be included in marginal_cost at scenario level.
    - waste_brine_bus is the correct optional OUTPUT for spent regenerant /
      brine reject. Its conversion factor is derived from water_recovery.
      Disposal cost is placed as variable_costs on output_parameters_1.
    - availability is implemented as an activity_bound_max constraint, not as
      a hidden multiplier inside SEC or water_recovery, to keep cost
      interpretation unambiguous.
    - Resin capacity, selectivity, breakthrough, and service-loading fields
      are kept for scenario documentation and future extension. They are not
      enforced as hard optimization constraints in v3.0 because breakthrough
      prediction requires cycle-state or contaminant-balance logic (WaterTAP).
    - The backward-compatible alias 'efficiency' is accepted as water_recovery
      to allow drop-in replacement of v2.0 instances.
    - Characterization values (resin capacity, selectivity, service loading,
      breakthrough metrics) are stored as documentation/calibration defaults.
      They are not enforced as hard optimization constraints in v3.0.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "ion_exchange"
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
    electricity_bus: Bus = None  # kWh
    water_in_bus: Bus = None  # m³
    water_out_bus: Bus = None  # m³  (PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    regenerant_bus: Optional[Bus] = None  # kg  (HCl / NaOH / NaCl supply)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    waste_brine_bus: Optional[Bus] = None  # m³  (spent regenerant / brine reject)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.06  # kWh / m³ treated water (WaterTAP)
    water_recovery: float = 0.96  # m³ treated / m³ feed (Veolia / PDH)
    availability: float = 1.0  # fraction of productive operating time
    regenerant_dose_kg_per_m3_product: float = 0.0  # kg chemical / m³ treated water (Veolia; PDH)

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0  # €/m³ treated water (incl. regenerant cost at scenario level)
    carrier_cost: float = 0.0  # €/kWh electricity
    waste_disposal_cost_per_m3_product: float = 0.0  # €/m³ brine (Veolia handbook)
    resin_replacement_cost_per_m3_product: float = 0.0  # €/m³ treated water (WaterTAP costing)

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # WaterTAP and Veolia handbook style characterization fields
    # ------------------------------------------------------------------
    sec_typical_min: float = 0.03  # kWh/m³  literature lower bound (WaterTAP)
    sec_typical_max: float = 0.10  # kWh/m³  literature upper bound (WaterTAP)
    target_ion: str = ""  # e.g. "nitrate", "hardness", "fluoride"
    influent_concentration_mgL: float = None  # mg/L   feed contaminant concentration
    resin_capacity_eq_per_m3_resin: float = None  # eq/m³  total exchange capacity of resin bed
    resin_selectivity: float = None  # –      selectivity coefficient vs. reference ion
    service_flow_rate_bvph: float = None  # BV/hr  bed volumes per hour in service mode
    breakthrough_fraction: float = None  # –      C/C₀ at which bed is considered exhausted
    breakthrough_time_h: float = None  # hr     time to breakthrough at design conditions
    regenerant_type: str = ""  # e.g. "HCl", "NaOH", "NaCl"
    regenerant_dose_relative_to_stoich: float = None  # –  excess factor over stoichiometric dose
    bed_volumes_per_cycle: float = None  # BV     service BV between two regenerations
    parallel_trains: int = None  # –      number of parallel IX vessels

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
        self.regenerant_bus = attributes.pop("regenerant_bus", None)
        self.waste_brine_bus = attributes.pop("waste_brine_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.water_recovery = attributes.pop("water_recovery", self.water_recovery)

        self.availability = attributes.pop("availability", self.availability)
        self.regenerant_dose_kg_per_m3_product = attributes.pop(
            "regenerant_dose_kg_per_m3_product", self.regenerant_dose_kg_per_m3_product
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.waste_disposal_cost_per_m3_product = attributes.pop(
            "waste_disposal_cost_per_m3_product", self.waste_disposal_cost_per_m3_product
        )
        self.resin_replacement_cost_per_m3_product = attributes.pop(
            "resin_replacement_cost_per_m3_product", self.resin_replacement_cost_per_m3_product
        )
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

        # --------------------------------------------------------------
        # documentation / calibration defaults
        # --------------------------------------------------------------
        self.sec_typical_min = attributes.pop("sec_typical_min", self.sec_typical_min)
        self.sec_typical_max = attributes.pop("sec_typical_max", self.sec_typical_max)
        self.target_ion = attributes.pop("target_ion", self.target_ion)
        self.influent_concentration_mgL = attributes.pop(
            "influent_concentration_mgL", self.influent_concentration_mgL
        )
        self.resin_capacity_eq_per_m3_resin = attributes.pop(
            "resin_capacity_eq_per_m3_resin", self.resin_capacity_eq_per_m3_resin
        )
        self.resin_selectivity = attributes.pop(
            "resin_selectivity", self.resin_selectivity
        )
        self.service_flow_rate_bvph = attributes.pop(
            "service_flow_rate_bvph", self.service_flow_rate_bvph
        )
        self.breakthrough_fraction = attributes.pop(
            "breakthrough_fraction", self.breakthrough_fraction
        )
        self.breakthrough_time_h = attributes.pop(
            "breakthrough_time_h", self.breakthrough_time_h
        )
        self.regenerant_type = attributes.pop(
            "regenerant_type", self.regenerant_type
        )
        self.regenerant_dose_relative_to_stoich = attributes.pop(
            "regenerant_dose_relative_to_stoich",
            self.regenerant_dose_relative_to_stoich,
        )
        self.bed_volumes_per_cycle = attributes.pop(
            "bed_volumes_per_cycle", self.bed_volumes_per_cycle
        )
        self.parallel_trains = attributes.pop(
            "parallel_trains", self.parallel_trains
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # (WaterTAP steady-state convention; Veolia / PDH mass balance)
        # --------------------------------------------------------------
        self._feedwater_per_output = 1.0 / self.water_recovery
        self._brine_per_output = self._feedwater_per_output - 1.0

        # --------------------------------------------------------------
        # conversion factors
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.regenerant_bus is not None:
            attributes[f"conversion_factor_{self.regenerant_bus.label}"] = sequence(
                self.regenerant_dose_kg_per_m3_product
            )

        if self.waste_brine_bus is not None:
            attributes[f"conversion_factor_{self.waste_brine_bus.label}"] = sequence(
                self._brine_per_output
            )

        # --------------------------------------------------------------
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # (WaterTAP costing; Veolia handbook)
        # --------------------------------------------------------------
        if self.resin_replacement_cost_per_m3_product > 0:
            attributes.setdefault("output_parameters", {})
            attributes["output_parameters"].update(
                {"variable_costs": self.resin_replacement_cost_per_m3_product}
            )

        if self.waste_disposal_cost_per_m3_product > 0 and self.waste_brine_bus is not None:
            attributes.setdefault("output_parameters_1", {})
            attributes["output_parameters_1"].update(
                {"variable_costs": self.waste_disposal_cost_per_m3_product}
            )

        # --------------------------------------------------------------
        # availability as activity bound
        # derates maximum productive output without distorting SEC or
        # water_recovery (PDH design note; Veolia operational derating)
        # --------------------------------------------------------------
        if self.availability < 1.0:
            attributes["activity_bound_max"] = sequence(self.availability)

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

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 1
        # inputs
        for bus in [
            self.regenerant_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.waste_brine_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.water_recovery <= 1:
            raise ValueError("water_recovery must be in (0, 1].")
        if not 0 < self.availability <= 1:
            raise ValueError("availability must be in (0, 1].")
        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")
        if self.regenerant_dose_kg_per_m3_product < 0:
            raise ValueError("regenerant_dose_kg_per_m3_product must be >= 0.")

        if self.regenerant_bus is not None and self.regenerant_dose_kg_per_m3_product == 0:
            warnings.warn(
                "regenerant_bus connected but regenerant_dose_kg_per_m3_product=0 "
                "— regenerant flow will be zero.",
                UserWarning,
            )
        if self.regenerant_dose_kg_per_m3_product > 0 and self.regenerant_bus is None:
            warnings.warn(
                "regenerant_dose_kg_per_m3_product > 0 but no regenerant_bus — "
                "chemical consumption not tracked in the energy system graph.",
                UserWarning,
            )

        if self.breakthrough_fraction is not None and not 0 < self.breakthrough_fraction <= 1:
            raise ValueError("breakthrough_fraction must be in (0, 1].")
        if self.resin_capacity_eq_per_m3_resin is not None and self.resin_capacity_eq_per_m3_resin <= 0:
            raise ValueError("resin_capacity_eq_per_m3_resin must be > 0.")
