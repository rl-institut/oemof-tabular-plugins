import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO

@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class CentralizedWWTP(MIMO):
    """
    Literature-informed centralized wastewater treatment plant facade (v3.0).

    Purpose
    -------
    Linear whole-plant surrogate for a centralized municipal WWTP based on
    activated-sludge treatment concepts. Designed as a planning-level component
    for oemof-style optimization, not a full mechanistic ASM model.
    All conversion factors are normalized to 1 m³ influent wastewater.

    Core references
    ---------------
    1. Tchobanoglous et al. (2014): Wastewater Engineering: Treatment and
       Resource Recovery (5th ed.) — core mass-balance, biomass yield, oxygen
       requirements, and activated-sludge design.
    2. WEF OM-9 (3rd ed.): Activated Sludge and Nutrient Removal — operational
       logic, sludge-age selection, wasting rates, and nutrient-removal control.
    3. Nguyen et al. (2025): Energy efficiency evaluation of a centralised WWTP
       in an industrial zone — empirical SEC per m³ and per kg pollutant removed.
    4. PCA et al. (2024): Energy efficiency benchmarking of WWTPs — whole-plant
       SEC benchmarking methodology and DEA-based performance framing.
    5. Abbadi et al. (2025): Comprehensive assessment of WWTP contributions to
       urban GHG and ammonia emissions — direct CH4 / N2O emission factors.

    Main equations
    --------------
    Whole-plant electricity demand per m³ influent
    (Tchobanoglous et al. 2014; Nguyen et al. 2025):
        E_total(t) = E_base
                   + E_aer  * BOD_removed
                   + E_nit  * TN_removed
        [kWh/m³_in]

    Hydraulic recovery:
        V_treated(t) = water_recovery * V_influent(t)
        [m³/hr]

    Sludge generation (independent of hydraulic recovery):
        V_sludge(t) = sludge_yield * V_influent(t)
        [m³/hr]

    Optional direct emissions (Abbadi et al. 2025):
        E_CH4(t) = direct_ch4_kgco2e_per_m3_in * V_influent(t)
        E_N2O(t) = direct_n2o_kgco2e_per_m3_in * V_influent(t)

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum
      treated-water output of the plant.
    - Sludge yield is an independent parameter, not derived from water recovery.
      Actual biological sludge production is governed by observed yield and SRT
      (Tchobanoglous et al. 2014, Ch. 7), not hydraulic loss.
    - Electricity demand is decomposed into a base load, an aeration-linked
      BOD-removal term, and a nitrification-linked TN-removal term. If only a
      lumped SEC is available, set aeration and nitrification terms to 0 and
      use base_energy_kwh_per_m3_in alone.
    - Optional direct-emissions buses (CH4, N2O) can be connected for
      environmental accounting. If buses are absent, emissions are ignored.
    - Operational variables (SRT, HRT, F:M, DO, RAS, WAS) are stored as
      documentation / calibration metadata. They are not enforced as hard
      optimization constraints in v3.0.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "centralized_WWTP"
    name: str = ""
    tech: str = "wastewater-treatment"
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
    electricity_bus: Bus = None      # kWh
    water_in_bus: Bus = None         # m³ influent wastewater
    water_out_bus: Bus = None        # m³ treated water
    sludge_out_bus: Bus = None       # m³ sludge

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    ch4_emissions_bus: Optional[Bus] = None    # kgCO2e
    n2o_emissions_bus: Optional[Bus] = None    # kgCO2e

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    water_recovery: float = 0.98                             # m³_out / m³_in
    sludge_yield: float = 0.01                               # m³_sludge / m³_in
    base_energy_kwh_per_m3_in: float = 0.05                  # kWh / m³_in
    aeration_energy_kwh_per_kg_bod_removed: float = 0.0      # kWh / kgBOD
    removed_bod_kg_per_m3_in: float = 0.0                    # kgBOD / m³_in
    nitrification_energy_kwh_per_kg_tn_removed: float = 0.0  # kWh / kgTN
    removed_tn_kg_per_m3_in: float = 0.0                     # kgTN / m³_in
    direct_ch4_kgco2e_per_m3_in: float = 0.0                 # kgCO2e / m³_in
    direct_n2o_kgco2e_per_m3_in: float = 0.0                 # kgCO2e / m³_in

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0          # €/m³ treated water
    carrier_cost: float = 0.0           # €/kWh electricity
    sludge_disposal_cost: float = 0.0   # €/m³ sludge
    ch4_emissions_cost: float = 0.0     # €/kgCO2e CH4
    n2o_emissions_cost: float = 0.0     # €/kgCO2e N2O

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # Tchobanoglous et al. (2014) and WEF OM-9 style operational descriptors
    # ------------------------------------------------------------------
    srt_days: float = None
    hrt_hours: float = None
    fm_ratio: float = None
    do_setpoint_mg_per_l: float = None
    ras_ratio: float = None
    was_ratio: float = None

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
        self.sludge_out_bus = attributes.pop("sludge_out_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.ch4_emissions_bus = attributes.pop("ch4_emissions_bus", None)
        self.n2o_emissions_bus = attributes.pop("n2o_emissions_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.water_recovery = attributes.pop(
            "water_recovery", self.water_recovery
        )
        self.sludge_yield = attributes.pop(
            "sludge_yield", self.sludge_yield
        )
        self.base_energy_kwh_per_m3_in = attributes.pop(
            "base_energy_kwh_per_m3_in", self.base_energy_kwh_per_m3_in
        )
        self.aeration_energy_kwh_per_kg_bod_removed = attributes.pop(
            "aeration_energy_kwh_per_kg_bod_removed",
            self.aeration_energy_kwh_per_kg_bod_removed,
        )
        self.removed_bod_kg_per_m3_in = attributes.pop(
            "removed_bod_kg_per_m3_in", self.removed_bod_kg_per_m3_in
        )
        self.nitrification_energy_kwh_per_kg_tn_removed = attributes.pop(
            "nitrification_energy_kwh_per_kg_tn_removed",
            self.nitrification_energy_kwh_per_kg_tn_removed,
        )
        self.removed_tn_kg_per_m3_in = attributes.pop(
            "removed_tn_kg_per_m3_in", self.removed_tn_kg_per_m3_in
        )
        self.direct_ch4_kgco2e_per_m3_in = attributes.pop(
            "direct_ch4_kgco2e_per_m3_in", self.direct_ch4_kgco2e_per_m3_in
        )
        self.direct_n2o_kgco2e_per_m3_in = attributes.pop(
            "direct_n2o_kgco2e_per_m3_in", self.direct_n2o_kgco2e_per_m3_in
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.sludge_disposal_cost = attributes.pop(
            "sludge_disposal_cost", self.sludge_disposal_cost
        )
        self.ch4_emissions_cost = attributes.pop(
            "ch4_emissions_cost", self.ch4_emissions_cost
        )
        self.n2o_emissions_cost = attributes.pop(
            "n2o_emissions_cost", self.n2o_emissions_cost
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
        self.srt_days = attributes.pop("srt_days", self.srt_days)
        self.hrt_hours = attributes.pop("hrt_hours", self.hrt_hours)
        self.fm_ratio = attributes.pop("fm_ratio", self.fm_ratio)
        self.do_setpoint_mg_per_l = attributes.pop(
            "do_setpoint_mg_per_l", self.do_setpoint_mg_per_l
        )
        self.ras_ratio = attributes.pop("ras_ratio", self.ras_ratio)
        self.was_ratio = attributes.pop("was_ratio", self.was_ratio)

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived whole-plant electricity demand per m³ influent
        # E_total = E_base + E_aer * BOD_removed + E_nit * TN_removed
        # (Tchobanoglous et al. 2014; Nguyen et al. 2025)
        # --------------------------------------------------------------
        self._electricity_per_m3_in = (
            self.base_energy_kwh_per_m3_in
            + self.aeration_energy_kwh_per_kg_bod_removed
            * self.removed_bod_kg_per_m3_in
            + self.nitrification_energy_kwh_per_kg_tn_removed
            * self.removed_tn_kg_per_m3_in
        )

        # --------------------------------------------------------------
        # conversion factors
        # Normalization basis: 1 m³ influent wastewater (water_in_bus)
        # electricity:  kWh per m³ influent
        # water_in:     1.0 (reference flow)
        # water_out:    water_recovery  [m³_out / m³_in]
        # sludge_out:   sludge_yield    [m³_sludge / m³_in]
        # --------------------------------------------------------------
        attributes[
            f"conversion_factor_{self.electricity_bus.label}"
        ] = sequence(self._electricity_per_m3_in)
        attributes[
            f"conversion_factor_{self.water_in_bus.label}"
        ] = sequence(1.0)
        attributes[
            f"conversion_factor_{self.water_out_bus.label}"
        ] = sequence(self.water_recovery)
        attributes[
            f"conversion_factor_{self.sludge_out_bus.label}"
        ] = sequence(self.sludge_yield)

        # --------------------------------------------------------------
        # optional direct emissions conversion factors
        # (Abbadi et al. 2025)
        # --------------------------------------------------------------
        if self.ch4_emissions_bus is not None:
            attributes[
                f"conversion_factor_{self.ch4_emissions_bus.label}"
            ] = sequence(self.direct_ch4_kgco2e_per_m3_in)

        if self.n2o_emissions_bus is not None:
            attributes[
                f"conversion_factor_{self.n2o_emissions_bus.label}"
            ] = sequence(self.direct_n2o_kgco2e_per_m3_in)

        # --------------------------------------------------------------
        # output-specific variable costs
        # --------------------------------------------------------------
        attributes.setdefault("output_parameters", {})
        attributes.setdefault("output_parameters_1", {})

        if self.sludge_disposal_cost != 0:
            attributes["output_parameters_1"].update(
                {"variable_costs": self.sludge_disposal_cost}
            )

        # --------------------------------------------------------------
        # primary bus label resolution
        # --------------------------------------------------------------
        if self.primary == "water_out_bus":
            primary_label = self.water_out_bus.label
        elif self.primary == "water_in_bus":
            primary_label = self.water_in_bus.label
        elif self.primary == "sludge_out_bus":
            primary_label = self.sludge_out_bus.label
        else:
            primary_label = self.primary

        # --------------------------------------------------------------
        # initialize base MIMO facade
        # --------------------------------------------------------------
        super().__init__(
            from_bus_0=self.electricity_bus,
            from_bus_1=self.water_in_bus,
            to_bus_0=self.water_out_bus,
            to_bus_1=self.sludge_out_bus,
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
        idx_out = 2
        # inputs
        for bus in []:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.ch4_emissions_bus,
            self.n2o_emissions_bus,
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        # --- active physical parameters ---
        if not 0 < self.water_recovery <= 1:
            raise ValueError("water_recovery must be in (0, 1].")

        if self.sludge_yield < 0:
            raise ValueError("sludge_yield must be >= 0.")

        nonneg = {
            "base_energy_kwh_per_m3_in": self.base_energy_kwh_per_m3_in,
            "aeration_energy_kwh_per_kg_bod_removed": self.aeration_energy_kwh_per_kg_bod_removed,
            "removed_bod_kg_per_m3_in": self.removed_bod_kg_per_m3_in,
            "nitrification_energy_kwh_per_kg_tn_removed": self.nitrification_energy_kwh_per_kg_tn_removed,
            "removed_tn_kg_per_m3_in": self.removed_tn_kg_per_m3_in,
            "direct_ch4_kgco2e_per_m3_in": self.direct_ch4_kgco2e_per_m3_in,
            "direct_n2o_kgco2e_per_m3_in": self.direct_n2o_kgco2e_per_m3_in,
        }
        for name, value in nonneg.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")

        # --- warn if emissions bus is set but factor is zero ---
        if self.ch4_emissions_bus is not None and self.direct_ch4_kgco2e_per_m3_in == 0:
            warnings.warn(
                "ch4_emissions_bus is set but direct_ch4_kgco2e_per_m3_in is 0. "
                "The CH4 emission output will always be zero.",
                UserWarning,
            )
        if self.n2o_emissions_bus is not None and self.direct_n2o_kgco2e_per_m3_in == 0:
            warnings.warn(
                "n2o_emissions_bus is set but direct_n2o_kgco2e_per_m3_in is 0. "
                "The N2O emission output will always be zero.",
                UserWarning,
            )

        # --- documentation / calibration only ---
        if self.srt_days is not None and self.srt_days <= 0:
            raise ValueError("srt_days must be > 0 if provided.")
        if self.hrt_hours is not None and self.hrt_hours <= 0:
            raise ValueError("hrt_hours must be > 0 if provided.")
        if self.fm_ratio is not None and self.fm_ratio < 0:
            raise ValueError("fm_ratio must be >= 0 if provided.")
        if self.do_setpoint_mg_per_l is not None and self.do_setpoint_mg_per_l < 0:
            raise ValueError("do_setpoint_mg_per_l must be >= 0 if provided.")
        if self.ras_ratio is not None and self.ras_ratio < 0:
            raise ValueError("ras_ratio must be >= 0 if provided.")
        if self.was_ratio is not None and self.was_ratio < 0:
            raise ValueError("was_ratio must be >= 0 if provided.")

