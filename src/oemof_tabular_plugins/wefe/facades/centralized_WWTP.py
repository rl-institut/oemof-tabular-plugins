import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO

@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class CentralizedWWTP(MIMO):
    """
    Literature-informed generic centralized WWTP facade based on MIMO.

    Purpose
    -------
    Generic whole-plant surrogate for a centralized municipal WWTP based on
    activated-sludge treatment concepts. The model is designed as a process-yield
    unit representing a WWTP as a water-energy-resource intervention. It is not a
    full mechanistic biological reactor model or pollutant-state model.

    Core references
    ---------------
    1. Core mass-balance principles, biosolids yield relationships, and activated-sludge design fundamentals; basis for
       the efficiency (hydraulic recovery) and sludge_yield parameters.
       Metcalf & Eddy, Inc., Tchobanoglous, G., Stensel, H. F., Tsuchihashi, R., & Burton, F. L. (2014). Wastewater
       engineering: Treatment and resource recovery (5th ed.). McGraw-Hill Education.
    2. Activated-sludge operational logic: sludge-age (SRT) selection, wasting-rate calculation, RAS/WAS flow optimization,
       and DO/ORP set-point control; basis for srt_days, hrt_hours, fm_ratio, do_setpoint_mg_per_l, ras_ratio, was_ratio,
       and the aeration/nitrification-linked energy terms.
       Water Environment Federation. (2025). Activated sludge and nutrient removal (4th ed.). Water Environment Federation.
       https://prod.wef.org/publications/publications/books/activated-sludge-and-nutrient-removal-mop-om-9-4th-edition/
    3. Empirical specific energy consumption (SEC) per m³ treated and per kg pollutant removed, measured at a real
       centralized industrial WWTP; basis for base_energy_kwh_per_m3_out (SEC) and the BOD/TN-removal energy decomposition.
       Nguyen, V. T., Anh, L. H., Dao, T. M., Nguyen, T. A., & Le, T. T. (2025). Energy efficiency evaluation of a
       centralised wastewater treatment plant in an industrial zone of former Binh Duong province, Vietnam. Vietnam
       Journal of Science, Technology and Engineering. https://doaj.org/article/1b4f35d46a4242e88d761a93ece06c15
    4. Whole-plant SEC benchmarking methodology and DEA-based performance framing; supports documentation-only calibration
       checks against fleet-wide efficiency norms.
       Gallo, M., Malluta, D., Del Borghi, A., & Gagliano, E. (2024). A critical review on methodologies for the energy
       benchmarking of wastewater treatment plants. Sustainability, 16(5), 1922. https://doi.org/10.3390/su16051922
    5. Measured direct CH4/N2O emission factors normalized to influent BOD/TN loading, across 96 real US water resource
       recovery facilities; basis for direct_ch4_kgco2e_per_m3_out and direct_n2o_kgco2e_per_m3_out.
       Moore, D. P., Li, N., Song, C., Zhu, J.-J., Yi, H., Tao, L., McSpiritt, J., Sevostianov, V. I., Wendt, L. P.,
       Rojas-Robles, N. E., Hopkins, F. M., Ren, Z. J., & Zondlo, M. A. (2025). Comprehensive assessment of the
       contribution of wastewater treatment to urban greenhouse gas and ammonia emissions. Nature Water, 3, 1114–1124.
       https://doi.org/10.1038/s44221-025-00490-z

    Main equations
    --------------
    All flows normalized to treated water output = 1 [m³/hr]:

    Whole-plant electricity demand per m³ effluent (Tchobanoglous et al. 2014 [1]; Nguyen et al. 2025 [3]):
        E_total(t) = E_base (SEC) + E_aer  * BOD_removed + E_nit  * TN_removed
        [kWh/m³_out]

    Hydraulic recovery:
        V_treated(t) = efficiency * V_influent(t)
        [m³/hr]

    Sludge generation (independent of hydraulic recovery):
        V_sludge(t) = sludge_yield * V_treated(t)
        [m³/hr]

    Optional direct emissions (Moore et al. 2025 [5]):
        E_CH4(t) = direct_ch4_kgco2e_per_m3_out * V_treated(t)
        E_N2O(t) = direct_n2o_kgco2e_per_m3_out * V_treated(t)

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum treated-water output of the plant.
    - Sludge yield is an independent parameter, not derived from water recovery. Actual biological sludge production is
      governed by observed yield and SRT (Tchobanoglous et al. (2014) [1] and WEF OM-9 (2025) [2]), not hydraulic loss.
    - Electricity demand is decomposed into a base load, an aeration-linked BOD-removal term, and a nitrification-linked
      TN-removal term. If only a lumped SEC is available, set aeration and nitrification terms to 0.0 and use
      SEC (base_energy_kwh_per_m3_out) alone.
    - Optional direct-emissions buses (CH4, N2O) can be connected for detailed environmental accounting.
      If buses are absent, emissions are ignored.
    - Operational variables (SRT, HRT, F:M, DO, RAS, WAS) are stored as documentation / calibration metadata.
      They are not enforced as hard optimization constraints.
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
    water_out_bus: Bus = None        # m³ treated water (PRIMARY)
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
    efficiency: float = 0.98                                    # m³_out / m³_in (water recovery) [1]
    sludge_yield: float = 0.0102                                # m³_sludge / m³_out [1, 2]
    specific_energy_consumption: float = 0.0510                  # kWh / m³_out (base_energy_kwh_per_m3_out)[3, 4]
    aeration_energy_kwh_per_kg_bod_removed: float = 0.0         # kWh / kgBOD [2, 3]
    removed_bod_kg_per_m3_out: float = 0.0                      # kgBOD / m³_out [3, 4]
    nitrification_energy_kwh_per_kg_tn_removed: float = 0.0     # kWh / kgTN [2, 3]
    removed_tn_kg_per_m3_out: float = 0.0                       # kgTN / m³_out [3, 4]
    direct_ch4_kgco2e_per_m3_out: float = 0.0                   # kgCO2e / m³_out [5]
    direct_n2o_kgco2e_per_m3_out: float = 0.0                   # kgCO2e / m³_out [5]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0          # USD/m³ treated water
    carrier_cost: float = 0.0           # USD/m³ influent water
    sludge_disposal_cost: float = 0.0   # USD/m³ sludge
    ch4_emissions_cost: float = 0.0     # USD/kgCO2e CH4
    n2o_emissions_cost: float = 0.0     # USD/kgCO2e N2O

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints)
    # Based on Tchobanoglous et al. (2014) [1] and WEF OM-9 (2025) [2]
    # ------------------------------------------------------------------
    srt_days: float = None              # days, solids retention time / sludge age [1, 2]
    hrt_hours: float = None             # hours, hydraulic retention time (reactor sizing) [1]
    fm_ratio: float = None              # kgBOD/kgMLVSS/day, food-to-microorganism ratio [1, 2]
    do_setpoint_mg_per_l: float = None  # mg/L, aeration basin dissolved oxygen set point [2]
    ras_ratio: float = None             # Q_RAS/Q_in, return activated sludge flow ratio [2]
    was_ratio: float = None             # fraction, waste activated sludge rate [1, 2]

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
        self.efficiency = attributes.pop(
            "efficiency", self.efficiency
        )
        self.sludge_yield = attributes.pop(
            "sludge_yield", self.sludge_yield
        )
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.aeration_energy_kwh_per_kg_bod_removed = attributes.pop(
            "aeration_energy_kwh_per_kg_bod_removed",
            self.aeration_energy_kwh_per_kg_bod_removed,
        )
        self.removed_bod_kg_per_m3_out = attributes.pop(
            "removed_bod_kg_per_m3_out", self.removed_bod_kg_per_m3_out
        )
        self.nitrification_energy_kwh_per_kg_tn_removed = attributes.pop(
            "nitrification_energy_kwh_per_kg_tn_removed",
            self.nitrification_energy_kwh_per_kg_tn_removed,
        )
        self.removed_tn_kg_per_m3_out = attributes.pop(
            "removed_tn_kg_per_m3_out", self.removed_tn_kg_per_m3_out
        )
        self.direct_ch4_kgco2e_per_m3_out = attributes.pop(
            "direct_ch4_kgco2e_per_m3_out", self.direct_ch4_kgco2e_per_m3_out
        )
        self.direct_n2o_kgco2e_per_m3_out = attributes.pop(
            "direct_n2o_kgco2e_per_m3_out", self.direct_n2o_kgco2e_per_m3_out
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
        self.output_parameters = attributes.pop("output_parameters", {})

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
        # derived whole-plant electricity demand per m³ effluent
        # E_total = E_base (SEC) + E_aer * BOD_removed + E_nit * TN_removed
        # (Tchobanoglous et al., 2014 [1]; Nguyen et al., 2025 [3])
        # feedwater ratio: m³ influent per m³ treated water output
        # --------------------------------------------------------------
        self._electricity_per_m3_out = (
            self.specific_energy_consumption
            + self.aeration_energy_kwh_per_kg_bod_removed
            * self.removed_bod_kg_per_m3_out
            + self.nitrification_energy_kwh_per_kg_tn_removed
            * self.removed_tn_kg_per_m3_out
        )

        self._feedwater_per_output = 1.0 / self.efficiency

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._electricity_per_m3_out
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.sludge_out_bus.label}"] = sequence(
            self.sludge_yield
        )

        if self.ch4_emissions_bus is not None:
            attributes[f"conversion_factor_{self.ch4_emissions_bus.label}"] = sequence(
                self.direct_ch4_kgco2e_per_m3_out
            )

        if self.n2o_emissions_bus is not None:
            attributes[f"conversion_factor_{self.n2o_emissions_bus.label}"] = sequence(
                self.direct_n2o_kgco2e_per_m3_out
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

        if self.water_out_bus in self.outputs:
            out_flow = self.outputs[self.water_out_bus]
            out_flow.variable_costs = sequence(self.marginal_cost)
            if not self.expandable and self.capacity is not None:
                out_flow.nominal_value = self.capacity
            custom_attrs = (getattr(self, "output_parameters", None) or {}).get(
                "custom_attributes"
            )
            if custom_attrs:
                for attribute, value in custom_attrs.items():
                    setattr(out_flow, attribute, value)

        if self.sludge_out_bus in self.outputs:
            self.outputs[self.sludge_out_bus].variable_costs = sequence(
                self.sludge_disposal_cost
            )

        if self.ch4_emissions_bus is not None and self.ch4_emissions_bus in self.outputs:
            self.outputs[self.ch4_emissions_bus].variable_costs = sequence(
                self.ch4_emissions_cost
            )

        if self.n2o_emissions_bus is not None and self.n2o_emissions_bus in self.outputs:
            self.outputs[self.n2o_emissions_bus].variable_costs = sequence(
                self.n2o_emissions_cost
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
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1].")

        if self.sludge_yield < 0:
            raise ValueError("sludge_yield must be >= 0.")

        nonneg = {
            "specific_energy_consumption": self.specific_energy_consumption,
            "aeration_energy_kwh_per_kg_bod_removed": self.aeration_energy_kwh_per_kg_bod_removed,
            "removed_bod_kg_per_m3_out": self.removed_bod_kg_per_m3_out,
            "nitrification_energy_kwh_per_kg_tn_removed": self.nitrification_energy_kwh_per_kg_tn_removed,
            "removed_tn_kg_per_m3_out": self.removed_tn_kg_per_m3_out,
            "direct_ch4_kgco2e_per_m3_out": self.direct_ch4_kgco2e_per_m3_out,
            "direct_n2o_kgco2e_per_m3_out": self.direct_n2o_kgco2e_per_m3_out,
        }
        for name, value in nonneg.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")

        # --- warn if emissions bus is set but factor is zero ---
        if self.ch4_emissions_bus is not None and self.direct_ch4_kgco2e_per_m3_out == 0:
            warnings.warn(
                "ch4_emissions_bus is set but direct_ch4_kgco2e_per_m3_out is 0. "
                "The CH4 emission output will always be zero.",
                UserWarning,
            )
        if self.n2o_emissions_bus is not None and self.direct_n2o_kgco2e_per_m3_out == 0:
            warnings.warn(
                "n2o_emissions_bus is set but direct_n2o_kgco2e_per_m3_out is 0. "
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

