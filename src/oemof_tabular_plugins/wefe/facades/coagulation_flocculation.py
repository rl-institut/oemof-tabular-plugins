import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class CoagulationFlocculation(MIMO):
    """
    Literature-informed coagulation-flocculation facade based on MIMO.

    Purpose
    -------
    Generic chemical-assisted particle destabilization and floc aggregation
    facade for coagulation-flocculation units used in water treatment systems.
    The model is designed as a bookkeeping/process-yield unit representing a
    coagulation-flocculation stage as a water-treatment intervention. It is
    not a full mechanistic colloidal chemistry simulator or dynamic jar-test
    model.

    Core references
    ---------------
    1. Process framing, treatment-stage logic, turbidity suitability guidance, and practical assumptions for placing
       coagulation-flocculation within the drinking-water treatment chain; part of the same Irish EPA Water Treatment
       Manuals series as the Filtration and Disinfection manuals.
       Environmental Protection Agency (Ireland). (2011). Water treatment manuals: Coagulation, flocculation and clarification. EPA.
       https://www.epa.ie/publications/compliance--enforcement/drinking-water/advice--guidance/EPA_water_treatment_mgt_coag_flocc_clar2.pdf
    2. Variable definitions and process logic distinguishing destabilization (coagulation) from aggregate growth under
       mixing energy (flocculation).
       TU Delft OpenCourseWare. (2014). CTB3365DWx: Coagulation and flocculation. Delft University of Technology.
       https://ocw.tudelft.nl/wp-content/uploads/2014-CTB3365DWx-Coagulation-flocculation.pdf
    3. Jar-testing and monitoring logic for operational control of coagulation and filtration processes.
       American Water Works Association. (2011). M37: Operational control of coagulation and filtration processes (3rd ed.).
       AWWA. ISBN 9781583218013.
    4. Textbook-level treatment of coagulant/flocculant dosing ranges, rapid mixing velocity gradient ranges, flocculation
       mixing intensity, and specialist process refinement.
       Bratby, J. (2016). Coagulation and flocculation in water and wastewater treatment (3rd ed., Vol. 15).
       IWA Publishing. https://doi.org/10.2166/9781780407500

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Feedwater requirement:
        feedwater_per_output = 1 / efficiency      [m³_feed / m³_product]

    Electricity demand:
        electricity_per_output = specific_energy_consumption
                                                       [kWh / m³_product]

    Coagulant demand (active only if coagulant_bus is provided):
        coagulant_per_output = coagulant_dose * 1e-3 * dose_factor
                                                       [kg / m³_product]

    Flocculant demand (active only if flocculant_bus is provided):
        flocculant_per_output = flocculant_dose * 1e-3 * dose_factor
                                                       [kg / m³_product]

    Spent chemical output (active only if spent_chemical_bus is provided):
        spent_per_output = spent_chemical_factor       [kg / m³_product]

        Physical lower bound: (coagulant_dose + flocculant_dose) * 1e-3 [kg/m³].
        Practical values are higher because precipitation reactions
        (e.g. Al³⁺ + 3OH⁻ → Al(OH)₃) add hydroxide mass to the floc.

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. capacity constrains the maximum treated-water throughput of the unit.
    - dosing_mode controls how chemical consumption is represented:
        "cost_only"          — costs folded into output variable_costs (default)
        "tracked_coagulant"  — coagulant_bus required; explicit kg/hr input flow
        "tracked_flocculant" — flocculant_bus required; explicit kg/hr input flow
        "tracked_both"       — both coagulant_bus and flocculant_bus required
      In tracked modes the cost for that chemical is not added to variable_costs; it is expected to be handled by the
      upstream commodity node.
    - If coagulant_bus or flocculant_bus is active, chemical cost must come from the upstream supply node. Chemical costs
      are only applied as output-side variable cost surcharges when the respective bus is absent.
    - Characterization values (g_rapid_s_inv, g_floc_s_inv, rapid_mix_time_s, flocculation_time_min, raw_water_turbidity_ntu)
      are stored as documentation/calibration defaults. They are not enforced as hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "coagulation_flocculation"
    name: str = ""
    tech: str = "water-treatment"
    carrier: str = "water"
    dosing_mode: str = "cost_only"  # "cost_only" | "tracked_coagulant" | "tracked_flocculant" | "tracked_both"
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
    electricity_bus: Bus = None             # kWh
    water_in_bus: Bus = None                # m³ (untreated feedwater)
    water_out_bus: Bus = None               # m³ (treated water — PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    coagulant_bus: Optional[Bus] = None  # kg  (tracked_coagulant / tracked_both modes)
    flocculant_bus: Optional[Bus] = None  # kg  (tracked_flocculant / tracked_both modes)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    spent_chemical_bus: Optional[Bus] = None  # kg  precipitated solids / hydroxide flocs / residual chemical mass

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.04               # kWh/m³ treated water [1, 2]
    efficiency: float = 0.99                            # m³ treated water /m³ feed water (water recovery) [1]
    coagulant_dose: float = 20.0                            # g/m³ treated water (= mg/L) [3, 4]
    flocculant_dose: float = 2.55                           # g/m³ treated water (= mg/L) [4]
    dose_factor: float = 1.0                                # jar-test / seasonal adjustment [-] [3]
    spent_chemical_factor: float = 0.0                      # kg spent chemical output / m³ treated water [1]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                              # USD/m³ treated water (excl. chemical costs)
    carrier_cost: float = 0.0                               # USD/m³ feed
    coagulant_cost: float = 1.5                             # USD/kg
    flocculant_cost: float = 2.0                            # USD/kg
    spent_chemical_disposal_cost: float = 0.0               # USD/kg spent chemical output

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
    coagulant_dose_typical_min: float = 5.0                 # g/m³, lower bound from literature [4]
    coagulant_dose_typical_max: float = 80.0                # g/m³, upper bound from literature [4]
    jar_test_dose_mg_per_l: float = None                    # mg/L, reference jar-test coagulant result, documentation only [3]
    raw_water_turbidity_ntu: float = None                   # NTU, influent turbidity, documentation only [1]
    enforce_turbidity_check: bool = False                   # bool, promote turbidity warning to ValueError [1]
    rapid_mix_time_s: float = None                          # s, rapid mixing duration, typical: 10–60 s, documentation only [2, 4]
    flocculation_time_min: float = None                     # min, flocculation duration, typical: 20–40 min, documentation only [2, 4]
    g_rapid_s_inv: float = None                             # /s, velocity gradient rapid mix, typical: 300–1500 /s, documentation only [4]
    g_floc_s_inv: float = None                              # /s, velocity gradient flocculation, typical: 10–100 /s, documentation only [4]
    flocculation_stages: int = None                         # -, number of tapered mixing stages, documentation only [4]

    def __init__(self, **attributes):
        # --------------------------------------------------------------
        # identity
        # --------------------------------------------------------------
        self.type = attributes.pop("type", self.type)
        self.name = attributes.pop("name", self.name)
        self.tech = attributes.pop("tech", self.tech)
        self.carrier = attributes.pop("carrier", self.carrier)
        self.dosing_mode = attributes.pop("dosing_mode", self.dosing_mode)
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
        self.coagulant_bus = attributes.pop("coagulant_bus", None)
        self.flocculant_bus = attributes.pop("flocculant_bus", None)
        self.spent_chemical_bus = attributes.pop("spent_chemical_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop(
            "efficiency", self.efficiency
        )
        self.coagulant_dose = attributes.pop("coagulant_dose", self.coagulant_dose)
        self.flocculant_dose = attributes.pop("flocculant_dose", self.flocculant_dose)
        self.dose_factor = attributes.pop("dose_factor", self.dose_factor)
        self.spent_chemical_factor = attributes.pop(
            "spent_chemical_factor", self.spent_chemical_factor
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.coagulant_cost = attributes.pop("coagulant_cost", self.coagulant_cost)
        self.flocculant_cost = attributes.pop("flocculant_cost", self.flocculant_cost)
        self.spent_chemical_disposal_cost = attributes.pop(
            "spent_chemical_disposal_cost", self.spent_chemical_disposal_cost
        )
        self.expandable = attributes.pop("expandable", self.expandable)
        self.capacity = attributes.pop("capacity", self.capacity)
        self.capacity_cost = attributes.pop("capacity_cost", self.capacity_cost)
        self.capacity_minimum = attributes.pop("capacity_minimum", self.capacity_minimum)
        self.capacity_potential = attributes.pop("capacity_potential", self.capacity_potential)

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
        self.coagulant_dose_typical_min = attributes.pop(
            "coagulant_dose_typical_min", self.coagulant_dose_typical_min
        )
        self.coagulant_dose_typical_max = attributes.pop(
            "coagulant_dose_typical_max", self.coagulant_dose_typical_max
        )
        self.jar_test_dose_mg_per_l = attributes.pop(
            "jar_test_dose_mg_per_l", self.jar_test_dose_mg_per_l
        )
        self.raw_water_turbidity_ntu = attributes.pop(
            "raw_water_turbidity_ntu", self.raw_water_turbidity_ntu
        )
        self.enforce_turbidity_check = attributes.pop(
            "enforce_turbidity_check", self.enforce_turbidity_check
        )
        self.rapid_mix_time_s = attributes.pop("rapid_mix_time_s", self.rapid_mix_time_s)
        self.flocculation_time_min = attributes.pop(
            "flocculation_time_min", self.flocculation_time_min
        )
        self.g_rapid_s_inv = attributes.pop("g_rapid_s_inv", self.g_rapid_s_inv)
        self.g_floc_s_inv = attributes.pop("g_floc_s_inv", self.g_floc_s_inv)
        self.flocculation_stages = attributes.pop("flocculation_stages", self.flocculation_stages)

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._coagulant_kg_per_m3 = self.coagulant_dose * 1e-3 * self.dose_factor
        self._flocculant_kg_per_m3 = self.flocculant_dose * 1e-3 * self.dose_factor
        self._coagulant_cost_per_m3 = self._coagulant_kg_per_m3 * self.coagulant_cost
        self._flocculant_cost_per_m3 = self._flocculant_kg_per_m3 * self.flocculant_cost

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            1.0 / self.efficiency
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.coagulant_bus is not None:
            attributes[f"conversion_factor_{self.coagulant_bus.label}"] = sequence(
                self._coagulant_kg_per_m3
            )
        if self.flocculant_bus is not None:
            attributes[f"conversion_factor_{self.flocculant_bus.label}"] = sequence(
                self._flocculant_kg_per_m3
            )
        if self.spent_chemical_bus is not None:
            attributes[f"conversion_factor_{self.spent_chemical_bus.label}"] = sequence(
                max(self.spent_chemical_factor, 1e-9)
            )

        # --------------------------------------------------------------
        # output-specific variable costs
        # --------------------------------------------------------------
        self.chemical_cost_per_m3 = 0.0
        if self.coagulant_bus is None:
            self.chemical_cost_per_m3 += self._coagulant_cost_per_m3
        if self.flocculant_bus is None:
            self.chemical_cost_per_m3 += self._flocculant_cost_per_m3

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
        total_marginal_cost = self.marginal_cost + self.chemical_cost_per_m3

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

        if self.spent_chemical_bus is not None and self.spent_chemical_bus in self.outputs:
            self.outputs[self.spent_chemical_bus].variable_costs = sequence(
                self.spent_chemical_disposal_cost
            )

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 1
        # inputs
        for bus in [
            self.coagulant_bus,
            self.flocculant_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.spent_chemical_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        allowed_modes = {"cost_only", "tracked_coagulant", "tracked_flocculant", "tracked_both"}
        if self.dosing_mode not in allowed_modes:
            raise ValueError(f"dosing_mode must be one of {sorted(allowed_modes)}, got '{self.dosing_mode}'.")
        if self.dosing_mode in {"tracked_coagulant", "tracked_both"} and self.coagulant_bus is None:
            raise ValueError(f"dosing_mode='{self.dosing_mode}' requires coagulant_bus.")
        if self.dosing_mode in {"tracked_flocculant", "tracked_both"} and self.flocculant_bus is None:
            raise ValueError(f"dosing_mode='{self.dosing_mode}' requires flocculant_bus.")

        if not (0 < self.efficiency <= 1):
            raise ValueError("efficiency must be in (0, 1].")
        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")
        for n, v in {"coagulant_dose": self.coagulant_dose, "flocculant_dose": self.flocculant_dose}.items():
            if v <= 0:
                raise ValueError(f"{n} must be > 0.")
        for n, v in {
            "dose_factor": self.dose_factor, "spent_chemical_factor": self.spent_chemical_factor,
            "coagulant_cost": self.coagulant_cost, "flocculant_cost": self.flocculant_cost,
            "spent_chemical_disposal_cost": self.spent_chemical_disposal_cost,
        }.items():
            if v < 0:
                raise ValueError(f"{n} must be >= 0.")

        if not (self.coagulant_dose_typical_min <= self.coagulant_dose <= self.coagulant_dose_typical_max):
            warnings.warn(
                f"coagulant_dose={self.coagulant_dose} mg/L outside typical range "
                f"[{self.coagulant_dose_typical_min}, {self.coagulant_dose_typical_max}] mg/L "
                f"(Bratby, 2016; AWWA M37).", UserWarning,
            )

        for bus, dose, cost, label in [
            (self.coagulant_bus, self.coagulant_dose, self.coagulant_cost, "coagulant"),
            (self.flocculant_bus, self.flocculant_dose, self.flocculant_cost, "flocculant"),
        ]:
            if bus is None and cost == 0.0 and dose > 0:
                warnings.warn(f"{label}_bus is absent and {label}_cost == 0.0: physically dosed but uncosted.",
                              UserWarning)

        # spent_chemical_bus without factor
        if self.spent_chemical_bus is not None and self.spent_chemical_factor == 0.0:
            warnings.warn(
                "spent_chemical_bus provided but spent_chemical_factor == 0.0 "
                "(clamped to 1e-9 kg/m³). Set spent_chemical_factor [kg/m³].", UserWarning,
            )

        for val, lo, hi, label in [
            (self.g_rapid_s_inv, 300, 1500, "g_rapid_s_inv"),
            (self.g_floc_s_inv, 10, 100, "g_floc_s_inv"),
        ]:
            if val is not None and not (lo <= val <= hi):
                warnings.warn(f"{label}={val} /s outside {lo}–{hi} /s (Bratby, 2016).", UserWarning)

        if self.raw_water_turbidity_ntu is not None:
            if self.raw_water_turbidity_ntu < 0:
                raise ValueError("raw_water_turbidity_ntu must be >= 0.")
            if self.raw_water_turbidity_ntu > 500:
                msg = (f"raw_water_turbidity_ntu={self.raw_water_turbidity_ntu} NTU is very high. "
                       "Pre-screening or sedimentation may be required (EPA Ireland, 2011).")
                if self.enforce_turbidity_check:
                    raise ValueError(msg)
                warnings.warn(msg, UserWarning)

        if self.jar_test_dose_mg_per_l is not None:
            if abs(self.jar_test_dose_mg_per_l - self.coagulant_dose) / max(self.coagulant_dose, 1e-9) > 0.5:
                warnings.warn(
                    f"coagulant_dose ({self.coagulant_dose} mg/L) deviates >50% from "
                    f"jar_test_dose_mg_per_l ({self.jar_test_dose_mg_per_l} mg/L). "
                    "Consider aligning with jar-test results (AWWA M37).", UserWarning,
                )

        for bus, mode_set, label in [
            (self.coagulant_bus, {"tracked_coagulant", "tracked_both"}, "coagulant"),
            (self.flocculant_bus, {"tracked_flocculant", "tracked_both"}, "flocculant"),
        ]:
            if bus is not None and self.dosing_mode in mode_set:
                warnings.warn(f"{label}_bus in tracked mode — verify commodity bus unit is kg/hr.", UserWarning)