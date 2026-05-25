import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class SimpleOxidation(MIMO):
    """
    Literature-informed oxidation/disinfection facade based on MIMO.

    Purpose
    -------
    Generic oxidation/disinfection facade for water treatment systems.
    The model is designed as a bookkeeping/process-yield unit representing
    an oxidation unit as a water-treatment intervention for pathogen control,
    micro-pollutant removal, or advanced oxidation. It is not a full
    reaction-kinetics or CT-compliance model.

    Core references
    ---------------
    1. WHO, Treatment methods and performance (2017): process framing and
       technology-train placement for chlorination, ozonation, UV-based
       and hydrogen-peroxide-based oxidation/disinfection.
    2. EPA Ireland, Water Treatment Manual: Disinfection (2011): contact-time
       and residual design logic, technology differentiation, and
       disinfection verification guidance.
    3. Metcalf & Eddy, Wastewater Engineering — Treatment and Resource
       Recovery (5th ed.): engineering backbone for oxidation, ozone, H2O2,
       and AOP process families.
    4. PMC 2024 AOP review: justification for generalized AOP variant
       representation and scalable planning-oriented parameters.

    Main equations
    --------------
    All flows normalized to 1 m³ net treated-water output (primary):

    Feedwater requirement:
        water_in_per_output = 1 / water_recovery      [m³_feed / m³_treated]

    Net specific energy consumption [kWh / m³ treated water]:
        net_SEC = specific_energy_consumption          [kWh / m³]

    Oxidant demand (if oxidant_bus is provided):
        oxidant_per_output = soc * oxidant_demand_factor
        [kg / m³_treated]   [kg/m³]  [dimensionless]
        where soc = oxidant_dose [mg/L] * 1e-3  [kg/m³ per mg/L]

    Offgas generation (optional placeholder for ozone variants):
        offgas_per_output = specific_offgas_generation [unit / m³_treated]

    Concentration factor (dimensionless, reporting only):
        CF = 1 / water_recovery

    CT value (reporting / design verification only):
        CT = residual_target_mgL * contact_time_min   [mg·min/L]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the
      maximum treated-water throughput of the unit.
    - Each bus is treated as its own MIMO group (no additive grouping
      is used here; all input-output coupling is pairwise linear).
    - If oxidant_bus is provided, oxidant is modeled as a real input
      flow and its cost must be captured via chemical_carrier_cost or
      upstream supply economics. If oxidant_bus is absent, oxidant cost
      is embedded into the output variable costs as a fallback to avoid
      silent omission from the objective.
    - For backward compatibility, "oxidant_dose" and
      "specific_oxidant_consumption" are mutually exclusive; providing
      both raises a ValueError at instantiation.
    - contact_time_min, residual_target_mgL, ct_target_mg_min_L, and
      dbp_risk_class are design-verification / documentation fields.
      They are not enforced as hard optimization constraints in v3.0.
    - Mode-specific metadata (post_treatment_required, residual_provided,
      dbp_risk_class) is populated automatically inside
      _validate_parameters() based on the selected mode.
    - Characterization values (sec_typical_min/max, contact_time_min,
      residual_target_mgL) are stored as documentation/calibration defaults.
      They are not enforced as hard optimization constraints in v3.0.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "oxidation"
    name: str = ""
    tech: str = "water-treatment"
    carrier: str = "water"
    mode: str = "chlorine_dioxide"  # see _validate_parameters for allowed values
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
    electricity_bus: Bus = None         # kWh/hr
    water_in_bus: Bus = None            # m³/hr (raw / pretreated feedwater)
    water_out_bus: Bus = None           # m³/hr (treated water — PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    oxidant_bus: Optional[Bus] = None   # kg/hr (oxidant as explicit input flow)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    offgas_bus: Optional[Bus] = None    # kg / hr (ozone off-gas)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.07       # kWh / m³ treated water
    water_recovery: float = 1.0                     # m³ treated / m³ feed, (0, 1]
    oxidant_dose: float = None                      # mg/L = g/m³; converted to soc internally
    specific_oxidant_consumption: float = None      # kg / m³ treated water (alternative to dose)
    oxidant_demand_factor: float = 1.0              # dimensionless raw-water quality multiplier
    specific_offgas_generation: float = 0.0         # kg / m³ treated water

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # € / m³ treated water
    carrier_cost: float = 0.0               # € / kWh electricity
    oxidant_cost: float = None              # € / kg oxidant (fallback if no oxidant_bus)
    chemical_carrier_cost: float = 0.0      # € / kg oxidant via modeled oxidant_bus

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # ------------------------------------------------------------------
    contact_time_min: float = None              # minutes; design verification only
    residual_target_mgL: float = None           # mg/L; design verification only
    ct_target_mg_min_L: float = None            # mg·min/L; design verification only
    sec_typical_min: float = 0.02               # kWh/m³, lower bound from literature
    sec_typical_max: float = 1.5                # kWh/m³, upper bound from literature
    dbp_risk_class: str = ""                    # populated by mode default if not set
    post_treatment_required: bool = False
    residual_provided: bool = False

    def __init__(self, **attributes):
        # --------------------------------------------------------------
        # identity
        # --------------------------------------------------------------
        self.type = attributes.pop("type", self.type)
        self.name = attributes.pop("name", self.name)
        self.tech = attributes.pop("tech", self.tech)
        self.carrier = attributes.pop("carrier", self.carrier)
        self.mode = attributes.pop("mode", self.mode).lower().strip()
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
        self.oxidant_bus = attributes.pop("oxidant_bus", None)
        self.offgas_bus = attributes.pop("offgas_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.water_recovery = attributes.pop("water_recovery", self.water_recovery)
        self.oxidant_dose = attributes.pop("oxidant_dose", self.oxidant_dose)
        self.specific_oxidant_consumption = attributes.pop(
            "specific_oxidant_consumption", self.specific_oxidant_consumption
        )
        self.oxidant_demand_factor = attributes.pop(
            "oxidant_demand_factor", self.oxidant_demand_factor
        )
        self.specific_offgas_generation = attributes.pop(
            "specific_offgas_generation", self.specific_offgas_generation
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.oxidant_cost = attributes.pop("oxidant_cost", self.oxidant_cost)
        self.chemical_carrier_cost = attributes.pop(
            "chemical_carrier_cost", self.chemical_carrier_cost
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
        self.contact_time_min = attributes.pop(
            "contact_time_min", self.contact_time_min
        )
        self.residual_target_mgL = attributes.pop(
            "residual_target_mgL", self.residual_target_mgL
        )
        self.ct_target_mg_min_L = attributes.pop(
            "ct_target_mg_min_L", self.ct_target_mg_min_L
        )
        self.sec_typical_min = attributes.pop("sec_typical_min", self.sec_typical_min)
        self.sec_typical_max = attributes.pop("sec_typical_max", self.sec_typical_max)
        self.dbp_risk_class = attributes.pop("dbp_risk_class", self.dbp_risk_class)
        self.post_treatment_required = attributes.pop(
            "post_treatment_required", self.post_treatment_required
        )
        self.residual_provided = attributes.pop(
            "residual_provided", self.residual_provided
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # (WHO 2017; EPA Ireland 2011; Metcalf & Eddy 5th ed.)
        # --------------------------------------------------------------
        self._feedwater_per_output = 1.0 / self.water_recovery

        # derive specific oxidant consumption (soc) [kg/m³ treated]
        # 1 mg/L = 1 g/m³ = 1e-3 kg/m³
        if self.specific_oxidant_consumption is not None:
            soc = self.specific_oxidant_consumption
        elif self.oxidant_dose is not None:
            soc = self.oxidant_dose * 1e-3
        else:
            _defaults_mgL = {
                "chlorine": 1.0,
                "chlorine_dioxide": 0.8,
                "hydrogen_peroxide": 5.0,
                "ozone": 3.0,
                "uv_h2o2": 5.0,
            }
            self.oxidant_dose = _defaults_mgL.get(self.mode, 0.0)
            soc = self.oxidant_dose * 1e-3

        self._soc = soc

        if self.oxidant_bus is None and self.oxidant_cost is None and soc > 0:
            warnings.warn(
                f"[SimpleOxidation '{self.name}'] Oxidant consumption (soc={soc} kg/m³) is "
                "defined but neither oxidant_bus nor oxidant_cost is set. "
                "Chemical use will not appear in the objective.",
                UserWarning,
            )

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

        if self.oxidant_bus is not None:
            attributes[f"conversion_factor_{self.oxidant_bus.label}"] = sequence(
                self._soc * self.oxidant_demand_factor
            )
        if self.offgas_bus is not None and self.specific_offgas_generation > 0:
            attributes[f"conversion_factor_{self.offgas_bus.label}"] = sequence(
                self.specific_offgas_generation
            )

        # --------------------------------------------------------------
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # --------------------------------------------------------------
        attributes.setdefault("output_parameters", {})

        if self.oxidant_bus is None and self.oxidant_cost is not None:
            embedded = self._soc * self.oxidant_demand_factor * self.oxidant_cost
            attributes["output_parameters"].update(
                {"variable_costs": embedded}
            )

        if self.offgas_bus is not None:
            attributes.setdefault("output_parameters_1", {})

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
            self.oxidant_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.offgas_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        allowed_modes = {
            "chlorine",
            "chlorine_dioxide",
            "hydrogen_peroxide",
            "ozone",
            "uv_h2o2",
        }
        if self.mode not in allowed_modes:
            raise ValueError(
                f"mode must be one of {sorted(allowed_modes)}, got '{self.mode}'."
            )

        if not 0 < self.water_recovery <= 1:
            raise ValueError("water_recovery must be in (0, 1].")

        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        if self.oxidant_demand_factor is not None and self.oxidant_demand_factor <= 0:
            raise ValueError("oxidant_demand_factor must be > 0.")

        if (
                self.oxidant_dose is not None
                and self.specific_oxidant_consumption is not None
        ):
            raise ValueError(
                "Provide either oxidant_dose or specific_oxidant_consumption, not both."
            )

        bounded = {
            "oxidant_dose": self.oxidant_dose,
            "specific_oxidant_consumption": self.specific_oxidant_consumption,
            "specific_offgas_generation": self.specific_offgas_generation,
            "contact_time_min": self.contact_time_min,
            "residual_target_mgL": self.residual_target_mgL,
            "ct_target_mg_min_L": self.ct_target_mg_min_L,
        }
        for field_name, value in bounded.items():
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be >= 0.")

        if self.oxidant_bus is not None and (
                self.specific_oxidant_consumption is None and self.oxidant_dose is None
        ):
            warnings.warn(
                "oxidant_bus is set but neither oxidant_dose nor "
                "specific_oxidant_consumption is provided. "
                "A mode-default dose will be used.",
                UserWarning,
            )

        if self.mode == "chlorine":
            if not self.residual_provided:
                self.residual_provided = True
            if not self.dbp_risk_class:
                self.dbp_risk_class = "chlorinated_dbp"

        elif self.mode == "chlorine_dioxide":
            if not self.residual_provided:
                self.residual_provided = True
            if not self.dbp_risk_class:
                self.dbp_risk_class = "chlorite_chlorate"

        elif self.mode == "hydrogen_peroxide":
            self.residual_provided = False
            if not self.dbp_risk_class:
                self.dbp_risk_class = "low_direct_residual"

        elif self.mode == "ozone":
            self.residual_provided = False
            self.post_treatment_required = True
            if not self.dbp_risk_class:
                self.dbp_risk_class = "ozone_byproduct_risk"

        elif self.mode == "uv_h2o2":
            self.residual_provided = False
            if not self.dbp_risk_class:
                self.dbp_risk_class = "aop_site_specific"