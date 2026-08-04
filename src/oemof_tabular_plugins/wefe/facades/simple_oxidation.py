import dataclasses
import warnings
from typing import Sequence, Union, Optional
import numpy as np
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
    1. Process framing and technology-train placement for chlorination, ozonation, UV-based, and hydrogen-peroxide-based
       oxidation/disinfection, including the combined ozone/UV + H2O2 dosing ratio (~0.4 mg H2O2 per mg O3, the theoretical
       optimum for hydroxyl radical formation).
       World Health Organization. (2017). Treatment methods and performance (Annex 5). In Guidelines for drinking-water
       quality (4th ed., incorporating the 1st addendum). WHO.
       https://www.who.int/docs/default-source/wash-documents/wash-chemicals/treatment-methods-and-performance.pdf
    2. Contact-time and residual design logic, technology differentiation, and disinfection verification guidance for real
       plant operation, management, and maintenance.
       Environmental Protection Agency (Ireland). (2011). Water treatment manual: Disinfection. EPA. ISBN 978-184095-421-0.
       https://www.epa.ie/publications/compliance--enforcement/drinking-water/advice--guidance/Disinfection2_web.pdf
    3. Engineering backbone for oxidation, ozone, H2O2, and AOP process families — general process-role and mass-balance justification.
       Metcalf & Eddy, Inc., Tchobanoglous, G., Stensel, H. F., Tsuchihashi, R., & Burton, F. L. (2014).
       Wastewater engineering: Treatment and resource recovery (5th ed.). McGraw-Hill Education.
    4. Tutorial review proposing comparable, scalable evaluation parameters (e.g. UV fluence, ozone consumption) across
       catalytic, ozone-based, and radiation-driven AOP variants at lab through pilot scale.
       Hübner, U., Spahr, S., Lutze, H., Wieland, A., Rüting, S., Gernjak, W., & Wenk, J. (2024). Advanced oxidation processes
       for water and wastewater treatment – Guidance for systematic future research. Heliyon, 10(9), e30402.
       https://doi.org/10.1016/j.heliyon.2024.e30402

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Feedwater requirement:
        water_in_per_output = 1 / water_recovery      [m³_feed / m³_treated]

    Net specific energy consumption:
        net_SEC = specific_energy_consumption          [kWh / m³ treated water]

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
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum treated-water throughput of the unit.
    - If oxidant_bus is provided, oxidant is modeled as a real input flow and its cost must be captured via upstream supply
      economics. If oxidant_bus is absent, oxidant cost is embedded into the output variable costs as a fallback to avoid
      silent omission from the objective.
    - Characterization values (sec_typical_min/max, contact_time_min, residual_target_mgL) are stored as documentation/calibration
      defaults. They are not enforced as hard optimization constraints.
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
    electricity_bus: Bus = None         # kWh
    water_in_bus: Bus = None            # m³ (raw / pretreated feedwater)
    water_out_bus: Bus = None           # m³ (treated water — PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    oxidant_bus: Optional[Bus] = None   # kg (oxidant as explicit input flow)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    offgas_bus: Optional[Bus] = None    # kg (ozone off-gas)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.07       # kWh / m³ treated water [3]
    water_recovery: float = 1.0                     # m³ treated / m³ feed, (0, 1] [1, 3]
    oxidant_dose: float = None                      # mg/L = g/m³; converted to soc internally [1]
    specific_oxidant_consumption: float = None      # kg / m³ treated water (alternative to dose) [1]
    oxidant_demand_factor: float = 1.0              # dimensionless raw-water quality multiplier [4]
    specific_offgas_generation: float = 0.0         # kg / m³ treated water [3]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # USD/m³ treated water
    carrier_cost: float = 0.0               # USD/m³ feed
    oxidant_cost: float = 2.5               # USD/kg oxidant (fallback if no oxidant_bus)

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
    contact_time_min: float = None              # minutes; design verification only [2]
    residual_target_mgL: float = None           # mg/L; design verification only [2]
    ct_target_mg_min_L: float = None            # mg·min/L; design verification only [1, 2]
    sec_typical_min: float = 0.02               # kWh/m³, lower bound from literature [3]
    sec_typical_max: float = 1.5                # kWh/m³, upper bound from literature [3]
    dbp_risk_class: str = ""                    # populated by mode default if not set [4]
    post_treatment_required: bool = False       # [2]
    residual_provided: bool = False             # [2]

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
        # All normalized to treated water output = 1 [m³/hr].
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
        total_marginal_cost = np.add(self.marginal_cost, self._soc * self.oxidant_demand_factor * self.oxidant_cost)

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

        if self.oxidant_bus is not None and self.oxidant_cost not in (0, 0.0, None):
            self.oxidant_cost = 0.0

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