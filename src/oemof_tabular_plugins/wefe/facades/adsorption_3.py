import dataclasses
import warnings
from typing import Sequence, Union, Optional
import math

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class Adsorption(MIMO):
    """
    Literature-informed fixed-bed adsorption facade based on MIMO.

    Purpose
    -------
    Simplified engineering fixed-bed adsorption unit for contaminant removal
    from water. The model represents an adsorption step as a fixed-ratio
    process-yield unit, not a full mechanistic breakthrough simulator or
    dynamic PDE model.

    Core references
    ---------------
    1. Core design methodology, capacity estimation, breakthrough prediction, and scale-up workflow from laboratory to
       full-scale plant design.
       Thomas, W. J., & Crittenden, B. (1998). Adsorption technology and design. Butterworth-Heinemann.
       https://www.sciencedirect.com/book/monograph/9780750619592/adsorption-technology-and-design
    2. Operating limits, process engineering ranges, regeneration cycles, and validated operating envelopes for real
       wastewater matrices.
       Worch, E. (2012). Adsorption technology in water treatment: Fundamentals, processes, and modeling. De Gruyter.
       https://doi.org/10.1515/9783110240238
    3. BDST, Thomas, Yoon-Nelson, and Bohart-Adams model equations with linearization methods and design variable
       definitions (EBCT, breakthrough ratio, bed depth, flow rate).
       Abin-Bazaine, A. A., Olmos-Marquez, M. A., & Campos-Trujillo, A. (2024). A fixed-bed column sorption: Breakthrough
       curves modeling. In K. Margeta & A. Farkas (Eds.), Sorption — new perspectives and applications. IntechOpen.
       https://doi.org/10.5772/intechopen.1004446
    4. Specific energy consumption value.
       Gibellini, S., Gialdini, F., Masoud, A., & Sorlini, S. (2024). Electric energy consumption (EEC) in groundwater-based
       drinking water supply systems: Analysis and comparison of seven Italian case studies. Water Supply, 24(10), 3456–3466.
       https://doi.org/10.2166/ws.2024.216

    Main equations
    --------------
    All flows normalized to 1 m3 net treated-water output (primary):

    Electricity coupling:
        Q_elec(t) = net_SEC * Q_water_out(t)
        [kWh/hr]    [kWh/m3]   [m3/hr]

    Hydraulic mass balance:
        Q_water_in(t) = Q_water_out(t)
        [m3/hr]         [m3/hr]

    Output concentration:
        C_out = C_in * (1 - eta_rem)
        [mg/L]   [mg/L]              [-]

    Adsorbent commodity coupling (tracked_adsorbent mode only):
        Q_ads(t) = (adsorbent_dose * 1e-3) * Q_water_out(t)
        [kg/hr]     [kg/m3]                   [m3/hr]

    EBCT:
        EBCT = V_bed / Q = (pi * D^2 / 4 * Z) / Q
        [h]    [m3]   [m3/h]

    BDST service time:
        t = (N0 * Z) / (C_in * U0) - (1 / (k * C_in)) * ln(C_in / C_b - 1)
        [h]

    Notes
    -----
    - Primary flow is water_out_bus [m3/hr]. Capacity constrains the maximum treated-water throughput of the unit.
    - Adsorbent chemical input can be modelled either as an output-side variable cost (default, no adsorbent_bus) or as
      a tracked third input commodity (tracked_adsorbent mode, requires adsorbent_bus).
    - kinetic_model_params accepts a dict of Thomas, Yoon-Nelson, or Adams-Bohart coefficients as calibration metadata for
      future extensions. They do not affect the optimization.
    - Design parameters (EBCT, bed geometry, BDST coefficients) are stored as metadata and do not affect the optimization
      unless design_model='bdst' derives an activity_bound_max from service time.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "adsorption"
    name: str = ""
    tech: str = "water-treatment"
    carrier: str = "water"
    dosing_mode: str = "cost_only"      # "cost_only" | "tracked_adsorbent"
    design_model: str = "simple"        # "simple" | "bdst" | "thomas" | "yoon_nelson" | "adams_bohart"
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
    electricity_bus: Bus = None                     # kWh
    water_in_bus: Bus = None                        # m³ (untreated feedwater)
    water_out_bus: Bus = None                       # m³ (treated water — PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    adsorbent_bus: Optional[Bus] = None             # kg only in tracked_adsorbent mode

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    spent_adsorbent_bus: Optional[Bus] = None       # kg  spent adsorbent for disposal / regeneration

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.06               # kWh/m³ treated water [4]
    Cin: float = 10.0                                       # mg/L influent concentration [1, 2]
    removal_efficiency: float = 0.8                         # fraction [-] [1, 2]
    adsorbent_dose: float = 25                              # mg/L == g/m³ [1, 2] (based on per volume of treated water)
    regeneration_factor: float = 1.0                        # multiplier on adsorbent cost [-] [1, 2]
    max_throughput_before_regen: Optional[float] = None     # m³ [1, 2]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                      # USD/m³ treated water
    carrier_cost: float = 0.0                       # USD/m³ feed
    adsorbent_cost: float = 5.0                     # USD/kg adsorbent
    spent_adsorbent_disposal_cost: float = 0.0      # USD/kg spent adsorbent

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
    breakthrough_ratio: float = 0.1                     # C_b / C_in [-]                        [3]
    exhaustion_ratio: float = 0.9                       # C_e / C_in [-]                        [3]
    target_Cout: float = None                           # mg/L                                  [1, 2]
    enforce_removal_check: bool = False                 #                                       [1, 2]
    flow_rate: float = None                             # m³/h                                  [1, 3]
    bed_depth: float = None                             # m                                     [1, 3]
    column_diameter: float = None                       # m                                     [1]
    bed_porosity: float = None                          # fraction [-]                          [1]
    adsorbent_bulk_density: float = None                # kg/m³                                 [1]
    ebct: float = None                                  # h                                     [1, 3]
    N0_bdst: float = None                               #                                       [3]
    k_bdst: float = None                                #                                       [3]
    linear_velocity: float = None                       #                                       [1, 3]
    service_time: float = None                          # h                                     [3]
    kinetic_model_params: Optional[dict] = None         # Thomas / Yoon-Nelson /                [3]
                                                        # Adams-Bohart coefficients

    def __init__(self, **attributes):
        # --------------------------------------------------------------
        # identity
        # --------------------------------------------------------------
        self.type = attributes.pop("type", self.type)
        self.name = attributes.pop("name", self.name)
        self.tech = attributes.pop("tech", self.tech)
        self.carrier = attributes.pop("carrier", self.carrier)
        self.dosing_mode = attributes.pop("dosing_mode", self.dosing_mode)
        self.design_model = attributes.pop("design_model", self.design_model)
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
        self.adsorbent_bus = attributes.pop("adsorbent_bus", None)
        self.spent_adsorbent_bus = attributes.pop("spent_adsorbent_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.Cin = attributes.pop("Cin", self.Cin)
        self.removal_efficiency = attributes.pop(
            "removal_efficiency", self.removal_efficiency
        )
        self.adsorbent_dose = attributes.pop(
            "adsorbent_dose",
            attributes.pop("adsorbent_dose", self.adsorbent_dose),
        )
        self.regeneration_factor = attributes.pop(
            "regeneration_factor", self.regeneration_factor
        )
        self.max_throughput_before_regen = attributes.pop(
            "max_throughput_before_regen", self.max_throughput_before_regen
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.adsorbent_cost = attributes.pop("adsorbent_cost", self.adsorbent_cost)
        self.spent_adsorbent_disposal_cost = attributes.pop(
            "spent_adsorbent_disposal_cost", self.spent_adsorbent_disposal_cost
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
        self.breakthrough_ratio = attributes.pop(
            "breakthrough_ratio", self.breakthrough_ratio
        )
        self.exhaustion_ratio = attributes.pop(
            "exhaustion_ratio", self.exhaustion_ratio
        )
        self.target_Cout = attributes.pop("target_Cout", self.target_Cout)
        self.enforce_removal_check = attributes.pop(
            "enforce_removal_check", self.enforce_removal_check
        )
        self.flow_rate = attributes.pop("flow_rate", self.flow_rate)
        self.bed_depth = attributes.pop("bed_depth", self.bed_depth)
        self.column_diameter = attributes.pop("column_diameter", self.column_diameter)
        self.bed_porosity = attributes.pop("bed_porosity", self.bed_porosity)
        self.adsorbent_bulk_density = attributes.pop(
            "adsorbent_bulk_density", self.adsorbent_bulk_density
        )
        self.ebct = attributes.pop("ebct", self.ebct)
        self.N0_bdst = attributes.pop("N0_bdst", self.N0_bdst)
        self.k_bdst = attributes.pop("k_bdst", self.k_bdst)
        self.linear_velocity = attributes.pop("linear_velocity", self.linear_velocity)
        self.service_time = attributes.pop("service_time", self.service_time)
        self.kinetic_model_params = attributes.pop(
            "kinetic_model_params", self.kinetic_model_params
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._adsorbent_kg_per_m3 = self.adsorbent_dose * 1e-3
        self._adsorbent_cost_per_m3 = (
                self._adsorbent_kg_per_m3 * self.adsorbent_cost * self.regeneration_factor
        )

        derived_max_throughput = None
        if self.max_throughput_before_regen is not None:
            derived_max_throughput = self.max_throughput_before_regen
        elif self.service_time is not None and self.flow_rate is not None:
            derived_max_throughput = self.service_time * self.flow_rate
        if derived_max_throughput is not None:
            attributes["activity_bound_max"] = sequence(derived_max_throughput)

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to permeate output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.adsorbent_bus is not None:
            attributes[f"conversion_factor_{self.adsorbent_bus.label}"] = sequence(
                self._adsorbent_kg_per_m3
            )
        if self.spent_adsorbent_bus is not None:
            attributes[f"conversion_factor_{self.spent_adsorbent_bus.label}"] = sequence(
                self._adsorbent_kg_per_m3
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

        if self.water_out_bus in self.outputs:
            out_flow = self.outputs[self.water_out_bus]
            out_flow.variable_costs = sequence(
                self.marginal_cost + self._adsorbent_cost_per_m3
            )
            if not self.expandable and self.capacity is not None:
                out_flow.nominal_value = self.capacity
            custom_attrs = (getattr(self, "output_parameters", None) or {}).get(
                "custom_attributes"
            )
            if custom_attrs:
                for attribute, value in custom_attrs.items():
                    setattr(out_flow, attribute, value)

        if self.spent_adsorbent_bus is not None and self.spent_adsorbent_bus in self.outputs:
            self.outputs[self.spent_adsorbent_bus].variable_costs = sequence(
                self.spent_adsorbent_disposal_cost * self._adsorbent_kg_per_m3 * self.regeneration_factor
            )

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 1
        # inputs
        for bus in [
            self.adsorbent_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.spent_adsorbent_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        allowed_modes = {"cost_only", "tracked_adsorbent"}
        if self.dosing_mode not in allowed_modes:
            raise ValueError(f"dosing_mode must be one of {sorted(allowed_modes)}, got '{self.dosing_mode}'.")

        if self.dosing_mode == "tracked_adsorbent" and self.adsorbent_bus is None:
            raise ValueError("dosing_mode='tracked_adsorbent' requires adsorbent_bus.")

        valid_design_models = {"simple", "bdst", "thomas", "yoon_nelson", "adams_bohart"}
        if self.design_model not in valid_design_models:
            raise ValueError(f"design_model must be one of {sorted(valid_design_models)}, got '{self.design_model}'.")

        if self.adsorbent_dose is None:
            raise ValueError("adsorbent_dose must be provided (mg/L).")

        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        if self.Cin < 0:
            raise ValueError("Cin must be >= 0.")

        if not (0 <= self.removal_efficiency <= 1):
            raise ValueError("removal_efficiency must be in [0, 1].")

        if self.adsorbent_dose < 0:
            raise ValueError("adsorbent_dose must be >= 0.")

        if self.adsorbent_cost < 0:
            raise ValueError("adsorbent_cost must be >= 0.")

        if self.adsorbent_bus is not None and self.adsorbent_cost not in (0, 0.0, None):
            self.adsorbent_cost = 0.0

        if self.regeneration_factor <= 0:
            raise ValueError("regeneration_factor must be > 0.")

        if not (0 < self.breakthrough_ratio < self.exhaustion_ratio < 1):
            raise ValueError("Require 0 < breakthrough_ratio < exhaustion_ratio < 1.")

        positive_if_given = {
            "flow_rate": self.flow_rate,
            "bed_depth": self.bed_depth,
            "column_diameter": self.column_diameter,
            "adsorbent_bulk_density": self.adsorbent_bulk_density,
            "ebct": self.ebct,
            "service_time": self.service_time,
            "max_throughput_before_regen": self.max_throughput_before_regen,
            "linear_velocity": self.linear_velocity,
        }
        for k, v in positive_if_given.items():
            if v is not None and v <= 0:
                raise ValueError(f"{k} must be > 0 if provided.")

        if self.bed_porosity is not None and not (0 < self.bed_porosity < 1):
            raise ValueError("bed_porosity must be in (0, 1).")

        _Cout = self.Cin * (1.0 - self.removal_efficiency)
        if self.target_Cout is not None:
            if abs(_Cout - self.target_Cout) > max(1e-9, 0.05 * max(self.Cin, 1e-9)):
                msg = (
                    f"target_Cout ({self.target_Cout} mg/L) differs from Cout implied "
                    f"by removal_efficiency ({_Cout:.4f} mg/L). v3.0 treats "
                    f"removal_efficiency as the active simplified quality model."
                )
                if self.enforce_removal_check:
                    raise ValueError(msg)
                warnings.warn(msg, UserWarning)

        area = None
        bed_volume = None
        if self.column_diameter is not None:
            area = math.pi * (self.column_diameter ** 2) / 4.0
        if area is not None and self.bed_depth is not None:
            bed_volume = area * self.bed_depth
        if bed_volume is not None and self.flow_rate not in (None, 0):
            if self.ebct is None:
                self.ebct = bed_volume / self.flow_rate
        if bed_volume is not None and self.adsorbent_bulk_density is not None:
            self._adsorbent_mass = bed_volume * self.adsorbent_bulk_density
        self._bed_volume = bed_volume

        if self.design_model == "bdst":
            _required = [
                self.Cin, self.breakthrough_ratio, self.bed_depth,
                self.N0_bdst, self.k_bdst, self.linear_velocity,
            ]
            if all(v is not None for v in _required):
                _Cb = self.breakthrough_ratio * self.Cin
                _t = (
                        (self.N0_bdst * self.bed_depth) / (self.Cin * self.linear_velocity)
                        - (1.0 / (self.k_bdst * self.Cin)) * math.log(self.Cin / _Cb - 1.0)
                )
                if _t > 0:
                    if self.service_time is None:
                        self.service_time = _t
                else:
                    warnings.warn(
                        "Computed BDST service time is negative. "
                        "Check coefficient units and calibration.",
                        UserWarning,
                    )
            else:
                missing = [
                    n for n, v in {
                        "bed_depth": self.bed_depth,
                        "N0_bdst": self.N0_bdst,
                        "k_bdst": self.k_bdst,
                        "linear_velocity": self.linear_velocity,
                    }.items() if v is None
                ]
                if missing and self.service_time is None and self.max_throughput_before_regen is None:
                    warnings.warn(
                        f"design_model='bdst' selected but BDST inputs are incomplete "
                        f"({missing}). No BDST operating cap will be derived.",
                        UserWarning,
                    )

        if self.adsorbent_bus is not None and self.dosing_mode == "tracked_adsorbent":
            warnings.warn(
                "adsorbent_bus is set in tracked_adsorbent mode. "
                "Verify that the adsorbent commodity bus unit is kg/hr.",
                UserWarning,
            )