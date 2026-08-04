import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class Chlorination(MIMO):
    """
    Literature-informed chlorination facade based on MIMO.

    Purpose
    -------
    Simplified engineering chlorination unit for primary or secondary
    disinfection of treated water. The model represents a chlorination step
    as a fixed-ratio process-yield unit, not a full mechanistic chlorine-decay
    or kinetics reactor model.

    Core references
    ---------------
    1. Core design logic for chlorination verification, CT, contact time, and breakpoint chlorination concepts, as part of
       drinking-water disinfection practice.
       Environmental Protection Agency (Ireland). (2011). Water treatment manual: Disinfection. Environmental Protection Agency.
       https://www.epa.ie/publications/compliance--enforcement/drinking-water/advice--guidance/Disinfection2_web.pdf
    2. Dose-demand-residual logic, target CT, effective CT (Ct), and dosing calculations for primary and secondary chlorination
       verification in utility-scale practice.
       Irish Water. (n.d.). Disinfection of drinking water training manual (Document No. IW-AMT-MAN-002). Irish Water.
       https://www.water.ie/sites/default/files/iwstandards/IW-AMT-MAN-002.pdf
    3. Internationally recognized minimum free-chlorine residual targets and public-health design assumptions for drinking-water
       disinfection; basis for target_residual_chlorine and residual_minimum.
       World Health Organization. (2017). Guidelines for drinking-water quality (4th ed., incorporating the 1st addendum).
       World Health Organization. https://www.who.int/publications/i/item/9789241549950
    4. Professional-standard reference for chlorination/chloramination dosing practices, CT values for Giardia/virus
       inactivation, and chlorine residual test methods.
       American Water Works Association. (2006). M20: Water chlorination/chloramination practices and principles (2nd ed.).
       American Water Works Association. https://kh.aquaenergyexpo.com/wp-content/uploads/2024/01/Water-Chlorination-Chloramination-Practices-and-Principles.pdf
    5. Specific energy consumption value.
       Skoczko, I. (2025). Energy efficiency analysis of water treatment plants: Current status and future trends. Energies,
       18(5), Article 1086. https://doi.org/10.3390/en18051086

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Electricity coupling:
        Q_elec(t) = net_SEC * Q_water_out(t)
        [kWh/hr]    [kWh/m3]   [m3/hr]

    Hydraulic mass balance:
        Q_water_in(t) = Q_water_out(t)
        [m3/hr]         [m3/hr]

    Dose-demand-residual balance:
        D_applied = wqf * D_demand + C_res_target
        [mg/L]       [-]   [mg/L]     [mg/L]

    Chlorine commodity coupling (tracked_chemical mode only):
        Q_chlorine(t) = (D_applied * 1e-3) * Q_water_out(t)
        [kg/hr]          [kg/m3]              [m3/hr]

    CT adequacy check (documentation-only; not a hard constraint):
        CT_achieved = C_res_target * T_effective  >=  CT_target
        [mg.min/L]    [mg/L]          [min]           [mg.min/L]

    Notes
    -----
    - Primary flow is water_out_bus [m3/hr]. Capacity constrains the maximum treated-water throughput of the unit.
    - Chlorine chemical input can be modelled either as an output-side variable cost (default, no chlorine_bus) or as a
      tracked third input commodity (tracked_chemical mode, requires chlorine_bus).
    - CT adequacy and WHO residual checks are implemented as validation warnings or optional errors, not as hard optimization
      constraints. Full CT compliance requires temperature, pH, baffling, disinfectant species, and pathogen-specific target
      data not included in the linear model.
    - Water-quality surrogates (ct_achieved, effective residual) are stored as reporting-only values in custom_attributes.
      They are not enforced as hard optimization constraints.
    - Characterization values (water_temperature_c, water_pH, turbidity_ntu, contact_tank_baffling_factor) are stored as
      documentation/calibration defaults. They are not enforced as hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "chlorination"
    name: str = ""
    tech: str = "water-treatment"
    carrier: str = "water"
    dosing_mode: str = "demand_residual"  # "fixed_dose" | "demand_residual" | "tracked_chemical"
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
    water_in_bus: Bus = None                # m³  (raw / pre-treated feed)
    water_out_bus: Bus = None               # m³  (disinfected water — PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    chlorine_bus: Optional[Bus] = None      # kg — only in tracked_chemical mode

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    # reserved for future extension (e.g. DBP / THM by-product tracking)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.02           # kWh/m³ treated water [5]
    applied_chlorine_dose: float = None                 # mg/L — set directly in fixed_dose mode [2]
    chlorine_demand: float = 0.0                        # mg/L — consumed before residual remains [2]
    target_residual_chlorine: float = 0.2               # mg/L — desired free residual after contact [3]
    water_quality_factor: float = 1.0                   # dimensionless — safety scaling on demand term [4]
    max_applied_dose: Optional[float] = None            # mg/L — design upper bound check [4]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # USD/m³ treated water
    carrier_cost: float = 0.0               # USD/m³ feed
    chlorine_cost: float = 0.5              # USD/kg chlorine

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
    effective_contact_time: float = None                # min [1, 2]
    ct_target: float = None                             # mg·min/L [1, 2]
    residual_minimum: float = 0.2                       # mg/L — based on WHO guidance [3]
    enforce_ct_check: bool = False                      # [1, 2]
    enforce_residual_check: bool = False                # [3]
    water_temperature_c: float = None                   # degC — affects CT and decay rate [4]
    water_pH: float = None                              # affects HOCl/OCl- speciation [4]
    turbidity_ntu: float = None                         # NTU — indicator of NOM and demand load [1]
    contact_tank_baffling_factor: float = None          # T10/T — hydraulic efficiency factor [2]

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
        self.chlorine_bus = attributes.pop("chlorine_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.applied_chlorine_dose = attributes.pop(
            "applied_chlorine_dose",
            attributes.pop("chlorine_dose", self.applied_chlorine_dose),  # backward compat alias
        )
        self.chlorine_demand = attributes.pop(
            "chlorine_demand", self.chlorine_demand
        )
        self.target_residual_chlorine = attributes.pop(
            "target_residual_chlorine", self.target_residual_chlorine
        )
        self.water_quality_factor = attributes.pop(
            "water_quality_factor", self.water_quality_factor
        )
        self.max_applied_dose = attributes.pop(
            "max_applied_dose", self.max_applied_dose
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.chlorine_cost = attributes.pop("chlorine_cost", self.chlorine_cost)
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
        self.effective_contact_time = attributes.pop(
            "effective_contact_time", self.effective_contact_time
        )
        self.ct_target = attributes.pop("ct_target", self.ct_target)
        self.residual_minimum = attributes.pop(
            "residual_minimum", self.residual_minimum
        )
        self.enforce_ct_check = attributes.pop(
            "enforce_ct_check", self.enforce_ct_check
        )
        self.enforce_residual_check = attributes.pop(
            "enforce_residual_check", self.enforce_residual_check
        )
        self.water_temperature_c = attributes.pop(
            "water_temperature_c", self.water_temperature_c
        )
        self.water_pH = attributes.pop("water_pH", self.water_pH)
        self.turbidity_ntu = attributes.pop("turbidity_ntu", self.turbidity_ntu)
        self.contact_tank_baffling_factor = attributes.pop(
            "contact_tank_baffling_factor", self.contact_tank_baffling_factor
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        if self.dosing_mode == "fixed_dose":
            pass                                                                    # applied_chlorine_dose already set and validated
        else:
            self.applied_chlorine_dose = (
                    self.water_quality_factor * self.chlorine_demand
                    + self.target_residual_chlorine
            )

        if self.max_applied_dose is not None and self.applied_chlorine_dose is not None:
            if self.applied_chlorine_dose > self.max_applied_dose:
                raise ValueError(
                    f"Configured applied_chlorine_dose ({self.applied_chlorine_dose} mg/L) "
                    f"exceeds max_applied_dose ({self.max_applied_dose} mg/L)."
                )

        self._chlorine_kg_per_m3 = self.applied_chlorine_dose * 1e-3                # mg/L -> kg/m3
        self._chlorine_cost_per_m3 = self._chlorine_kg_per_m3 * self.chlorine_cost

        if (
                self.effective_contact_time is not None
                and self.target_residual_chlorine is not None
        ):
            self._ct_achieved = (
                    self.target_residual_chlorine * self.effective_contact_time
            )
        else:
            self._ct_achieved = None

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to permeate output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.chlorine_bus is not None:
            attributes[f"conversion_factor_{self.chlorine_bus.label}"] = sequence(
                self._chlorine_kg_per_m3
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
        total_marginal_cost = self.marginal_cost + self._chlorine_cost_per_m3

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
            self.chlorine_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in []:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        allowed_modes = {"fixed_dose", "demand_residual", "tracked_chemical"}
        if self.dosing_mode not in allowed_modes:
            raise ValueError(
                f"dosing_mode must be one of {sorted(allowed_modes)}, "
                f"got '{self.dosing_mode}'."
            )

        if self.dosing_mode == "tracked_chemical" and self.chlorine_bus is None:
            raise ValueError(
                "dosing_mode='tracked_chemical' requires chlorine_bus."
            )

        if self.dosing_mode == "fixed_dose" and self.applied_chlorine_dose is None:
            raise ValueError(
                "dosing_mode='fixed_dose' requires applied_chlorine_dose."
            )

        if self.chlorine_cost < 0:
            raise ValueError("chlorine_cost must be >= 0.")

        if self.chlorine_bus is not None and self.chlorine_cost not in (0, 0.0, None):
            self.chlorine_cost = 0.0

        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        if self.chlorine_demand < 0:
            raise ValueError("chlorine_demand must be non-negative.")

        if self.target_residual_chlorine < 0:
            raise ValueError("target_residual_chlorine must be non-negative.")

        if self.water_quality_factor <= 0:
            raise ValueError("water_quality_factor must be > 0.")

        if (
                self.applied_chlorine_dose is not None
                and self.applied_chlorine_dose < self.chlorine_demand
        ):
            raise ValueError(
                "applied_chlorine_dose is lower than chlorine_demand; "
                "no residual chlorine would remain under this simplified mass balance."
            )

        if (
                self.contact_tank_baffling_factor is not None
                and not 0 < self.contact_tank_baffling_factor <= 1
        ):
            raise ValueError("contact_tank_baffling_factor must be in (0, 1].")

        if self.target_residual_chlorine < self.residual_minimum:
            msg = (
                f"target_residual_chlorine={self.target_residual_chlorine} mg/L is "
                f"below residual_minimum={self.residual_minimum} mg/L (WHO guidance)."
            )
            if self.enforce_residual_check:
                raise ValueError(msg)
            warnings.warn(msg, UserWarning)

        if self.ct_target is not None and self.effective_contact_time is not None:
            ct_achieved = self.target_residual_chlorine * self.effective_contact_time
            if ct_achieved < self.ct_target:
                msg = (
                    f"CT check not met: ct_achieved={ct_achieved:.3f} mg.min/L < "
                    f"ct_target={self.ct_target} mg.min/L (EPA / Irish Water guidance)."
                )
                if self.enforce_ct_check:
                    raise ValueError(msg)
                warnings.warn(msg, UserWarning)

        if self.chlorine_bus is not None and self.dosing_mode == "tracked_chemical":
            warnings.warn(
                "chlorine_bus is set in tracked_chemical mode. "
                "Verify that the chlorine commodity bus unit is kg/hr.",
                UserWarning,
            )