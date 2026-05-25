import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class PhotocatalyticUnit(MIMO):
    """
    Literature-informed photocatalytic water treatment facade based on MIMO.

    Purpose
    -------
    Generic electricity-driven oxidative pollutant removal facade for
    photocatalytic units used in water treatment systems. The model is
    designed as a bookkeeping/process-yield unit representing a photocatalytic
    reactor as a water-treatment intervention. It is not a full mechanistic
    kinetic reactor model.

    Core references
    ---------------
    1. Wang et al. (2023), Catalysts: reactor design variables, optimization
       logic, catalyst loading/recovery, reaction-condition control, and
       cost-aware design for practical deployment.
    2. Espíndola & Vilar (2020), JCTB: dominant engineering parameters for
       photoreactor design and scale-up — light source, catalyst dosage,
       reactor configuration, and hydrodynamics.
    3. Pichat ed. (2013), Photocatalysis and Water Purification (Wiley-VCH):
       fundamentals, reactor-design framing, and engineering scope justification.

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Electricity demand:
        E = SEC_base * performance_factor * hydraulic_factor * water_quality_factor
                                                               [kWh / m³_product]

    Catalyst demand (active only if catalyst_bus is provided):
        C = catalyst_dose_base * 1e-3 * deactivation_factor * water_quality_factor
                                                               [kg / m³_product]

    Spent catalyst output (active only if spent_catalyst_bus is provided):
        S = C * spent_catalyst_yield                           [kg / m³_product]

    Output water quality:
        Cout = Cin * (1 - target_removal)                      [mg/L]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum
      treated-water throughput of the unit.
    - UV lamp electricity is intentionally NOT modeled as a separate input bus.
      It is assumed to be fully included in specific_energy_consumption_base,
      which covers all electrical demand (pumping, UV lamps, ancillaries).
      A separate UV bus would cause double-counting. For solar photocatalysis
      variants where UV is externally sourced, set solar_mode=True
      (documentation flag only — no effect on optimization equations).
    - If catalyst_bus is active, catalyst cost must come from the upstream
      catalyst supply node. catalyst_cost is only applied as an output-side
      variable cost surcharge when catalyst_bus is absent.
    - spent_catalyst_bus requires catalyst_bus to be active. A warning is
      raised if spent_catalyst_bus is provided without catalyst_bus.
    - Engineering proxy scalers (performance_factor, hydraulic_factor,
      water_quality_factor, deactivation_factor) keep the model linear while
      reflecting operating-condition sensitivity identified in the literature.
    - Characterization values (Cin, target_removal, reactor_type, catalyst_mode,
      light_source_type) are stored as documentation/calibration defaults.
      They are not enforced as hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "photocatalysis"
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
    electricity_bus: Bus = None         # kWh
    water_in_bus: Bus = None            # m³
    water_out_bus: Bus = None           # m³  (PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    catalyst_bus: Optional[Bus] = None  # kg — if absent, cost is output-side only

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    spent_catalyst_bus: Optional[Bus] = None  # kg — requires catalyst_bus to be active

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption_base: float = 0.12      # kWh / m³ treated water
    catalyst_dose_base: float = 10.0                    # g / m³ treated water
    spent_catalyst_yield: float = 1.0                   # kg spent / kg dosed (0–1, or >1 with adsorbed mass)
    performance_factor: float = 1.0                     # non-ideal operation / operating-condition scaling
    hydraulic_factor: float = 1.0                       # residence-time / flow-regime adequacy proxy
    water_quality_factor: float = 1.0                   # influent difficulty / pollutant-class proxy
    deactivation_factor: float = 1.0                    # catalyst aging / recovery loss proxy

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                  # €/m³ treated water
    carrier_cost: float = 0.0                   # €/kWh electricity
    catalyst_cost: float = 0.0                  # €/kg — used only if catalyst_bus is absent
    spent_catalyst_disposal_cost: float = 0.0   # €/kg spent catalyst
    cleaning_cost: float = 0.0                  # €/m³ treated water (O&M surcharge)

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # Wang et al. (2023) / Espindola & Vilar (2020) style characterization
    # ------------------------------------------------------------------
    Cin: float = 10.0                   # mg/L influent pollutant concentration
    target_removal: float = 0.8         # fraction [0, 1]
    sec_typical_min: float = 0.05       # kWh/m³, lower bound from literature
    sec_typical_max: float = 0.5        # kWh/m³, upper bound from literature
    reactor_type: str = "generic"       # e.g. slurry / annular / thin-film / CPC
    catalyst_mode: str = "suspended"    # suspended / immobilized
    light_source_type: str = "UV"       # UV / LED / solar
    solar_mode: bool = False            # True = SEC covers pump-only; UV is externally sourced

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
        self.catalyst_bus = attributes.pop("catalyst_bus", None)
        self.spent_catalyst_bus = attributes.pop("spent_catalyst_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption_base = attributes.pop(
            "specific_energy_consumption_base",
            self.specific_energy_consumption_base,
        )
        self.catalyst_dose_base = attributes.pop(
            "catalyst_dose_base", self.catalyst_dose_base
        )
        self.spent_catalyst_yield = attributes.pop(
            "spent_catalyst_yield", self.spent_catalyst_yield
        )
        self.performance_factor = attributes.pop(
            "performance_factor", self.performance_factor
        )
        self.hydraulic_factor = attributes.pop(
            "hydraulic_factor", self.hydraulic_factor
        )
        self.water_quality_factor = attributes.pop(
            "water_quality_factor", self.water_quality_factor
        )
        self.deactivation_factor = attributes.pop(
            "deactivation_factor", self.deactivation_factor
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.catalyst_cost = attributes.pop("catalyst_cost", self.catalyst_cost)
        self.spent_catalyst_disposal_cost = attributes.pop(
            "spent_catalyst_disposal_cost", self.spent_catalyst_disposal_cost
        )
        self.cleaning_cost = attributes.pop("cleaning_cost", self.cleaning_cost)
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
        self.Cin = attributes.pop("Cin", self.Cin)
        self.target_removal = attributes.pop("target_removal", self.target_removal)
        self.sec_typical_min = attributes.pop("sec_typical_min", self.sec_typical_min)
        self.sec_typical_max = attributes.pop("sec_typical_max", self.sec_typical_max)
        self.reactor_type = attributes.pop("reactor_type", self.reactor_type)
        self.catalyst_mode = attributes.pop("catalyst_mode", self.catalyst_mode)
        self.light_source_type = attributes.pop(
            "light_source_type", self.light_source_type
        )
        self.solar_mode = attributes.pop("solar_mode", self.solar_mode)

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # (Wang et al., 2023; Espindola & Vilar, 2020)
        # --------------------------------------------------------------
        Cout = self.Cin * (1 - self.target_removal)

        self._electricity_per_output = (
                self.specific_energy_consumption_base
                * self.performance_factor
                * self.hydraulic_factor
                * self.water_quality_factor
        )

        self._catalyst_per_output = (
                self.catalyst_dose_base
                * 1e-3
                * self.deactivation_factor
                * self.water_quality_factor
        )

        self._spent_catalyst_per_output = (
                self._catalyst_per_output * self.spent_catalyst_yield
        )

        # --------------------------------------------------------------
        # conversion factors
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._electricity_per_output
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.catalyst_bus is not None:
            attributes[f"conversion_factor_{self.catalyst_bus.label}"] = sequence(
                self._catalyst_per_output
            )
        if self.spent_catalyst_bus is not None:
            attributes[f"conversion_factor_{self.spent_catalyst_bus.label}"] = sequence(
                self._spent_catalyst_per_output
            )

        # --------------------------------------------------------------
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # --------------------------------------------------------------
        residual_output_variable_costs = self.cleaning_cost
        if self.catalyst_bus is None:
            residual_output_variable_costs += (
                    self._catalyst_per_output * self.catalyst_cost
            )

        if residual_output_variable_costs > 0:
            attributes.setdefault("output_parameters", {})
            attributes["output_parameters"].update(
                {
                    "variable_costs": residual_output_variable_costs,
                    "custom_attributes": {"Cout_mg_per_L": Cout},
                }
            )

        if self.spent_catalyst_bus is not None:
            attributes.setdefault("output_parameters_1", {})
            if self.spent_catalyst_disposal_cost > 0:
                attributes["output_parameters_1"].update(
                    {"variable_costs": self.spent_catalyst_disposal_cost}
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

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 1
        # inputs
        for bus in [
            self.catalyst_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.spent_catalyst_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if self.specific_energy_consumption_base < 0:
            raise ValueError("specific_energy_consumption_base must be >= 0.")
        if self.catalyst_dose_base < 0:
            raise ValueError("catalyst_dose_base must be >= 0.")
        if self.spent_catalyst_yield < 0:
            raise ValueError("spent_catalyst_yield must be >= 0.")
        if self.Cin < 0:
            raise ValueError("Cin must be >= 0.")
        if not 0 <= self.target_removal <= 1:
            raise ValueError("target_removal must be in [0, 1].")

        for factor_name, factor_value in {
            "performance_factor": self.performance_factor,
            "hydraulic_factor": self.hydraulic_factor,
            "water_quality_factor": self.water_quality_factor,
            "deactivation_factor": self.deactivation_factor,
        }.items():
            if factor_value <= 0:
                raise ValueError(f"{factor_name} must be > 0.")

        if self.spent_catalyst_bus is not None and self.catalyst_bus is None:
            warnings.warn(
                "spent_catalyst_bus is provided but catalyst_bus is absent. "
                "Spent catalyst conversion factor is derived from catalyst_dose_base "
                "but no explicit catalyst input flow is modeled.",
                UserWarning,
            )
        if self.target_removal > 0.95:
            warnings.warn(
                "target_removal > 0.95 may be optimistic for a generic model.",
                UserWarning,
            )
        if self.solar_mode and self.light_source_type.lower() != "solar":
            warnings.warn(
                "solar_mode=True but light_source_type is not 'solar'. "
                "Check consistency of documentation fields.",
                UserWarning,
            )
        if self.catalyst_mode.lower() == "immobilized" and self.catalyst_bus is not None:
            warnings.warn(
                "catalyst_bus is active but catalyst_mode='immobilized'. "
                "Verify that catalyst replenishment is correctly modeled as a consumable input flow.",
                UserWarning,
            )
        if (
                self.catalyst_bus is None
                and self.catalyst_cost == 0.0
                and self.catalyst_dose_base > 0
        ):
            warnings.warn(
                "No catalyst_bus and catalyst_cost == 0: catalyst use is "
                "physically represented but economically uncosted.",
                UserWarning,
            )
        if self.sec_typical_min is not None and self.sec_typical_max is not None:
            if not (
                    self.sec_typical_min
                    <= self.specific_energy_consumption_base
                    <= self.sec_typical_max
            ):
                warnings.warn(
                    f"specific_energy_consumption_base="
                    f"{self.specific_energy_consumption_base} kWh/m³ is outside "
                    f"the typical literature range "
                    f"[{self.sec_typical_min}, {self.sec_typical_max}] kWh/m³.",
                    UserWarning,
                )

