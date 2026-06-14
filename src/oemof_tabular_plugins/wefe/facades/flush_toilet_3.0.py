import dataclasses
import warnings
from typing import Sequence, Union, Optional, ClassVar

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class FlushToilet(MIMO):
    """
    Literature-informed flush-toilet facade based on MIMO.

    Purpose
    -------
    Sanitation-interface facade representing a cistern or tank-fed flush toilet
    that combines human feces and urine with pressurised service water to
    produce a single blackwater stream. The model is a bookkeeping /
    process-yield unit that converts excreta mass flows and flush-water volume
    into blackwater volume at a configurable flush-water intensity. It is not
    a full mechanistic biological reactor model.

    Core references
    ---------------
    1. Rose et al. (2015): excreta generation, characterization defaults, and
       per-capita feces/urine ratios used for calibration fields.
    2. Eawag Compendium of Sanitation Systems and Technologies (2nd ed.):
       sanitation-chain definitions, system boundary guidance, and blackwater
       characterization.
    3. Blackwater nutrient-recovery review (Frontiers Env. Sci., 2022):
       blackwater volume fractions and nutrient load basis for future
       recovery outputs.
    4. Emergency Sanitation Compendium / GIZ low-flush & vacuum guidance:
       flush-volume presets for low_flush, vacuum, and water_scarce modes.
    5. AD/BMP stoichiometry: reporting-only methane/energy potential — basis
       for future anaerobic-digestion outputs (not enforced in v3.0).

    Main equations
    --------------
    Feces volume conversion:
        V_feces(t) = m_feces(t) / rho_feces
        [m³/hr]     [kg/hr]       [kg/m³]

    Blackwater volume fractions (Rose et al., 2015; Frontiers Env. Sci., 2022):
        f_feces + f_urine + f_service_water = 1.0
        defaults: f_feces = 0.01, f_urine = 0.11, f_service_water = 0.88

    Flush dose — service water per unit combined excreta:
        flush_dose = f_service_water / (f_feces + f_urine)
        [m³_flush / m³_excreta]
        Enforced via conversion_factor on service_water_bus:
        f_flush(t) = flush_dose × (V_feces(t) + V_urine(t))

    Blackwater output:
        V_blackwater(t) = V_feces(t) + V_urine(t) + V_flush(t)
                        = GROUP_FLOW_in_main(t)  [m³/hr]

    Urine/feces coupling (Rose et al., 2015):
        f_urine(t) ≤ urine_per_feces_ratio × f_feces(t)
        where urine_per_feces_ratio
            = urine_volume_per_cap_per_day / feces_wet_mass_per_cap_per_day
            = 0.00142 / 0.128
            = 0.01109  [m³_urine / kg_feces]
        Enforced via flow_share_max on human_urine_bus.

    Flush-mode presets (GIZ / Emergency Sanitation Compendium):
        mode            f_service_water
        ──────────────  ───────────────
        standard        0.88
        low_flush       0.80
        vacuum          0.55
        water_scarce    0.55
    When a non-standard mode is selected, f_feces and f_urine are rescaled
    proportionally so that all three fractions continue to sum to 1.0.

    Notes
    -----
    - Primary flow is human_feces_bus [kg/hr]. Capacity constrains the maximum
      feces throughput of the toilet, representing sanitation service capacity
      (persons served × feces generation rate per capita).
    - service_water_bus is intentionally excluded from the in_main group and
      linked via flush_dose conversion factor. This keeps the group activity
      basis in excreta-volume units and couples flush-water demand to excreta
      flow without making service water a co-equal group member.
    - Grouped MIMO flows (in_main) make the input side additive; feces [kg]
      is normalized to [m³] by feces_density before summation.
    - Urine inflow is bounded by the physiological per-capita feces/urine ratio
      (Rose et al., 2015). This constraint is inactive when system-level
      penalties already enforce the ratio, but is required in multi-toilet
      systems to prevent the optimizer from routing urine independently of
      feces across competing toilet facades.
    - Characterization values (pH, dry solids, per-capita generation rates)
      are stored as metadata for scenario documentation and calibration.
      They are not enforced as hard optimization constraints in v3.0.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "flush_toilet"
    name: str = ""
    tech: str = "toilet"
    carrier: str = "water"
    mode: str = "standard"  # standard / low_flush / vacuum / water_scarce
    primary: str = "human_feces_bus"

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
    human_feces_bus: Bus = None          # kg
    human_urine_bus: Bus = None          # m³
    service_water_bus: Bus = None        # m³ (flush water)
    black_water_bus: Bus = None          # m³

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    # reserved for future extension

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    feces_density: float = 1060.0           # kg/m³ wet feces
    feces_fraction: float = 0.01            # m³ wet feces /m³ blackwater
    urine_fraction: float = 0.11            # m³ urine /m³ blackwater
    service_water_fraction: float = 0.88    # m³ flush water/m³ blackwater

    # flush presets: service-water fraction per mode (drives the dose).
    flush_presets: ClassVar[dict] = {
        "standard": 0.88,
        "low_flush": 0.80,
        "vacuum": 0.55,
        "water_scarce": 0.55,
    }

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # €/m³ blackwater

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # Rose et al. (2015) style characterization fields
    # ------------------------------------------------------------------
    population_equivalent: float = 1.0
    feces_wet_mass_per_cap_per_day: float = 0.128
    feces_dry_mass_per_cap_per_day: float = 0.029
    feces_water_fraction: float = 0.746
    urine_volume_per_cap_per_day: float = 0.00142
    feces_pH: float = 6.64
    urine_pH: float = 6.2
    urine_nitrogen_g_per_cap_per_day: float = 10.98


    def __init__(self, **attributes):
        # --------------------------------------------------------------
        # identity
        # --------------------------------------------------------------
        self.type = attributes.pop("type", self.type)
        self.name = attributes.pop("name", self.name)
        self.tech = attributes.pop("tech", self.tech)
        self.carrier = attributes.pop("carrier", self.carrier)
        self.mode = attributes.pop("mode", self.mode)
        self.primary = attributes.pop("primary", self.primary)

        # --------------------------------------------------------------
        # mandatory buses
        # --------------------------------------------------------------
        self.human_feces_bus = attributes.pop("human_feces_bus")
        self.human_urine_bus = attributes.pop("human_urine_bus")
        self.service_water_bus = attributes.pop("service_water_bus")
        self.black_water_bus = attributes.pop("black_water_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        # reserved for future extension

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.feces_density = attributes.pop("feces_density", self.feces_density)
        self.feces_fraction = attributes.pop("feces_fraction", self.feces_fraction)
        self.urine_fraction = attributes.pop("urine_fraction", self.urine_fraction)
        self.service_water_fraction = attributes.pop(
            "service_water_fraction", self.service_water_fraction
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
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
        self.population_equivalent = attributes.pop(
            "population_equivalent", self.population_equivalent
        )
        self.feces_wet_mass_per_cap_per_day = attributes.pop(
            "feces_wet_mass_per_cap_per_day", self.feces_wet_mass_per_cap_per_day
        )
        self.feces_dry_mass_per_cap_per_day = attributes.pop(
            "feces_dry_mass_per_cap_per_day", self.feces_dry_mass_per_cap_per_day
        )
        self.feces_water_fraction = attributes.pop(
            "feces_water_fraction", self.feces_water_fraction
        )
        self.urine_volume_per_cap_per_day = attributes.pop(
            "urine_volume_per_cap_per_day", self.urine_volume_per_cap_per_day
        )
        self.feces_pH = attributes.pop("feces_pH", self.feces_pH)
        self.urine_pH = attributes.pop("urine_pH", self.urine_pH)
        self.urine_nitrogen_g_per_cap_per_day = attributes.pop(
            "urine_nitrogen_g_per_cap_per_day",
            self.urine_nitrogen_g_per_cap_per_day,
        )

        # --------------------------------------------------------------
        # apply flush presets and validate parameters
        # --------------------------------------------------------------
        self._apply_flush_presets()
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # flush dose: m³ service water (flush) per m³ combined excreta
        # per-capita urine/feces coupling ratio (Rose et al., 2015)
        # urine inflow is bounded by the physiological feces-linked ratio
        # urine_volume_per_cap_per_day / feces_wet_mass_per_cap_per_day
        # = 0.00142 / 0.128 = 0.01109 m³_urine / kg_feces
        # --------------------------------------------------------------
        excreta_fraction = self.feces_fraction + self.urine_fraction
        self.flush_dose = self.service_water_fraction / excreta_fraction
        self._urine_per_feces_ratio = (
                self.urine_volume_per_cap_per_day
                / self.feces_wet_mass_per_cap_per_day
        )

        # --------------------------------------------------------------
        # groups
        # main additive input group + main output group + single buses
        # --------------------------------------------------------------
        attributes["groups"] = {
            "in_main": [
                self.human_feces_bus.label,
                self.human_urine_bus.label,
            ],
            "out_main": [self.black_water_bus.label],
        }

        # --------------------------------------------------------------
        # conversion factors (division semantic GROUP_FLOW = sum flow_i / cf_i)
        # bus-level factors convert flow into common activity basis
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.human_feces_bus.label}"] = sequence(
            self.feces_density
        )
        attributes[f"conversion_factor_{self.human_urine_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.service_water_bus.label}"] = sequence(
            self.flush_dose
        )
        attributes[f"conversion_factor_{self.black_water_bus.label}"] = sequence(1.0)

        # Group-level normalization: no extra scaling on the group
        attributes["conversion_factor_in_main"] = sequence(1.0)
        attributes["conversion_factor_out_main"] = sequence(1.0)

        # --------------------------------------------------------------
        # flow-share constraints: urine inflow bounded by feces-linked ratio
        # --------------------------------------------------------------
        attributes[f"flow_share_max_{self.human_urine_bus.label}"] = sequence(
            self._urine_per_feces_ratio
        )

        # --------------------------------------------------------------
        # primary bus should point to actual bus label
        # --------------------------------------------------------------
        if self.primary == "human_feces_bus":
            primary_label = self.human_feces_bus.label
        elif self.primary == "human_urine_bus":
            primary_label = self.human_urine_bus.label
        elif self.primary == "service_water_bus":
            primary_label = self.service_water_bus.label
        elif self.primary == "black_water_bus":
            primary_label = self.black_water_bus.label
        else:
            primary_label = self.primary

        # --------------------------------------------------------------
        # initialize base MIMO facade
        # --------------------------------------------------------------
        super().__init__(
            from_bus_0=self.human_feces_bus,
            from_bus_1=self.human_urine_bus,
            from_bus_2=self.service_water_bus,
            to_bus_0=self.black_water_bus,
            primary=primary_label,
            marginal_cost=self.marginal_cost,
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
        idx_in = 3
        idx_out = 1
        # inputs
        for bus in []:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in []:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _apply_flush_presets(self):
        if self.mode != "standard" and self.mode in self.flush_presets:
            new_water_fraction = self.flush_presets[self.mode]
            remaining = 1.0 - new_water_fraction
            ratio_sum = self.urine_fraction + self.feces_fraction
            self.urine_fraction = remaining * (self.urine_fraction / ratio_sum)
            self.feces_fraction = remaining * (self.feces_fraction / ratio_sum)
            self.service_water_fraction = new_water_fraction

    def _validate_parameters(self):
        if self.mode not in self.flush_presets:
            raise ValueError(f"mode must be one of {sorted(self.flush_presets)}.")

        if self.feces_density <= 0:
            raise ValueError("feces_density must be > 0.")

        fractions = (
            self.feces_fraction,
            self.urine_fraction,
            self.service_water_fraction,
        )
        if any(f < 0 for f in fractions):
            raise ValueError("Volume fractions must be non-negative.")

        if (self.feces_fraction + self.urine_fraction) <= 0:
            raise ValueError("feces_fraction + urine_fraction must be > 0.")

        total = sum(fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"feces + urine + service_water fractions must sum to 1.0 (got {total:.6f})."
            )
        if not 0 <= self.feces_water_fraction <= 1:
            raise ValueError("feces_water_fraction must be between 0 and 1.")
