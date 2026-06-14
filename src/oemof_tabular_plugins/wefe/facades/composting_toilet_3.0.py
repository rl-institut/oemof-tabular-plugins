import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class CompostingToilet(MIMO):
    """
    Literature-informed composting-toilet facade based on MIMO.

    Purpose
    -------
    Fixed-recipe sanitation-interface facade representing a urine-diverting or
    mixed composting toilet. The model is a bookkeeping / process-yield unit
    that converts human feces and urine into compost and leachate at fixed
    stoichiometric ratios. It is not a mechanistic biological reactor model.
    Inputs are pairwise-coupled to the compost output via fixed conversion
    factors; the solver decides how much throughput flows here, and the
    bulking agent and marginal_cost express the operating cost of the pathway.

    Core references
    ---------------
    1. Rose et al. (2015): excreta generation, characterization defaults, and
       per-capita feces/urine ratios used for calibration fields and the
       urine/feces coupling constraint.
    2. Eawag Compendium of Sanitation Systems and Technologies (2nd ed.,
       Tilley et al. 2014): sanitation-chain definitions, system-boundary
       guidance, and stoichiometric defaults for composting
       (feces_compost_fraction, urine_compost_fraction,
       leachate_compost_relation).
    3. Joensson et al. (2004), EcoSanRes 2004-2: N/P recovery coefficients
       used as defaults for optional nutrient accounting buses.
    4. Anand & Apul (2014), Waste Management: composting design drivers
       (moisture content, C:N ratio, bulking agent dose); basis for
       moisture and C:N calibration defaults.
    5. Vinneras & Joensson (2002), Bioresource Technology: faecal/urine
       nutrient mass balance; basis for n_from_urine default.

    Main equations
    --------------
    Stoichiometry (conversion_factor divisor semantics):
        f_feces(t) / feces_compost_fraction    = C(t)   [kg compost / hr]
        f_urine(t) / urine_compost_fraction    = C(t)   [kg compost / hr]
        f_water(t) / leachate_compost_relation = C(t)   [kg compost / hr]
        defaults: feces_compost_fraction = 3.91,
                  urine_compost_fraction = 2.17,
                  leachate_compost_relation = 1.74

    Bulking agent commodity coupling (tracked_bulking_agent mode only):
        Q_bulk(t) = bulking_agent_dose * C(t)
        [kg/hr]     [kg/kg compost]   [kg compost/hr]

    Urine/feces coupling (Rose et al., 2015):
        f_urine(t) ≤ urine_per_feces_ratio × f_feces(t)
        where urine_per_feces_ratio
            = urine_volume_per_cap_per_day / feces_wet_mass_per_cap_per_day
            = 0.00142 / 0.128
            = 0.01109  [m³_urine / kg_feces]
        Enforced via flow_share_max on human_urine_bus.

    Optional nutrient recovery and GHG emission factors:
        R_N(t) = n_from_urine * f_urine(t) + n_from_feces * f_feces(t)
        R_P(t) = p_from_feces * f_feces(t) + p_from_urine * f_urine(t)
        E_CH4(t) = ch4_factor * f_feces(t)
        Only active when the corresponding bus AND factor are both set.

    Notes
    -----
    - Primary flow is human_feces_bus [kg/hr]. Capacity constrains the maximum
      feces throughput of the toilet, representing sanitation service capacity
      (persons served × feces generation rate per capita).
    - Bulking agent input can be modelled either as an output-side variable
      cost embedded in the compost flow (default, dosing_mode='cost_only',
      no bulking_agent_bus) or as a tracked third input commodity
      (dosing_mode='tracked_bulking_agent', requires bulking_agent_bus).
    - marginal_cost is charged on the primary input flow (kg feces / hr),
      not on the compost output. It represents a collection or handling cost
      tied to input volume.
    - Inputs are pairwise-coupled, not additive. The conversion factors enforce
      a fixed feces-to-compost and urine-to-compost recipe. Any deviation from
      the recipe makes the system infeasible for that timestep.
    - Urine inflow is bounded by the physiological per-capita feces/urine ratio
      (Rose et al., 2015). This constraint is inactive when system-level
      penalties already enforce the ratio, but is required in multi-toilet
      systems to prevent the optimizer from routing urine independently of
      feces.
    - Accounting buses (N, P, CH4) must be created with balanced=False in the
      datapackage unless a real downstream sink consumes them.
    - Characterization values (pH, dry solids, C:N, per-capita generation rates)
      are stored as metadata for scenario documentation and calibration.
      They are not enforced as hard optimization constraints.
    - UNIT NOTE: feces flows are in kg; urine, leachate, and water flows are
      in m³. Compost output is in kg. Conversion factors carry the unit bridge.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "composting_toilet"
    name: str = ""
    tech: str = "toilet"
    carrier: str = "water"
    dosing_mode: str = "cost_only"  # "cost_only" | "tracked_bulking_agent"
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
    human_feces_bus: Bus = None         # kg
    human_urine_bus: Bus = None         # m³
    compost_out_bus: Bus = None         # kg
    water_out_bus: Bus = None           # m³ (leachate)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    bulking_agent_bus: Optional[Bus] = None  # kg  only in tracked_bulking_agent mode

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    n_recovery_bus: Optional[Bus] = None       # kg N
    p_recovery_bus: Optional[Bus] = None       # kg P
    ch4_bus: Optional[Bus] = None              # kg CH4

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    feces_compost_fraction: float = 3.91        # kg feces/kg compost   [Eawag T14]
    urine_compost_fraction: float = 2.17        # m³ urine/kg compost   [Eawag T14]
    leachate_compost_relation: float = 1.74     # m³ leach/kg compost   [Eawag T14]
    feces_density: float = 1060.0               # kg/m³                 [Rose 2015]
    bulking_agent_dose: float = 2.61            # kg bulking agent/kg compost  [GTZ]

    n_from_urine: Optional[float] = 7.73        # kg N/m³ urine         [Rose 2015]
    n_from_feces: Optional[float] = None        # kg N/kg feces         [Joensson 2004]
    p_from_feces: Optional[float] = None        # kg P/kg feces         [Joensson 2004]
    p_from_urine: Optional[float] = None        # kg P/m³ urine         [Joensson 2004]
    ch4_factor: Optional[float] = None          # kg CH4/kg feces

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    bulking_agent_cost: float = 0.2             # USD/kg bulking agent   [GTZ]
    marginal_cost: float = 0.0                  # USD/kg human feces (charged on primary input flow)

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # ------------------------------------------------------------------
    feces_wet_mass_per_cap_per_day: float = 0.128           # kg/cap/day       [Rose 2015]
    feces_dry_mass_per_cap_per_day: float = 0.029           # kg/cap/day       [Rose 2015]
    feces_water_fraction: float = 0.746                     # mass fraction    [Rose 2015]
    urine_volume_per_cap_per_day: float = 0.00142           # m³/cap/day       [Rose 2015]
    urine_nitrogen_g_per_cap_per_day: float = 10.98         # g N/cap/day      [Rose 2015]
    feces_pH: float = 6.64                                  # [Rose 2015]
    urine_pH: float = 6.2                                   # [Rose 2015]
    cn_target: float = 25.0                                 # target C:N ratio [Anand 2014]
    moisture_target_min: float = 0.45                       # [Anand 2014]
    moisture_target_max: float = 0.70                       # [Anand 2014]

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
        self.human_feces_bus = attributes.pop("human_feces_bus")
        self.human_urine_bus = attributes.pop("human_urine_bus")
        self.compost_out_bus = attributes.pop("compost_out_bus")
        self.water_out_bus = attributes.pop("water_out_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.bulking_agent_bus = attributes.pop("bulking_agent_bus", None)
        self.n_recovery_bus = attributes.pop("n_recovery_bus", None)
        self.p_recovery_bus = attributes.pop("p_recovery_bus", None)
        self.ch4_bus = attributes.pop("ch4_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.feces_compost_fraction = attributes.pop("feces_compost_fraction", self.feces_compost_fraction)
        self.urine_compost_fraction = attributes.pop("urine_compost_fraction", self.urine_compost_fraction)
        self.leachate_compost_relation = attributes.pop("leachate_compost_relation", self.leachate_compost_relation)
        self.feces_density = attributes.pop("feces_density", self.feces_density)
        self.bulking_agent_dose = attributes.pop("bulking_agent_dose", self.bulking_agent_dose)
        self.n_from_urine = attributes.pop("n_from_urine", self.n_from_urine)
        self.n_from_feces = attributes.pop("n_from_feces", self.n_from_feces)
        self.p_from_feces = attributes.pop("p_from_feces", self.p_from_feces)
        self.p_from_urine = attributes.pop("p_from_urine", self.p_from_urine)
        self.ch4_factor = attributes.pop("ch4_factor", self.ch4_factor)

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.bulking_agent_cost = attributes.pop("bulking_agent_cost", self.bulking_agent_cost)
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
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

        # --------------------------------------------------------------
        # documentation / calibration defaults
        # --------------------------------------------------------------
        self.feces_wet_mass_per_cap_per_day = attributes.pop("feces_wet_mass_per_cap_per_day",
                                                             self.feces_wet_mass_per_cap_per_day)
        self.feces_dry_mass_per_cap_per_day = attributes.pop("feces_dry_mass_per_cap_per_day",
                                                             self.feces_dry_mass_per_cap_per_day)
        self.feces_water_fraction = attributes.pop("feces_water_fraction", self.feces_water_fraction)
        self.urine_volume_per_cap_per_day = attributes.pop("urine_volume_per_cap_per_day",
                                                           self.urine_volume_per_cap_per_day)
        self.urine_nitrogen_g_per_cap_per_day = attributes.pop("urine_nitrogen_g_per_cap_per_day",
                                                               self.urine_nitrogen_g_per_cap_per_day)
        self.feces_pH = attributes.pop("feces_pH", self.feces_pH)
        self.urine_pH = attributes.pop("urine_pH", self.urine_pH)
        self.cn_target = attributes.pop("cn_target", self.cn_target)
        self.moisture_target_min = attributes.pop("moisture_target_min", self.moisture_target_min)
        self.moisture_target_max = attributes.pop("moisture_target_max", self.moisture_target_max)

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._bulking_cost_per_kg_compost = (
                self.bulking_agent_dose * self.bulking_agent_cost
        )
        self._urine_per_feces_ratio = (
                self.urine_volume_per_cap_per_day
                / self.feces_wet_mass_per_cap_per_day
        )

        # --------------------------------------------------------------
        # conversion factors
        # bus-level factors convert flow into common activity basis
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.human_feces_bus.label}"] = sequence(
            self.feces_compost_fraction
        )
        attributes[f"conversion_factor_{self.human_urine_bus.label}"] = sequence(
            self.urine_compost_fraction
        )
        attributes[f"conversion_factor_{self.compost_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(
            self.leachate_compost_relation
        )

        if self.bulking_agent_bus is not None:
            attributes[f"conversion_factor_{self.bulking_agent_bus.label}"] = sequence(
                self.bulking_agent_dose
            )

        # --------------------------------------------------------------
        # flow-share constraints: urine inflow bounded by feces-linked ratio
        # --------------------------------------------------------------
        attributes[f"flow_share_max_{self.human_urine_bus.label}"] = sequence(
            self._urine_per_feces_ratio
        )

        # --------------------------------------------------------------
        # optional proxy emissions via existing emission-factor logic
        # key: emission_factor_<source_bus_label>_<target_bus_label>
        # applied per source bus independently [Rose 2015, Joensson 2004]
        # --------------------------------------------------------------
        emission_pairs = [
            (self.human_urine_bus, self.n_recovery_bus, self.n_from_urine),
            (self.human_feces_bus, self.n_recovery_bus, self.n_from_feces),
            (self.human_feces_bus, self.p_recovery_bus, self.p_from_feces),
            (self.human_urine_bus, self.p_recovery_bus, self.p_from_urine),
            (self.human_feces_bus, self.ch4_bus, self.ch4_factor),
        ]
        for source_bus, target_bus, factor in emission_pairs:
            if target_bus is not None and factor is not None:
                attributes[
                    f"emission_factor_{source_bus.label}_{target_bus.label}"
                ] = sequence(float(factor))

        # --------------------------------------------------------------
        # output-specific variable costs
        # cost_only mode: bulking agent cost embedded in compost output flow
        # tracked_bulking_agent mode: cost borne by the input commodity bus
        # --------------------------------------------------------------
        attributes.setdefault("output_parameters", {})
        if self.dosing_mode == "cost_only":
            attributes["output_parameters"].update({
                "variable_costs": self._bulking_cost_per_kg_compost,
                "custom_attributes": {"bulking_agent_dose": self.bulking_agent_dose},
            })

        # --------------------------------------------------------------
        # primary bus should point to actual bus label
        # --------------------------------------------------------------
        if self.primary == "human_feces_bus":
            primary_label = self.human_feces_bus.label
        elif self.primary == "human_urine_bus":
            primary_label = self.human_urine_bus.label
        elif self.primary == "compost_out_bus":
            primary_label = self.compost_out_bus.label
        elif self.primary == "water_out_bus":
            primary_label = self.water_out_bus.label
        else:
            primary_label = self.primary

        # --------------------------------------------------------------
        # initialize base MIMO facade
        # --------------------------------------------------------------
        super().__init__(
            from_bus_0=self.human_feces_bus,
            from_bus_1=self.human_urine_bus,
            to_bus_0=self.compost_out_bus,
            to_bus_1=self.water_out_bus,
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
        idx_in = 2
        idx_out = 2
        # inputs
        for bus in [
            self.bulking_agent_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.n_recovery_bus,
            self.p_recovery_bus,
            self.ch4_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        allowed_modes = {"cost_only", "tracked_bulking_agent"}
        if self.dosing_mode not in allowed_modes:
            raise ValueError(
                f"dosing_mode must be one of {sorted(allowed_modes)}, got '{self.dosing_mode}'."
            )

        if self.dosing_mode == "tracked_bulking_agent" and self.bulking_agent_bus is None:
            raise ValueError("dosing_mode='tracked_bulking_agent' requires bulking_agent_bus.")

        if self.dosing_mode == "cost_only" and self.bulking_agent_bus is not None:
            warnings.warn(
                "bulking_agent_bus set but dosing_mode='cost_only' — bus cost not tracked. "
                "Use dosing_mode='tracked_bulking_agent' to activate.",
                UserWarning,
            )

        positive = {
            "feces_compost_fraction": self.feces_compost_fraction,
            "urine_compost_fraction": self.urine_compost_fraction,
            "leachate_compost_relation": self.leachate_compost_relation,
        }
        for name, value in positive.items():
            if value is None or value <= 0:
                raise ValueError(f"{name} must be > 0, got {value!r}.")

        if self.feces_density is None or self.feces_density <= 0:
            raise ValueError(f"feces_density must be > 0, got {self.feces_density!r}.")

        if self.feces_wet_mass_per_cap_per_day is None or self.feces_wet_mass_per_cap_per_day <= 0:
            raise ValueError("feces_wet_mass_per_cap_per_day must be > 0 (flow_share_max denominator).")

        if self.urine_volume_per_cap_per_day is None or self.urine_volume_per_cap_per_day <= 0:
            raise ValueError("urine_volume_per_cap_per_day must be > 0 (flow_share_max numerator).")

        non_negative = {
            "bulking_agent_dose": self.bulking_agent_dose,
            "bulking_agent_cost": self.bulking_agent_cost,
            "marginal_cost": self.marginal_cost,
        }
        for name, value in non_negative.items():
            if value is None or value < 0:
                raise ValueError(f"{name} must be >= 0, got {value!r}.")

        optional_factors = {
            "n_from_urine": self.n_from_urine,
            "n_from_feces": self.n_from_feces,
            "p_from_feces": self.p_from_feces,
            "p_from_urine": self.p_from_urine,
            "ch4_factor": self.ch4_factor,
        }
        for name, value in optional_factors.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 when set, got {value!r}.")

        recovery_pairs = [
            ("n_recovery_bus", self.n_recovery_bus, "n_from_urine", self.n_from_urine),
            ("n_recovery_bus", self.n_recovery_bus, "n_from_feces", self.n_from_feces),
            ("p_recovery_bus", self.p_recovery_bus, "p_from_feces", self.p_from_feces),
            ("p_recovery_bus", self.p_recovery_bus, "p_from_urine", self.p_from_urine),
            ("ch4_bus", self.ch4_bus, "ch4_factor", self.ch4_factor),
        ]
        for bus_name, bus, factor_name, factor in recovery_pairs:
            if bus is not None and factor is None:
                warnings.warn(f"{bus_name} set but {factor_name} is None — no emission injected.", UserWarning)
            if bus is None and factor is not None:
                warnings.warn(f"{factor_name} set but {bus_name} is None — factor has no effect.", UserWarning)

        if not 0.0 <= self.feces_water_fraction <= 1.0:
            raise ValueError(f"feces_water_fraction must be in [0, 1], got {self.feces_water_fraction!r}.")

        if not 0.0 < self.moisture_target_min < self.moisture_target_max <= 1.0:
            raise ValueError(
                f"Require 0 < moisture_target_min < moisture_target_max <= 1, "
                f"got [{self.moisture_target_min}, {self.moisture_target_max}]."
            )

        if self.cn_target is not None and self.cn_target <= 0:
            raise ValueError(f"cn_target must be > 0, got {self.cn_target!r}.")
