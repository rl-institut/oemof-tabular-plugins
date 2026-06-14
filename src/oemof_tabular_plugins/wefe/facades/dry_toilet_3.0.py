import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class DryToilet(MIMO):
    """
    Literature-informed dry toilet (UDDT) facade based on MIMO.

    Purpose
    -------
    Fixed-recipe sanitation-interface facade representing a urine-diverting dry
    toilet (UDDT) or mixed dry latrine. The model is a bookkeeping / process-yield
    unit that converts human feces and urine into dried feces and leachate at fixed
    stoichiometric ratios. It is not a mechanistic desiccation or biological reactor
    model. Inputs are pairwise-coupled to the dry feces output via fixed conversion
    factors; the solver decides how much throughput flows here, and the bulking
    agent and marginal_cost express the operating cost of the pathway.

    Core references
    ---------------
    1. Rose et al. (2015): excreta generation, characterization defaults, and
       per-capita feces/urine ratios used for calibration fields and the
       urine/feces coupling constraint.
    2. Eawag Compendium of Sanitation Systems and Technologies (2nd ed.,
       Tilley et al. 2014): sanitation-chain definitions, system-boundary
       guidance, and stoichiometric defaults for dry toilet
       (wet_feces_dry_feces_fraction, urine_dry_feces_fraction,
       leachate_dry_feces_relation).
    3. Berger, W. (2011), GIZ/SuSanA Technology Review — Composting Toilets:
       bulking agent dose design range (0.2–0.5 kg/kg feces), urine diversion
       efficiency design expectation (85–95%); basis for bulking_agent_dose
       and urine_diversion_efficiency defaults.
    4. Huussi et al. (2012/2013), WECF UDDT Use & Maintenance Guide:
       urine_diversion_efficiency field-conditions range (0.80–0.95);
       hygienisation retention time guidance.
    5. Radha et al. (2022), IJERT Vol. 14 — Advancements in Dry Toilet
       Technologies: wet-to-dry feces fraction range (3.0–4.5 kg/kg)
       cross-validating Tilley 2014 defaults; GHG yield factor ranges for
       ch4_yield_factor and n2o_yield_factor.

    Main equations
    --------------
    Stoichiometry (conversion_factor divisor semantics):
        f_feces(t)    / wet_feces_dry_feces_fraction = C(t)  [kg dry feces / hr]
        f_urine(t)    / urine_dry_feces_fraction      = C(t)  [kg dry feces / hr]
        f_dry(t)      / 1.0                           = C(t)  [kg dry feces / hr]
        f_leachate(t) / leachate_cf                   = C(t)  [kg dry feces / hr]
        defaults: wet_feces_dry_feces_fraction = 3.3,
                  urine_dry_feces_fraction     = 2.0,
                  leachate_dry_feces_relation  = 0.1  (mixed mode baseline)

    Effective leachate conversion factor by mode:
        mode = "mixed"          : leachate_cf = leachate_dry_feces_relation
        mode = "urine_diverting": leachate_cf = leachate_dry_feces_relation
                                              * (1 + urine_diversion_efficiency)

    Bulking agent commodity coupling (when bulking_agent_bus is connected):
        Q_bulk(t) = bulking_agent_dose * C(t)
        [kg/hr]     [kg/kg dry feces]   [kg dry feces/hr]

    Urine/feces coupling (Rose et al., 2015):
        f_urine(t) ≤ urine_per_feces_ratio × f_feces(t)
        where urine_per_feces_ratio
            = urine_volume_per_cap_per_day / feces_wet_mass_per_cap_per_day
            = 0.00142 / 0.128
            = 0.01109  [m³_urine / kg_feces]
        Enforced via flow_share_max on human_urine_bus.

    Optional proxy GHG emissions (IPCC 2019, waste sector convention):
        E_NH3(t) = nh3_loss_fraction  * f_feces(t)   [kg NH3 / hr]
        E_CH4(t) = ch4_yield_factor   * f_feces(t)   [kg CH4 / hr]
        E_N2O(t) = n2o_yield_factor   * f_feces(t)   [kg N2O / hr]
        Source bus is human_feces_bus (wet feces input mass).
        Only active when the corresponding bus AND a non-zero factor are set.

    Notes
    -----
    - Primary flow is human_feces_bus [kg/hr]. Capacity constrains the maximum
      feces throughput of the toilet, representing sanitation service capacity
      (persons served × feces generation rate per capita).
    - Bulking agent input can be modelled either as an output-side variable
      cost embedded in the dry feces flow (default, no bulking_agent_bus) or
      as a tracked third input commodity (requires bulking_agent_bus). When
      the bus is connected, variable_costs on dry_feces_out_bus is set to 0
      to avoid double-charging.
    - marginal_cost is charged on the primary input flow (kg feces / hr),
      representing a collection or handling cost tied to input volume.
    - Inputs are pairwise-coupled, not additive. The conversion factors enforce
      a fixed feces-to-dry-feces and urine-to-dry-feces recipe. Any deviation
      from the recipe makes the system infeasible for that timestep.
    - Urine inflow is bounded by the physiological per-capita feces/urine ratio
      (Rose et al., 2015). This constraint is inactive when system-level
      penalties already enforce the ratio, but is required in multi-toilet
      systems to prevent the optimizer from routing urine independently of
      feces.
    - In urine_diverting mode the diverted urine fraction is not routed to a
      separate bus — it increases the leachate conversion factor. Connect a
      dedicated urine bus upstream if separate urine accounting is needed.
    - Emission buses (nh3_loss_bus, ch4_bus, n2o_bus) must be created with
      balanced=False in the datapackage unless a real downstream sink consumes
      them.
    - Characterization values (pH, dry solids, per-capita generation rates)
      are stored as metadata for scenario documentation and calibration.
      They are not enforced as hard optimization constraints.
    - UNIT NOTE: feces flows are in kg; urine and leachate flows are in m³.
      Dry feces output is in kg. Conversion factors carry the unit bridge.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "dry_toilet"
    name: str = ""
    tech: str = "toilet"
    carrier: str = "water"
    mode: str = "mixed"  # "mixed" or "urine_diverting"
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
    human_feces_bus: Bus = None             # kg
    human_urine_bus: Bus = None             # m³
    dry_feces_out_bus: Bus = None           # kg
    water_out_bus: Bus = None               # m³ (leachate)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    bulking_agent_bus: Optional[Bus] = None  # kg (explicit material flow)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    nh3_loss_bus: Optional[Bus] = None      # kg
    ch4_bus: Optional[Bus] = None           # kg
    n2o_bus: Optional[Bus] = None           # kg

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    wet_feces_dry_feces_fraction: float = 3.3    # kg wet feces/kg dry feces  [Tilley 2014][Radha 2022]
    urine_dry_feces_fraction: float = 2.0        # m³ urine/kg dry feces      [Tilley 2014]
    leachate_dry_feces_relation: float = 0.1     # m³ leachate/kg dry feces — mixed mode baseline [Tilley 2014]
    feces_density: float = 1060.0                # kg/m³ wet feces              [Rose 2015]
    urine_diversion_efficiency: float = 0.85     # fraction of urine retained → leachate [0,1]  [Berger 2011][Huussi 2013]
    nh3_loss_fraction: float = 0.0               # kg NH3/kg dry feces
    ch4_yield_factor: float = 0.0                # kg CH4/kg dry feces        [IPCC 2019]
    n2o_yield_factor: float = 0.0                # kg N2O/kg dry feces        [IPCC 2019]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    bulking_agent_dose: float = 0.25                # kg bulking agent/kg dry feces  [Berger 2011]
    bulking_agent_cost: float = 0.1                 # USD/kg bulking agent
    marginal_cost: float = 0.0                      # USD/kg human feces

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # ------------------------------------------------------------------
    population_equivalent: float = 1.0
    feces_wet_mass_per_cap_per_day: float = 0.128       # kg/cap/day       [Rose 2015]
    feces_dry_mass_per_cap_per_day: float = 0.029       # kg/cap/day       [Rose 2015]
    feces_water_fraction: float = 0.746                 # mass fraction    [Rose 2015]
    urine_volume_per_cap_per_day: float = 0.00142       # m³/cap/day       [Rose 2015]
    urine_nitrogen_g_per_cap_per_day: float = 10.98     # g N/cap/day      [Rose 2015]
    feces_pH: float = 6.64                              # [Rose 2015]
    urine_pH: float = 6.2                               # [Rose 2015]

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
        self.dry_feces_out_bus = attributes.pop("dry_feces_out_bus")
        self.water_out_bus = attributes.pop("water_out_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.bulking_agent_bus = attributes.pop("bulking_agent_bus", None)
        self.nh3_loss_bus = attributes.pop("nh3_loss_bus", None)
        self.ch4_bus = attributes.pop("ch4_bus", None)
        self.n2o_bus = attributes.pop("n2o_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.wet_feces_dry_feces_fraction = attributes.pop(
            "wet_feces_dry_feces_fraction", self.wet_feces_dry_feces_fraction)
        self.urine_dry_feces_fraction = attributes.pop(
            "urine_dry_feces_fraction", self.urine_dry_feces_fraction)
        self.leachate_dry_feces_relation = attributes.pop(
            "leachate_dry_feces_relation", self.leachate_dry_feces_relation)
        self.feces_density = attributes.pop(
            "feces_density", self.feces_density)
        self.urine_diversion_efficiency = attributes.pop(
            "urine_diversion_efficiency", self.urine_diversion_efficiency)
        self.nh3_loss_fraction = attributes.pop(
            "nh3_loss_fraction", self.nh3_loss_fraction)
        self.ch4_yield_factor = attributes.pop(
            "ch4_yield_factor", self.ch4_yield_factor)
        self.n2o_yield_factor = attributes.pop(
            "n2o_yield_factor", self.n2o_yield_factor)

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.bulking_agent_dose = attributes.pop("bulking_agent_dose", self.bulking_agent_dose)
        self.bulking_agent_cost = attributes.pop("bulking_agent_cost", self.bulking_agent_cost)
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
        self.population_equivalent = attributes.pop("population_equivalent", self.population_equivalent)
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

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._bulking_cost_per_kg_dry = (
                self.bulking_agent_dose * self.bulking_agent_cost
        )
        self._urine_per_feces_ratio = (
                self.urine_volume_per_cap_per_day
                / self.feces_wet_mass_per_cap_per_day
        )
        if self.mode == "urine_diverting":
            self._effective_leachate_cf = (
                    self.leachate_dry_feces_relation * (1.0 + self.urine_diversion_efficiency)
            )
        else:
            self._effective_leachate_cf = self.leachate_dry_feces_relation

        # --------------------------------------------------------------
        # conversion factors
        # bus-level factors convert flow into common activity basis
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.human_feces_bus.label}"] = sequence(
            self.wet_feces_dry_feces_fraction
        )
        attributes[f"conversion_factor_{self.human_urine_bus.label}"] = sequence(
            self.urine_dry_feces_fraction
        )
        attributes[f"conversion_factor_{self.dry_feces_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(
            self._effective_leachate_cf
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
        # source is human_feces_bus (wet feces input), not the output —
        # emission factors are per kg wet feces in, matching IPCC waste sector convention [IPCC 2019]
        # --------------------------------------------------------------
        if self.nh3_loss_bus is not None:
            attributes[
                f"emission_factor_{self.human_feces_bus.label}_{self.nh3_loss_bus.label}"
            ] = sequence(self.nh3_loss_fraction)
        if self.ch4_bus is not None:
            attributes[
                f"emission_factor_{self.human_feces_bus.label}_{self.ch4_bus.label}"
            ] = sequence(self.ch4_yield_factor)
        if self.n2o_bus is not None:
            attributes[
                f"emission_factor_{self.human_feces_bus.label}_{self.n2o_bus.label}"
            ] = sequence(self.n2o_yield_factor)

        # --------------------------------------------------------------
        # output-specific variable costs
        # --------------------------------------------------------------
        bulking_variable_cost = (
            0.0 if self.bulking_agent_bus is not None
            else self._bulking_cost_per_kg_dry
        )
        attributes.setdefault("output_parameters", {})
        attributes["output_parameters"].update({
            "variable_costs": bulking_variable_cost,
            "custom_attributes": {"bulking_agent_dose": self.bulking_agent_dose},
        })

        # --------------------------------------------------------------
        # primary bus should point to actual bus label
        # --------------------------------------------------------------
        if self.primary == "human_feces_bus":
            primary_label = self.human_feces_bus.label
        elif self.primary == "human_urine_bus":
            primary_label = self.human_urine_bus.label
        elif self.primary == "dry_feces_out_bus":
            primary_label = self.dry_feces_out_bus.label
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
            to_bus_0=self.dry_feces_out_bus,
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
            self.nh3_loss_bus,
            self.ch4_bus,
            self.n2o_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if self.mode not in {"mixed", "urine_diverting"}:
            raise ValueError(
                f"mode must be 'mixed' or 'urine_diverting', got '{self.mode}'."
            )

        for name, value in {
            "wet_feces_dry_feces_fraction": self.wet_feces_dry_feces_fraction,
            "urine_dry_feces_fraction": self.urine_dry_feces_fraction,
            "leachate_dry_feces_relation": self.leachate_dry_feces_relation,
            "feces_density": self.feces_density,
        }.items():
            if value is None or value <= 0:
                raise ValueError(f"{name} must be > 0, got {value!r}.")

        if not 0.0 <= self.urine_diversion_efficiency <= 1.0:
            raise ValueError(
                f"urine_diversion_efficiency must be in [0, 1], got {self.urine_diversion_efficiency!r}."
            )

        for name, value in {
            "bulking_agent_dose": self.bulking_agent_dose,
            "bulking_agent_cost": self.bulking_agent_cost,
            "marginal_cost": self.marginal_cost,
        }.items():
            if value is None or value < 0:
                raise ValueError(f"{name} must be >= 0, got {value!r}.")

        if self.feces_wet_mass_per_cap_per_day <= 0:
            raise ValueError("feces_wet_mass_per_cap_per_day must be > 0.")
        if self.urine_volume_per_cap_per_day <= 0:
            raise ValueError("urine_volume_per_cap_per_day must be > 0.")

        for bus_name, bus, factor_name, factor in [
            ("nh3_loss_bus", self.nh3_loss_bus, "nh3_loss_fraction", self.nh3_loss_fraction),
            ("ch4_bus", self.ch4_bus, "ch4_yield_factor", self.ch4_yield_factor),
            ("n2o_bus", self.n2o_bus, "n2o_yield_factor", self.n2o_yield_factor),
        ]:
            if bus is not None and factor == 0.0:
                warnings.warn(
                    f"{bus_name} connected but {factor_name}=0.0 — bus carries zero flow.",
                    UserWarning,
                )
            if bus is None and factor != 0.0:
                warnings.warn(
                    f"{factor_name} set but {bus_name} is None — factor has no effect.",
                    UserWarning,
                )