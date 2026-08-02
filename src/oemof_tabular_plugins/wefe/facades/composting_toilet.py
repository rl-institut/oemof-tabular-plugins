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

    Core references
    ---------------
    1. Per-capita excreta generation rates and physicochemical characterization defaults (wet/dry mass, water fraction,
       pH, nitrogen content).
       Rose, C., Parker, A., Jefferson, B., & Cartmell, E. (2015). The characterization of feces and urine: A review of
       the literature to inform advanced treatment technology. Critical Reviews in Environmental Science and Technology,
       45(17), 1827-1879. https://doi.org/10.1080/10643389.2014.1000761
    2. Sanitation-chain and system-boundary definitions; composting-toilet processing stoichiometry.
       Tilley, E., Ulrich, L., Lüthi, C., Reymond, P., & Zurbrügg, C. (2014). Compendium of sanitation systems and technologies
       (2nd rev. ed.). Swiss Federal Institute of Aquatic Science and Technology (Eawag).
       https://www.eawag.ch/fileadmin/Domain1/Abteilungen/sandec/schwerpunkte/sesp/CLUES/Compendium_2nd_pdfs/Compendium_2nd_Ed_Lowres_1p.pdf
    3. Nitrogen and phosphorus content of source-separated urine and feces.
       Jönsson, H., Richert Stintzing, A., Vinnerås, B., & Salomon, E. (2004). Guidelines on the use of urine and faeces
       in crop production (EcoSanRes Publication Series, Report 2004-2). Stockholm Environment Institute.
       https://sswm.info/sites/default/files/reference_attachments/JOENSSON%202004%20Guidelines%20on%20the%20use%20of%20urine%20and%20faeces%20in%20crop%20production.pdf
    4. Faecal/urine nutrient mass-balance data from a source-separation trial.
       Vinnerås, B., & Jönsson, H. (2002). The performance and potential of faecal separation and urine diversion to recycle
       plant nutrients in household wastewater. Bioresource Technology, 84(3), 275-282.
       https://doi.org/10.1016/S0960-8524(02)00054-8
    5. Composting design drivers — moisture content, temperature, and carbon-to-nitrogen ratio.
       Anand, C. K., & Apul, D. S. (2014). Composting toilets as a sustainable alternative to urban sanitation - A review.
       Waste Management, 34(2), 329-343. https://doi.org/10.1016/j.wasman.2013.10.006
    6. Default CH4 emission factors for on-site/dry sanitation excreta management.
       IPCC. (2019). 2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories, Volume 5: Waste,
       Chapter 6: Wastewater Treatment and Discharge. Intergovernmental Panel on Climate Change.
       https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/5_Volume5/19R_V5_6_Ch06_Wastewater.pdf
    7. UDDT field-practice additive dosing ratio (2:1 additive:feces, w/w) — input-basis component of the derived
       bulking_agent_dose default.
       Niwagaba, C., Kulabako, R. N., Mugala, P., & Jönsson, H. (2009). Comparing microbial die-off in separately collected
       faeces with ash and sawdust additives. Waste Management, 29(7), 2214-2219.
       https://doi.org/10.1016/j.wasman.2009.02.010
    8. Generic composting mass-loss rate (~19.4%, range 11.5-31.4%) — used to convert the Niwagaba et al. (2009) feces-basis
       ratio onto a compost-output basis for bulking_agent_dose.
       Breitenbeck, G. A., & Schellinger, D. (2004). Calculating the reduction in material mass and volume during composting.
       Compost Science & Utilization, 12(4), 365-371. https://doi.org/10.1080/1065657X.2004.10702206

    Main equations
    --------------
    All normalized to compost output = 1 [kg/hr].

    Stoichiometry:
        Additive input group: feces + urine both feed "in_main"
        f_feces(t) / feces_compost_fraction    = C(t)   [kg compost / hr]
        f_urine(t) / feces_compost_fraction    = C(t)   [kg compost / hr]
        f_compost(t) / 1.0                     = C(t)   [kg compost / hr]
        f_water(t) / leachate_compost_relation = C(t)   [kg compost / hr]
        defaults: feces_compost_fraction = 3.91,
                  leachate_compost_relation = 1.74

    Bulking agent commodity coupling (tracked_bulking_agent mode only):
        Q_bulk(t) = bulking_agent_dose * C(t)
        [kg/hr]     [kg/kg compost]   [kg compost/hr]

    Urine/feces coupling:
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
    - Primary flow is human_feces_bus [kg/hr]. Capacity constrains the maximum feces throughput of the toilet, representing
      sanitation service capacity (persons served × feces generation rate per capita).
    - Bulking agent input can be modelled either as an output-side variable cost embedded in the compost flow (default,
      dosing_mode='cost_only', no bulking_agent_bus) or as a tracked third input commodity (dosing_mode='tracked_bulking_agent',
      requires bulking_agent_bus).
    - marginal_cost is charged on the primary input flow (kg feces / hr).
    - Characterization values (pH, dry solids, C:N, per-capita generation rates) are stored as metadata for scenario documentation
      and calibration. They are not enforced as hard optimization constraints.
    - UNIT NOTE: feces flows are in kg; urine, leachate, and water flows are in m³. Compost output is in kg. Conversion
      factors carry the unit bridge.
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
    human_feces_bus: Bus = None         # kg (PRIMARY)
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
    leachate_compost_relation: float = 1.74     # m³ leach/kg compost   [Eawag T14]
    feces_density: float = 1060.0               # kg/m³                 [Rose 2015]
    bulking_agent_dose: float = 0.83            # kg bulking agent/kg compost [derived: Niwagaba et al. 2009, Breitenbeck
    # & Schellinger 2004] Derived (not directly reported): 2:1 additive:feces input ratio (Niwagaba 2009)
    # combined with ~19.4% composting mass loss (Breitenbeck & Schellinger 2004): 2/(3×0.806) ≈ 0.83.
    n_from_urine: Optional[float] = 7.73        # kg N/m³ urine         [Joensson 2004][Vinnerås & Joensson 2002]
    n_from_feces: Optional[float] = None        # kg N/kg feces         [Joensson 2004]
    p_from_feces: Optional[float] = None        # kg P/kg feces         [Joensson 2004]
    p_from_urine: Optional[float] = None        # kg P/m³ urine         [Joensson 2004]
    ch4_factor: Optional[float] = None          # kg CH4/kg feces       [IPCC 2019]

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
    # documentation / calibration defaults (not hard constraints)
    # Based on the core literature references
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
        self.output_parameters = attributes.pop("output_parameters", {})

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
        # groups
        # Additive input group: feces + urine both feed "in_main".
        # --------------------------------------------------------------
        groups = {
            "in_main": [self.human_feces_bus.label, self.human_urine_bus.label],
            "out_main": [self.compost_out_bus.label],
            "out_liquid": [self.water_out_bus.label],
        }
        attributes["groups"] = groups

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to compost output = 1 [kg/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.human_feces_bus.label}"] = sequence(
            self.feces_compost_fraction
        )
        attributes[f"conversion_factor_{self.human_urine_bus.label}"] = sequence(
            self.feces_compost_fraction # same basis as feces within in_main
        )
        attributes["conversion_factor_in_main"] = sequence(1.0)
        attributes["conversion_factor_out_main"] = sequence(1.0)
        attributes[f"conversion_factor_{self.compost_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        attributes["conversion_factor_out_liquid"] = sequence(self.leachate_compost_relation)

        if self.bulking_agent_bus is not None:
            attributes[f"conversion_factor_{self.bulking_agent_bus.label}"] = sequence(
                self.bulking_agent_dose
            )

        # --------------------------------------------------------------
        # flow-share constraints: urine inflow bounded by feces-linked ratio (approximately)
        # --------------------------------------------------------------
        attributes[f"flow_share_max_{self.human_urine_bus.label}"] = sequence(
            self._urine_per_feces_ratio
        )

        # --------------------------------------------------------------
        # optional proxy emissions via existing emission-factor logic
        # key: emission_factor_<source_bus_label>_<target_bus_label>
        # source is human_feces_bus (wet feces input), the dominant in the in_main
        # emission factors are per kg wet feces in, [Rose 2015, Joensson 2004]
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
        # primary bus label resolution
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

        if self.human_feces_bus in self.inputs:
            self.inputs[self.human_feces_bus].variable_costs = sequence(self.marginal_cost)

        bulking_variable_cost = (
            0.0 if self.dosing_mode == "tracked_bulking_agent" else self._bulking_cost_per_kg_compost
        )

        if self.compost_out_bus in self.outputs:
            out_flow = self.outputs[self.compost_out_bus]
            out_flow.variable_costs = sequence(bulking_variable_cost)
            custom_attrs = (getattr(self, "output_parameters", None) or {}).get(
                "custom_attributes"
            )
            if custom_attrs:
                for attribute, value in custom_attrs.items():
                    setattr(out_flow, attribute, value)

        if not self.expandable and self.capacity is not None:
            primary_bus_map = {
                "human_feces_bus": self.human_feces_bus,
                "human_urine_bus": self.human_urine_bus,
                "compost_out_bus": self.compost_out_bus,
                "water_out_bus": self.water_out_bus,
            }
            primary_bus = primary_bus_map.get(self.primary)
            if primary_bus is not None:
                flow = self.inputs.get(primary_bus, self.outputs.get(primary_bus))
                if flow is not None:
                    flow.nominal_value = self.capacity

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
