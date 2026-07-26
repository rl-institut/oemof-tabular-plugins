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
    model.

    Core references
    ---------------
    1. Per-capita excreta generation rates and physicochemical characterization defaults (wet/dry mass, water fraction,
       pH, nitrogen content).
       Rose, C., Parker, A., Jefferson, B., & Cartmell, E. (2015). The characterization of feces and urine: A review of
       the literature to inform advanced treatment technology. Critical Reviews in Environmental Science and Technology,
       45(17), 1827-1879. https://doi.org/10.1080/10643389.2014.1000761
    2. Sanitation-chain and system-boundary definitions; dry-toilet processing stoichiometry.
       Tilley, E., Ulrich, L., Lüthi, C., Reymond, P., Schertenleib, R., & Zurbrügg, C. (2014). Compendium of sanitation
       systems and technologies (2nd rev. ed.). Swiss Federal Institute of Aquatic Science and Technology (Eawag).
       https://www.eawag.ch/fileadmin/Domain1/Abteilungen/sandec/schwerpunkte/sesp/CLUES/Compendium_2nd_pdfs/Compendium_2nd_Ed_Lowres_1p.pdf
    3. Bulking-agent dose design range (0.2-0.5 kg/kg feces) and urine diversion efficiency design expectation (85-95%).
       Berger, W. (2011). Technology review of composting toilets: Basic overview of composting toilets (with or without
       urine diversion). Deutsche Gesellschaft für Internationale Zusammenarbeit (GIZ) GmbH.
       https://www.susana.org/_resources/documents/default/2-878-2-1383-gtz2011-en-technology-review-composting-toilets1.pdf
    4. Field-conditions urine diversion efficiency range (0.80-0.95) and practical use/maintenance guidance for UDDTs.
       Käymäläseura Huussi ry / Global Dry Toilet Association of Finland. (2013). Use and maintenance of urine diversion
       dry toilets (UDDTs) and composting toilets. A series of educational manuals on ecological sanitation and hygiene.
       https://www.pseau.org/outils/ouvrages/huussi_use_and_maintenance_of_urine_diversion_dry_toilets_uddts_and_composting_toilets_2013.pdf
    5. Default CH4 and N2O emission factors for on-site/dry sanitation (including pit latrines).
       IPCC. (2019). 2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories, Volume 5: Waste,
       Chapter 6: Wastewater Treatment and Discharge. Intergovernmental Panel on Climate Change.
       https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/5_Volume5/19R_V5_6_Ch06_Wastewater.pdf

    Main equations
    --------------
    All normalized to dry feces output = 1 [kg/hr].

    Stoichiometry:
        Additive input group: feces + urine both feed "in_main"
        f_feces(t)    / wet_feces_dry_feces_fraction = C(t)  [kg dry feces / hr]
        f_urine(t)    / wet_feces_dry_feces_fraction  = C(t)  [kg dry feces / hr]
        f_dry(t)      / 1.0                           = C(t)  [kg dry feces / hr]
        f_leachate(t) / leachate_cf                   = C(t)  [kg dry feces / hr]
        defaults: wet_feces_dry_feces_fraction = 3.3,
                  leachate_dry_feces_relation  = 0.1  (mixed mode baseline)

    Effective leachate conversion factor by mode:
        mode = "mixed"          : leachate_cf = leachate_dry_feces_relation
        mode = "urine_diverting": leachate_cf = leachate_dry_feces_relation
                                              * (1 + urine_diversion_efficiency)

    Bulking agent commodity coupling (when bulking_agent_bus is connected):
        Q_bulk(t) = bulking_agent_dose * C(t)
        [kg/hr]     [kg/kg dry feces]   [kg dry feces/hr]

    Urine/feces coupling:
        f_urine(t) ≤ urine_per_feces_ratio × f_feces(t)
        where urine_per_feces_ratio
            = urine_volume_per_cap_per_day / feces_wet_mass_per_cap_per_day
            = 0.00142 / 0.128
            = 0.01109  [m³_urine / kg_feces]
        Enforced via flow_share_max on human_urine_bus.

    Optional proxy GHG emissions:
        E_NH3(t) = nh3_loss_fraction  * f_feces(t)   [kg NH3 / hr]
        E_CH4(t) = ch4_yield_factor   * f_feces(t)   [kg CH4 / hr]
        E_N2O(t) = n2o_yield_factor   * f_feces(t)   [kg N2O / hr]
        Source bus is human_feces_bus (wet feces input mass).
        Only active when the corresponding bus AND a non-zero factor are set.

    Notes
    -----
    - Primary flow is human_feces_bus [kg/hr]. Capacity constrains the maximum feces throughput of the toilet, representing
      sanitation service capacity (persons served × feces generation rate per capita).
    - Bulking agent input can be modelled either as an output-side variable cost embedded in the dry feces flow (default,
      no bulking_agent_bus) or as a tracked third input commodity (requires bulking_agent_bus). When the bus is connected,
      variable_costs on dry_feces_out_bus is set to 0 to avoid double-charging.
    - marginal_cost is charged on the primary input flow (kg feces / hr).
    - Characterization values (pH, dry solids, per-capita generation rates) are stored as metadata for scenario documentation
      and calibration. They are not enforced as hard optimization constraints.
    - UNIT NOTE: feces flows are in kg; urine and leachate flows are in m³. Dry feces output is in kg. Conversion factors
      carry the unit bridge.
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
    human_feces_bus: Bus = None             # kg (PRIMARY)
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
    wet_feces_dry_feces_fraction: float = 3.3    # kg wet feces/kg dry feces  [Tilley 2014]
    leachate_dry_feces_relation: float = 0.1     # m³ leachate/kg dry feces — mixed mode baseline [Tilley 2014]
    feces_density: float = 1060.0                # kg/m³ wet feces              [Rose 2015]
    urine_diversion_efficiency: float = 0.85     # fraction of urine retained → leachate [0,1]  [Berger 2011][Huussi 2013]
    nh3_loss_fraction: float = 0.0               # kg NH3/kg wet feces
    ch4_yield_factor: float = 0.0                # kg CH4/kg wet feces        [IPCC 2019]
    n2o_yield_factor: float = 0.0                # kg N2O/kg wet feces        [IPCC 2019]

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
    # documentation / calibration defaults (not hard constraints)
    # Based on the core literature references
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
        self.output_parameters = attributes.pop("output_parameters", {})

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
        # groups
        # Additive input group: feces + urine both feed "in_main".
        # --------------------------------------------------------------
        groups = {
            "in_main": [self.human_feces_bus.label, self.human_urine_bus.label],
            "out_main": [self.dry_feces_out_bus.label],
            "out_liquid": [self.water_out_bus.label],
        }
        attributes["groups"] = groups

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to dry feces output = 1 [kg/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.human_feces_bus.label}"] = sequence(
            self.wet_feces_dry_feces_fraction
        )
        attributes[f"conversion_factor_{self.human_urine_bus.label}"] = sequence(
            self.wet_feces_dry_feces_fraction  # same basis as feces within in_main
        )
        attributes["conversion_factor_in_main"] = sequence(1.0)
        attributes["conversion_factor_out_main"] = sequence(1.0)
        attributes[f"conversion_factor_{self.dry_feces_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        attributes["conversion_factor_out_liquid"] = sequence(self._effective_leachate_cf)

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
        # emission factors are per kg wet feces in, [IPCC 2019]
        # --------------------------------------------------------------
        if self.nh3_loss_bus is not None:
            attributes[
                f"emission_factor_in_main_{self.nh3_loss_bus.label}"
            ] = sequence(self.nh3_loss_fraction)
        if self.ch4_bus is not None:
            attributes[
                f"emission_factor_in_main_{self.ch4_bus.label}"
            ] = sequence(self.ch4_yield_factor)
        if self.n2o_bus is not None:
            attributes[
                f"emission_factor_in_main_{self.n2o_bus.label}"
            ] = sequence(self.n2o_yield_factor)

        # --------------------------------------------------------------
        # primary bus label resolution
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
            0.0 if self.bulking_agent_bus is not None else self._bulking_cost_per_kg_dry
        )

        if self.dry_feces_out_bus in self.outputs:
            out_flow = self.outputs[self.dry_feces_out_bus]
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
                "dry_feces_out_bus": self.dry_feces_out_bus,
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