import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class OpenField(MIMO):
    """
    Literature-informed catch-all open-field disposal facade based on MIMO.

    Purpose
    -------
    Generic "open field" / open-defecation sink representing the *unsafe fallback*
    pathway in a sanitation system: it absorbs whatever human and animal excreta
    the toilet/latrine facades cannot handle. The model is a bookkeeping/mass-balance
    unit representing open defecation as the non-preferred terminal product flow in
    a sanitation chain. It is not a mechanistic soil or biological model. The solver
    decides how much flows here; a positive marginal_cost expresses the disutility
    of the pathway.

    Core references
    ---------------
    1. Human feces/urine generation, density, and characterization defaults.
       Rose, C., Parker, A., Jefferson, B., & Cartmell, E. (2015). The characterization of feces and urine: A review of
       the literature to inform advanced treatment technology. Critical Reviews in Environmental Science and Technology,
       45(17), 1827-1879. https://doi.org/10.1080/10643389.2014.1000761
    2. As-excreted manure moisture content (75-90%) and specific gravity (~1.0) — basis for animal_feces_density and
       animal_feces_water_fraction.
       ASABE. (2005, reaffirmed 2019). ASAE D384.2 MAR2005 (R2019): Manure production and characteristics. American Society
       of Agricultural and Biological Engineers. https://webstore.ansi.org/preview-pages/ASABE/preview_ASAE+D384.2+MAR2005+(R2019).pdf
    3. Methodological basis for excluding CH4 from open defecation (no anaerobic containment) and default N2O emission
       factor for nitrogen in domestic wastewater/excreta pathways.
       IPCC. (2019). 2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories, Volume 5: Waste,
       Chapter 6: Wastewater Treatment and Discharge. Intergovernmental Panel on Climate Change.
       https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/5_Volume5/19R_V5_6_Ch06_Wastewater.pdf
    4. Sanitation-chain definitions; open field/open defecation as the non-preferred terminal product flow.
       Tilley, E., Ulrich, L., Lüthi, C., Reymond, P., & Zurbrügg, C. (2014). Compendium of sanitation systems and technologies
       (2nd rev. ed.). Swiss Federal Institute of Aquatic Science and Technology (Eawag).
       https://sswm.info/sites/default/files/reference_attachments/TILLEY%20et%20al%202014%20Compendium%20of%20Sanitation%20Systems%20and%20Technologies%202nd%20Revised%20Edition.pdf
    5. Rationale for treating open defecation as the unsafe sanitation fallback.
       https://iris.who.int/handle/10665/39313

    Main equations
    --------------
    # All normalized to biomass output = 1 [m³/hr].

    Feces volume conversion:
        V_feces(t) = m_feces(t) / rho_feces
        [m³/hr]     [kg/hr]       [kg/m³]
        Animal feces use rho_animal_feces.

    Catch-all additive balance (single input group "in_main"):
        GROUP_FLOW_in_main(t) = sum_i V_i(t) = V_biomass(t)
        where V_i(t) = flow_i(t) / cf_i,  i in {human_feces, human_urine,
                                                  animal_feces, animal_urine}
        -> inputs are SUMMED, never pairwise-equalized, and their ratio is free.

    Optional proxy emissions (group-level):
        E_k(t) = beta_k * GROUP_FLOW_in_main(t),   k in {NH3, CH4, N2O}

    Optional policy cap on total disposal volume:
        GROUP_FLOW_in_main(t) <= max_open_field_load(t)   [m³/hr]
        Only active when max_open_field_load is set; None = unconstrained.

    Notes
    -----
    - Primary flow is biomass_waste_bus [m³/hr]. Because open field is a penalty-driven sink with no real operating cost,
      marginal_cost is attached to total output volume rather than a single input stream. This is the correct cost basis
      for a volumetric disposal penalty [WHO 1992].
    - NO urine<->feces coupling. The open field is a sink for "all remaining" excreta, so it deliberately omits any flow_share
      linking urine to feces. Any input ratio (including all-of-one) is permitted.
    - Group-level emissions apply one factor to the summed input volume. Because the open field mixes human+animal feces+urine
      with very different nitrogen contents, this is a coarser proxy than for a single-stream unit. For N-precise N2O, switch
      to per-stream emission_factor_<bus>_<emission_bus> keyed on each input's nitrogen content (deferred refinement).
    - Characterization values (pH, dry solids, per-capita / per-animal generation rates) are stored as metadata for scenario
      documentation. They size the upstream sources and are not enforced as hard optimization constraints.
    - Expected bus units: feces buses in kg, urine + biomass buses in m³.
    - max_open_field_load only constrains human excreta rerouting in practice. Animal waste buses connect exclusively to
      OpenField; if animal waste volume alone exceeds max_open_field_load at any timestep, the model is infeasible.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "open_field"
    name: str = ""
    tech: str = "toilet"
    carrier: str = "water"
    primary: str = "biomass_waste_bus"

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
    human_feces_bus: Bus = None            # kg
    human_urine_bus: Bus = None            # m³
    animal_feces_bus: Bus = None           # kg
    animal_urine_bus: Bus = None           # m³
    biomass_waste_bus: Bus = None          # m³ (PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    nh3_loss_bus: Optional[Bus] = None          # kg
    ch4_bus: Optional[Bus] = None               # kg
    n2o_bus: Optional[Bus] = None               # kg

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    human_feces_density: float = 1060.0         # kg/m³  wet feces [Rose et al. 2015]
    animal_feces_density: float = 1000.0        # kg/m³  as-excreted manure [ASABE D384.2]
    nh3_loss_fraction: float = 0.0              # [IPCC 2019]
    ch4_yield_factor: float = 0.0               # 0 for true open field [IPCC 2019]
    n2o_yield_factor: float = 0.0               # [IPCC 2019]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                  # Penalty cost per m³ of unsafe biomass disposal.
                                                # Set >0 to discourage open field use [WHO 1992].
    max_open_field_load: Union[float, Sequence[float]] = None  # m³/hr policy cap; None = unconstrained

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
    feces_wet_mass_per_cap_per_day: float = 0.128       # kg   [Rose 2015]
    feces_dry_mass_per_cap_per_day: float = 0.029       # kg   [Rose 2015]
    feces_water_fraction: float = 0.746                 # -    [Rose 2015]
    urine_volume_per_cap_per_day: float = 0.00142       # m³   [Rose 2015]
    animal_feces_water_fraction: float = 0.85           # -    [ASABE D384.2]
    feces_pH: float = 6.64                              # [Rose 2015]
    urine_pH: float = 6.2                               # [Rose 2015]
    urine_nitrogen_g_per_cap_per_day: float = 10.98     # g N  [Rose 2015]
    ef_n2o_n: float = 0.005                             # kg N2O-N / kg N [IPCC 2019]


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
        self.human_feces_bus = attributes.pop("human_feces_bus")
        self.human_urine_bus = attributes.pop("human_urine_bus")
        self.animal_feces_bus = attributes.pop("animal_feces_bus")
        self.animal_urine_bus = attributes.pop("animal_urine_bus")
        self.biomass_waste_bus = attributes.pop("biomass_waste_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.nh3_loss_bus = attributes.pop("nh3_loss_bus", None)
        self.ch4_bus = attributes.pop("ch4_bus", None)
        self.n2o_bus = attributes.pop("n2o_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.human_feces_density = attributes.pop("human_feces_density", self.human_feces_density)
        self.animal_feces_density = attributes.pop(
            "animal_feces_density", self.animal_feces_density
        )
        self.nh3_loss_fraction = attributes.pop(
            "nh3_loss_fraction", self.nh3_loss_fraction
        )
        self.ch4_yield_factor = attributes.pop(
            "ch4_yield_factor", self.ch4_yield_factor
        )
        self.n2o_yield_factor = attributes.pop(
            "n2o_yield_factor", self.n2o_yield_factor
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
        self.max_open_field_load = attributes.pop(
            "max_open_field_load", self.max_open_field_load
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
        self.animal_feces_water_fraction = attributes.pop(
            "animal_feces_water_fraction", self.animal_feces_water_fraction
        )
        self.feces_pH = attributes.pop("feces_pH", self.feces_pH)
        self.urine_pH = attributes.pop("urine_pH", self.urine_pH)
        self.urine_nitrogen_g_per_cap_per_day = attributes.pop(
            "urine_nitrogen_g_per_cap_per_day",
            self.urine_nitrogen_g_per_cap_per_day,
        )
        self.ef_n2o_n = attributes.pop("ef_n2o_n", self.ef_n2o_n)

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # groups
        # main additive input group + main output group
        # ALL four inputs share one group so the input side is additive,
        # not pairwise-equalized. The ratio between inputs is left free.
        # --------------------------------------------------------------
        in_main = [
            self.human_feces_bus.label,
            self.human_urine_bus.label,
            self.animal_feces_bus.label,
            self.animal_urine_bus.label,
        ]
        groups = {
            "in_main": in_main,
            "out_main": [self.biomass_waste_bus.label],
        }
        attributes["groups"] = groups

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to biomass output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.human_feces_bus.label}"] = sequence(
            self.human_feces_density
        )
        attributes[f"conversion_factor_{self.animal_feces_bus.label}"] = sequence(
            self.animal_feces_density
        )
        attributes[f"conversion_factor_{self.human_urine_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.animal_urine_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.biomass_waste_bus.label}"] = sequence(1.0)

        # Group-level normalization: no extra scaling on the group
        attributes["conversion_factor_in_main"] = sequence(1.0)
        attributes["conversion_factor_out_main"] = sequence(1.0)

        # --------------------------------------------------------------
        # optional proxy emissions via existing emission-factor logic
        # key: emission_factor_<source_bus_label>_<target_bus_label>
        # --------------------------------------------------------------
        if self.nh3_loss_bus is not None:
            attributes[
                f"emission_factor_in_main_{self.nh3_loss_bus.label}"
            ] = sequence(self.nh3_loss_fraction)

        if self.ch4_bus is not None:
            # IPCC 2019: true open field is not a CH4 source; default factor 0.
            attributes[
                f"emission_factor_in_main_{self.ch4_bus.label}"
            ] = sequence(self.ch4_yield_factor)

        if self.n2o_bus is not None:
            attributes[
                f"emission_factor_in_main_{self.n2o_bus.label}"
            ] = sequence(self.n2o_yield_factor)

        # --------------------------------------------------------------
        # activity bound - policy cap on open-field throughput [WHO 1992]
        # max_open_field_load limits total disposal volume [m³/hr]
        # without distorting conversion factors or the group balance
        # --------------------------------------------------------------
        if self.max_open_field_load is not None:
            attributes["activity_bound_max"] = sequence(self.max_open_field_load)

        # --------------------------------------------------------------
        # primary bus label resolution
        # --------------------------------------------------------------
        if self.primary == "biomass_waste_bus":
            primary_label = self.biomass_waste_bus.label
        elif self.primary == "human_feces_bus":
            primary_label = self.human_feces_bus.label
        elif self.primary == "human_urine_bus":
            primary_label = self.human_urine_bus.label
        elif self.primary == "animal_feces_bus":
            primary_label = self.animal_feces_bus.label
        elif self.primary == "animal_urine_bus":
            primary_label = self.animal_urine_bus.label
        else:
            primary_label = self.primary

        # --------------------------------------------------------------
        # initialize base MIMO facade
        # --------------------------------------------------------------
        super().__init__(
            from_bus_0=self.human_feces_bus,
            from_bus_1=self.human_urine_bus,
            from_bus_2=self.animal_feces_bus,
            from_bus_3=self.animal_urine_bus,
            to_bus_0=self.biomass_waste_bus,
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

        if self.biomass_waste_bus in self.outputs:
            out_flow = self.outputs[self.biomass_waste_bus]
            out_flow.variable_costs = sequence(self.marginal_cost)
            custom_attrs = (getattr(self, "output_parameters", None) or {}).get(
                "custom_attributes"
            )
            if custom_attrs:
                for attribute, value in custom_attrs.items():
                    setattr(out_flow, attribute, value)

        if not self.expandable and self.capacity is not None:
            primary_bus_map = {
                "biomass_waste_bus": self.biomass_waste_bus,
                "human_feces_bus": self.human_feces_bus,
                "human_urine_bus": self.human_urine_bus,
                "animal_feces_bus": self.animal_feces_bus,
                "animal_urine_bus": self.animal_urine_bus,
            }
            primary_bus = primary_bus_map.get(self.primary)
            if primary_bus is not None:
                flow = self.inputs.get(primary_bus, self.outputs.get(primary_bus))
                if flow is not None:
                    flow.nominal_value = self.capacity

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 4
        idx_out = 1
        # inputs
        for bus in []:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.nh3_loss_bus,
            self.ch4_bus,
            self.n2o_bus,
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if self.human_feces_density <= 0:
            raise ValueError("human_feces_density must be > 0.")
        if self.animal_feces_density <= 0:
            raise ValueError("animal_feces_density must be > 0.")

        bounded = {
            "nh3_loss_fraction": self.nh3_loss_fraction,
            "feces_water_fraction": self.feces_water_fraction,
            "animal_feces_water_fraction": self.animal_feces_water_fraction,
        }
        for name, value in bounded.items():
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")

        non_negative = {
            "ch4_yield_factor": self.ch4_yield_factor,
            "n2o_yield_factor": self.n2o_yield_factor,
            "marginal_cost": self.marginal_cost,
            "ef_n2o_n": self.ef_n2o_n,
        }
        for name, value in non_negative.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0.")

        if self.max_open_field_load is not None:
            if isinstance(self.max_open_field_load, (int, float)):
                if self.max_open_field_load <= 0:
                    raise ValueError("max_open_field_load must be > 0.")
