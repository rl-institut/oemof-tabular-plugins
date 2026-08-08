import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class Latrine(MIMO):
    """
    Literature-informed generic latrine facade based on MIMO.

    Purpose
    -------
    Generic sanitation-interface / storage facade for mixed or urine-diverting
    latrine concepts. The model is designed as a bookkeeping/process-yield unit
    representing a latrine as a feces-management intervention for the prevention
    of open defecation. It is not a full mechanistic biological reactor model.

    Core references
    ---------------
    1. Per-capita excreta generation rates and physicochemical characterization defaults.
       Rose, C., Parker, A., Jefferson, B., & Cartmell, E. (2015). The characterization of feces and urine: A review of
       the literature to inform advanced treatment technology. Critical Reviews in Environmental Science and Technology,
       45(17), 1827-1879. https://doi.org/10.1080/10643389.2014.1000761
    2. Sanitation-chain and system-boundary definitions — basis for the general latrine input/output model structure.
       Tilley, E., Ulrich, L., Lüthi, C., Reymond, P., & Zurbrügg, C. (2014). Compendium of sanitation systems and technologies
       (2nd rev. ed.). Swiss Federal Institute of Aquatic Science and Technology (Eawag).
       https://sswm.info/sites/default/files/reference_attachments/TILLEY%20et%20al%202014%20Compendium%20of%20Sanitation%20Systems%20and%20Technologies%202nd%20Revised%20Edition.pdf
    3. Pit-latrine storage volume and accumulation-rate design guidance.
       Reed, B., & Shaw, R. (2014). Technical brief: Simple pit latrines (WEDC Guide No. 25). Water, Engineering and
       Development Centre (WEDC), Loughborough University.
       https://wedc-knowledge.lboro.ac.uk/resources/booklets/G025-Simple-pit-latrines-booklet.pdf
    4. Bulking-agent dose and urine-diversion efficiency design range (85-95%).
       Berger, W. (2011). Technology review of composting toilets: Basic overview of composting toilets (with or without
       urine diversion). Deutsche Gesellschaft für Internationale Zusammenarbeit (GIZ) GmbH.
       https://www.susana.org/_resources/documents/default/2-878-2-1383-gtz2011-en-technology-review-composting-toilets1.pdf
    5. Pathogen (bacteria, viruses, protozoa, Ascaris) decay rates and T99 values as a function of pH, temperature, and
       moisture content in pit latrines and other onsite sanitation systems.
       Musaazi, I. G., McLoughlin, S., Murphy, H. M., Rose, J. B., Hofstra, N., Tumwebaze, I. K., & Verbyla, M. E. (2023).
       A systematic review and meta-analysis of pathogen reduction in onsite sanitation systems. Water Research X, 18,
       100171. https://doi.org/10.1016/j.wroa.2023.100171
    6. Default CH4 and N2O emission factors for on-site/dry sanitation excreta management.
       IPCC. (2019). 2019 Refinement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories, Volume 5: Waste,
       Chapter 6: Wastewater Treatment and Discharge. Intergovernmental Panel on Climate Change.
       https://www.ipcc-nggip.iges.or.jp/public/2019rf/pdf/5_Volume5/19R_V5_6_Ch06_Wastewater.pdf
    7. Risk-management framework for excreta and greywater reuse in agriculture.
       World Health Organization. (2006). WHO guidelines for the safe use of wastewater, excreta and greywater, Volume 4:
       Excreta and greywater use in agriculture. WHO. ISBN 92-4-154685-9.
       https://www.who.int/publications/i/item/9241546859

    Main equations
    --------------
    # All normalized to biomass output = 1 [m³/hr].

    Feces volume conversion:
        V_feces(t) = m_feces(t) / rho_feces
        [m³/hr]     [kg/hr]       [kg/m³]

    Additive input basis (in_main), all volumes in m³/hr:
        GROUP_FLOW_in_main(t) =
            V_feces(t)
          + V_urine(t)
          + V_flush(t)  × r_flush     (if flushwater_bus provided)
          + V_clean(t)  × r_clean     (if cleaning_water_bus provided)
          + V_cover(t)  × r_cover     (if cover_material_bus provided)
          where:
          r_flush, r_clean, r_cover are dimensionless retention fractions in
          [0, 1]

    Urine-feces coupling:
        f_urine(t) ≤ (V_urine_cap / m_feces_cap) * f_feces(t)
        where:
            V_urine_cap = urine_volume_per_cap_per_day   = 0.00142  [m³/cap/day]
            m_feces_cap = feces_wet_mass_per_cap_per_day = 0.128    [kg/cap/day]
            ratio       = 0.011094                        [m³/kg]
        Enforced via flow_share_max on human_urine_bus.

    Biomass output:
        V_biomass(t) = GROUP_FLOW_in_main(t)

    Leachate (if leachate_bus provided):
        V_leachate(t) = liquid_loss_fraction * GROUP_FLOW_in_main(t)

    Diverted urine (urine-diverting mode only):
        V_diverted_urine(t) = GROUP_FLOW_in_main(t) * urine_diversion_efficiency * _urine_fraction_in_common_input

    Optional proxy emissions:
        E_k(t) = beta_k * GROUP_FLOW_in_main(t),  k in {NH3, CH4, N2O}

    Notes
    -----
    - Primary flow is human_feces_bus [kg/hr]. Capacity constrains the maximum feces throughput of the latrine, representing
      sanitation service capacity (persons served × feces generation rate per capita).
    - Urine inflow is constrained to be at most proportional to feces inflow. This allows urine flow to be lower than the ratio,
      including zero, but prevents this latrine from absorbing more urine than its feces-linked service level. Excess or
      zero‑urine scenarios are handled by the wider system of toilet facades, not by changing this constraint.
    - Grouped MIMO flows (in_main) make the input side additive, not pairwise equalized.
    - Characterization values (pH, dry solids, per-capita generation rates) are stored as metadata for scenario documentation.
      They are not enforced as hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "latrine"
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
    human_feces_bus: Bus = None          # kg (PRIMARY)
    human_urine_bus: Bus = None          # m³
    biomass_waste_bus: Bus = None        # m³

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    flushwater_bus: Optional[Bus] = None           # m³
    cleaning_water_bus: Optional[Bus] = None       # m³
    cover_material_bus: Optional[Bus] = None       # m³

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    leachate_bus: Optional[Bus] = None             # m³
    diverted_urine_bus: Optional[Bus] = None       # m³
    nh3_loss_bus: Optional[Bus] = None             # kg
    ch4_bus: Optional[Bus] = None                  # kg
    n2o_bus: Optional[Bus] = None                  # kg

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------

    feces_density: float = 1060.0                   # kg/m³ [Rose 2015]
    flushwater_retention: float = 1.0               # [Tilley 2014]
    cleaning_water_retention: float = 1.0           # [Tilley 2014]
    cover_material_retention: float = 1.0           # [Tilley 2014]
    urine_diversion_efficiency: float = 0.0         # fraction [0,1] [Berger 2011]
    liquid_loss_fraction: float = 0.0               # [Reed & Shaw 2014]
    nh3_loss_fraction: float = 0.0                  # kg NH3/kg wet feces
    ch4_yield_factor: float = 0.0                   # kg CH4/kg wet feces        [IPCC 2019]
    n2o_yield_factor: float = 0.0                   # kg CH4/kg wet feces        [IPCC 2019]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0      # USD/kg human feces

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
    feces_pH: float = 6.64                              # [Rose 2015]
    urine_pH: float = 6.2                               # [Rose 2015]
    urine_nitrogen_g_per_cap_per_day: float = 10.98     # g N/cap/day      [Rose 2015]
    retention_time_days: float = None                   # [Musaazi et al. 2023]
    temperature_c: float = None                         # [Musaazi et al. 2023]

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
        self.biomass_waste_bus = attributes.pop("biomass_waste_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.flushwater_bus = attributes.pop("flushwater_bus", None)
        self.cleaning_water_bus = attributes.pop("cleaning_water_bus", None)
        self.cover_material_bus = attributes.pop("cover_material_bus", None)

        self.leachate_bus = attributes.pop("leachate_bus", None)
        self.diverted_urine_bus = attributes.pop("diverted_urine_bus", None)
        self.nh3_loss_bus = attributes.pop("nh3_loss_bus", None)
        self.ch4_bus = attributes.pop("ch4_bus", None)
        self.n2o_bus = attributes.pop("n2o_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.feces_density = attributes.pop("feces_density", self.feces_density)
        self.flushwater_retention = attributes.pop(
            "flushwater_retention", self.flushwater_retention
        )
        self.cleaning_water_retention = attributes.pop(
            "cleaning_water_retention", self.cleaning_water_retention
        )
        self.cover_material_retention = attributes.pop(
            "cover_material_retention", self.cover_material_retention
        )
        self.urine_diversion_efficiency = attributes.pop(
            "urine_diversion_efficiency", self.urine_diversion_efficiency
        )
        self.liquid_loss_fraction = attributes.pop(
            "liquid_loss_fraction", self.liquid_loss_fraction
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
        self.feces_pH = attributes.pop("feces_pH", self.feces_pH)
        self.urine_pH = attributes.pop("urine_pH", self.urine_pH)
        self.urine_nitrogen_g_per_cap_per_day = attributes.pop(
            "urine_nitrogen_g_per_cap_per_day",
            self.urine_nitrogen_g_per_cap_per_day,
        )
        self.retention_time_days = attributes.pop(
            "retention_time_days", self.retention_time_days
        )
        self.temperature_c = attributes.pop("temperature_c", self.temperature_c)

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._urine_per_feces_ratio = (
                self.urine_volume_per_cap_per_day
                / self.feces_wet_mass_per_cap_per_day
        )

        self._urine_fraction_in_common_input = 0.9216

        # --------------------------------------------------------------
        # groups
        # main additive input group + main output group
        # --------------------------------------------------------------
        in_main = [
            self.human_feces_bus.label,
            self.human_urine_bus.label,
        ]
        if self.flushwater_bus is not None:
            in_main.append(self.flushwater_bus.label)
        if self.cleaning_water_bus is not None:
            in_main.append(self.cleaning_water_bus.label)
        if self.cover_material_bus is not None:
            in_main.append(self.cover_material_bus.label)

        groups = {
            "in_main": in_main,
            "out_main": [self.biomass_waste_bus.label],
        }
        if self.leachate_bus is not None:
            groups["out_liquid"] = [self.leachate_bus.label]
        if self.diverted_urine_bus is not None:
            groups["out_urine"] = [self.diverted_urine_bus.label]

        attributes["groups"] = groups  # ← single assignment, never overwritten

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to biomass output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.human_feces_bus.label}"] = sequence(
            self.feces_density
        )
        attributes[f"conversion_factor_{self.human_urine_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.biomass_waste_bus.label}"] = sequence(1.0)
        attributes["conversion_factor_in_main"] = sequence(1.0)
        attributes["conversion_factor_out_main"] = sequence(1.0)

        if self.flushwater_bus is not None:
            attributes[f"conversion_factor_{self.flushwater_bus.label}"] = sequence(
                1.0 / max(self.flushwater_retention, 1e-9)
            )
        if self.cleaning_water_bus is not None:
            attributes[f"conversion_factor_{self.cleaning_water_bus.label}"] = sequence(
                1.0 / max(self.cleaning_water_retention, 1e-9)
            )
        if self.cover_material_bus is not None:
            attributes[f"conversion_factor_{self.cover_material_bus.label}"] = sequence(
                1.0 / max(self.cover_material_retention, 1e-9)
            )
        if self.leachate_bus is not None:
            attributes[f"conversion_factor_{self.leachate_bus.label}"] = sequence(1.0)
            attributes["conversion_factor_out_liquid"] = sequence(
                max(self.liquid_loss_fraction, 1e-9)
            )
        if self.diverted_urine_bus is not None:
            attributes[f"conversion_factor_{self.diverted_urine_bus.label}"] = sequence(1.0)
            attributes["conversion_factor_out_urine"] = sequence(
                max(self.urine_diversion_efficiency * self._urine_fraction_in_common_input, 1e-9)
            )

        # --------------------------------------------------------------
        # flow-share constraints: urine inflow bounded by feces-linked ratio (approximately)
        # --------------------------------------------------------------
        attributes[
            f"flow_share_max_{self.human_urine_bus.label}"
        ] = sequence(self._urine_per_feces_ratio)

        # --------------------------------------------------------------
        # optional proxy emissions via existing emission-factor logic
        # key: emission_factor_<source_bus_label>_<target_bus_label>
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
        if self.primary == "biomass_waste_bus":
            primary_label = self.biomass_waste_bus.label
        elif self.primary == "human_feces_bus":
            primary_label = self.human_feces_bus.label
        elif self.primary == "human_urine_bus":
            primary_label = self.human_urine_bus.label
        else:
            primary_label = self.primary

        # --------------------------------------------------------------
        # initialize base MIMO facade
        # --------------------------------------------------------------
        super().__init__(
            from_bus_0=self.human_feces_bus,
            from_bus_1=self.human_urine_bus,
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

        if self.human_feces_bus in self.inputs:
            self.inputs[self.human_feces_bus].variable_costs = sequence(self.marginal_cost)

        if self.biomass_waste_bus in self.outputs:
            out_flow = self.outputs[self.biomass_waste_bus]
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
            }
            primary_bus = primary_bus_map.get(self.primary)
            if primary_bus is not None:
                flow = self.inputs.get(primary_bus, self.outputs.get(primary_bus))
                if flow is not None:
                    flow.nominal_value = self.capacity

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 1
        # inputs
        for bus in [
            self.flushwater_bus,
            self.cleaning_water_bus,
            self.cover_material_bus,
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.leachate_bus,
            self.diverted_urine_bus,
            self.nh3_loss_bus,
            self.ch4_bus,
            self.n2o_bus,
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if self.mode not in {"mixed", "urine_diverting"}:
            raise ValueError("mode must be 'mixed' or 'urine_diverting'.")

        if self.feces_density <= 0:
            raise ValueError("feces_density must be > 0.")

        bounded = {
            "flushwater_retention": self.flushwater_retention,
            "cleaning_water_retention": self.cleaning_water_retention,
            "cover_material_retention": self.cover_material_retention,
            "urine_diversion_efficiency": self.urine_diversion_efficiency,
            "liquid_loss_fraction": self.liquid_loss_fraction,
            "nh3_loss_fraction": self.nh3_loss_fraction,
        }
        for name, value in bounded.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")

        if self.mode == "urine_diverting" and self.diverted_urine_bus is None:
            raise ValueError(
                "mode='urine_diverting' requires diverted_urine_bus."
            )

        if self.feces_water_fraction is not None and not 0 <= self.feces_water_fraction <= 1:
            raise ValueError("feces_water_fraction must be between 0 and 1.")