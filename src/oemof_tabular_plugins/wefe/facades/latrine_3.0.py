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
    1. Rose et al. (2015): excreta generation, characterization defaults, and
       urine-feces per-capita ratios.
    2. Eawag Compendium of Sanitation Systems and Technologies (2nd ed.):
       sanitation-chain definitions and system boundary guidance.
    3. WEDC pit-latrine sizing guidance: storage volume and accumulation realism.
    4. Pathogen reduction meta-analysis: supports future hygiene extensions.
    5. UN/WHO Guide to sanitation resource recovery products & technologies:
       supports future recovery outputs.

    Main equations
    --------------
    Feces volume conversion:
        V_feces(t) = m_feces(t) / rho_feces
        [m³/hr]     [kg/hr]       [kg/m³]

    Urine-feces coupling (Rose et al., 2015):
        f_urine(t) ≤ (V_urine_cap / m_feces_cap) * f_feces(t)
        where:
            V_urine_cap = urine_volume_per_cap_per_day   = 0.00142  [m³/cap/day]
            m_feces_cap = feces_wet_mass_per_cap_per_day = 0.128    [kg/cap/day]
            ratio       = 0.011094                        [m³/kg]
        Enforced via flow_share_max on human_urine_bus:
        This limits the urine inflow this latrine can absorb to at most the
        feces-linked per-capita ratio; any excess urine can be handled by
        other toilet facades in the model.

    Stored excreta bookkeeping:
        V_stored(t) =
            V_feces(t)
          + (1 - alpha_ud) * V_urine(t)
          + r_flush  * V_flush(t)
          + r_clean  * V_clean(t)
          + r_cover  * V_cover(t)
          - V_leachate(t)

    Diverted urine (urine-diverting mode only):
        V_diverted_urine(t) = alpha_ud * V_urine(t)

    Optional proxy emissions:
        E_k(t) = beta_k * GROUP_FLOW_in_main(t),  k in {NH3, CH4, N2O}

    Notes
    -----
    - Primary flow is human_feces_bus [kg/hr]. Capacity constrains the maximum
      feces throughput of the latrine, representing sanitation service capacity
      (persons served × feces generation rate per capita).
    - Urine inflow is constrained to be at most proportional to feces inflow
      (Rose et al., 2015). This allows urine flow to be lower than the ratio,
      including zero, but prevents this latrine from absorbing more urine than
      its feces-linked service level. Excess or zero‑urine scenarios are handled
      by the wider system of toilet facades, not by changing this constraint.
    - Grouped MIMO flows (in_main) make the input side additive, not pairwise
      equalized.
    - Characterization values (pH, dry solids, per-capita generation rates) are
      stored as metadata for scenario documentation. They are not enforced as
      hard optimization constraints in v3.0.
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
    human_feces_bus: Bus = None          # expected unit: kg
    human_urine_bus: Bus = None          # expected unit: m³
    biomass_waste_bus: Bus = None       # expected unit: m³

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    flushwater_bus: Optional[Bus] = None           # m³
    cleaning_water_bus: Optional[Bus] = None       # m³
    cover_material_bus: Optional[Bus] = None       # assumed already in retained-volume basis

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    leachate_bus: Optional[Bus] = None             # m³
    diverted_urine_bus: Optional[Bus] = None       # m³
    nh3_loss_bus: Optional[Bus] = None             # proxy unit
    ch4_bus: Optional[Bus] = None                  # proxy unit
    n2o_bus: Optional[Bus] = None                  # proxy unit

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    feces_density: float = 1060.0
    flushwater_retention: float = 1.0
    cleaning_water_retention: float = 1.0
    cover_material_retention: float = 1.0
    urine_diversion_efficiency: float = 0.0
    liquid_loss_fraction: float = 0.0
    nh3_loss_fraction: float = 0.0
    ch4_yield_factor: float = 0.0
    n2o_yield_factor: float = 0.0

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0

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
    retention_time_days: float = None
    temperature_c: float = None

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
        # per-capita urine/feces coupling ratio (Rose et al. 2015)
        # feces is primary — urine is coupled to feces flow
        # ratio: urine [m³/hr] per unit feces [kg/hr]
        # = urine_volume_per_cap_per_day / feces_wet_mass_per_cap_per_day
        # = 0.00142 / 0.128 = 0.011094 m³_urine / kg_feces
        # --------------------------------------------------------------
        self._urine_per_feces_ratio = (
                self.urine_volume_per_cap_per_day
                / self.feces_wet_mass_per_cap_per_day
        )

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
        # bus-level factors convert flow into common activity basis
        # --------------------------------------------------------------
        # Bus-level conversion factors: normalize each input to m³ basis
        # MIMO divides flow by conversion_factor internally (GROUP_FLOW = sum(f_i / eta_i))
        # so pass density directly — MIMO computes: f_feces [kg/hr] / 1060 [kg/m³] = m³/hr
        attributes[f"conversion_factor_{self.human_feces_bus.label}"] = sequence(
            self.feces_density  # kg/m³ — NOTE: MIMO semantic: eta is the DENOMINATOR, not multiplier
        )
        attributes[f"conversion_factor_{self.human_urine_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.biomass_waste_bus.label}"] = sequence(1.0)

        # Group-level normalization: no extra scaling on the group
        attributes["conversion_factor_in_main"] = sequence(1.0)
        attributes["conversion_factor_out_main"] = sequence(1.0)

        if self.flushwater_bus is not None:
            attributes[f"conversion_factor_{self.flushwater_bus.label}"] = sequence(
                max(self.flushwater_retention, 1e-9)
            )
        if self.cleaning_water_bus is not None:
            attributes[f"conversion_factor_{self.cleaning_water_bus.label}"] = sequence(
                max(self.cleaning_water_retention, 1e-9)
            )
        if self.cover_material_bus is not None:
            attributes[f"conversion_factor_{self.cover_material_bus.label}"] = sequence(
                max(self.cover_material_retention, 1e-9)
            )
        if self.leachate_bus is not None:
            attributes[f"conversion_factor_{self.leachate_bus.label}"] = sequence(1.0)
            attributes["conversion_factor_out_liquid"] = sequence(
                max(self.liquid_loss_fraction, 1e-9)
            )
        if self.diverted_urine_bus is not None:
            attributes[f"conversion_factor_{self.diverted_urine_bus.label}"] = sequence(1.0)
            attributes["conversion_factor_out_urine"] = sequence(
                max(self.urine_diversion_efficiency, 1e-9)
            )

        # --------------------------------------------------------------
        # flow-share constraints
        # --------------------------------------------------------------
        # [MANDATORY] Urine inflow is limited by the feces-derived per-capita ratio.
        # Enforces: f_urine(t) <= urine_per_feces_ratio * f_feces(t)
        # This allows urine to be lower than the ratio at a given timestep, while
        # preventing this latrine from absorbing more urine than its feces-linked
        # service level. Any remaining urine can be handled by other toilet facades.
        attributes[
            f"flow_share_max_{self.human_urine_bus.label}"
        ] = sequence(self._urine_per_feces_ratio)

        # [OPTIONAL] Urine diversion split — only in urine_diverting mode
        if self.diverted_urine_bus is not None and self.mode == "urine_diverting":
            attributes[
                f"flow_share_fix_{self.diverted_urine_bus.label}"
            ] = sequence(self.urine_diversion_efficiency)

        # [OPTIONAL] Leachate loss — upper bound on liquid loss fraction
        if self.leachate_bus is not None:
            attributes[
                f"flow_share_max_{self.leachate_bus.label}"
            ] = sequence(self.liquid_loss_fraction)

        # --------------------------------------------------------------
        # optional proxy emissions via existing emission-factor logic
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
        # primary bus should point to actual bus label
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