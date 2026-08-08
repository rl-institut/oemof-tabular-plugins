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
    1. Per-capita excreta generation rates and physicochemical characterization defaults.
       Rose, C., Parker, A., Jefferson, B., & Cartmell, E. (2015). The characterization of feces and urine: A review of
       the literature to inform advanced treatment technology. Critical Reviews in Environmental Science and Technology,
       45(17), 1827-1879. https://doi.org/10.1080/10643389.2014.1000761
    2. Sanitation-chain definitions, system-boundary guidance, and blackwater characterization — basis for the general
       flush-toilet/blackwater model structure.
       Tilley, E., Ulrich, L., Lüthi, C., Reymond, P., & Zurbrügg, C. (2014). Compendium of sanitation systems and technologies
       (2nd rev. ed.). Swiss Federal Institute of Aquatic Science and Technology (Eawag).
       https://sswm.info/sites/default/files/reference_attachments/TILLEY%20et%20al%202014%20Compendium%20of%20Sanitation%20Systems%20and%20Technologies%202nd%20Revised%20Edition.pdf
    3. Blackwater volume fractions and nutrient load basis — basis for feces_fraction, urine_fraction, service_water_fraction
       Wang, X., Chen, J., Li, Z., Cheng, S., Mang, H.-P., Zheng, L., Jan, I., & Harada, H. (2022). Nutrient recovery technologies
       for management of blackwater: A review. Frontiers in Environmental Science, 10, 1080536.
       https://doi.org/10.3389/fenvs.2022.1080536
    4. Measured flush-volume ranges for standard cistern (6-9 L), pour-flush (1-3 L), and low-volume (down to 1.5 L)
       toilets in emergency/water-scarce contexts.
       Gensch, R., Jennings, A., Renggli, S., & Reymond, P. (2018). Compendium of sanitation technologies in emergencies
       (1st ed.). German WASH Network (GWN); Eawag Sandec; Global WASH Cluster; Sustainable Sanitation Alliance (SuSanA).
       https://www.washnet.de/wp-content/uploads/emergency-sanitation-compendium.pdf
    5. Measured water savings (25-50% lower flush volume) and resource recovery potential (biogas, phosphorus) of dual-flush
       vacuum toilets.
       Todt, D., Bisschops, I., Chatzopoulos, P., & van Eekert, M. H. A. (2021). Practical performance and user experience
       of novel dual-flush vacuum toilets. Water, 13(16), 2228.
       https://doi.org/10.3390/w13162228

    Main equations
    --------------
    All normalized to black water output = 1 [m³/hr].

    Feces volume conversion:
        V_feces(t) = m_feces(t) / rho_feces
        [m³/hr]     [kg/hr]       [kg/m³]

    Blackwater volume fractions:
        f_feces + f_urine + f_service_water = 1.0
        defaults: f_feces = 0.01, f_urine = 0.11, f_service_water = 0.88

    Flush dose — service water per unit combined excreta:
        flush_dose = f_service_water / (f_feces + f_urine)
        [m³_flush / m³_excreta]
        Enforced via flow_share_fix on service_water_bus:
            f_flush(t) = f_service_water × GROUP_FLOW_in_main(t)

    Blackwater output:
        V_blackwater(t) = V_feces(t) + V_urine(t) + V_flush(t)
                        = GROUP_FLOW_in_main(t)  [m³/hr]

    Urine/feces coupling:
        f_urine(t) ≤ urine_per_feces_ratio × f_feces(t)
        where urine_per_feces_ratio
            = urine_volume_per_cap_per_day / feces_wet_mass_per_cap_per_day
            = 0.00142 / 0.128
            = 0.01109  [m³_urine / kg_feces]
        Enforced via flow_share_max on human_urine_bus.
        flow_share_max = (rho_feces × urine_per_feces_ratio × (1 - f_service_water))
                          / (1 + rho_feces × urine_per_feces_ratio)

    Flush-mode presets:
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
    - Primary flow is human_feces_bus [kg/hr]. Capacity constrains the maximum feces throughput of the toilet, representing
      sanitation service capacity (persons served × feces generation rate per capita).
    - Grouped MIMO flows (in_main) make the input side additive; feces [kg] is normalized to [m³] by feces_density before summation.
    - Urine inflow is bounded by the physiological per-capita feces/urine ratio. This constraint is inactive when system-level
      penalties already enforce the ratio, but is required in multi-toilet systems to prevent the optimizer from routing
      urine independently of feces across competing toilet facades.
    - Characterization values (pH, dry solids, per-capita generation rates) are stored as metadata for scenario documentation
      and calibration. They are not enforced as hard optimization constraints.
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
    human_feces_bus: Bus = None          # kg (PRIMARY)
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
    feces_density: float = 1060.0           # kg/m³ wet feces [Rose 2015]
    feces_fraction: float = 0.01            # m³ wet feces /m³ blackwater [Tilley 2014][Wang 2022]
    urine_fraction: float = 0.11            # m³ urine /m³ blackwater [Tilley 2014][Wang 2022]
    service_water_fraction: float = 0.88    # m³ flush water/m³ blackwater [Gensch 2018]

    # flush presets: service-water fraction per mode (drives the dose).
    flush_presets: ClassVar[dict] = {
        "standard": 0.88,               # [Gensch 2018] — modern cistern, ~6-9 L/flush
        "low_flush": 0.80,              # [Gensch 2018] — pour-flush / low-volume, 1-3 L/flush
        "vacuum": 0.55,                 # [Todt et al. 2021] — 25-50% reduction vs. standard
        "water_scarce": 0.55,           # [Todt et al. 2021] — vacuum-equivalent water saving
    }

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # USD/kg human feces

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
    population_equivalent: float = 1.0                  # [Rose 2015]
    feces_wet_mass_per_cap_per_day: float = 0.128       # [Rose 2015]
    feces_dry_mass_per_cap_per_day: float = 0.029       # [Rose 2015]
    feces_water_fraction: float = 0.746                 # [Rose 2015]
    urine_volume_per_cap_per_day: float = 0.00142       # [Rose 2015]
    feces_pH: float = 6.64                              # [Rose 2015]
    urine_pH: float = 6.2                               # [Rose 2015]
    urine_nitrogen_g_per_cap_per_day: float = 10.98     # [Rose 2015]

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

        # --------------------------------------------------------------
        # apply flush presets and validate parameters
        # --------------------------------------------------------------
        self._apply_flush_presets()
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        excreta_fraction = self.feces_fraction + self.urine_fraction
        self.flush_dose = self.service_water_fraction / excreta_fraction
        self._urine_per_feces_ratio = (
                self.urine_volume_per_cap_per_day
                / self.feces_wet_mass_per_cap_per_day
        )
        _k = 1.0 - self.service_water_fraction
        _corrected_urine_share = (
                self.feces_density * self._urine_per_feces_ratio * _k
                / (1.0 + self.feces_density * self._urine_per_feces_ratio)
        )

        # --------------------------------------------------------------
        # groups
        # main additive input group + main output group
        # --------------------------------------------------------------
        attributes["groups"] = {
            "in_main": [
                self.human_feces_bus.label,
                self.human_urine_bus.label,
                self.service_water_bus.label,
            ],
            "out_main": [self.black_water_bus.label],
        }

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to black water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.human_feces_bus.label}"] = sequence(
            self.feces_density
        )
        attributes[f"conversion_factor_{self.human_urine_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.service_water_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.black_water_bus.label}"] = sequence(1.0)

        # Group-level normalization: no extra scaling on the group
        attributes["conversion_factor_in_main"] = sequence(1.0)
        attributes["conversion_factor_out_main"] = sequence(1.0)

        # --------------------------------------------------------------
        # flow-share constraints
        # --------------------------------------------------------------
        attributes[f"flow_share_max_{self.human_urine_bus.label}"] = sequence(
            _corrected_urine_share
        )
        attributes[f"flow_share_fix_{self.service_water_bus.label}"] = sequence(self.service_water_fraction)

        # --------------------------------------------------------------
        # primary bus label resolution
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

        if self.black_water_bus in self.outputs:
            out_flow = self.outputs[self.black_water_bus]
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
                "service_water_bus": self.service_water_bus,
                "black_water_bus": self.black_water_bus,
            }
            primary_bus = primary_bus_map.get(self.primary)
            if primary_bus is not None:
                flow = self.inputs.get(primary_bus, self.outputs.get(primary_bus))
                if flow is not None:
                    flow.nominal_value = self.capacity

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
