import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class NanoFiltration(MIMO):
    """
    Literature-informed nanofiltration facade based on MIMO.

    Purpose
    -------
    Engineering-grade NF water treatment facade representing a pressure-driven
    membrane unit as a four-port MIMO component. The model is designed as a
    recovery-and-energy bookkeeping unit for system optimization, not a full
    mechanistic membrane transport solver.

    Core references
    ---------------
    1. DuPont / FilmTec Principle of RO and NF Technical Manual Excerpt:
       recovery definition, osmotic-pressure limitation logic, and engineering
       operating intuition.
    2. Nanofiltration: Principles, Process Modeling, and Applications
       (Taylor & Francis, 2021): terminology, process-design framing, and
       variable definitions.
    3. Schaefer et al. (2021) — Modeling of NF using DSPM-DE model (PMC):
       mechanistic NF transport boundary and justification of reduced-order
       modeling scope.
    4. Discussion on calculation of maximum water recovery in NF systems
       (Desalination, 2014): recovery bounds and concentration-factor logic.

    Main equations
    --------------
    All flows normalized to 1 m³ net permeate output (primary output):

    Feedwater requirement:
        feedwater_per_output = 1 / water_recovery     [m³_feed / m³_permeate]

    Brine / concentrate output:
        brine_per_output = (1 - water_recovery) / water_recovery
                                                       [m³_brine / m³_permeate]

    Electricity demand (Mode A — direct SEC):
        electricity_per_output = SEC                   [kWh / m³_permeate]

    Electricity demand (Mode B — pressure-derived SEC):
        delta_P_eff = max(0, P_feed - Pi_osm)          [bar]
        SEC         = delta_P_eff / (36 * eta_pump)    [kWh / m³_permeate]

    Antiscalant dosing (optional input):
        antiscalant_per_output = antiscalant_dose_per_m3
                                                       [m³_chem / m³_permeate]

    Backwash water demand (optional input):
        backwash_per_output = backwash_fraction        [m³_bw / m³_permeate]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum
      permeate output of the unit.
    - SEC can be provided directly (Mode A) or derived from effective pressure
      and pump efficiency (Mode B). Provide either specific_energy_consumption
      or feed_pressure_bar; both at the same time defaults to Mode A.
    - Optional antiscalant_bus and backwash_water_bus extend the four-port base
      for more detailed process configurations; not required for standard use.
    - Detailed solute rejection, concentration polarization, and ion-transport
      physics (DSPM-DE) are intentionally excluded from v3.0. Their effects
      should be reflected through water_recovery, SEC, and operating-cost
      parameters calibrated from literature.
    - For backward compatibility, `efficiency` may be passed as an alias for
      `water_recovery` when `water_recovery` is not explicitly provided.
    - Characterization values (typical SEC range, design flux, TMP) are stored
      as documentation/calibration defaults. They are not enforced as hard
      optimization constraints in v3.0.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "nanofiltration"
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
    water_in_bus: Bus = None            # m³ feedwater
    water_out_bus: Bus = None           # m³ permeate (PRIMARY)
    brine_out_bus: Bus = None           # m³ concentrate

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    antiscalant_bus: Optional[Bus] = None       # m³  chemical dosing input
    backwash_water_bus: Optional[Bus] = None    # m³  membrane cleaning water

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    # reserved for future extension

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    water_recovery: float = None                         # m³ permeate / m³ feed [-]
    specific_energy_consumption: float = None            # kWh / m³ permeate
    feed_pressure_bar: float = None                      # bar
    osmotic_pressure_bar: float = 0.0                    # bar
    pump_efficiency: float = 0.80                        # [-]
    max_recovery: Optional[float] = None                 # upper validation bound [-]
    antiscalant_dose_per_m3: float = 0.0                 # m³ antiscalant / m³ permeate
    backwash_fraction: float = 0.0                       # m³ backwash / m³ permeate

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # €/m³ permeate; exclude cleaning O&M (use cleaning_cost)
    carrier_cost: float = 0.0               # €/kWh electricity
    brine_disposal_cost: float = 0.0        # €/m³ brine
    cleaning_cost: float = 0.0              # €/m³ permeate (O&M surcharge)

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # DuPont / FilmTec NF Technical Manual; Schaefer et al. (2021) DSPM-DE
    # ------------------------------------------------------------------
    sec_typical_min: float = 0.3                # kWh/m³, lower bound from literature
    sec_typical_max: float = 1.5                # kWh/m³, upper bound from literature
    design_flux_lmh: float = None               # L/m²/hr, documentation only
    tmp_bar: float = None                       # bar, transmembrane pressure, documentation only
    cp_factor: Optional[float] = None           # concentration polarization [-], documentation only
    feed_tds: Optional[float] = None            # mg/L, documentation only
    solute_rejection: Optional[float] = None    # lumped rejection [-], documentation only
    temperature_c: Optional[float] = None       # °C, documentation only
    membrane_area_m2: Optional[float] = None    # m², documentation only

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
        self.brine_out_bus = attributes.pop("brine_out_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.antiscalant_bus = attributes.pop("antiscalant_bus", None)
        self.backwash_water_bus = attributes.pop("backwash_water_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.water_recovery = attributes.pop("water_recovery", self.water_recovery)
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.feed_pressure_bar = attributes.pop(
            "feed_pressure_bar", self.feed_pressure_bar
        )
        self.osmotic_pressure_bar = attributes.pop(
            "osmotic_pressure_bar", self.osmotic_pressure_bar
        )
        self.pump_efficiency = attributes.pop("pump_efficiency", self.pump_efficiency)
        self.max_recovery = attributes.pop("max_recovery", self.max_recovery)

        # optional input bus parameters
        self.antiscalant_dose_per_m3 = attributes.pop(
            "antiscalant_dose_per_m3", self.antiscalant_dose_per_m3
        )
        self.backwash_fraction = attributes.pop(
            "backwash_fraction", self.backwash_fraction
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.brine_disposal_cost = attributes.pop(
            "brine_disposal_cost", self.brine_disposal_cost
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
        self.sec_typical_min = attributes.pop("sec_typical_min", self.sec_typical_min)
        self.sec_typical_max = attributes.pop("sec_typical_max", self.sec_typical_max)
        self.design_flux_lmh = attributes.pop("design_flux_lmh", self.design_flux_lmh)
        self.tmp_bar = attributes.pop("tmp_bar", self.tmp_bar)
        self.cp_factor = attributes.pop("cp_factor", self.cp_factor)
        self.feed_tds = attributes.pop("feed_tds", self.feed_tds)
        self.solute_rejection = attributes.pop("solute_rejection", self.solute_rejection)
        self.temperature_c = attributes.pop("temperature_c", self.temperature_c)
        self.membrane_area_m2 = attributes.pop("membrane_area_m2", self.membrane_area_m2)

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # (DuPont / FilmTec NF Technical Manual, ref. 1;
        #  Nanofiltration: Principles, Process Modeling, ref. 2)
        # --------------------------------------------------------------
        if self.specific_energy_consumption is not None:
            self._sec = float(self.specific_energy_consumption)
        else:
            _eff_p = self.feed_pressure_bar - self.osmotic_pressure_bar
            self._sec = float(_eff_p / (36.0 * self.pump_efficiency))

        self._feedwater_per_output = 1.0 / self.water_recovery
        self._brine_per_output = (1.0 - self.water_recovery) / self.water_recovery

        # --------------------------------------------------------------
        # conversion factors
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._sec
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.brine_out_bus.label}"] = sequence(
            self._brine_per_output
        )
        if self.antiscalant_bus is not None:
            attributes[f"conversion_factor_{self.antiscalant_bus.label}"] = sequence(
                max(self.antiscalant_dose_per_m3, 1e-9)
            )
        if self.backwash_water_bus is not None:
            attributes[f"conversion_factor_{self.backwash_water_bus.label}"] = sequence(
                max(self.backwash_fraction, 1e-9)
            )

        # --------------------------------------------------------------
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # --------------------------------------------------------------
        if self.cleaning_cost > 0:
            attributes.setdefault("output_parameters", {})
            attributes["output_parameters"].update(
                {"variable_costs": self.cleaning_cost}
            )

        attributes.setdefault("output_parameters_1", {})
        if self.brine_disposal_cost > 0:
            attributes["output_parameters_1"].update(
                {"variable_costs": self.brine_disposal_cost}
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
        elif self.primary == "brine_out_bus":
            primary_label = self.brine_out_bus.label
        else:
            primary_label = self.primary

        # --------------------------------------------------------------
        # initialize base MIMO facade
        # --------------------------------------------------------------
        super().__init__(
            from_bus_0=self.electricity_bus,
            from_bus_1=self.water_in_bus,
            to_bus_0=self.water_out_bus,
            to_bus_1=self.brine_out_bus,
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
        idx_out = 2
        # inputs
        for bus in [
            self.antiscalant_bus,
            self.backwash_water_bus,
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in []:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if self.water_recovery is None:
            raise ValueError(
                "water_recovery must be provided "
                "(defined as V_permeate / V_feed, per DuPont/FilmTec NF manual)."
            )
        if not 0 < self.water_recovery < 1:
            raise ValueError("water_recovery must be in the open interval (0, 1).")

        if self.max_recovery is not None:
            if not 0 < self.max_recovery < 1:
                raise ValueError("max_recovery must be in the open interval (0, 1).")
            if self.water_recovery > self.max_recovery:
                raise ValueError(
                    f"water_recovery ({self.water_recovery}) exceeds max_recovery "
                    f"({self.max_recovery}). Revise design or use a lower recovery value."
                )

        if self.specific_energy_consumption is None and self.feed_pressure_bar is None:
            raise ValueError(
                "Provide either specific_energy_consumption [kWh/m³] directly "
                "or feed_pressure_bar [bar] for pressure-derived SEC (Mode B)."
            )
        if self.specific_energy_consumption is not None:
            if self.specific_energy_consumption < 0:
                raise ValueError("specific_energy_consumption must be >= 0.")

        if self.feed_pressure_bar is not None:
            if self.feed_pressure_bar < 0:
                raise ValueError("feed_pressure_bar must be >= 0.")
            if self.specific_energy_consumption is None:
                if self.feed_pressure_bar <= self.osmotic_pressure_bar:
                    raise ValueError(
                        "feed_pressure_bar must be strictly greater than "
                        "osmotic_pressure_bar for a positive effective driving "
                        "pressure (Mode B SEC)."
                    )

        if self.osmotic_pressure_bar < 0:
            raise ValueError("osmotic_pressure_bar must be >= 0.")
        if not 0 < self.pump_efficiency <= 1:
            raise ValueError("pump_efficiency must be in (0, 1].")

        if self.solute_rejection is not None:
            if not 0 <= self.solute_rejection <= 1:
                warnings.warn(
                    f"solute_rejection ({self.solute_rejection}) is outside [0, 1]. "
                    "This field is for documentation only and has no effect on the "
                    "optimization, but the value appears physically unrealistic.",
                    UserWarning,
                )
        if self.cp_factor is not None and self.cp_factor < 1.0:
            warnings.warn(
                f"cp_factor ({self.cp_factor}) is below 1.0. "
                "Concentration polarization factor is expected to be >= 1.0. "
                "This field is for documentation only and has no effect on the optimization.",
                UserWarning,
            )

        if self.antiscalant_bus is None and self.antiscalant_dose_per_m3 > 0:
            warnings.warn(
                "antiscalant_dose_per_m3 is set but antiscalant_bus is None. "
                "Dose parameter will be ignored.",
                UserWarning,
            )
        if self.backwash_water_bus is None and self.backwash_fraction > 0:
            warnings.warn(
                "backwash_fraction > 0 but no backwash_water_bus provided. "
                "Backwash demand parameter will be ignored.",
                UserWarning,
            )