import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class UltraFiltration(MIMO):
    """
    Literature-informed ultrafiltration facade based on MIMO.

    Purpose
    -------
    Generic pressure-driven membrane filtration facade for ultrafiltration
    units used in water treatment systems. The model is designed as a
    bookkeeping/process-yield unit representing a UF membrane as a
    water-treatment intervention. It is not a full mechanistic membrane
    transport model.

    Core references
    ---------------
    1. Christensen & Gilabert-Oriol (2024): recovery, availability, net flux,
       cleaning/fouling concepts, and feed-filtrate-concentrate mass balances.
    2. TU Delft OCW - Micro- and ultrafiltration (Water Treatment):
       dead-end UF mass balance, backwash-adjusted recovery, and
       filtration/backwash cycle logic.
    3. DuPont UF process design and backwash technical guidance:
       practical availability, backwash procedure, and net water production.

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Feedwater requirement:
        feedwater_per_output = 1 / recovery           [m³_feed / m³_product]

    Backwash wastewater:
        backwash_per_output = backwash_ratio           [m³_bw / m³_product]

    Brine / reject output:
        brine_per_output = 1/recovery - 1 - backwash_ratio  [m³_brine / m³_product]

    Electricity demand:
        electricity_per_output = SEC                   [kWh / m³_product]

    Availability derating (activity bound):
        Q_net(t) <= availability * Q_cap               [m³/hr]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum
      net treated water throughput of the unit.
    - availability is implemented as an activity_bound_max constraint, not as
      a hidden multiplier inside SEC or recovery, to keep cost interpretation
      unambiguous.
    - TMP, flux, membrane area, and detailed fouling dynamics are intentionally
      excluded from v3.0. Their effects should be reflected through recovery,
      availability, SEC, and operating-cost parameters calibrated from literature.
    - For backward compatibility, `efficiency` may be passed as an alias for
      `recovery` when `recovery` is not explicitly provided.
    - Characterization values (typical SEC range, design flux, TMP) are stored
      as documentation/calibration defaults. They are not enforced as hard
      optimization constraints in v3.0.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "ultrafiltration"
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
    water_in_bus: Bus = None            # m³
    water_out_bus: Bus = None           # m³  (PRIMARY)
    brine_out_bus: Bus = None           # m³

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    backwash_out_bus: Optional[Bus] = None      # m³  backwash wastewater

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.25       # kWh / m³ net treated water
    recovery: float = 0.98                          # m³ net treated water / m³ feedwater
    backwash_ratio: float = 0.0                     # m³ backwash water / m³ net treated water
    availability: float = 1.0                       # fraction of productive operating time

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # €/m³ net treated water
    carrier_cost: float = 0.0               # €/kWh electricity
    brine_disposal_cost: float = 0.0        # €/m³ brine
    backwash_disposal_cost: float = 0.0     # €/m³ backwash wastewater
    cleaning_cost: float = 0.0              # €/m³ net treated water (O&M surcharge)

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # Christensen & Gilabert-Oriol (2024) / TU Delft style characterization
    # ------------------------------------------------------------------
    sec_typical_min: float = 0.1            # kWh/m³, lower bound from literature
    sec_typical_max: float = 0.5            # kWh/m³, upper bound from literature
    design_flux_lmh: float = None           # L/m²/hr, design flux, documentation only
    tmp_bar: float = None                   # bar, transmembrane pressure, documentation only
    backwash_duration_s: float = None       # s, duration per backwash event, documentation only
    backwash_interval_min: float = None     # min, interval between backwash events, documentation only

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
        self.backwash_out_bus = attributes.pop("backwash_out_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.recovery = attributes.pop("recovery", self.recovery)
        self.backwash_ratio = attributes.pop("backwash_ratio", self.backwash_ratio)
        self.availability = attributes.pop("availability", self.availability)

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.brine_disposal_cost = attributes.pop(
            "brine_disposal_cost", self.brine_disposal_cost
        )
        self.backwash_disposal_cost = attributes.pop(
            "backwash_disposal_cost", self.backwash_disposal_cost
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
        self.backwash_duration_s = attributes.pop(
            "backwash_duration_s", self.backwash_duration_s
        )
        self.backwash_interval_min = attributes.pop(
            "backwash_interval_min", self.backwash_interval_min
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # (Christensen & Gilabert - Oriol, 2024; TU Delft OCW)
        # --------------------------------------------------------------
        self._feedwater_per_output = 1.0 / self.recovery
        self._backwash_per_output = self.backwash_ratio
        self._brine_per_output = (
                self._feedwater_per_output - 1.0 - self._backwash_per_output
        )

        # --------------------------------------------------------------
        # conversion factors
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.brine_out_bus.label}"] = sequence(
            self._brine_per_output
        )
        if self.backwash_out_bus is not None:
            attributes[f"conversion_factor_{self.backwash_out_bus.label}"] = sequence(
                self._backwash_per_output
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

        if self.backwash_out_bus is not None:
            attributes.setdefault("output_parameters_2", {})
            if self.backwash_disposal_cost > 0:
                attributes["output_parameters_2"].update(
                    {"variable_costs": self.backwash_disposal_cost}
                )

        # --------------------------------------------------------------
        # availability as activity bound (Christensen & Gilabert-Oriol, 2024;
        # DuPont UF design guidance)
        # derates maximum productive output without distorting SEC or recovery
        # --------------------------------------------------------------
        if self.availability < 1.0:
            attributes["activity_bound_max"] = sequence(self.availability)

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
        for bus in []:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.backwash_out_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.recovery <= 1:
            raise ValueError("recovery must be in (0, 1].")
        if self.backwash_ratio < 0:
            raise ValueError("backwash_ratio must be >= 0.")
        if not 0 < self.availability <= 1:
            raise ValueError("availability must be in (0, 1].")
        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        brine_check = 1.0 / self.recovery - 1.0 - self.backwash_ratio
        if brine_check < 0:
            raise ValueError(
                "Invalid parameter combination: brine fraction becomes negative. "
                "Reduce backwash_ratio or increase recovery."
            )

        if self.backwash_ratio > 0 and self.backwash_out_bus is None:
            warnings.warn(
                "backwash_ratio > 0 but no backwash_out_bus provided. "
                "Backwash water loss is absorbed into the brine output stream.",
                UserWarning,
            )