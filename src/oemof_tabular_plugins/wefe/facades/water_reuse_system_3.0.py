import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO

@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class WaterReuseSystem(MIMO):
    """
    Literature-informed generic water reuse system facade based on MIMO.

    Purpose
    -------
    Generic membrane-based water reuse treatment train facade for planning-level
    WEFE nexus modeling. The model is designed as a bookkeeping / flow-split
    unit representing a complete reuse treatment train normalized to treated
    water output. It is not a full mechanistic membrane transport model or
    RO design tool.

    Core references
    ---------------
    1. AWWA M62 — Membrane Processes for Water Reuse (2020):
       treatment train design, SEC ranges (0.3–1.5 kWh/m³), recovery
       operating envelope (70–85%), and fouling impact on capacity.
    2. DuPont RO Operations Advisor User Manual:
       operating limits (recovery ≤ 80%), fouling effects on flux and SEC,
       cleaning frequency and maintenance realism.
    3. Kehrein et al. (2021), Water Reuse:
       reuse-target differentiation (potable / industrial / agricultural),
       SEC by application, and concentrate / brine management costs.
    4. Tang et al. (2018), Environmental Science & Technology:
       advanced membrane reuse energy fundamentals, fouling–energy coupling,
       and RO as standard technology for advanced reuse trains.
    5. Potable water reuse energy modeling review (2021):
       train-level SEC abstraction and energy-recovery feasibility.

    Main equations
    --------------
    Feedwater input per unit treated water:
        f_feed(t) = f_product(t) / recovery_ratio
        [m³/hr]     [m³/hr]        [-]

    Reject output per unit treated water (only when reject_bus is set):
        f_reject(t) = f_product(t) * (1 - recovery_ratio) / recovery_ratio
        [m³/hr]        [m³/hr]        [-]

    Effective specific energy consumption:
        SEC_eff = specific_energy_consumption
                  * fouling_factor
                  * (1 - energy_recovery_factor)
        [kWh/m³]   [kWh/m³]              [-]           [-]

    Electricity input per unit treated water:
        f_elec(t) = SEC_eff * f_product(t)
        [kWh/hr]    [kWh/m³]  [m³/hr]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains maximum
      treated water production of the treatment train.
    - recovery_ratio replaces the "efficiency" concept from v2.0, following
      standard membrane reuse terminology (AWWA M62; Kehrein et al., 2021).
    - reject_bus is optional: when provided, concentrate/brine becomes a
      first-class output. When absent, the facade behaves like v1.0 topology
      (2 inputs → 1 output).
    - fouling_factor and energy_recovery_factor are reduced-order surrogates
      for operational deterioration and energy integration benefits. They are
      not mechanistic fouling or pressure models.
    - Characterization values (SEC reference ranges, recovery envelope) are
      stored as metadata for scenario documentation. They are enforced only
      as soft warnings, not as hard optimization constraints in v3.0.
    - reuse_mode provides optional initialization presets for agricultural,
      industrial, and potable reuse targets (Kehrein et al., 2021).
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "water_reuse_system"
    name: str = ""
    tech: str = "wastewater-treatment"
    carrier: str = "water"
    reuse_mode: Optional[str] = None  # "agricultural", "industrial", "potable"
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
    electricity_bus: Bus = None     # expected unit: kWh
    water_in_bus: Bus = None        # expected unit: m³  (secondary effluent / feedwater)
    water_out_bus: Bus = None       # expected unit: m³  (product / reclaimed water)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension, e.g. heat_bus, chemical_bus

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    reject_bus: Optional[Bus] = None  # m³  (concentrate / brine)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.90      # kWh / m³ treated water (gross, baseline)
    recovery_ratio: float = 0.80                   # treated water / feedwater [-], (0, 1)
    fouling_factor: float = 1.00                   # multiplier on SEC [-], must be > 0
    energy_recovery_factor: float = 0.00           # fractional SEC reduction [-], [0, 1)

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0      # €/m³ treated water
    carrier_cost: float = 0.0       # €/kWh electricity
    chemical_cost: float = 0.0      # €/m³ treated water (OPEX proxy)
    reject_cost: float = 0.0        # €/m³ reject water (disposal / brine handling)

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # literature-backed reference values for scenario documentation
    # ------------------------------------------------------------------
    sec_reference_potable: float = 1.20         # kWh/m³
    sec_reference_industrial: float = 0.80      # kWh/m³
    sec_reference_agricultural: float = 0.50    # kWh/m³
    recovery_recommended_min: float = 0.50      # [-]
    recovery_recommended_max: float = 0.85      # [-]
    recovery_warning_threshold: float = 0.85    # [-]

    def __init__(self, **attributes):
        # --------------------------------------------------------------
        # identity
        # --------------------------------------------------------------
        self.type = attributes.pop("type", self.type)
        self.name = attributes.pop("name", self.name)
        self.tech = attributes.pop("tech", self.tech)
        self.carrier = attributes.pop("carrier", self.carrier)
        self.reuse_mode = attributes.pop("reuse_mode", self.reuse_mode)
        self.primary = attributes.pop("primary", self.primary)

        # --------------------------------------------------------------
        # mandatory buses
        # --------------------------------------------------------------
        self.electricity_bus = attributes.pop("electricity_bus")
        self.water_in_bus = attributes.pop("water_in_bus")
        self.water_out_bus = attributes.pop("water_out_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.reject_bus = attributes.pop("reject_bus", None)

        # --------------------------------------------------------------
        # optional reuse-mode presets
        # applied before physics parameters so explicit user values override
        # --------------------------------------------------------------
        preset_map = {
            "agricultural": {
                "recovery_ratio": 0.85,
                "specific_energy_consumption": 0.50,
            },
            "industrial": {
                "recovery_ratio": 0.80,
                "specific_energy_consumption": 0.80,
            },
            "potable": {
                "recovery_ratio": 0.75,
                "specific_energy_consumption": 1.20,
            },
        }
        preset = {}
        if self.reuse_mode is not None:
            if self.reuse_mode not in preset_map:
                raise ValueError(
                    "reuse_mode must be one of "
                    "{'agricultural', 'industrial', 'potable'}."
                )
            preset = preset_map[self.reuse_mode]

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption",
            preset.get("specific_energy_consumption", self.specific_energy_consumption),
        )
        self.recovery_ratio = attributes.pop(
            "recovery_ratio",
            preset.get("recovery_ratio", self.recovery_ratio),
        )
        self.fouling_factor = attributes.pop("fouling_factor", self.fouling_factor)
        self.energy_recovery_factor = attributes.pop(
            "energy_recovery_factor", self.energy_recovery_factor
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.chemical_cost = attributes.pop("chemical_cost", self.chemical_cost)
        self.reject_cost = attributes.pop("reject_cost", self.reject_cost)
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
        self.sec_reference_potable = attributes.pop(
            "sec_reference_potable", self.sec_reference_potable
        )
        self.sec_reference_industrial = attributes.pop(
            "sec_reference_industrial", self.sec_reference_industrial
        )
        self.sec_reference_agricultural = attributes.pop(
            "sec_reference_agricultural", self.sec_reference_agricultural
        )
        self.recovery_recommended_min = attributes.pop(
            "recovery_recommended_min", self.recovery_recommended_min
        )
        self.recovery_recommended_max = attributes.pop(
            "recovery_recommended_max", self.recovery_recommended_max
        )
        self.recovery_warning_threshold = attributes.pop(
            "recovery_warning_threshold", self.recovery_warning_threshold
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived conversion quantities
        # --------------------------------------------------------------
        self._feedwater_per_output = 1.0 / self.recovery_ratio
        self._reject_per_output = (1.0 - self.recovery_ratio) / self.recovery_ratio
        self._effective_sec = (
                self.specific_energy_consumption
                * self.fouling_factor
                * (1.0 - self.energy_recovery_factor)
        )

        # --------------------------------------------------------------
        # conversion factors
        # normalized to treated liquid output [m³/hr] = 1
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._effective_sec
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        if self.reject_bus is not None:
            attributes[f"conversion_factor_{self.reject_bus.label}"] = sequence(
                self._reject_per_output
            )

        # --------------------------------------------------------------
        # output-specific costs
        # --------------------------------------------------------------
        attributes.setdefault("output_parameters", {})

        # --------------------------------------------------------------
        # primary bus label resolution
        # --------------------------------------------------------------
        if self.primary == "water_out_bus":
            primary_label = self.water_out_bus.label
        elif self.primary == "water_in_bus":
            primary_label = self.water_in_bus.label
        elif self.primary == "reject_bus" and self.reject_bus is not None:
            primary_label = self.reject_bus.label
        else:
            primary_label = self.primary

        # --------------------------------------------------------------
        # initialize base MIMO facade
        # --------------------------------------------------------------
        super().__init__(
            from_bus_0=self.electricity_bus,
            from_bus_1=self.water_in_bus,
            to_bus_0=self.water_out_bus,
            primary=primary_label,
            marginal_cost=self.marginal_cost + self.chemical_cost,
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
        idx_out = 1
        # inputs
        for bus in []:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.reject_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.recovery_ratio < 1:
            raise ValueError("recovery_ratio must be in (0, 1).")

        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        if self.fouling_factor <= 0:
            raise ValueError("fouling_factor must be > 0.")

        if not 0 <= self.energy_recovery_factor < 1:
            raise ValueError("energy_recovery_factor must be in [0, 1).")

        for param_name, value in {
            "reject_cost": self.reject_cost,
            "chemical_cost": self.chemical_cost,
        }.items():
            if value < 0:
                raise ValueError(f"{param_name} must be >= 0.")

        if self.primary == "reject_bus" and self.reject_bus is None:
            raise ValueError(
                "primary='reject_bus' requires reject_bus to be provided."
            )

        # ----------------------------------------------------------
        # soft warnings (DuPont manual; AWWA M62)
        # ----------------------------------------------------------
        if self.recovery_ratio > self.recovery_warning_threshold:
            warnings.warn(
                f"recovery_ratio={self.recovery_ratio:.2f} exceeds the recommended "
                f"threshold of {self.recovery_warning_threshold:.2f}. "
                "High-recovery operation increases fouling and scaling risk "
                "(DuPont RO Operations Advisor; AWWA M62).",
                UserWarning,
            )

        if (
                self.reuse_mode == "potable"
                and self.specific_energy_consumption < 0.50
        ):
            warnings.warn(
                f"specific_energy_consumption={self.specific_energy_consumption:.2f} kWh/m³ "
                "is unusually low for a potable reuse train. Full advanced treatment "
                "trains are typically 1.1–1.4 kWh/m³ "
                "(Tang et al. 2018; potable reuse energy modeling review 2021).",
                UserWarning,
            )