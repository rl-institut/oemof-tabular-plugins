import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class Boiling(MIMO):
    """
    Literature-informed simple boiling facade based on MIMO.

    Purpose
    -------
    Generic thermal-disinfection facade representing boiling as an
    energy-driven pathogen inactivation step used in household or
    community-scale water treatment systems. The model is designed as a
    bookkeeping/process-yield unit representing a boiling unit as a
    water-treatment intervention. It is not a full thermodynamic
    phase-change or mechanistic microbial-inactivation model.

    Core references
    ---------------
    1. WHO (2016). Household water treatment and safe storage guidance:
       boiling as a valid treatment step, treatment-chain context, and
       safe storage requirement.
    2. WHO (2015). Technical brief — Boil water: rolling-boil treatment
       condition, pre-clarification for turbid water, pathogen inactivation.
    3. CAWST (2008). Household water treatment manual: practical
       implementation guidance, pretreatment dependency, and energy framing.
    4. WHO (2011). Evaluating household water treatment options: performance
       framing and barrier-based technology assessment methodology.

    Main equations
    --------------
    All flows normalized to 1 m3 net treated water output (primary):

    Feedwater requirement:
        feedwater_per_output = 1 / treated_water_fraction   [m3_feed / m3_treated]

    Steam / vapour loss (optional output):
        steam_per_output = steam_loss_fraction / treated_water_fraction
                                                            [m3_steam / m3_treated]

    Electric-only mode (fuel_bus is None):
        electricity_per_output =
            specific_thermal_energy_demand / heater_efficiency
            + specific_electricity_auxiliaries
                                            [kWh_el / m3_treated]

    External fuel mode (fuel_bus is provided):
        fuel_per_output        = specific_fuel_consumption   [kWh_fuel / m3_treated]
        electricity_per_output = specific_electricity_auxiliaries
                                            [kWh_el / m3_treated]

    Boiling conditions (documentation only — no state variable in v3.0):
        Rolling boil >= 1 min at sea level (WHO, 2015)
        Turbidity pre-clarification recommended if NTU > threshold (CAWST, 2008)

    Notes
    -----
    - Primary flow is water_out_bus [m3/hr]. Capacity constrains the maximum
      treated-water throughput of the boiling unit.
    - treated_water_fraction represents the availability of treated water after
      handling losses including steam evaporation.
    - electricity_bus carries electrical energy converted to useful process
      heat via heater_efficiency in electric-only mode. This decouples process
      thermodynamics from the electric heating technology type (resistance
      heater, heat pump, etc.).
    - When fuel_bus is provided, it supplies the full thermal duty
      (specific_fuel_consumption) directly. electricity_bus then carries only
      auxiliary electricity (specific_electricity_auxiliaries). heater_efficiency
      has no effect in this mode and a UserWarning is raised.
    - fuel_carrier_cost [€/kWh_fuel] is folded into output_parameters as a
      variable cost on water_out_bus (per m3 treated) when fuel_bus is active
      and fuel_carrier_cost > 0. It has no effect when fuel_bus is None.
    - steam_loss_bus is an optional output representing water lost as vapour
      during boiling. steam_loss_fraction defines the share of feedwater
      evaporated. When omitted, steam losses are implicitly absorbed into the
      treated_water_fraction and no explicit vapour stream is tracked.
    - Boiling conditions (rolling boil, >= 1 min retention, turbidity pre-
      treatment) are documented here but cannot be enforced as hard constraints
      without explicit temperature or residence-time state variables.
    - WHO and CAWST guidance consistently note that safe storage after boiling
      is as critical as the boiling step itself; this is stored as metadata only.
    - Characterization values (boiling time, storage requirements, turbidity
      threshold) are stored as metadata for scenario documentation and are not
      enforced as hard optimization constraints in v3.0.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "boiling"
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
    electricity_bus: Bus = None             # kWh_el  (primary energy input)
    water_in_bus: Bus = None                # m³      (raw / untreated feedwater)
    water_out_bus: Bus = None               # m³      (boiled treated water — PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    fuel_bus: Optional[Bus] = None          # kWh_fuel / MJ  (biomass, charcoal, LPG, etc.)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    steam_loss_bus: Optional[Bus] = None    # m³  (vapour / steam evaporation loss)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    treated_water_fraction: float = 1.0                 # m³ treated / m³ feed
    steam_loss_fraction: float = 0.0                    # m³ steam / m³ feed (set > 0 when steam_loss_bus used)
    specific_thermal_energy_demand: float = 0.93        # kWh_th / m³ treated  (WHO, 2015; CAWST, 2008)
    heater_efficiency: float = 0.95                     # kWh_th / kWh_el  (0, 1]
    specific_electricity_auxiliaries: float = 0.03      # kWh_el / m³ treated  (pumping, controls)
    specific_fuel_consumption: float = 0.0              # kWh_fuel / m³ treated (set > 0 when fuel_bus used)

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # €/m³ treated water
    carrier_cost: float = 0.0               # €/kWh_el
    fuel_carrier_cost: float = 0.0          # €/kWh_fuel  (only relevant when fuel_bus is set)

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # ------------------------------------------------------------------
    pretreatment_required_if_turbid: bool = True        # WHO (2015), CAWST (2008)
    safe_storage_required: bool = True                  # WHO (2016)
    reference_boiling_time_minutes: float = 1.0         # WHO (2015): rolling boil >= 1 min at sea level
    max_treated_water_fraction: Optional[float] = None  # design upper bound (validation only)
    min_treated_water_fraction: Optional[float] = None  # technology lower bound (validation only)
    sec_typical_min: float = 0.5                        # kWh_th/m³, lower bound from literature
    sec_typical_max: float = 3.0                        # kWh_th/m³, upper bound from literature

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

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.fuel_bus = attributes.pop("fuel_bus", None)
        self.steam_loss_bus = attributes.pop("steam_loss_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.treated_water_fraction = attributes.pop(
            "treated_water_fraction", self.treated_water_fraction
        )
        self.steam_loss_fraction = attributes.pop(
            "steam_loss_fraction", self.steam_loss_fraction
        )
        self.specific_thermal_energy_demand = attributes.pop(
            "specific_thermal_energy_demand", self.specific_thermal_energy_demand
        )
        self.heater_efficiency = attributes.pop(
            "heater_efficiency", self.heater_efficiency
        )
        self.specific_electricity_auxiliaries = attributes.pop(
            "specific_electricity_auxiliaries", self.specific_electricity_auxiliaries
        )
        self.specific_fuel_consumption = attributes.pop(
            "specific_fuel_consumption", self.specific_fuel_consumption
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.fuel_carrier_cost = attributes.pop(
            "fuel_carrier_cost", self.fuel_carrier_cost
        )
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
        self.pretreatment_required_if_turbid = attributes.pop(
            "pretreatment_required_if_turbid", self.pretreatment_required_if_turbid
        )
        self.safe_storage_required = attributes.pop(
            "safe_storage_required", self.safe_storage_required
        )
        self.reference_boiling_time_minutes = attributes.pop(
            "reference_boiling_time_minutes", self.reference_boiling_time_minutes
        )
        self.max_treated_water_fraction = attributes.pop(
            "max_treated_water_fraction", self.max_treated_water_fraction
        )
        self.min_treated_water_fraction = attributes.pop(
            "min_treated_water_fraction", self.min_treated_water_fraction
        )
        self.sec_typical_min = attributes.pop("sec_typical_min", self.sec_typical_min)
        self.sec_typical_max = attributes.pop("sec_typical_max", self.sec_typical_max)

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._feedwater_per_output = 1.0 / self.treated_water_fraction

        if self.steam_loss_bus is not None:
            self._steam_per_output = (
                    self.steam_loss_fraction / self.treated_water_fraction
            )
        else:
            self._steam_per_output = None

        if self.fuel_bus is not None:
            self._electricity_per_output = self.specific_electricity_auxiliaries
            self._fuel_per_output = self.specific_fuel_consumption
        else:
            self._electricity_per_output = (
                    self.specific_thermal_energy_demand / self.heater_efficiency
                    + self.specific_electricity_auxiliaries
            )
            self._fuel_per_output = None

        # --------------------------------------------------------------
        # conversion factors
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._electricity_per_output
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.fuel_bus is not None:
            attributes[f"conversion_factor_{self.fuel_bus.label}"] = sequence(
                self._fuel_per_output
            )

        if self.steam_loss_bus is not None:
            attributes[f"conversion_factor_{self.steam_loss_bus.label}"] = sequence(
                self._steam_per_output
            )

        # --------------------------------------------------------------
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # --------------------------------------------------------------
        attributes.setdefault("output_parameters", {})

        if self.fuel_bus is not None and self.fuel_carrier_cost > 0:
            fuel_variable_cost = self._fuel_per_output * self.fuel_carrier_cost
            attributes["output_parameters"].update(
                {"variable_costs": fuel_variable_cost}
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
        idx_out = 1
        # inputs
        for bus in [
            self.fuel_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.steam_loss_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.treated_water_fraction <= 1:
            raise ValueError("treated_water_fraction must be in (0, 1].")

        if not 0 <= self.steam_loss_fraction < 1:
            raise ValueError("steam_loss_fraction must be in [0, 1).")

        if self.steam_loss_fraction + self.treated_water_fraction > 1.0 + 1e-9:
            raise ValueError(
                f"steam_loss_fraction ({self.steam_loss_fraction}) + "
                f"treated_water_fraction ({self.treated_water_fraction}) "
                f"must not exceed 1.0 — total water balance violated."
            )

        if not 0 < self.heater_efficiency <= 1:
            raise ValueError("heater_efficiency must be in (0, 1].")

        bounded_nonneg = {
            "specific_thermal_energy_demand": self.specific_thermal_energy_demand,
            "specific_electricity_auxiliaries": self.specific_electricity_auxiliaries,
            "specific_fuel_consumption": self.specific_fuel_consumption,
            "carrier_cost": self.carrier_cost,
            "fuel_carrier_cost": self.fuel_carrier_cost,
            "marginal_cost": self.marginal_cost,
        }
        for name, value in bounded_nonneg.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")

        if self.fuel_bus is not None:
            if self.specific_fuel_consumption <= 0:
                raise ValueError(
                    "specific_fuel_consumption must be > 0 when fuel_bus is provided."
                )
            warnings.warn(
                "fuel_bus is set — heater_efficiency has no effect in this mode. "
                "Thermal duty is supplied via fuel_bus.",
                UserWarning,
            )

        if self.fuel_bus is None and self.specific_fuel_consumption > 0:
            warnings.warn(
                "specific_fuel_consumption > 0 but fuel_bus is not set. "
                "Fuel consumption will be ignored.",
                UserWarning,
            )

        if self.fuel_bus is None and self.fuel_carrier_cost > 0:
            warnings.warn(
                "fuel_carrier_cost is set but fuel_bus is None. "
                "fuel_carrier_cost will have no effect.",
                UserWarning,
            )

        if self.steam_loss_bus is not None and self.steam_loss_fraction <= 0:
            warnings.warn(
                "steam_loss_bus is set but steam_loss_fraction is 0. "
                "No steam loss flow will be enforced.",
                UserWarning,
            )

        if self.steam_loss_bus is None and self.steam_loss_fraction > 0:
            warnings.warn(
                "steam_loss_fraction > 0 but steam_loss_bus is not set. "
                "Steam loss is implicitly absorbed into treated_water_fraction.",
                UserWarning,
            )

        if self.max_treated_water_fraction is not None:
            if self.treated_water_fraction > self.max_treated_water_fraction:
                raise ValueError(
                    f"treated_water_fraction ({self.treated_water_fraction}) exceeds "
                    f"max_treated_water_fraction ({self.max_treated_water_fraction})."
                )

        if self.min_treated_water_fraction is not None:
            if self.treated_water_fraction < self.min_treated_water_fraction:
                raise ValueError(
                    f"treated_water_fraction ({self.treated_water_fraction}) is below "
                    f"min_treated_water_fraction ({self.min_treated_water_fraction})."
                )

        if (
                self.specific_thermal_energy_demand < self.sec_typical_min
                or self.specific_thermal_energy_demand > self.sec_typical_max
        ):
            warnings.warn(
                f"specific_thermal_energy_demand ({self.specific_thermal_energy_demand} kWh_th/m³) "
                f"is outside the typical literature range [{self.sec_typical_min}, {self.sec_typical_max}] kWh_th/m³ "
                f"(CAWST, 2008; WHO, 2016). Verify this is intentional.",
                UserWarning,
            )