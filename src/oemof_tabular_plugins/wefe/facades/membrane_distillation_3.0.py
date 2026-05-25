import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class MembraneDistillation(MIMO):
    """
    Literature-informed membrane distillation facade based on MIMO.

    Purpose
    -------
    Generic thermally driven membrane distillation facade for water
    treatment units used in off-grid and rural water supply systems.
    The model is designed as a bookkeeping/process-yield unit representing
    a membrane distillation module (e.g. DCMD, AGMD, VMD, SGMD) as a
    water-treatment intervention. It is not a full mechanistic
    heat-and-mass-transfer membrane transport model.

    Core references
    ---------------
    1. Jantaporn et al. (2017): recovery-based mass balance, specific
       thermal and electrical energy demand, heat-recovery adjustment,
       and process-design calculation chain for DCMD — primary source
       for feedwater, brine, STEC, and SEC equations.
    2. Warsinger et al. (2018): MD at the water-energy nexus — system-
       level framing, heat-driven nature of MD, and operating limitations;
       justifies heat_in_bus as the primary optional thermal source
       enabling hybrid electric-plus-thermal or waste-heat-driven modes.
    3. Tijing et al. (2015): fouling and its control in MD — justification
       for performance_factor as a derating surrogate for flux
       deterioration due to fouling, wetting, scaling, or membrane aging.
    4. Adeleke et al. (2024); Patil et al. (2024): electrification of
       distillation — justifies modeling electricity as an upstream carrier
       for useful process heat and separating heater efficiency from
       process thermal demand when no dedicated thermal bus is available.

    Main equations
    --------------
    All flows normalized to 1 m³ net permeate output (primary):

    Feedwater requirement:
        feedwater_per_output = 1 / recovery_ratio
                                            [m³_feed / m³_permeate]

    Brine / concentrate output:
        brine_per_output = 1 / recovery_ratio - 1
                                            [m³_brine / m³_permeate]

    Net specific thermal energy consumption:
        STEC_net = specific_thermal_energy_demand
                   * (1 - heat_recovery_factor)
                   / performance_factor
                                            [kWh_th / m³_permeate]

    Net specific electricity consumption:
        SEC_net = specific_electricity_consumption / performance_factor
                                            [kWh_el / m³_permeate]

    Electric-only mode (heat_in_bus is None):
        electricity_per_output = STEC_net / heater_efficiency + SEC_net
                                            [kWh_el / m³_permeate]

    External heat mode (heat_in_bus is provided):
        heat_per_output        = STEC_net   [kWh_th / m³_permeate]
        electricity_per_output = SEC_net    [kWh_el / m³_permeate]

    CIP cleaning waste (time-averaged over cleaning cycles):
        cip_waste_per_output = cleaning_waste_ratio
                                            [m³_cip / m³_permeate]

    Brine concentration factor (dimensionless, reporting only):
        CF_brine = 1 / (1 - recovery_ratio)

    GOR proxy (reporting only):
        GOR_implied = 627.0 / specific_thermal_energy_demand
        [627.0 = 2257 kJ/kg x 1000 kg/m³ / 3600 kJ/kWh]

    Notes
    -----
    - Primary flow is water_out_bus [m³]. Capacity constrains the maximum
      permeate output of the unit.
    - electricity_bus carries electrical energy for auxiliary loads
      (pumps, controls, vacuum). In electric-only mode (heat_in_bus is
      None), it also carries the thermal duty converted via
      heater_efficiency, representing electric heating or a heat-pump
      assumption. A UserWarning is raised to flag this fallback
      (Warsinger et al., 2018).
    - When heat_in_bus is provided, it supplies the full net thermal duty
      (STEC_net) directly. electricity_bus then carries only SEC_net.
      heater_efficiency has no effect in this mode and a UserWarning is
      raised if a non-default value is passed alongside heat_in_bus.
    - heat_carrier_cost [€/kWh_th] is folded into output_parameters as
      a variable cost on water_out_bus (per m³ permeate) when
      heat_in_bus is active and heat_carrier_cost > 0. It has no effect
      when heat_in_bus is None.
    - brine_out_bus carries the concentrate/reject stream.
      brine_disposal_cost is applied as a variable cost on this output
      flow via output_parameters_1.
    - cleaning_waste_bus is an optional output representing CIP
      (Clean-In-Place) effluent from membrane cleaning. If omitted, CIP
      waste is implicitly absorbed into the brine stream.
      cleaning_waste_disposal_cost is applied via output_parameters_2.
    - performance_factor and heat_recovery_factor are MD-specific
      extensions not present in the Distillation facade. performance_factor
      derates both STEC_net and SEC_net for fouling and aging effects
      (Tijing et al., 2015). heat_recovery_factor reduces gross STEC for
      internal condenser-side heat reuse. Both are fixed parameters in
      v3.0, not decision variables.
    - Engineering bounds (max_recovery_ratio, min_recovery_ratio,
      max_brine_concentration_factor) are implemented as Python-side
      validation checks at instantiation, not as Pyomo constraints,
      because recovery_ratio is a fixed input parameter in v3.0, not a
      decision variable.
    - A GOR sanity check is run at instantiation. A UserWarning is raised
      if the implied GOR falls below 1.0, which indicates a physically
      implausible specific_thermal_energy_demand value or a unit error.
    - md_configuration, gor_reference, sec_typical_min, sec_typical_max,
      stec_typical_min, and stec_typical_max are stored as documentation
      and calibration metadata only. They are not enforced as hard
      optimization constraints in v3.0.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "membrane_distillation"
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
    electricity_bus: Bus = None  # kWh_el  (auxiliaries; also thermal if heat_in_bus is None)
    water_in_bus: Bus = None  # m³      (feedwater)
    water_out_bus: Bus = None  # m³      (permeate — PRIMARY)
    brine_out_bus: Bus = None  # m³      (concentrate)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    heat_in_bus: Optional[Bus] = None  # kWh_th — explicit thermal source

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    cleaning_waste_bus: Optional[Bus] = None  # m³ — CIP effluent from membrane cleaning

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    recovery_ratio: float = 0.75  # m³_permeate / m³_feed
    specific_thermal_energy_demand: float = 100.0  # kWh_th / m³ permeate (gross)
    heater_efficiency: float = 0.95  # kWh_th / kWh_el  (0, 1]
    specific_electricity_consumption: float = 2.0  # kWh_el / m³ permeate (gross, auxiliaries)
    performance_factor: float = 1.0  # fouling / non-ideal derating [-]
    heat_recovery_factor: float = 0.0  # fraction of thermal demand recovered [-]
    cleaning_waste_ratio: float = 0.0  # m³_cip / m³_permeate (time-averaged)
    max_recovery_ratio: Optional[float] = None  # design upper bound (validation only)
    max_brine_concentration_factor: Optional[float] = None  # design upper bound (validation only)

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0  # €/m³ permeate
    carrier_cost: float = 0.0  # €/kWh_el
    heat_carrier_cost: float = 0.0  # €/kWh_th
    brine_disposal_cost: float = 0.0  # €/m³ brine
    cleaning_waste_disposal_cost: float = 0.0  # €/m³ CIP effluent

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # ------------------------------------------------------------------
    md_configuration: str = "DCMD"  # Direct Contact MD | Air Gap MD | Vacuum MD | Sweeping Gas MD | generic
    gor_reference: Optional[float] = None  # gained output ratio reference value
    min_recovery_ratio: Optional[float] = None  # technology lower bound (validation only)
    sec_typical_min: float = 0.5  # kWh_el/m³, lower bound from literature
    sec_typical_max: float = 10.0  # kWh_el/m³, upper bound from literature
    stec_typical_min: float = 40.0  # kWh_th/m³, lower bound from literature
    stec_typical_max: float = 200.0  # kWh_th/m³, upper bound from literature
    feed_temperature_c: Optional[float] = None  # °C, documentation only
    feed_salinity_g_per_l: Optional[float] = None  # g/L, documentation only

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
        self.heat_in_bus = attributes.pop("heat_in_bus", None)
        self.cleaning_waste_bus = attributes.pop("cleaning_waste_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.recovery_ratio = attributes.pop(
            "recovery_ratio", self.recovery_ratio
        )
        self.specific_thermal_energy_demand = attributes.pop(
            "specific_thermal_energy_demand", self.specific_thermal_energy_demand
        )
        self.heater_efficiency = attributes.pop(
            "heater_efficiency", self.heater_efficiency
        )
        self.specific_electricity_consumption = attributes.pop(
            "specific_electricity_consumption",
            attributes.pop(
                "specific_energy_consumption",
                self.specific_electricity_consumption,
            ),
        )
        self.performance_factor = attributes.pop(
            "performance_factor", self.performance_factor
        )
        self.heat_recovery_factor = attributes.pop(
            "heat_recovery_factor", self.heat_recovery_factor
        )
        self.cleaning_waste_ratio = attributes.pop(
            "cleaning_waste_ratio", self.cleaning_waste_ratio
        )
        self.max_recovery_ratio = attributes.pop(
            "max_recovery_ratio", self.max_recovery_ratio
        )
        self.max_brine_concentration_factor = attributes.pop(
            "max_brine_concentration_factor", self.max_brine_concentration_factor
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.heat_carrier_cost = attributes.pop(
            "heat_carrier_cost", self.heat_carrier_cost
        )
        self.brine_disposal_cost = attributes.pop(
            "brine_disposal_cost", self.brine_disposal_cost
        )
        self.cleaning_waste_disposal_cost = attributes.pop(
            "cleaning_waste_disposal_cost", self.cleaning_waste_disposal_cost
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
        self.md_configuration = attributes.pop(
            "md_configuration", self.md_configuration
        )
        self.gor_reference = attributes.pop("gor_reference", self.gor_reference)
        self.min_recovery_ratio = attributes.pop(
            "min_recovery_ratio", self.min_recovery_ratio
        )
        self.sec_typical_min = attributes.pop("sec_typical_min", self.sec_typical_min)
        self.sec_typical_max = attributes.pop("sec_typical_max", self.sec_typical_max)
        self.stec_typical_min = attributes.pop(
            "stec_typical_min", self.stec_typical_min
        )
        self.stec_typical_max = attributes.pop(
            "stec_typical_max", self.stec_typical_max
        )
        self.feed_temperature_c = attributes.pop(
            "feed_temperature_c", self.feed_temperature_c
        )
        self.feed_salinity_g_per_l = attributes.pop(
            "feed_salinity_g_per_l", self.feed_salinity_g_per_l
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._feedwater_per_output = 1.0 / self.recovery_ratio
        self._brine_per_output = self._feedwater_per_output - 1.0
        self._brine_concentration_factor = 1.0 / (1.0 - self.recovery_ratio)

        self._stec_net = (
                self.specific_thermal_energy_demand
                * (1.0 - self.heat_recovery_factor)
                / self.performance_factor
        )
        self._sec_net = (
                self.specific_electricity_consumption / self.performance_factor
        )

        if self.heat_in_bus is not None:
            self._electricity_per_output = self._sec_net
            self._heat_per_output = self._stec_net
        else:
            self._electricity_per_output = (
                    self._stec_net / self.heater_efficiency + self._sec_net
            )
            self._heat_per_output = None

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
        attributes[f"conversion_factor_{self.brine_out_bus.label}"] = sequence(
            self._brine_per_output
        )
        if self.heat_in_bus is not None:
            attributes[f"conversion_factor_{self.heat_in_bus.label}"] = sequence(
                self._heat_per_output
            )
        if self.cleaning_waste_bus is not None:
            attributes[f"conversion_factor_{self.cleaning_waste_bus.label}"] = sequence(
                self.cleaning_waste_ratio
            )

        # --------------------------------------------------------------
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # --------------------------------------------------------------
        attributes.setdefault("output_parameters", {})
        attributes.setdefault("output_parameters_1", {})

        if self.heat_in_bus is not None and self.heat_carrier_cost > 0:
            heat_variable_cost = self._heat_per_output * self.heat_carrier_cost
            attributes["output_parameters"].update(
                {"variable_costs": heat_variable_cost}
            )

        if self.brine_disposal_cost > 0:
            attributes["output_parameters_1"].update(
                {"variable_costs": self.brine_disposal_cost}
            )

        if self.cleaning_waste_bus is not None:
            attributes.setdefault("output_parameters_2", {})
            if self.cleaning_waste_disposal_cost > 0:
                attributes["output_parameters_2"].update(
                    {"variable_costs": self.cleaning_waste_disposal_cost}
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
            self.heat_in_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.cleaning_waste_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.recovery_ratio < 1:
            raise ValueError("recovery_ratio must be in (0, 1).")

        if not 0 < self.heater_efficiency <= 1:
            raise ValueError("heater_efficiency must be in (0, 1].")

        if not 0 < self.performance_factor <= 1:
            raise ValueError("performance_factor must be in (0, 1].")

        if not 0 <= self.heat_recovery_factor < 1:
            raise ValueError("heat_recovery_factor must be in [0, 1).")

        bounded_nonneg = {
            "specific_thermal_energy_demand": self.specific_thermal_energy_demand,
            "specific_electricity_consumption": self.specific_electricity_consumption,
            "cleaning_waste_ratio": self.cleaning_waste_ratio,
            "carrier_cost": self.carrier_cost,
            "heat_carrier_cost": self.heat_carrier_cost,
            "marginal_cost": self.marginal_cost,
            "brine_disposal_cost": self.brine_disposal_cost,
            "cleaning_waste_disposal_cost": self.cleaning_waste_disposal_cost,
        }
        for name, value in bounded_nonneg.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")

        if self.heat_in_bus is not None:
            warnings.warn(
                "heat_in_bus is set — heater_efficiency has no effect in this mode. "
                "Thermal duty is supplied directly via heat_in_bus.",
                UserWarning,
            )

        if self.heat_in_bus is None and self.heat_carrier_cost > 0:
            warnings.warn(
                "heat_carrier_cost is set but heat_in_bus is None. "
                "heat_carrier_cost will have no effect.",
                UserWarning,
            )

        if self.heat_in_bus is None and self.specific_thermal_energy_demand > 0:
            warnings.warn(
                f"No heat_in_bus provided. Thermal demand "
                f"({self.specific_thermal_energy_demand} kWh_th/m³ gross) will be "
                "converted to electricity via heater_efficiency and absorbed into "
                "electricity_bus. For more physical accuracy, provide a heat_in_bus "
                "(Warsinger et al., 2018).",
                UserWarning,
            )

        if self.max_recovery_ratio is not None:
            if self.recovery_ratio > self.max_recovery_ratio:
                raise ValueError(
                    f"recovery_ratio ({self.recovery_ratio}) exceeds max_recovery_ratio ({self.max_recovery_ratio})."
                )

        if self.min_recovery_ratio is not None:
            if self.recovery_ratio < self.min_recovery_ratio:
                raise ValueError(
                    f"recovery_ratio ({self.recovery_ratio}) is below min_recovery_ratio ({self.min_recovery_ratio})."
                )

        if self.max_brine_concentration_factor is not None:
            cf = 1.0 / (1.0 - self.recovery_ratio)
            if cf > self.max_brine_concentration_factor:
                raise ValueError(
                    f"Implied brine_concentration_factor ({cf:.3f}) exceeds "
                    f"max_brine_concentration_factor ({self.max_brine_concentration_factor})."
                )

        if self.cleaning_waste_bus is not None and self.cleaning_waste_ratio <= 0:
            warnings.warn(
                "cleaning_waste_bus is set but cleaning_waste_ratio is 0. "
                "No CIP waste flow will be enforced.",
                UserWarning,
            )

        if self.md_configuration not in {"DCMD", "AGMD", "VMD", "SGMD", "generic"}:
            warnings.warn(
                f"Unknown md_configuration '{self.md_configuration}'. "
                "Accepted values are: DCMD, AGMD, VMD, SGMD, generic.",
                UserWarning,
            )

        # GOR sanity check — 627.0 = 2257 kJ/kg × 1000 kg/m³ / 3600 kJ/kWh
        gor_implied = 627.0 / self.specific_thermal_energy_demand
        if gor_implied < 1.0:
            warnings.warn(
                f"Implied GOR ({gor_implied:.2f}) is below 1.0 — "
                "specific_thermal_energy_demand may be too high. Check units (expected kWh_th/m³).",
                UserWarning,
            )