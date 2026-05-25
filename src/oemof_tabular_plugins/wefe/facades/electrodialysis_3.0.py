import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class ElectrodialysisUnit(MIMO):
    """
    Literature-informed electrodialysis facade based on MIMO.

    Purpose
    -------
    Generic ion-exchange membrane electrodialysis facade for water desalination
    systems. The model is designed as a bookkeeping/process-yield unit
    representing an ED stack as a water-treatment intervention for brackish or
    low-salinity feed water. It is not a full electrochemical membrane-transport
    model.

    Core references
    ---------------
    1. Al-Amshawee et al. (2020): water recovery, current efficiency,
       desalination metrics, limiting current, and energy consumption framework.
    2. Ahdab & Lienhard (2021): design and optimization of ED process
       parameters: limiting current density, current efficiency, operating
       constraints, and energy use.
    3. Bdiri et al. (2018): determination of limiting current density and
       current efficiency; operating-window justification.
    4. Chehayeb et al. (2024): hypersaline ED performance trade-offs,
       salinity-dependent SEC, and concentration-performance coupling.

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Feedwater requirement:
        feedwater_per_output = 1 / water_recovery           [m³_feed / m³_product]

    Brine / concentrate output:
        brine_per_output = 1 / water_recovery - 1           [m³_brine / m³_product]

    Chemical dosing input (optional):
        chemical_per_output = chemical_dose_ratio           [m³_chem / m³_product]

    Electrode rinse output (optional):
        rinse_per_output = electrode_rinse_ratio            [m³_rinse / m³_product]

    Effective specific energy consumption
    (Al-Amshawee et al., 2020; Chehayeb et al., 2024):
        SEC_eff = (SEC_base + alpha_s * S_f) / xi           [kWh / m³_product]
        where:
            SEC_base  = specific_energy_base                (anchor energy demand)
            alpha_s   = sec_per_salinity                    (linear salinity correction)
            S_f       = feed_salinity                       (feed water salinity, g/L)
            xi        = current_efficiency                  (useful ion transport fraction)

    Effective water recovery (Al-Amshawee et al., 2020):
        r_eff = water_recovery - water_transport_loss       [—]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum
      net treated water throughput of the unit.
    - chemical_dosing_bus is optional. When provided, chemical_dose_ratio
      defines how much dosing chemical is consumed per m³ of treated water.
      Typical use: antiscalant or acid dosing for scale prevention at higher
      recovery rates (Al-Amshawee et al., 2020).
    - electrode_rinse_out_bus is optional. When provided, electrode_rinse_ratio
      defines the electrode rinse water volume produced per m³ of treated water.
      This stream is physically distinct from brine and requires separate handling
      in the system model.
    - Limiting current and stack design parameters (stack_voltage, membrane_area,
      limiting_current_density, superficial_velocity, number_of_cell_pairs) are
      intentionally excluded from v3.0 as hard constraints. Their effects should
      be reflected through water_recovery, current_efficiency, and SEC parameters
      calibrated from literature.
    - Salinity-dependent SEC correction is optional. If feed_salinity is not
      provided, the salinity correction term is zero.
    - Characterization values (typical SEC range, limiting current density, target
      salt removal) are stored as documentation/calibration defaults. They are not
      enforced as hard optimization constraints in v3.0.
    - Each mandatory bus is its own singleton MIMO group. No explicit groups
      argument is used. This is equivalent to a standard converter when all
      group conversion factors are 1.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "electrodialysis"
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
    electricity_bus: Bus = None                 # kWh
    water_in_bus: Bus = None                    # m³
    water_out_bus: Bus = None                   # m³  (PRIMARY)
    brine_out_bus: Bus = None                   # m³

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    chemical_dosing_bus: Optional[Bus] = None   # m³  antiscalant / acid dosing

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    electrode_rinse_out_bus: Optional[Bus] = None  # m³  electrode rinse wastewater

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_base: float = 0.9           # kWh / m³ net treated water
    water_recovery: float = 0.75                # m³ net treated water / m³ feedwater
    current_efficiency: float = 0.90            # fraction of applied current doing useful ion transport
    water_transport_loss: float = 0.0           # osmotic / electro-osmotic loss as fraction of feedwater
    feed_salinity: Optional[float] = None       # g/L — activates salinity correction if provided
    sec_per_salinity: float = 0.0               # kWh / (m³ · g/L) — linear salinity correction slope
    chemical_dose_ratio: float = 0.0            # m³ chemical / m³ treated water
    electrode_rinse_ratio: float = 0.0          # m³ electrode rinse / m³ treated water

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                  # €/m³ net treated water
    carrier_cost: float = 0.0                   # €/kWh electricity
    brine_disposal_cost: float = 0.0            # €/m³ brine
    chemical_cost: float = 0.0                  # €/m³ chemical dosed
    electrode_rinse_disposal_cost: float = 0.0  # €/m³ electrode rinse wastewater

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # Al-Amshawee et al. (2020) / Chehayeb et al. (2024) style characterization
    # ------------------------------------------------------------------
    sec_typical_min: float = 0.5                            # kWh/m³, lower bound from literature
    sec_typical_max: float = 2.5                            # kWh/m³, upper bound from literature
    target_salt_removal: Optional[float] = None             # fraction of feed TDS removed
    limiting_current_utilization: Optional[float] = None    # fraction of limiting current used
    limiting_current_density: Optional[float] = None        # A/m²
    stack_voltage: Optional[float] = None                   # V
    membrane_area: Optional[float] = None                   # m² per cell pair
    number_of_cell_pairs: Optional[int] = None              # —
    superficial_velocity: Optional[float] = None            # m/s
    feed_temperature: Optional[float] = None                # °C

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
        self.chemical_dosing_bus = attributes.pop("chemical_dosing_bus", None)
        self.electrode_rinse_out_bus = attributes.pop("electrode_rinse_out_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_base = attributes.pop(
            "specific_energy_base", self.specific_energy_base
        )
        self.water_recovery = attributes.pop("water_recovery", self.water_recovery)

        self.current_efficiency = attributes.pop(
            "current_efficiency", self.current_efficiency
        )
        self.water_transport_loss = attributes.pop(
            "water_transport_loss", self.water_transport_loss
        )
        self.feed_salinity = attributes.pop("feed_salinity", self.feed_salinity)
        self.sec_per_salinity = attributes.pop(
            "sec_per_salinity", self.sec_per_salinity
        )
        self.chemical_dose_ratio = attributes.pop(
            "chemical_dose_ratio", self.chemical_dose_ratio
        )
        self.electrode_rinse_ratio = attributes.pop(
            "electrode_rinse_ratio", self.electrode_rinse_ratio
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.brine_disposal_cost = attributes.pop(
            "brine_disposal_cost", self.brine_disposal_cost
        )
        self.chemical_cost = attributes.pop("chemical_cost", self.chemical_cost)
        self.electrode_rinse_disposal_cost = attributes.pop(
            "electrode_rinse_disposal_cost", self.electrode_rinse_disposal_cost
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
        self.sec_typical_min = attributes.pop("sec_typical_min", self.sec_typical_min)
        self.sec_typical_max = attributes.pop("sec_typical_max", self.sec_typical_max)
        self.target_salt_removal = attributes.pop(
            "target_salt_removal", self.target_salt_removal
        )
        self.limiting_current_utilization = attributes.pop(
            "limiting_current_utilization", self.limiting_current_utilization
        )
        self.limiting_current_density = attributes.pop(
            "limiting_current_density", self.limiting_current_density
        )
        self.stack_voltage = attributes.pop("stack_voltage", self.stack_voltage)
        self.membrane_area = attributes.pop("membrane_area", self.membrane_area)
        self.number_of_cell_pairs = attributes.pop(
            "number_of_cell_pairs", self.number_of_cell_pairs
        )
        self.superficial_velocity = attributes.pop(
            "superficial_velocity", self.superficial_velocity
        )
        self.feed_temperature = attributes.pop(
            "feed_temperature", self.feed_temperature
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # (Al-Amshawee et al., 2020; Chehayeb et al., 2024)
        # --------------------------------------------------------------
        self._effective_recovery = self.water_recovery - self.water_transport_loss
        salinity_sec = (
            0.0
            if self.feed_salinity is None
            else self.sec_per_salinity * self.feed_salinity
        )
        self._feedwater_per_output = 1.0 / self._effective_recovery
        self._brine_per_output = self._feedwater_per_output - 1.0
        self._electricity_per_output = (self.specific_energy_base + salinity_sec) / self.current_efficiency

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
        if self.chemical_dosing_bus is not None:
            attributes[f"conversion_factor_{self.chemical_dosing_bus.label}"] = sequence(
                max(self.chemical_dose_ratio, 1e-9)
            )
        if self.electrode_rinse_out_bus is not None:
            attributes[f"conversion_factor_{self.electrode_rinse_out_bus.label}"] = sequence(
                max(self.electrode_rinse_ratio, 1e-9)
            )

        # --------------------------------------------------------------
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # --------------------------------------------------------------
        attributes.setdefault("output_parameters", {})
        attributes.setdefault("output_parameters_1", {})
        if self.brine_disposal_cost > 0:
            attributes["output_parameters_1"].update(
                {"variable_costs": self.brine_disposal_cost}
            )
        if self.chemical_dosing_bus is not None:
            attributes.setdefault("input_parameters_1", {})
            if self.chemical_cost > 0:
                attributes["input_parameters_1"].update(
                    {"variable_costs": self.chemical_cost}
                )
        if self.electrode_rinse_out_bus is not None:
            attributes.setdefault("output_parameters_2", {})
            if self.electrode_rinse_disposal_cost > 0:
                attributes["output_parameters_2"].update(
                    {"variable_costs": self.electrode_rinse_disposal_cost}
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
            self.chemical_dosing_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.electrode_rinse_out_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.water_recovery <= 1:
            raise ValueError("water_recovery must be in (0, 1].")
        if not 0 < self.current_efficiency <= 1:
            raise ValueError("current_efficiency must be in (0, 1].")
        if self.water_transport_loss < 0:
            raise ValueError("water_transport_loss must be >= 0.")
        if self.water_transport_loss >= self.water_recovery:
            raise ValueError(
                "water_transport_loss must be strictly less than water_recovery."
            )

        bounded = {
            "specific_energy_base": self.specific_energy_base,
            "sec_per_salinity": self.sec_per_salinity,
            "brine_disposal_cost": self.brine_disposal_cost,
            "chemical_dose_ratio": self.chemical_dose_ratio,
            "chemical_cost": self.chemical_cost,
            "electrode_rinse_ratio": self.electrode_rinse_ratio,
            "electrode_rinse_disposal_cost": self.electrode_rinse_disposal_cost,
        }
        for param_name, value in bounded.items():
            if value < 0:
                raise ValueError(f"{param_name} must be >= 0.")

        if self.feed_salinity is not None and self.feed_salinity < 0:
            raise ValueError("feed_salinity must be >= 0 if provided.")

        effective_recovery = self.water_recovery - self.water_transport_loss
        if not 0 < effective_recovery <= 1:
            raise ValueError(
                "effective_recovery (water_recovery - water_transport_loss) "
                "must be in (0, 1]."
            )

        if self.chemical_dose_ratio > 0 and self.chemical_dosing_bus is None:
            warnings.warn(
                "chemical_dose_ratio > 0 but no chemical_dosing_bus provided. "
                "Chemical dosing is not tracked in the model.",
                UserWarning,
            )
        if self.electrode_rinse_ratio > 0 and self.electrode_rinse_out_bus is None:
            warnings.warn(
                "electrode_rinse_ratio > 0 but no electrode_rinse_out_bus provided. "
                "Electrode rinse water is not tracked in the model.",
                UserWarning,
            )

        if self.current_efficiency < 0.7:
            warnings.warn(
                "current_efficiency < 0.7 may be low for standard ED operation; "
                "please verify calibration against reference data.",
                UserWarning,
            )
        if self.water_recovery > 0.9:
            warnings.warn(
                "water_recovery > 0.9 may require careful calibration due to "
                "water transport effects, concentration polarization, and "
                "fouling/scaling risks at high recovery.",
                UserWarning,
            )