import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class Ozonation(MIMO):
    """
    Literature-informed ozonation facade based on MIMO.

    Purpose
    -------
    Generic ozone-based water/wastewater treatment facade for ozonation
    units used in water treatment systems. The model is designed as a
    bookkeeping/process-yield unit representing an ozonation contactor
    as a water-treatment intervention. It is not a full mechanistic
    ozone-reaction or mass-transfer model.

    Core references
    ---------------
    1. von Sonntag & von Gunten (2012): ozone chemistry, kinetics,
       disinfection, micropollutant transformation, energy requirements,
       and by-product (bromate) context.
    2. Rakness (2011): process design, operating parameters, transfer
       efficiency, contactor realism, specific energy per kg ozone,
       and operational optimization.
    3. Lazarova et al. (2013): wastewater-oriented design variables,
       transferred dose and residual logic, dose-mode justification,
       and water-quality-driven dose adaptation.

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water output (primary):

    Water balance:
        water_in_per_output = 1.0                   [m³_feed / m³_treated]

    Effective applied ozone dose — fixed mode:
        d_O3_app = applied_ozone_dose_mg_per_l      [mg/L]

    Effective applied ozone dose — demand_residual mode:
        d_O3_app = (ozone_demand_mg_per_l + target_residual_ozone_mg_per_l)
                   / transfer_efficiency            [mg/L]

    Specific ozone consumption:
        q_O3 = d_O3_app * 1e-6                     [kg_O3 / m³_treated]

    Net electricity demand per m³ treated water:
        e_m3 = q_O3 * specific_energy_per_kg_ozone  [kWh / m³_treated]

    Off-gas output (optional, reporting + cost only):
        off_gas_per_output = 1.0 - transfer_efficiency  [m³ unit / m³_treated]

    Transferred ozone dose (reporting only):
        d_O3_trans = d_O3_app * transfer_efficiency  [mg/L]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the
      maximum treated water throughput of the ozonation unit.
    - Raw water inflow is 1:1 with treated water outflow (mass balance).
    - Electricity represents ozone generation energy via electrolysis or
      corona discharge (Rakness, 2011). The ozone generation step is
      implicit — ozone is not modeled as an explicit intermediate flow
      in v3.0. A future version may introduce an ozone_bus to represent
      the generator-contactor split explicitly.
    - Two dose modes are supported: "fixed" (user-specified applied dose)
      and "demand_residual" (dose derived from water-quality demand,
      residual target, and transfer efficiency).
    - off_gas_bus is an optional output representing the residual ozone
      off-gas requiring destruction or venting treatment. Off-gas is
      produced when transfer_efficiency < 1.0. The off_gas_disposal_cost
      is a time-averaged cost coefficient (€/unit off-gas). If the bus
      is omitted, off-gas is implicitly absorbed into the system boundary
      with no explicit cost or flow tracking.
    - contact_time_min and reactor_volume_m3 are design/sizing checks
      only; they are not enforced as optimization constraints in v3.0.
    - bromide_mg_per_l triggers a bromate risk flag stored in metadata
      but does not add a bromate sub-model in v3.0.
    - For backward compatibility, "ozone_dose" may be passed as an alias
      for "applied_ozone_dose_mg_per_l" when the preferred name is not
      explicitly provided.
    - Derived design quantities (transferred dose, specific ozone
      consumption, electricity intensity) are stored as custom_attributes
      on water_out_bus for reporting and post-processing.
    - Characterization values (contact time, reactor volume, bromide)
      are stored as documentation/calibration defaults. They are not
      enforced as hard optimization constraints in v3.0.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "ozonation"
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
    electricity_bus: Bus = None             # kWh
    water_in_bus: Bus = None                # m³ (raw / pre-treated influent)
    water_out_bus: Bus = None               # m³ (ozone-treated effluent — PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension (e.g. oxygen feed, carrier gas)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    off_gas_bus: Optional[Bus] = None       # m³ — residual ozone off-gas

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_per_kg_ozone: float = 12.0              # kWh/kg ozone generated
    ozone_dose_mode: str = "fixed"                          # "fixed" | "demand_residual"
    applied_ozone_dose_mg_per_l: float = 1.0                # mg/L; required for mode="fixed"
    ozone_demand_mg_per_l: float = 0.0                      # mg/L; water-quality ozone demand
    target_residual_ozone_mg_per_l: float = 0.0             # mg/L; design residual setpoint
    transfer_efficiency: float = 1.0                        # dimensionless [0, 1]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # €/m³ treated water
    carrier_cost: float = 0.0               # €/kWh electricity
    off_gas_disposal_cost: float = 0.0      # €/unit off-gas (destruction/venting treatment)

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # Rakness (2011) / Lazarova et al. (2013) style design fields
    # ------------------------------------------------------------------
    contact_time_min: Optional[float] = None    # min; contactor sizing check only
    reactor_volume_m3: Optional[float] = None   # m³; contactor sizing check only
    bromide_mg_per_l: Optional[float] = None    # mg/L; triggers bromate risk flag
    sec_typical_min: float = 0.05               # kWh/m³; lower bound from literature
    sec_typical_max: float = 0.15               # kWh/m³; upper bound from literature
    water_quality_note: str = ""                # free-text scenario annotation

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
        self.off_gas_bus = attributes.pop("off_gas_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_per_kg_ozone = attributes.pop(
            "specific_energy_per_kg_ozone", self.specific_energy_per_kg_ozone
        )
        self.ozone_dose_mode = attributes.pop(
            "ozone_dose_mode", self.ozone_dose_mode
        )
        self.applied_ozone_dose_mg_per_l = attributes.pop(
            "applied_ozone_dose_mg_per_l",self.applied_ozone_dose_mg_per_l
        )
        self.ozone_demand_mg_per_l = attributes.pop(
            "ozone_demand_mg_per_l", self.ozone_demand_mg_per_l
        )
        self.target_residual_ozone_mg_per_l = attributes.pop(
            "target_residual_ozone_mg_per_l", self.target_residual_ozone_mg_per_l
        )
        self.transfer_efficiency = attributes.pop(
            "transfer_efficiency", self.transfer_efficiency
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.off_gas_disposal_cost = attributes.pop(
            "off_gas_disposal_cost", self.off_gas_disposal_cost
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
        self.contact_time_min = attributes.pop(
            "contact_time_min", self.contact_time_min
        )
        self.reactor_volume_m3 = attributes.pop(
            "reactor_volume_m3", self.reactor_volume_m3
        )
        self.bromide_mg_per_l = attributes.pop(
            "bromide_mg_per_l", self.bromide_mg_per_l
        )
        self.sec_typical_min = attributes.pop(
            "sec_typical_min", self.sec_typical_min
        )
        self.sec_typical_max = attributes.pop(
            "sec_typical_max", self.sec_typical_max
        )
        self.water_quality_note = attributes.pop(
            "water_quality_note", self.water_quality_note
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # (von Sonntag & von Gunten, 2012; Rakness, 2011;
        #  Lazarova et al., 2013)
        # --------------------------------------------------------------
        if self.ozone_dose_mode == "fixed":
            self._effective_applied_ozone_dose_mg_per_l = float(
                self.applied_ozone_dose_mg_per_l
            )
        else:  # demand_residual
            self._effective_applied_ozone_dose_mg_per_l = float(
                (self.ozone_demand_mg_per_l + self.target_residual_ozone_mg_per_l)
                / self.transfer_efficiency
            )

        self._transferred_ozone_dose_mg_per_l = (
                self._effective_applied_ozone_dose_mg_per_l * self.transfer_efficiency
        )
        self._specific_ozone_consumption_kg_per_m3 = (
                self._effective_applied_ozone_dose_mg_per_l * 1e-6
        )
        self._electricity_demand_per_m3 = (
                self._specific_ozone_consumption_kg_per_m3
                * self.specific_energy_per_kg_ozone
        )

        # --------------------------------------------------------------
        # conversion factors
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._electricity_demand_per_m3
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        if self.off_gas_bus is not None:
            attributes[f"conversion_factor_{self.off_gas_bus.label}"] = sequence(
                1.0 - self.transfer_efficiency
            )

        # --------------------------------------------------------------
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # --------------------------------------------------------------
        attributes.setdefault("output_parameters", {})

        if self.off_gas_bus is not None:
            attributes.setdefault("output_parameters_1", {})
            if self.off_gas_disposal_cost > 0:
                attributes["output_parameters_1"].update(
                    {"variable_costs": self.off_gas_disposal_cost}
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
        for bus in []:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.off_gas_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        allowed_modes = {"fixed", "demand_residual"}
        if self.ozone_dose_mode not in allowed_modes:
            raise ValueError(
                f"ozone_dose_mode must be one of {sorted(allowed_modes)}, got '{self.ozone_dose_mode}'."
            )

        if self.specific_energy_per_kg_ozone is None or self.specific_energy_per_kg_ozone <= 0:
            raise ValueError("specific_energy_per_kg_ozone must be > 0.")

        if not 0 < self.transfer_efficiency <= 1:
            raise ValueError("transfer_efficiency must be in (0, 1].")

        bounded_non_negative = {
            "ozone_demand_mg_per_l": self.ozone_demand_mg_per_l,
            "target_residual_ozone_mg_per_l": self.target_residual_ozone_mg_per_l,
        }
        for name, value in bounded_non_negative.items():
            if value is None or value < 0:
                raise ValueError(f"{name} must be >= 0.")

        if (
                self.applied_ozone_dose_mg_per_l is not None
                and self.applied_ozone_dose_mg_per_l < 0
        ):
            raise ValueError("applied_ozone_dose_mg_per_l must be >= 0.")

        if self.ozone_dose_mode == "fixed" and self.applied_ozone_dose_mg_per_l is None:
            raise ValueError(
                "ozone_dose_mode='fixed' requires applied_ozone_dose_mg_per_l to be set."
            )

        if self.contact_time_min is not None and self.contact_time_min <= 0:
            raise ValueError("contact_time_min must be > 0 if provided.")

        if self.reactor_volume_m3 is not None and self.reactor_volume_m3 <= 0:
            raise ValueError("reactor_volume_m3 must be > 0 if provided.")

        if self.bromide_mg_per_l is not None and self.bromide_mg_per_l < 0:
            raise ValueError("bromide_mg_per_l must be >= 0 if provided.")

        if self.off_gas_disposal_cost < 0:
            raise ValueError("off_gas_disposal_cost must be >= 0.")

        if self.off_gas_bus is not None and self.transfer_efficiency >= 1.0:
            warnings.warn(
                "off_gas_bus is set but transfer_efficiency is 1.0. "
                "No off-gas flow will be enforced.",
                UserWarning,
            )

        if self.off_gas_bus is None and self.off_gas_disposal_cost > 0:
            warnings.warn(
                "off_gas_disposal_cost is set but off_gas_bus is None. "
                "Disposal cost will have no effect.",
                UserWarning,
            )