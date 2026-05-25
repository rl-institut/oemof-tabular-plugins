import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class UVDisinfection(MIMO):
    """
    Literature-informed UV disinfection facade based on MIMO.

    Purpose
    -------
    Generic lamp-based photochemical disinfection facade for UV reactor units
    used in water treatment systems. The model is designed as a
    validated-operating-envelope unit representing a UV reactor as a
    water-treatment intervention for pathogen inactivation. It is not a full
    mechanistic UV reactor or fluence-rate distribution model.

    Core references
    ---------------
    1. U.S. EPA — Ultraviolet Disinfection Guidance Manual (2006):
       validated operating conditions, UV dose, UVT sensitivity, fouling/aging
       factor derivation, dose-monitoring strategy, and off-spec operation logic.
    2. Irish EPA — Water Treatment Manual: Disinfection:
       practical UV design, dose validation, operation, and monitoring guidance
       for real plant engineering; sludge/waste stream handling.
    3. NWRI — Ultraviolet Disinfection Guidelines for Drinking Water and
       Water Reuse: bridges potable and reuse-oriented UV operating envelopes;
       useful for mixed WEFE / non-potable contexts.
    4. AWWA — Ultraviolet (UV) Disinfection for Water Treatment, 2nd ed.
       (support source): professional engineering practice, maintenance cost
       ranges, and lamp replacement guidance.

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water output (primary):

    Water continuity:
        Q_in(t) = Q_out(t)                                [m³/hr]

    Net specific energy consumption [kWh / m³ treated water]:
        SEC_eff = SEC_base * (1 / fouling_aging_factor)   (EPA, 2006)

    Optional sludge output (time-averaged over cleaning cycles):
        sludge_per_output = sludge_generation_rate        [kg / m³ treated water]

    Validated throughput bound:
        Q_out(t) <= validated_max_flow * reactor_availability

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum
      treated water output of the unit.
    - fouling_aging_factor captures the combined effect of lamp aging and
      sleeve fouling as a linear SEC derating proxy (EPA, 2006). Values in
      (0, 1] — lower values imply worse lamp/sleeve performance and result
      in higher effective SEC. At fouling_aging_factor=1.0 (new lamp, clean
      sleeve), SEC_eff equals SEC_base exactly.
    - reactor_availability captures scheduled downtime, lamp replacement
      periods, and maintenance windows as a fractional throughput derating
      (Irish EPA). Values in (0, 1] — lower values reduce the effective
      validated throughput ceiling.
    - uv_sludge_bus is an optional output bus representing the solid/liquid
      waste stream from sleeve cleaning, lamp replacement, and reactor purging
      (Irish EPA; EPA, 2006). sludge_generation_rate is a time-averaged
      coefficient [kg / m³ treated water]. If the bus is omitted, sludge waste
      is not tracked in the model.
    - maintenance_cost and lamp_replacement_cost are applied as variable costs
      on the primary output flow via output_parameters, consistent with the
      RO facade pattern for per-output-bus cost assignment.
    - For backward compatibility, validated_max_flow=None disables the
      throughput bound entirely; the optimizer is free to dispatch up to
      the investment capacity.
    - Water-quality surrogates (uv_dose_target, uv_transmittance,
      target_pathogen, log_removal_target) are stored as documentation /
      calibration defaults. They are not enforced as hard optimization
      constraints in v3.0.
    - validated_min_uvt triggers ValueError at instantiation if
      uv_transmittance falls below it, acting as a design-envelope guard.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "uv_disinfection"
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
    water_out_bus: Bus = None           # m³

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    uv_sludge_bus: Optional[Bus] = None  # kg/hr; sleeve cleaning / lamp waste
                                         # (Irish EPA; EPA UV Manual, 2006)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.06       # kWh / m³ treated water
    fouling_aging_factor: float = 1.0               # fraction in (0, 1]; SEC derating proxy (EPA, 2006)
    reactor_availability: float = 1.0               # fraction in (0, 1]; throughput derating (Irish EPA)
    validated_max_flow: Optional[float] = None                # m³ / timestep; validated operating bound (EPA, 2006)
    sludge_generation_rate: float = 0.0             # kg sludge / m³ treated water (Irish EPA; EPA, 2006)

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                  # €/m³ treated water
    carrier_cost: float = 0.0                   # €/kWh electricity
    maintenance_cost: float = 0.0               # €/m³ treated water (Irish EPA, AWWA)
    lamp_replacement_cost: float = 0.0          # €/m³ treated water (Irish EPA, AWWA)

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # EPA UV Guidance Manual / NWRI style characterization fields
    # ------------------------------------------------------------------
    uv_dose_target: float = 40.0                # mJ/cm²; design basis (EPA, 2006; NWRI)
    uv_transmittance: float = 0.95              # fraction; key water-quality driver (EPA, 2006)
    validated_min_uvt: Optional[float] = None   # fraction; screening threshold only (EPA, 2006)
    target_pathogen: str = ""                   # e.g. "Cryptosporidium", "E. coli" (EPA, NWRI)
    log_removal_target: Optional[float] = None  # log-reduction design basis (EPA, NWRI)
    validation_method: str = ""                 # e.g. "validated dose", "setpoint" (EPA, 2006)
    uv_sensor_setpoint: Optional[float] = None  # mW/cm²; operational monitoring only (EPA, 2006)

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
        self.uv_sludge_bus = attributes.pop("uv_sludge_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.fouling_aging_factor = attributes.pop(
            "fouling_aging_factor", self.fouling_aging_factor
        )
        self.reactor_availability = attributes.pop(
            "reactor_availability", self.reactor_availability
        )
        self.validated_max_flow = attributes.pop(
            "validated_max_flow", self.validated_max_flow
        )
        self.sludge_generation_rate = attributes.pop(
            "sludge_generation_rate", self.sludge_generation_rate
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.maintenance_cost = attributes.pop(
            "maintenance_cost", self.maintenance_cost
        )
        self.lamp_replacement_cost = attributes.pop(
            "lamp_replacement_cost", self.lamp_replacement_cost
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
        self.uv_dose_target = attributes.pop(
            "uv_dose_target", self.uv_dose_target
        )
        self.uv_transmittance = attributes.pop(
            "uv_transmittance", self.uv_transmittance
        )
        self.validated_min_uvt = attributes.pop(
            "validated_min_uvt", self.validated_min_uvt
        )
        self.target_pathogen = attributes.pop(
            "target_pathogen", self.target_pathogen
        )
        self.log_removal_target = attributes.pop(
            "log_removal_target", self.log_removal_target
        )
        self.validation_method = attributes.pop(
            "validation_method", self.validation_method
        )
        self.uv_sensor_setpoint = attributes.pop(
            "uv_sensor_setpoint", self.uv_sensor_setpoint
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # (EPA UV Guidance Manual, 2006; Irish EPA; AWWA UV Disinfection)
        # --------------------------------------------------------------
        self._sec_effective = (
                self.specific_energy_consumption
                * (1.0 / self.fouling_aging_factor)
        )

        # --------------------------------------------------------------
        # conversion factors
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._sec_effective
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.uv_sludge_bus is not None:
            attributes[f"conversion_factor_{self.uv_sludge_bus.label}"] = sequence(
                self.sludge_generation_rate
            )

        # --------------------------------------------------------------
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # --------------------------------------------------------------
        attributes.setdefault("output_parameters", {})

        if self.maintenance_cost > 0 or self.lamp_replacement_cost > 0:
            attributes["output_parameters"].update(
                {"variable_costs": self.maintenance_cost + self.lamp_replacement_cost}
            )

        # --------------------------------------------------------------
        # validated throughput bound  (EPA UV Guidance Manual, 2006)
        # Q_out(t) <= validated_max_flow * reactor_availability
        # Enforced as activity_bound_max on the output group in MIMO block.
        # --------------------------------------------------------------
        if self.validated_max_flow is not None:
            attributes["activity_bound_max"] = sequence(
                self.validated_max_flow * self.reactor_availability
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
            self.uv_sludge_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if self.specific_energy_consumption is None or self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        if self.fouling_aging_factor is None or self.fouling_aging_factor <= 0:
            raise ValueError("fouling_aging_factor must be > 0.")

        if self.reactor_availability is None or not (0 < self.reactor_availability <= 1):
            raise ValueError("reactor_availability must be in the interval (0, 1].")

        if self.uv_transmittance is None or not (0 < self.uv_transmittance <= 1):
            raise ValueError("uv_transmittance must be in the interval (0, 1].")

        if self.validated_min_uvt is not None and not (0 < self.validated_min_uvt <= 1):
            raise ValueError("validated_min_uvt must be in the interval (0, 1].")

        if self.validated_max_flow is not None and self.validated_max_flow < 0:
            raise ValueError("validated_max_flow must be >= 0.")

        if self.capacity is not None and self.capacity < 0:
            raise ValueError("capacity must be >= 0.")

        if self.sludge_generation_rate < 0:
            raise ValueError("sludge_generation_rate must be >= 0.")

        if self.uv_sludge_bus is not None and self.sludge_generation_rate == 0.0:
            warnings.warn(
                "uv_sludge_bus is provided but sludge_generation_rate is 0. "
                "No sludge will be generated. Set sludge_generation_rate > 0 "
                "to track the sludge stream.",
                UserWarning,
            )

        if (
                self.validated_min_uvt is not None
                and self.uv_transmittance < self.validated_min_uvt
        ):
            raise ValueError(
                "uv_transmittance is below validated_min_uvt. "
                "This is outside the documented validated UV operating envelope."
            )