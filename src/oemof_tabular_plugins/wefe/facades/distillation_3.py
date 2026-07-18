import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class Distillation(MIMO):
    """
    Literature-informed thermal distillation facade based on MIMO.

    Purpose
    -------
    Generic thermally driven distillation facade for electrically heated
    water treatment units used in off-grid and rural water supply systems.
    The model is designed as a bookkeeping/process-yield unit representing
    a thermal distillation unit (e.g. MED, MSF, MVC-style) as a
    water-treatment intervention. It is not a full mechanistic
    stage-by-stage evaporation and condensation model.

    Core references
    ---------------
    1. Core thermodynamic/economic framework for thermal desalination: recovery, thermal duty, and GOR-based efficiency
       benchmarking across MSF/MED/TVC/MVC technologies.
       Al-Karaghouli, A., & Kazmerski, L. L. (2013). Energy consumption and water production cost of conventional and
       renewable-energy-powered desalination processes. Renewable and Sustainable Energy Reviews, 24, 343-356.
       https://doi.org/10.1016/j.rser.2012.12.064
    2. Electrification of distillation processes — justifies modeling electricity as an upstream carrier for useful
       process heat, and separating heater efficiency from process thermal demand.
       Sheng, M., Guo, Y., Lee, B., Epsztein, R., Wang, Z., & Wang, L. (2025). Electrified desalination processes: Where
       we are and where to go from performance and economic perspectives. Desalination, 600, 118486.
       https://doi.org/10.1016/j.desal.2024.118486
    3. Measured electricity intensity for electrically driven thermal desalination plants (intake through brine disposal,
       ~7-14 MJ/m³ / 2-4 kWh/m³)
       International Energy Agency. (2026). Wired for water: How electrification is transforming desalination. IEA Commentary.
       https://www.iea.org/commentaries/wired-for-water-how-electrification-is-transforming-desalination
    4. Global brine production and disposal-practice data.
       Jones, E., Qadir, M., van Vliet, M. T. H., Smakhtin, V., & Kang, S. (2019). The state of desalination and brine
       production: A global outlook. Science of the Total Environment, 657, 1343-1356.
       https://doi.org/10.1016/j.scitotenv.2018.12.076
    5. Brine management technology review — supports max_brine_concentration_factor as a design upper bound and reinforces
       disposal-cost framing.
       Bello, A. S., Zouari, N., Da'ana, D. A., Hahladakis, J. N., & Al-Ghouti, M. A. (2021). An overview of brine management:
       Emerging desalination technologies, life cycle assessment, and metal recovery methodologies. Journal of Environmental
       Management, 288, 112358. https://doi.org/10.1016/j.jenvman.2021.112358
    6. Waste-heat integration in thermal desalination — justifies heat_in_bus as an optional explicit thermal source enabling 
       hybrid electric-plus-thermal or waste-heat-driven operation modes.
       Charitar, D., & Madhlopa, A. (2022). Integration of waste heat in thermal desalination technologies: A review.
       Journal of Energy in Southern Africa, 33(1), 68-84. https://doi.org/10.17159/2413-3051/2022/v33i1a5434

    Main equations
    --------------
    All flows normalized to 1 m³ net distillate output (primary):

    Feedwater requirement:
        feedwater_per_output = 1 / recovery_ratio
                                            [m³_feed / m³_distillate]

    Brine / concentrate output:
        brine_per_output = 1 / recovery_ratio - 1
                                            [m³_brine / m³_distillate]

    Electric-only mode (heat_in_bus is None):
        electricity_per_output =
            specific_thermal_energy_demand / heater_efficiency
            + SEC (specific electricity auxiliaries)
                                            [kWh_el / m³_distillate]

    External heat mode (heat_in_bus is provided):
        heat_per_output        = specific_thermal_energy_demand
                                            [kWh_th / m³_distillate]
        electricity_per_output = SEC (specific electricity auxiliaries)
                                            [kWh_el / m³_distillate]

    Brine concentration factor (dimensionless, reporting only):
        CF_brine = 1 / (1 - recovery_ratio)

    GOR proxy (reporting only):
        GOR_implied = 627.0 / specific_thermal_energy_demand
        [627.0 = 2257 kJ/kg x 1000 kg/m³ / 3600 kJ/kWh]

    Notes
    -----
    - Primary flow is water_out_bus [m³]. Capacity constrains the maximum distillate output of the unit.
    - electricity_bus carries electrical energy converted to useful process heat via heater_efficiency. This decouples
      process thermodynamics from the electric heating technology type (resistance heater, heat pump, etc.).
    - When heat_in_bus is provided, it supplies the full thermal duty (specific_thermal_energy_demand) directly. electricity_bus
      then carries only auxiliary electricity SEC (specific electricity auxiliaries). heater_efficiency has no effect in this mode.
    - Engineering bounds (max_recovery_ratio, min_recovery_ratio, max_brine_concentration_factor) are implemented as Python-side
      validation checks at instantiation.
    - thermal_technology, gor_reference, sec_typical_min, and sec_typical_max are stored as documentation and calibration
      metadata only. They are not enforced as hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "distillation"
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
    electricity_bus: Bus = None         # kWh_el
    water_in_bus: Bus = None            # m³ (feedwater)
    water_out_bus: Bus = None           # m³ (distillate — PRIMARY)
    brine_out_bus: Bus = None           # m³ (concentrate)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    heat_in_bus: Optional[Bus] = None  # kWh_th — explicit thermal source

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    # reserved for future extension (e.g. waste_heat_bus, vapour_loss_bus)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    recovery_ratio: float = 0.70                            # m³_distillate / m³_feed (recovery ratio)[1]
    specific_thermal_energy_demand: float = 14.0            # kWh_th / m³ distillate [1]
    heater_efficiency: float = 0.95                         # kWh_th / kWh_el  (0, 1] [2]
    specific_energy_consumption: float = 0.8                # kWh_el / m³ distillate (specific electricity auxiliaries) [2, 3]
    max_recovery_ratio: Optional[float] = None              # design upper bound (validation only) [1]
    max_brine_concentration_factor: Optional[float] = None  # design upper bound (validation only) [4, 5]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0          # USD/m³ distillate
    carrier_cost: float = 0.0           # USD/m³ feedwater
    brine_disposal_cost: float = 0.0    # USD/m³ brine

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
    thermal_technology: str = "electric_evaporator"     # e.g. MVC_like, waste_heat, solar_thermal [1, 2]
    gor_reference: Optional[float] = None               # gained output ratio reference value [1]
    min_recovery_ratio: Optional[float] = None          # technology lower bound (validation only) [1]
    sec_typical_min: float = 5.0                        # kWh_th/m³, lower bound from literature [1, 3]
    sec_typical_max: float = 20.0                       # kWh_th/m³, upper bound from literature [1, 3]

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

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.recovery_ratio = attributes.pop(
            "recovery_ratio",self.recovery_ratio
        )
        self.specific_thermal_energy_demand = attributes.pop(
            "specific_thermal_energy_demand",
            self.specific_thermal_energy_demand
        )
        self.heater_efficiency = attributes.pop(
            "heater_efficiency", self.heater_efficiency
        )
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption",
            self.specific_energy_consumption,
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
        self.brine_disposal_cost = attributes.pop(
            "brine_disposal_cost", self.brine_disposal_cost
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
        self.output_parameters = attributes.pop("output_parameters", {})

        # --------------------------------------------------------------
        # documentation / calibration defaults
        # --------------------------------------------------------------
        self.thermal_technology = attributes.pop(
            "thermal_technology", self.thermal_technology
        )
        self.gor_reference = attributes.pop("gor_reference", self.gor_reference)
        self.min_recovery_ratio = attributes.pop(
            "min_recovery_ratio", self.min_recovery_ratio
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
        self._feedwater_per_output = 1.0 / self.recovery_ratio
        self._brine_per_output = self._feedwater_per_output - 1.0
        self._brine_concentration_factor = 1.0 / (1.0 - self.recovery_ratio)
        self._gor_implied = 627.0 / self.specific_thermal_energy_demand

        if self.heat_in_bus is not None:
            self._electricity_per_output = self.specific_energy_consumption
            self._heat_per_output = self.specific_thermal_energy_demand
        else:
            self._electricity_per_output = (
                    self.specific_thermal_energy_demand / self.heater_efficiency
                    + self.specific_energy_consumption
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

        if self.water_out_bus in self.outputs:
            out_flow = self.outputs[self.water_out_bus]
            out_flow.variable_costs = sequence(self.marginal_cost)
            if not self.expandable and self.capacity is not None:
                out_flow.nominal_value = self.capacity
            custom_attrs = (getattr(self, "output_parameters", None) or {}).get(
                "custom_attributes"
            )
            if custom_attrs:
                for attribute, value in custom_attrs.items():
                    setattr(out_flow, attribute, value)

        if self.brine_out_bus in self.outputs:
            self.outputs[self.brine_out_bus].variable_costs = sequence(
                self.brine_disposal_cost
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
        for bus in []:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.recovery_ratio < 1:
            raise ValueError("recovery_ratio must be in (0, 1).")

        if not 0 < self.heater_efficiency <= 1:
            raise ValueError("heater_efficiency must be in (0, 1].")

        bounded_nonneg = {
            "specific_thermal_energy_demand": self.specific_thermal_energy_demand,
            "specific_energy_consumption": self.specific_energy_consumption,
            "carrier_cost": self.carrier_cost,
            "marginal_cost": self.marginal_cost,
            "brine_disposal_cost": self.brine_disposal_cost,
        }
        for name, value in bounded_nonneg.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")

        if self.heat_in_bus is not None:
            warnings.warn(
                "heat_in_bus is set — heater_efficiency has no effect in this mode. "
                "Thermal duty is supplied via heat_in_bus.",
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

        # GOR sanity check — 627.0 = 2257 kJ/kg × 1000 kg/m³ / 3600 kJ/kWh
        gor_implied = 627.0 / self.specific_thermal_energy_demand
        if gor_implied < 1.0:
            warnings.warn(
                f"Implied GOR ({gor_implied:.2f}) is below 1.0 — "
                f"specific_thermal_energy_demand may be too high. Check units (expected kWh_th/m³).",
                UserWarning,
            )