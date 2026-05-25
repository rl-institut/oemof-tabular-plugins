import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class ConstructedWetland(MIMO):
    """
    Literature-informed constructed wetland facade based on MIMO.

    Purpose
    -------
    Generic water treatment facade representing a constructed wetland as a
    low-energy biological treatment unit for rural and peri-urban wastewater.
    The model is designed as a process-yield unit using fixed hydraulic
    conversion ratios. It is not a full mechanistic biological reactor model.

    Core references
    ---------------
    1. EPA Manual: Constructed Wetlands Treatment of Municipal Wastewaters —
       core process concept, wetland types, treatment mechanisms, and
       performance/modelling boundaries.
    2. EPA/NRCS Handbook of Constructed Wetlands — design-level fields and
       hydraulic design variables (HRT, HLR, hydrology, substrate, vegetation).
    3. Rahman et al. (2020): Design, Operation and Optimization of Constructed
       Wetland for Removal of Pollutant — parameter selection and validation
       logic for v3.0 fields.
    4. Meta-analysis review for pilot and large-scale CWs (2024) — evidence-
       backed design envelopes, performance ranges, and "what to leave out".

    Main equations
    --------------
    Core water recovery:
        water_in(t) = water_out(t) / recovery_rate
        [m³/hr]       [m³/hr]         [-]

    Electricity demand:
        electricity_in(t) = water_out(t) × specific_energy_consumption
        [kWh/hr]            [m³/hr]         [kWh/m³]

    Optional water loss side-stream (evapotranspiration / seepage)
        water_loss(t) = water_loss_fraction × water_out(t)

    Optional biomass side-stream (harvested macrophytes / retained solids)
        biomass_out(t) = biomass_yield_factor × water_out(t)

    Optional proxy greenhouse gas emissions (tied to treated water output)
        CH4_out(t) = ch4_emission_factor × water_out(t)
        N2O_out(t) = n2o_emission_factor × water_out(t)

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr] by default. Capacity constrains
      the maximum treated water throughput of the unit.
    - No explicit MIMO groups are defined. Each bus forms its own auto-group,
      preserving exact v1.0-style proportional coupling.
    - Hydraulic design variables (HLR, HRT, water depth, substrate, vegetation,
      wetland type) are stored as documentation/validation fields. They are not
      enforced as hard optimization constraints in v3.0.
    - specific_energy_consumption defaults to 0.07 kWh/m³ for pumped systems.
      Set to 0.0 for gravity-fed configurations.
    - Exact v1.0-equivalence holds only when all optional buses are None.
    - If an optional output bus is provided, its corresponding factor must be
      > 0. A factor of 0 with an active bus causes division-by-zero in the
      MIMO group-flow equations.
    - CH4 and N2O are modelled as proportional outputs via conversion_factor,
      coupled to the main activity chain. Their factors must be > 0 when
      the corresponding bus is active (enforced in _validate_parameters).
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "constructed_wetland"
    name: str = ""
    tech: str = "wastewater-treatment"
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
    electricity_bus: Bus = None     # kWh
    water_in_bus: Bus = None        # m³
    water_out_bus: Bus = None       # m³ (PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    water_loss_bus: Optional[Bus] = None        # m³ evapotranspiration / seepage / unrecovered water
    biomass_bus: Optional[Bus] = None           # kg harvested wetland biomass / retained solids
    ch4_bus: Optional[Bus] = None               # kg methane emissions
    n2o_bus: Optional[Bus] = None               # kg nitrous oxide emissions

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    recovery_rate: float = 0.90                 # treated water / influent water
    specific_energy_consumption: float = 0.07   # kWh / m³ treated water
    water_loss_fraction: float = 0.0            # m³ water loss / m³ treated water
    biomass_yield_factor: float = 0.0           # kg biomass / m³ treated water
    ch4_emission_factor: float = 0.0            # kgCO2e CH4 / m³ treated water
    n2o_emission_factor: float = 0.0            # kgCO2e N2O / m³ treated water

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # €/m³ treated water
    carrier_cost: float = 0.0               # €/kWh electricity

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / validation fields (not hard constraints in v3.0)
    # ------------------------------------------------------------------
    wetland_type: str = "subsurface_horizontal"         # free_water_surface | subsurface_horizontal | subsurface_vertical | hybrid
    hydraulic_loading_rate: Optional[float] = None      # m³/m²/day — design check only
    hydraulic_retention_time: Optional[float] = None    # days — design check only
    water_depth: Optional[float] = None                 # m — design check only
    substrate_type: str = ""                            # e.g. gravel, sand, zeolite
    vegetation_type: str = ""                           # e.g. Phragmites australis
    pretreatment_required: bool = True                  # safeguard against solids clogging
    climate_zone_note: str = ""                         # seasonal sensitivity documentation
    clogging_risk_note: str = ""                        # operational risk

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
        self.water_loss_bus = attributes.pop("water_loss_bus", None)
        self.biomass_bus = attributes.pop("biomass_bus", None)
        self.ch4_bus = attributes.pop("ch4_bus", None)
        self.n2o_bus = attributes.pop("n2o_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.recovery_rate = attributes.pop(
            "recovery_rate", self.recovery_rate
        )

        self.water_loss_fraction = attributes.pop(
            "water_loss_fraction", self.water_loss_fraction
        )
        self.biomass_yield_factor = attributes.pop(
            "biomass_yield_factor", self.biomass_yield_factor
        )
        self.ch4_emission_factor = attributes.pop(
            "ch4_emission_factor", self.ch4_emission_factor
        )
        self.n2o_emission_factor = attributes.pop(
            "n2o_emission_factor", self.n2o_emission_factor
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)

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
        # documentation / validation fields
        # --------------------------------------------------------------
        self.wetland_type = attributes.pop("wetland_type", self.wetland_type)
        self.hydraulic_loading_rate = attributes.pop(
            "hydraulic_loading_rate", self.hydraulic_loading_rate
        )
        self.hydraulic_retention_time = attributes.pop(
            "hydraulic_retention_time", self.hydraulic_retention_time
        )
        self.water_depth = attributes.pop("water_depth", self.water_depth)
        self.substrate_type = attributes.pop("substrate_type", self.substrate_type)
        self.vegetation_type = attributes.pop("vegetation_type", self.vegetation_type)
        self.pretreatment_required = attributes.pop(
            "pretreatment_required", self.pretreatment_required
        )
        self.climate_zone_note = attributes.pop(
            "climate_zone_note", self.climate_zone_note
        )
        self.clogging_risk_note = attributes.pop(
            "clogging_risk_note", self.clogging_risk_note
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived conversion quantities
        # feedwater ratio: m³ influent per m³ treated water output
        # --------------------------------------------------------------
        self._feedwater_per_output = 1.0 / self.recovery_rate

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.water_loss_bus is not None:
            attributes[f"conversion_factor_{self.water_loss_bus.label}"] = sequence(
                self.water_loss_fraction
            )

        if self.biomass_bus is not None:
            attributes[f"conversion_factor_{self.biomass_bus.label}"] = sequence(
                self.biomass_yield_factor
            )

        # --------------------------------------------------------------
        # optional direct emissions conversion factors
        # --------------------------------------------------------------
        if self.ch4_bus is not None:
            attributes[
                f"conversion_factor_{self.ch4_bus.label}"] = sequence(
                self.ch4_emission_factor
            )

        if self.n2o_bus is not None:
            attributes[
                f"conversion_factor_{self.n2o_bus.label}"] = sequence(
                self.n2o_emission_factor
            )

        # --------------------------------------------------------------
        # output-specific variable costs
        # --------------------------------------------------------------
        attributes.setdefault("output_parameters", {})

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
            self.water_loss_bus,
            self.biomass_bus,
            self.ch4_bus,
            self.n2o_bus,
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        valid_wetland_types = {
            "free_water_surface",
            "subsurface_horizontal",
            "subsurface_vertical",
            "hybrid",
        }
        if self.wetland_type not in valid_wetland_types:
            raise ValueError(
                f"wetland_type must be one of {sorted(valid_wetland_types)}."
            )

        if not 0 < self.recovery_rate <= 1:
            raise ValueError("recovery_rate must be in (0, 1].")

        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        if self.water_loss_bus is not None and self.water_loss_fraction <= 0:
            raise ValueError(
                "water_loss_fraction must be > 0 when water_loss_bus is provided."
            )
        if self.biomass_bus is not None and self.biomass_yield_factor <= 0:
            raise ValueError(
                "biomass_yield_factor must be > 0 when biomass_bus is provided."
            )
        if self.ch4_bus is not None and self.ch4_emission_factor <= 0:
            raise ValueError(
                "ch4_emission_factor must be > 0 when ch4_bus is provided."
            )
        if self.n2o_bus is not None and self.n2o_emission_factor <= 0:
            raise ValueError(
                "n2o_emission_factor must be > 0 when n2o_bus is provided."
            )

        for name, value in {
            "water_loss_fraction": self.water_loss_fraction,
            "biomass_yield_factor": self.biomass_yield_factor,
            "ch4_emission_factor": self.ch4_emission_factor,
            "n2o_emission_factor": self.n2o_emission_factor,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")

        if (
                self.hydraulic_loading_rate is not None
                and self.hydraulic_loading_rate <= 0
        ):
            raise ValueError("hydraulic_loading_rate must be > 0 if provided.")

        if (
                self.hydraulic_retention_time is not None
                and self.hydraulic_retention_time <= 0
        ):
            raise ValueError("hydraulic_retention_time must be > 0 if provided.")

        if self.water_depth is not None and self.water_depth <= 0:
            raise ValueError("water_depth must be > 0 if provided.")

