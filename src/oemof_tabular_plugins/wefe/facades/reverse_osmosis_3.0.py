import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class ReverseOsmosis(MIMO):
    """
    Literature-informed reverse osmosis (RO) facade based on MIMO.

    Purpose
    -------
    Generic pressure-driven membrane desalination facade for reverse osmosis
    units used in water treatment systems. The model is designed as a
    bookkeeping/process-yield unit representing an RO membrane as a
    water-treatment intervention. It is not a full mechanistic membrane
    transport model.

    Core references
    ---------------
    1. Salinas-Rodriguez, Kennedy, Schippers (2019): recovery, SEC,
       concentrate concentration, permeate quality, and concentration
       polarization equations; process-design calculation chain.
    2. DuPont FilmTec RO/NF Technical Manual (2023): operating limits,
       rejection trends, pretreatment guidance, and design interpretation.
    3. LANXESS/Lewabrane Guidelines for Design of RO Membrane Systems:
       engineering design rules, recommended flux and recovery ranges.
    4. DuPont FilmTec Design Equations Manual: SEC, net driving pressure
       (NDP), and system-design calculation chains.
    5. Carbotecnia / Morui RO CIP guidance: cleaning frequency, waste volumes,
       and CIP effluent characterization.

    Main equations
    --------------
    All flows normalized to 1 m3 net permeate output (primary):

    Feedwater requirement:
        feedwater_per_output = 1 / recovery           [m3_feed / m3_permeate]

    Brine / concentrate output:
        brine_per_output = 1/recovery - 1             [m3_brine / m3_permeate]

    Net specific energy consumption [kWh / m3 permeate]:
        net_SEC = SEC_gross * (1 - energy_recovery_efficiency)

    CIP cleaning waste (time-averaged over cleaning cycles):
        cip_waste_per_output = cleaning_waste_ratio   [m3_cip / m3_permeate]

    Concentration factor (dimensionless, reporting only):
        CF = 1 / (1 - recovery)

    Permeate TDS proxy (reporting only, requires tds_in):
        TDS_permeate = tds_in * (1 - salt_rejection)

    Brine TDS proxy (reporting only, requires tds_in):
        TDS_brine = tds_in * (1 - recovery * salt_rejection) / (1 - recovery)

    Notes
    -----
    - Primary flow is water_out_bus [m3/hr]. Capacity constrains the maximum
      permeate output of the unit.
    - Pretreatment (UF, chlorination, coagulation, etc.) is modelled as a
      separate upstream facade connected via its own bus. The RO facade
      receives already-pretreated feedwater on water_in_bus and has no
      pretreatment electricity input.
    - Energy recovery device (ERD) effects are captured through
      energy_recovery_efficiency as a fractional gross-SEC reduction.
      ERD hardware is considered internal to the RO unit boundary.
    - cleaning_waste_bus is an optional output representing the CIP
      (Clean-In-Place) effluent produced when the RO membrane is chemically
      cleaned. CIP is triggered when permeate flow drops by 10-15%,
      differential pressure rises by 15-20%, or on a scheduled cycle
      (typically every 3-6 months). The cleaning sequence consists of a
      pre-rinse, alkaline clean (pH 11-12) to remove organics/biofouling,
      acid clean (pH 2-3) to remove mineral scale, and a final quality rinse.
      cleaning_waste_ratio is a time-averaged coefficient (typically
      0.001-0.005 m3/m3 permeate for well-operated RO). If the bus is omitted,
      CIP waste is implicitly absorbed into the brine stream.
    - For backward compatibility, "efficiency" may be passed as an alias for
      "recovery", and "specific_energy_consumption" for
      "specific_energy_consumption_gross", when the preferred names are not
      explicitly provided.
    - Water-quality surrogates (estimated_permeate_tds, estimated_brine_tds,
      estimated_concentration_factor) are stored as reporting-only values.
      They are not enforced as hard optimization constraints in v3.0.
    - concentration_factor_limit and max_recovery trigger ValueError at
      instantiation if exceeded. Full osmotic-pressure and concentration-
      polarization effects are not modeled.
    - Characterization values (typical SEC range, salt_rejection, tds_in)
      are stored as documentation/calibration defaults. They are not enforced
      as hard optimization constraints in v3.0.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "reverse_osmosis"
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
    water_in_bus: Bus = None                # m³ (pretreated feedwater)
    water_out_bus: Bus = None               # m³ (permeate — PRIMARY)
    brine_out_bus: Bus = None               # m³ (concentrate)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension (e.g. antiscalant / acid dosing carrier)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    cleaning_waste_bus: Optional[Bus] = None  # m³  CIP effluent from membrane cleaning

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption_gross: float = 1.2          # kWh / m³ permeate
    recovery: float = 0.55                                  # m³ permeate / m³ feedwater
    energy_recovery_efficiency: float = 0.0                 # dimensionless [0, 1)
    salt_rejection: float = 0.99                            # dimensionless [0, 1]
    cleaning_waste_ratio: float = 0.0                       # m³ CIP effluent / m3 permeate (time-averaged)
    max_recovery: Optional[float] = None                    # design upper bound check
    concentration_factor_limit: Optional[float] = None      # design upper bound check

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                  # €/m³ permeate
    carrier_cost: float = 0.0                   # €/kWh electricity
    brine_disposal_cost: float = 0.0            # €/m³ brine
    cleaning_waste_disposal_cost: float = 0.0   # €/m³ CIP effluent

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # Salinas-Rodriguez et al. (2019) / DuPont design-equation style
    # ------------------------------------------------------------------
    tds_in: Optional[float] = None                              # kg/m³ or g/L feed TDS
    sec_typical_min: float = 0.5                                # kWh/m³, lower bound from literature
    sec_typical_max: float = 3.0                                # kWh/m³, upper bound from literature
    design_flux_lmh: Optional[float] = None                     # L/m²/hr, design flux, documentation only
    ndp_bar: Optional[float] = None                             # bar, net driving pressure, documentation only
    concentration_polarization_factor: Optional[float] = None   # dimensionless, documentation only

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
        self.cleaning_waste_bus = attributes.pop("cleaning_waste_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.recovery = attributes.pop("recovery", self.recovery)
        self.specific_energy_consumption_gross = attributes.pop(
            "specific_energy_consumption_gross",
            attributes.pop(
                "specific_energy_consumption",
                self.specific_energy_consumption_gross,
            ),
        )
        self.energy_recovery_efficiency = attributes.pop(
            "energy_recovery_efficiency", self.energy_recovery_efficiency
        )
        self.salt_rejection = attributes.pop("salt_rejection", self.salt_rejection)
        self.cleaning_waste_ratio = attributes.pop(
            "cleaning_waste_ratio", self.cleaning_waste_ratio
        )
        self.max_recovery = attributes.pop("max_recovery", self.max_recovery)
        self.concentration_factor_limit = attributes.pop(
            "concentration_factor_limit", self.concentration_factor_limit
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
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
        self.tds_in = attributes.pop("tds_in", self.tds_in)
        self.sec_typical_min = attributes.pop("sec_typical_min", self.sec_typical_min)
        self.sec_typical_max = attributes.pop("sec_typical_max", self.sec_typical_max)
        self.design_flux_lmh = attributes.pop("design_flux_lmh", self.design_flux_lmh)
        self.ndp_bar = attributes.pop("ndp_bar", self.ndp_bar)
        self.concentration_polarization_factor = attributes.pop(
            "concentration_polarization_factor", self.concentration_polarization_factor
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # (Salinas-Rodriguez et al., 2019; DuPont FilmTec Design Equations)
        # --------------------------------------------------------------
        self._feedwater_per_output = 1.0 / self.recovery
        self._brine_per_output = self._feedwater_per_output - 1.0

        self._net_specific_energy_consumption = (
                self.specific_energy_consumption_gross
                * (1.0 - self.energy_recovery_efficiency)
        )

        self._estimated_concentration_factor = 1.0 / (1.0 - self.recovery)

        if self.tds_in is not None:
            self._estimated_permeate_tds = self.tds_in * (1.0 - self.salt_rejection)
            self._estimated_brine_tds = (
                    self.tds_in
                    * (1.0 - self.recovery * self.salt_rejection)
                    / (1.0 - self.recovery)
            )
        else:
            self._estimated_permeate_tds = None
            self._estimated_brine_tds = None

        # --------------------------------------------------------------
        # conversion factors
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._net_specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.brine_out_bus.label}"] = sequence(
            self._brine_per_output
        )
        if self.cleaning_waste_bus is not None:
            attributes[
                f"conversion_factor_{self.cleaning_waste_bus.label}"
            ] = sequence(self.cleaning_waste_ratio)

        # --------------------------------------------------------------
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # --------------------------------------------------------------
        attributes.setdefault("output_parameters", {})
        attributes.setdefault("output_parameters_1", {})

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
        for bus in []:
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
        if not 0 < self.recovery < 1:
            raise ValueError("recovery must be in (0, 1).")

        if not 0 <= self.energy_recovery_efficiency < 1:
            raise ValueError("energy_recovery_efficiency must be in [0, 1).")

        if not 0 <= self.salt_rejection <= 1:
            raise ValueError("salt_rejection must be in [0, 1].")

        if self.specific_energy_consumption_gross < 0:
            raise ValueError("specific_energy_consumption_gross must be >= 0.")

        if self.cleaning_waste_ratio < 0:
            raise ValueError("cleaning_waste_ratio must be >= 0.")

        if self.max_recovery is not None and self.recovery > self.max_recovery:
            raise ValueError(
                f"Configured recovery ({self.recovery}) exceeds max_recovery ({self.max_recovery}).")

        if self.concentration_factor_limit is not None:
            cf = 1.0 / (1.0 - self.recovery)
            if cf > self.concentration_factor_limit:
                raise ValueError(f"Estimated concentration factor ({cf:.3f}) exceeds concentration_factor_limit ({self.concentration_factor_limit}).")

        if self.cleaning_waste_bus is not None and self.cleaning_waste_ratio <= 0:
            warnings.warn(
                "cleaning_waste_bus is set but cleaning_waste_ratio is 0. "
                "No CIP waste flow will be enforced.",
                UserWarning,
            )