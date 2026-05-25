import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class ActivatedCarbonFilter(MIMO):
    """
    Literature-informed activated carbon filter facade based on MIMO.

    Purpose
    -------
    Linearized process surrogate for granular activated carbon (GAC) water
    treatment. Represents the unit as a throughput/bookkeeping component
    relating electricity and raw water inputs to treated water output, with
    optional explicit waste, backwash, and carbon consumption streams.
    It is not a full adsorption-kinetics or breakthrough model.

    Core references
    ---------------
    1. US EPA Process Design Manual: Carbon Adsorption — design basis for
       EBCT, pretreatment requirements, regeneration/replacement costs, and
       plant configuration choices.
    2. Benstoem et al. (2017) — meta-analysis of pilot- and large-scale GAC
       performance; justifies surrogate modeling over universal breakthrough
       equations (Chemosphere).
    3. Bäthe et al. (2024) — continuous GAC filter operation; supports
       backwash-related parameters and prefiltration dependence (Water Sci. Tech.).
    4. Saialy et al. (2023) — backwashing of granular media filters; supports
       backwash water and energy burden parameters (J. Water Supply).

    Main equations
    --------------
    All conversion factors are normalized to treated water output = 1.

    Electricity demand:
        f_el(t) = (SEC + SEC_bw) * f_water_out(t)

    Raw water requirement:
        f_water_in(t) = (1 / eta) * f_water_out(t)

    Explicit hydraulic loss (wastewater_out_bus, if provided):
        f_waste(t) = (1/eta - 1) * f_water_out(t)

    Backwash water discharge (backwash_water_bus, if provided):
        f_bw(t) = beta_bw * solids_factor * f_water_out(t)

    Carbon consumption (carbon_bus, if provided):
        f_carbon(t) = s_carbon * pretreatment_factor * solids_factor * f_water_out(t)

    Notes
    -----
    - Primary flow is water_out_bus [m3/hr]. Capacity constrains the maximum
      treated-water throughput of the unit.
    - If wastewater_out_bus is not provided, hydraulic loss remains implicit
      in the water_in_bus / water_out_bus ratio via efficiency.
    - If carbon_bus is not provided, carbon burden can be folded into
      marginal_cost externally.
    - pretreatment_factor and solids_factor are multipliers modifying carbon
      demand and backwash burden; they are not standalone constraints.
    - Detailed adsorption breakthrough and contaminant-specific removal are
      deferred to v4.0 because GAC performance is highly context-specific.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "activated_carbon_filter"
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
    electricity_bus: Bus = None         # kWh/hr
    water_in_bus: Bus = None            # m³/hr  (feed water)
    water_out_bus: Bus = None           # m³/hr  (treated water)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    carbon_bus: Optional[Bus] = None    # kg/hr  activated carbon consumption

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    wastewater_out_bus: Optional[Bus] = None    # m³/hr  hydraulic loss / reject
    backwash_water_bus: Optional[Bus] = None    # m³/hr  backwash discharge

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.07       # kWh/m³ treated water
    efficiency: float = 0.97                        # treated water / feed water  [–]
    specific_backwash_energy: float = 0.0           # additional kWh/m³ treated
    backwash_water_share: float = 0.0               # m³ backwash /m³ treated   [–]
    specific_carbon_demand: float = 0.0             # kg carbon /m³ treated
    pretreatment_factor: float = 1.0                # multiplier on carbon demand  [–]
    solids_factor: float = 1.0                      # multiplier on backwash/carbon burden [–]

    # ------------------------------------------------------------------
    # optional activity bounds
    # ------------------------------------------------------------------
    activity_bound_max: Union[float, Sequence[float]] = None # upper throughput cap on water_out_bus [m3/hr];
                                                             # use to encode EBCT-derived or operator limits

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0          # €/m³ treated water
    carrier_cost: float = 0.0           # €/kWh electricity
    carbon_disposal_cost: float = 0.0   # €/kg spent carbon (wastewater_out_bus side)

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # ------------------------------------------------------------------
    min_ebct_minutes: float = None      # min. empty bed contact time [min]; use offline to derive activity_bound_max
    target_contaminant: str = ""        # target pollutant label; scenario documentation only
    influent_doc: float = None          # influent dissolved organic carbon [mg/L]; affects carbon exhaustion rate
    influent_turbidity: float = None    # influent turbidity [NTU]; indicates pre-filtration requirement

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
        self.carbon_bus = attributes.pop("carbon_bus", None)
        self.wastewater_out_bus = attributes.pop("wastewater_out_bus", None)
        self.backwash_water_bus = attributes.pop("backwash_water_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop("efficiency", self.efficiency)
        self.specific_backwash_energy = attributes.pop(
            "specific_backwash_energy", self.specific_backwash_energy
        )
        self.backwash_water_share = attributes.pop(
            "backwash_water_share", self.backwash_water_share
        )
        self.specific_carbon_demand = attributes.pop(
            "specific_carbon_demand", self.specific_carbon_demand
        )
        self.pretreatment_factor = attributes.pop(
            "pretreatment_factor", self.pretreatment_factor
        )
        self.solids_factor = attributes.pop("solids_factor", self.solids_factor)

        # --------------------------------------------------------------
        # activity bounds
        # --------------------------------------------------------------
        self.activity_bound_max = attributes.pop(
            "activity_bound_max", self.activity_bound_max
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.carbon_disposal_cost = attributes.pop(
            "carbon_disposal_cost", self.carbon_disposal_cost
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
        self.min_ebct_minutes = attributes.pop(
            "min_ebct_minutes", self.min_ebct_minutes
        )
        self.target_contaminant = attributes.pop(
            "target_contaminant", self.target_contaminant
        )
        self.influent_doc = attributes.pop("influent_doc", self.influent_doc)
        self.influent_turbidity = attributes.pop(
            "influent_turbidity", self.influent_turbidity
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._electricity_per_output = (
                self.specific_energy_consumption + self.specific_backwash_energy
        )
        self._water_in_per_output = 1.0 / self.efficiency
        self._waste_per_output = max(0.0, (1.0 / self.efficiency) - 1.0)
        self._backwash_per_output = self.backwash_water_share * self.solids_factor
        self._carbon_per_output = (
                self.specific_carbon_demand
                * self.pretreatment_factor
                * self.solids_factor
        )

        # --------------------------------------------------------------
        # conversion factors
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._electricity_per_output
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._water_in_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.carbon_bus is not None:
            attributes[f"conversion_factor_{self.carbon_bus.label}"] = sequence(
                self._carbon_per_output
            )
        if self.wastewater_out_bus is not None:
            attributes[f"conversion_factor_{self.wastewater_out_bus.label}"] = sequence(
                self._waste_per_output
            )
        if self.backwash_water_bus is not None:
            attributes[f"conversion_factor_{self.backwash_water_bus.label}"] = sequence(
                self._backwash_per_output
            )

        # --------------------------------------------------------------
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # --------------------------------------------------------------
        output_parameters = attributes.pop("output_parameters", {})
        wastewater_output_parameters = attributes.pop("wastewater_output_parameters", {})
        backwash_output_parameters = attributes.pop("backwash_output_parameters", {})

        if (
                self.wastewater_out_bus is None
                and self.carbon_disposal_cost not in (None, 0, 0.0)
        ):
            output_parameters.setdefault(
                "variable_costs", self.carbon_disposal_cost
            )

        attributes["output_parameters"] = output_parameters

        if self.wastewater_out_bus is not None:
            wastewater_output_parameters.setdefault(
                "variable_costs", self.carbon_disposal_cost
            )
            attributes["wastewater_output_parameters"] = wastewater_output_parameters

        if self.backwash_water_bus is not None:
            attributes["backwash_output_parameters"] = backwash_output_parameters

        # --------------------------------------------------------------
        # activity bounds — passed through to base MIMO as constraints
        # --------------------------------------------------------------
        if self.activity_bound_max is not None:
            attributes["activity_bound_max"] = sequence(self.activity_bound_max)

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
            self.carbon_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.wastewater_out_bus,
            self.backwash_water_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.efficiency <= 1:
            raise ValueError(f"efficiency must be in (0, 1], got {self.efficiency!r}.")
        if self.specific_energy_consumption < 0:
            raise ValueError(f"specific_energy_consumption must be >= 0, got {self.specific_energy_consumption!r}.")
        if self.specific_backwash_energy < 0:
            raise ValueError(f"specific_backwash_energy must be >= 0, got {self.specific_backwash_energy!r}.")
        if self.specific_carbon_demand < 0:
            raise ValueError(f"specific_carbon_demand must be >= 0, got {self.specific_carbon_demand!r}.")

        for name, value in (
                ("backwash_water_share", self.backwash_water_share),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}.")

        if self.pretreatment_factor <= 0:
            raise ValueError(f"pretreatment_factor must be > 0, got {self.pretreatment_factor!r}.")
        if self.solids_factor <= 0:
            raise ValueError(f"solids_factor must be > 0, got {self.solids_factor!r}.")

        # warn if carbon demand is set but no carbon bus is provided
        if self.carbon_bus is None and self.specific_carbon_demand > 0:
            warnings.warn(
                f"specific_carbon_demand={self.specific_carbon_demand!r} is set but "
                "no carbon_bus is provided. Carbon burden will not be represented "
                "explicitly in the optimization.",
                UserWarning,
            )

        # warn if EBCT is given without a concrete throughput cap
        if self.min_ebct_minutes is not None and self.activity_bound_max is None:
            warnings.warn(
                f"min_ebct_minutes={self.min_ebct_minutes!r} is set as documentation "
                "metadata. To enforce a hydraulic throughput cap, convert offline: "
                "Q_max = V_bed / EBCT and pass via activity_bound_max if needed.",
                UserWarning,
            )