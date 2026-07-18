import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class SlowSandFilter(MIMO):
    """
    Literature-informed slow sand filter facade based on MIMO.

    Purpose
    -------
    Linear aggregate representation of a slow sand filtration unit.
    The model is designed as a bookkeeping/process-yield unit representing
    a slow sand filter as a biological and physical treatment step for
    turbidity removal, pathogen reduction, and organic-matter degradation.
    It is not a mechanistic schmutzdecke or biofilm model.

    Core references
    ---------------
    1. Operational criteria, HLR limits, filter-to-waste guidance, ripening, cleaning triggers, temperature sensitivity,
       and dissolved-oxygen considerations for slow sand filtration optimization.
       U.S. Environmental Protection Agency (EPA), Area-Wide Optimization Program (AWOP). (2024). Water quality goals and
       operational criteria for optimization of slow sand filtration (EPA 815-B-24-011). U.S. EPA Office of Ground Water &
       Drinking Water. https://www.epa.gov/system/files/documents/2024-04/water-quality-goals-and-slow-sand-filtration.pdf
    2. Startup, resanding, biological maturity (schmutzdecke development), return-to-service (filtered-to-waste) protocols,
       and minimum/maximum sand-depth thresholds. Idaho Department of Environmental Quality (DEQ). (2022). Slow sand filter
       guidance (PID DWPR, CA code 82103). Idaho DEQ. https://www2.deq.idaho.gov/admin/LEIA/api/document/download/16620
    3. Classical design basis, media specifications (d10, uniformity coefficient, fines fraction), hydraulic design, and
       underdrainage. Barrett, J. M., & Hendricks, D. W. (1991). Manual of design for slow sand filtration. AWWA and AWWA
       Research Foundation. https://www.waterrf.org/research/projects/manual-design-slow-sand-filtration
    4. Biological maturity and performance variation over time in modern slow sand / biosand filter applications.
       Maiyo, J. K., Dasika, S., & Jafvert, C. T. (2023). Slow sand filters for the 21st century: A review. International
       Journal of Environmental Research and Public Health, 20(2), 1019. https://doi.org/10.3390/ijerph20021019

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Electricity demand:
        E(t) = specific_energy_consumption * Q_out(t)

    Raw water requirement:
        Q_in(t) = Q_out(t) / efficiency

    Treated water output share (1 in steady operation; < 1 during ripening):
        cf_treated(t) = treated_water_share ∈ [0, 1]

    Waste water share (filter-to-waste fraction during ripening/startup):
        cf_waste(t) = 1 − treated_water_share

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum treated-water throughput of the filter unit
      and must be set explicitly or left to the solver when expandable=True.
    - waste_out_bus is optional. If not provided, the facade behaves as a 2-input/1-output unit and treated_water_share should equal 1.0.
    - Documentation metadata fields (sand_depth, d10, uniformity_coefficient, fines_fraction, dissolved_oxygen_in, filter_area,
      filtration_rate_max) are stored for scenario documentation and design-check warnings only. They are not enforced as
      hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "slow_sand_filter"
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
    water_in_bus: Bus = None                # m³ (raw water)
    water_out_bus: Bus = None               # m³ (treated water - PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension (e.g. backwash water, chemical dosing)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    waste_out_bus: Optional[Bus] = None     # m³ filter-to-waste / ripening loss

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.015      # kWh / m³ treated water (mostly pumping/auxiliary) [1, 4]
    efficiency: float = 0.98                        # treated / feedwater [-]         [1]
    treated_water_share: float = 1.0                # fraction of throughput yielded as treated
                                                    # water [-]; < 1.0 during ripening or
                                                    # filter-to-waste operation [1, 2]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # USD/m³ treated water
    carrier_cost: float = 0.0               # USD/m³ feed
    waste_disposal_cost: float = 0.0        # USD/m³ waste out stream (filter-to-waste discharged)

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
    sand_depth: Optional[float] = None              # min 24 in                    [1, 2]
    design_sand_depth: Optional[float] = None       # min 30 in at resanding       [2]
    d10: Optional[float] = None                     # 0.15–0.35 mm                 [3]
    uniformity_coefficient: Optional[float] = None  # 1.5–3.0                      [3]
    fines_fraction: Optional[float] = None          # < 0.005                      [3]
    dissolved_oxygen_in: Optional[float] = None     # > 3 mg/L                     [1]
    filter_area: Optional[float] = None             # m² plan area                 [3]
    filtration_rate_max: Optional[float] = None     # upper HLR                    [1, 2]

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
        self.waste_out_bus = attributes.pop("waste_out_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop("efficiency", self.efficiency)
        self.treated_water_share = attributes.pop(
            "treated_water_share", self.treated_water_share
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.waste_disposal_cost = attributes.pop(
            "waste_disposal_cost", self.waste_disposal_cost
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
        self.sand_depth = attributes.pop("sand_depth", self.sand_depth)
        self.design_sand_depth = attributes.pop(
            "design_sand_depth", self.design_sand_depth
        )
        self.d10 = attributes.pop("d10", self.d10)
        self.uniformity_coefficient = attributes.pop(
            "uniformity_coefficient", self.uniformity_coefficient
        )
        self.fines_fraction = attributes.pop("fines_fraction", self.fines_fraction)
        self.dissolved_oxygen_in = attributes.pop(
            "dissolved_oxygen_in", self.dissolved_oxygen_in
        )
        self.filter_area = attributes.pop("filter_area", self.filter_area)
        self.filtration_rate_max = attributes.pop(
            "filtration_rate_max", self.filtration_rate_max
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._feedwater_per_treated_water = 1.0 / self.efficiency
        self._waste_share = 1.0 - self.treated_water_share

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_treated_water
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(
            self.treated_water_share
        )

        if self.waste_out_bus is not None:
            attributes[f"conversion_factor_{self.waste_out_bus.label}"] = sequence(
                self._waste_share
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

        if self.waste_out_bus is not None and self.waste_out_bus in self.outputs:
            self.outputs[self.waste_out_bus].variable_costs = sequence(
                self.waste_disposal_cost
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
            self.waste_out_bus
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

        if not 0.0 <= self.treated_water_share <= 1.0:
            raise ValueError(f"treated_water_share must be in [0, 1], got {self.treated_water_share!r}.")

        if self.filter_area is not None and self.filter_area <= 0:
            raise ValueError(f"filter_area must be > 0, got {self.filter_area!r}.")

        if self.filtration_rate_max is not None and self.filtration_rate_max <= 0:
            raise ValueError(f"filtration_rate_max must be > 0, got {self.filtration_rate_max!r}.")

        if (
                self.waste_out_bus is None
                and self.treated_water_share < 1.0
        ):
            warnings.warn(
                f"treated_water_share={self.treated_water_share!r} is below 1.0 "
                "but no waste_out_bus is provided. The rejected fraction will "
                "not be represented explicitly in the optimization. ",
                UserWarning,
            )

        if self.sand_depth is not None and self.sand_depth < 24:
            warnings.warn(
                f"sand_depth={self.sand_depth!r} inches is below the common "
                "minimum guidance threshold of 24 inches.",
                UserWarning,
            )

        if self.design_sand_depth is not None and self.design_sand_depth < 30:
            warnings.warn(
                f"design_sand_depth={self.design_sand_depth!r} inches is below "
                "the recommended initial/resanding depth of 30 inches.",
                UserWarning,
            )

        if self.d10 is not None and not (0.15 <= self.d10 <= 0.35):
            warnings.warn(
                f"d10={self.d10!r} mm is outside the common slow sand filter "
                "media design range of 0.15–0.35 mm.",
                UserWarning,
            )

        if (
                self.uniformity_coefficient is not None
                and not (1.5 <= self.uniformity_coefficient <= 3.0)
        ):
            warnings.warn(
                f"uniformity_coefficient={self.uniformity_coefficient!r} is "
                "outside the common guidance range of 1.5–3.0.",
                UserWarning,
            )

        if self.fines_fraction is not None and self.fines_fraction >= 0.005:
            warnings.warn(
                f"fines_fraction={self.fines_fraction!r} is at or above the "
                "recommended upper limit of 0.5% by weight.",
                UserWarning,
            )

        if self.dissolved_oxygen_in is not None and self.dissolved_oxygen_in < 3.0:
            warnings.warn(
                f"dissolved_oxygen_in={self.dissolved_oxygen_in!r} mg/L is "
                "below 3 mg/L — EPA guidance suggests DO > 3 mg/L for adequate "
                "biological activity in the filter.",
                UserWarning,
            )