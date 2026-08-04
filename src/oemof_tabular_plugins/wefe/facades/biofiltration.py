import dataclasses
import warnings
from typing import Sequence, Union, Optional
import numpy as np
from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class BioFiltration(MIMO):
    """
    Literature-informed biofiltration facade based on MIMO.

    Purpose
    -------
    Generic biological filtration facade for water treatment systems. The model
    is designed as a bookkeeping/process-yield unit representing a biofilter as
    a water-treatment intervention. It is not a mechanistic biofilm reactor model
    — biological kinetics, clogging dynamics, and microbial recovery after
    backwash are intentionally excluded.

    Core references
    ---------------
    1. Design fields, backwash procedure, empty bed contact time, and hydraulic loading rate guidance for drinking-water
       filtration in real plant operation.
       Environmental Protection Agency (Ireland). (2020). Water treatment manual: Filtration. Environmental Protection Agency.
       https://www.epa.ie/publications/compliance--enforcement/drinking-water/advice--guidance/EPA-Water-Filtration-Manual.pdf
    2. Full-scale manipulation of empty bed contact time (15-80 min) across four parallel biofilters, showing DOC and
       fluorescent organic-matter removal efficiency increasing with EBCT — primary basis for the EBCT geometry check and
       efficiency parameterization.
       Moona, N., Holmes, A., Wünsch, U. J., Pettersson, T. J. R., & Murphy, K. R. (2021). Full-scale manipulation of the
       empty bed contact time to optimize dissolved organic matter removal by drinking water biofilters. ACS ES&T Water,
       1(5), 1117–1126. https://doi.org/10.1021/acsestwater.0c00105
    3. Backwash water fraction conventions and recycle-stream management guidance for planning-scale models.
       U.S. Environmental Protection Agency. (2002). Filter backwash recycling rule technical guidance manual (EPA 816-R-02-014).
       U.S. EPA, Office of Water.
       https://nepis.epa.gov/Exe/ZyPDF.cgi?Dockey=200025V5.txt
    4. N/P nutrient requirements for biofilm sustenance and nutrient dose estimation.
       Rittmann, B. E., & McCarty, P. L. (2020). Environmental biotechnology: Principles and applications (2nd ed.).
       McGraw-Hill Education. https://www.accessengineeringlibrary.com/content/book/9781260441604

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Electricity demand:
        E = SEC + aeration_energy + backwash_energy    [kWh / m³_product]

    Feedwater requirement:
        no backwash_out_bus:   F = (1 / efficiency) * (1 + backwash_water_fraction)
        with backwash_out_bus: F = 1 / efficiency       [m³_feed / m³_product]

    Backwash output (only with backwash_out_bus):
        B = backwash_water_fraction                     [m³_bw / m³_product]

    Waste biomass output:
        W = biomass_waste_fraction                      [m³_biomass / m³_product]

    Nutrient flow (only with nutrient_in_bus):
        N = nutrient_dose_mg_per_L * 1e-3               [kg_nutrient / m³_product]

    Total variable cost:
        C = marginal_cost + c_nutrient
        where c_nutrient = N * nutrient_cost_per_kg     if nutrient_in_bus is None
                         = 0                             if nutrient_in_bus is provided

    EBCT geometry check (documentation-only):
        EBCT_calc = (filter_area * bed_depth * bed_porosity / capacity) * 60  [min]
        A warning is raised if EBCT_calc deviates >25 % from design_ebct.

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum net treated water throughput of the unit.
    - When backwash_out_bus is None, backwash volume is annualized into feedwater
      demand via the feedwater equation. When backwash_out_bus is provided, the
      backwash stream appears as an explicit output conversion factor.
    - When nutrient_in_bus is None, nutrient cost is captured as a scalar
      variable cost on water_out_bus. When nutrient_in_bus is provided, the
      nutrient stream is an explicit input conversion factor and no scalar cost
      is added.
    - EBCT, hydraulic_loading_rate, bed_depth, filter_area, bed_porosity, and media_type are stored as documentation/calibration
      defaults. They are not enforced as hard optimization constraints. Their effects should be reflected through efficiency,
      specific_energy_consumption, and operating cost parameters calibrated from literature.
    - Detailed fouling evolution, clogging, temperature-dependent kinetics, and dynamic biomass recovery after backwash are excluded.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "biofiltration"
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
    water_out_bus: Bus = None           # m³  (PRIMARY)
    waste_biomass_out_bus: Bus = None   # m³

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    nutrient_in_bus: Optional[Bus] = None  # kg nutrient (N/P source)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    backwash_out_bus: Optional[Bus] = None  # m³  backwash wastewater

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.12               # kWh / m³ net treated water [1]
    efficiency: float = 0.85                                # m³ net treated water / m³ feedwater [1, 2]
    biomass_waste_fraction: float = 0.005                   # m³ biomass / m³ net treated water [4]
    aeration_energy: float = 0.0                            # kWh / m³ net treated water [1]
    backwash_energy: float = 0.0                            # kWh / m³ net treated water [3]
    backwash_water_fraction: float = 0.0                    # m³ backwash / m³ net treated water [3]
    # scalar cost when nutrient_in_bus is None; conversion factor when provided.
    nutrient_dose_mg_per_L: float = 1.0                     # mg nutrient / L net treated water [4]
    nutrient_cost_per_kg: float = 1.0                       # USD/kg nutrient [4]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                              # USD/m³ net treated water
    carrier_cost: float = 0.0                               # USD/m³ feedwater
    biomass_disposal_cost: float = 0.0                      # USD/m³ waste biomass
    backwash_disposal_cost: float = 0.0                     # USD/m³ backwash wastewater

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
    design_ebct: float = None               # min — EBCT geometry check [2]
    hydraulic_loading_rate: float = None    # m/h — metadata only [1, 2]
    bed_depth: float = None                 # m — EBCT geometry check [1]
    filter_area: float = None               # m² — EBCT geometry check [1]
    bed_porosity: float = None              # void fraction of media bed [1]
    media_type: str = ""                    # e.g. "GAC", "anthracite", "sand" — descriptive only [1]
    target_contaminant: str = ""            # e.g. "DOC", "ammonia", "manganese" — descriptive only [2]
    backwash_trigger: str = ""              # e.g. "headloss", "turbidity", "time" — descriptive only [3]
    run_to_waste_bed_volumes: float = None  # bed volumes run to waste after backwash — metadata only [3]
    doc_removal_fraction: float = None      # reporting/calibration only [2]

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
        self.waste_biomass_out_bus = attributes.pop("waste_biomass_out_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.nutrient_in_bus = attributes.pop("nutrient_in_bus", None)
        self.backwash_out_bus = attributes.pop("backwash_out_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop("efficiency", self.efficiency)
        self.biomass_waste_fraction = attributes.pop(
            "biomass_waste_fraction", self.biomass_waste_fraction
        )
        self.aeration_energy = attributes.pop(
            "aeration_energy", self.aeration_energy
        )
        self.backwash_energy = attributes.pop(
            "backwash_energy", self.backwash_energy
        )
        self.backwash_water_fraction = attributes.pop(
            "backwash_water_fraction", self.backwash_water_fraction
        )
        self.nutrient_dose_mg_per_L = attributes.pop(
            "nutrient_dose_mg_per_L", self.nutrient_dose_mg_per_L
        )
        self.nutrient_cost_per_kg = attributes.pop(
            "nutrient_cost_per_kg", self.nutrient_cost_per_kg
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.biomass_disposal_cost = attributes.pop(
            "biomass_disposal_cost", self.biomass_disposal_cost
        )
        self.backwash_disposal_cost = attributes.pop(
            "backwash_disposal_cost", self.backwash_disposal_cost
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
        self.design_ebct = attributes.pop("design_ebct", self.design_ebct)
        self.hydraulic_loading_rate = attributes.pop(
            "hydraulic_loading_rate", self.hydraulic_loading_rate
        )
        self.bed_depth = attributes.pop("bed_depth", self.bed_depth)
        self.filter_area = attributes.pop("filter_area", self.filter_area)
        self.bed_porosity = attributes.pop("bed_porosity", self.bed_porosity)
        self.media_type = attributes.pop("media_type", self.media_type)
        self.target_contaminant = attributes.pop(
            "target_contaminant", self.target_contaminant
        )
        self.backwash_trigger = attributes.pop(
            "backwash_trigger", self.backwash_trigger
        )
        self.run_to_waste_bed_volumes = attributes.pop(
            "run_to_waste_bed_volumes", self.run_to_waste_bed_volumes
        )
        self.doc_removal_fraction = attributes.pop(
            "doc_removal_fraction", self.doc_removal_fraction
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # (EPA Filtration Manual, 2020 [1]; Moona et al., 2021 [2];
        # EPA Backwash Recycling Rule Guidance [3]; Rittmann & McCarty, 2020 [4])
        # --------------------------------------------------------------
        self._total_specific_electricity = (
            self.specific_energy_consumption
            + self.aeration_energy
            + self.backwash_energy
        )

        # no bus → annualize into feedwater; bus provided → explicit output only.
        if self.backwash_out_bus is None:
            self._feedwater_per_output = (
                    (1.0 / self.efficiency) * (1.0 + self.backwash_water_fraction)
            )
        else:
            self._feedwater_per_output = 1.0 / self.efficiency

        self._biomass_per_output = self.biomass_waste_fraction

        # 1 mg/L = 1 g/m³ = 0.001 kg/m³  →  factor = * 1e-3
        self._nutrient_per_output = self.nutrient_dose_mg_per_L * 1e-3  # kg/m³

        # geometry-derived EBCT consistency check (documentation-only)
        self._calculated_ebct = None
        if (
                self.filter_area is not None
                and self.bed_depth is not None
                and self.bed_porosity is not None
                and self.capacity is not None
                and self.capacity > 0
        ):
            self._calculated_ebct = (
                self.filter_area * self.bed_depth * self.bed_porosity / self.capacity
                ) * 60.0
            if self.design_ebct is not None:
                rel_dev = (
                        abs(self._calculated_ebct - self.design_ebct) / self.design_ebct
                )
                if rel_dev > 0.25:
                    warnings.warn(
                        "Calculated EBCT from geometry differs from design_ebct "
                        "by more than 25%. Check capacity, filter_area, bed_depth, "
                        "and bed_porosity for consistency.",
                        UserWarning,
                    )

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._total_specific_electricity
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.waste_biomass_out_bus.label}"] = sequence(
            self._biomass_per_output
        )
        if self.nutrient_in_bus is not None:
            attributes[f"conversion_factor_{self.nutrient_in_bus.label}"] = sequence(
                self._nutrient_per_output
            )
        if self.backwash_out_bus is not None:
            attributes[f"conversion_factor_{self.backwash_out_bus.label}"] = sequence(
                self.backwash_water_fraction
            )

        # --------------------------------------------------------------
        # output-specific variable costs
        # nutrient cost only in scalar mode (no nutrient_in_bus)
        # --------------------------------------------------------------
        self.nutrient_cost_per_m3 = 0.0
        if self.nutrient_in_bus is None:
            self.nutrient_cost_per_m3 = (
                    self._nutrient_per_output * self.nutrient_cost_per_kg
            )

        # --------------------------------------------------------------
        # primary bus label resolution
        # --------------------------------------------------------------
        if self.primary == "water_out_bus":
            primary_label = self.water_out_bus.label
        elif self.primary == "waste_biomass_out_bus":
            primary_label = self.waste_biomass_out_bus.label
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
            to_bus_1=self.waste_biomass_out_bus,
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
        total_marginal_cost = np.add(self.marginal_cost, self.nutrient_cost_per_m3)

        if self.water_out_bus in self.outputs:
            out_flow = self.outputs[self.water_out_bus]
            out_flow.variable_costs = sequence(total_marginal_cost)
            if not self.expandable and self.capacity is not None:
                out_flow.nominal_value = self.capacity
            custom_attrs = (getattr(self, "output_parameters", None) or {}).get(
                "custom_attributes"
            )
            if custom_attrs:
                for attribute, value in custom_attrs.items():
                    setattr(out_flow, attribute, value)

        if self.waste_biomass_out_bus in self.outputs:
            self.outputs[self.waste_biomass_out_bus].variable_costs = sequence(
                self.biomass_disposal_cost
            )

        if self.backwash_out_bus is not None and self.backwash_out_bus in self.outputs:
            self.outputs[self.backwash_out_bus].variable_costs = sequence(
                self.backwash_disposal_cost
            )

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 2
        # inputs
        for bus in [
            self.nutrient_in_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.backwash_out_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1].")
        if not 0 < self.biomass_waste_fraction <= 1:
            raise ValueError("biomass_waste_fraction must be in (0, 1].")
        if self.backwash_water_fraction < 0:
            raise ValueError("backwash_water_fraction must be >= 0.")
        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        non_negative = {
            "aeration_energy": self.aeration_energy,
            "backwash_energy": self.backwash_energy,
            "nutrient_dose_mg_per_L": self.nutrient_dose_mg_per_L,
            "nutrient_cost_per_kg": self.nutrient_cost_per_kg,
            "biomass_disposal_cost": self.biomass_disposal_cost,
            "backwash_disposal_cost": self.backwash_disposal_cost,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")

        if self.design_ebct is not None and self.design_ebct <= 0:
            raise ValueError("design_ebct must be > 0 if provided.")
        if self.hydraulic_loading_rate is not None and self.hydraulic_loading_rate <= 0:
            raise ValueError("hydraulic_loading_rate must be > 0 if provided.")
        if self.bed_depth is not None and self.bed_depth <= 0:
            raise ValueError("bed_depth must be > 0 if provided.")
        if self.filter_area is not None and self.filter_area <= 0:
            raise ValueError("filter_area must be > 0 if provided.")
        if self.bed_porosity is not None and not 0 < self.bed_porosity <= 1:
            raise ValueError("bed_porosity must be in (0, 1] if provided.")
        if (
                self.doc_removal_fraction is not None
                and not 0 <= self.doc_removal_fraction <= 1
        ):
            raise ValueError("doc_removal_fraction must be in [0, 1] if provided.")

        if self.backwash_out_bus is not None and self.backwash_water_fraction == 0:
            raise ValueError(
                "backwash_water_fraction cannot be 0 when backwash_out_bus "
                "is provided — a bus with zero flow is meaningless."
            )

        if self.backwash_water_fraction > 0.10:
            warnings.warn(
                "backwash_water_fraction > 0.10. This is high for an averaged "
                "planning representation; check whether recycle is being "
                "double-counted.",
                UserWarning,
            )
        if self.backwash_water_fraction > 0 and self.backwash_out_bus is None:
            warnings.warn(
                "backwash_water_fraction > 0 but no backwash_out_bus provided. "
                "Backwash water loss is annualized into feedwater demand.",
                UserWarning,
            )
        if self.design_ebct is not None and self.design_ebct < 5:
            warnings.warn(
                "design_ebct is very low for a biofiltration design "
                "representation; verify units and intent.",
                UserWarning,
            )
        if self.design_ebct is not None and self.design_ebct > 120:
            warnings.warn(
                "design_ebct is unusually high; verify whether this is "
                "true EBCT in minutes.",
                UserWarning,
            )