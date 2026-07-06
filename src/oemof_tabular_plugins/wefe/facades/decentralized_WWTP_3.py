import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO

@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class DecentralizedWWTP(MIMO):
    """
    Literature-informed generic decentralized WWTP facade based on MIMO.

    Purpose
    -------
    Generic wastewater treatment facade for mixed-influent decentralized systems
    (SBR, MBR, MBBR, ABR, extended aeration, and similar onsite/decentralized
    processes). The model is designed as a process-yield unit representing a WWTP
    as a water-energy-resource intervention. It is not a full mechanistic
    biological reactor model or pollutant-state model.

    Core references
    ---------------
    1. Standard mass-balance equations, per-capita wastewater generation rates (154–246 L/capita·day) used to convert
       sludge-volume data into m³/m³ terms, and general influent characterization/removal-performance context.
       Metcalf & Eddy, Inc., Tchobanoglous, G., Stensel, H. F., Tsuchihashi, R., & Burton, F. L. (2014). Wastewater
       engineering: Treatment and resource recovery (5th ed.). McGraw-Hill Education.
    2. Sludge volume production by treatment process (L/inhabitant·day), Table 2.1 — primary basis for sludge_specific_production.
       von Sperling, M., Andreoli, C. V., & Fernandes, F. (2007). Sludge treatment and disposal
       (Biological Wastewater Treatment Series, Vol. 6). IWA Publishing.
       https://kh.aquaenergyexpo.com/wp-content/uploads/2022/07/Sludge-Treatment-and-Disposal.pdf
    3. Removal-efficiency targets (COD, BOD, TN, TP) tied to the EU Urban Wastewater Treatment Directive;
       F/M ratios, oxygen transfer, and sludge production factors for single-stage activated sludge design.
       Normative Committee for Wastewater, Sludge and Waste Management (ATV-DVWK). (2000). ATV-DVWK-A 131E: Dimensioning
       of single-stage activated sludge plants. GFA-Gesellschaft zur Förderung der Abwassertechnik.
       https://kh.aquaenergyexpo.com/wp-content/uploads/2022/11/Dimensioning-of-Single-Stage-Activated-Sludge-Plants.pdf
    4. Septic tank and ABR design as gravity-fed/passive systems requiring no external energy input under normal operation.
       Sasse, L. (1998). DEWATS: Decentralized wastewater treatment in developing countries.
       Bremen Overseas Research and Development Association (BORDA).
       https://sswm.info/sites/default/files/reference_attachments/SASSE%201998%20DEWATS%20Decentralised%20Wastewater%20Treatment%20in%20Developing%20Countries_0.pdf
    5. Measured specific energy consumption (kWh/m³) from 8 real small-community plants (60–4,400 m³/d): extended aeration
       2.8 kWh/m³, MBR 6.6 kWh/m³, SBR 8.6–11.3 kWh/m³ — basis for generic_decentralized, mbr, and sbr energy defaults and ranges.
       Hamza, R., Hamoda, M. F., & Elassar, M. (2022). Energy and reliability analysis of wastewater treatment plants in
       small communities in Ontario. Water Science and Technology, 85(6), 1824–1839. https://doi.org/10.2166/wst.2022.093
    6. Pilot-scale decentralized MBBR greywater treatment, measured power consumption 0.09–0.25 kWh/m³
       — basis for the mbbr energy default and range.
       Al Hosani, N., Fathelrahman, E., Ahmed, H., & Rikabi, E. (2022). Moving bed biofilm reactor (MBBR) for decentralized
       grey water treatment: Technical, ecological and cost efficiency comparison for domestic applications.
       Emirates Journal of Food and Agriculture, 34(9), 731–742.
       https://research.uaeu.ac.ae/en/publications/moving-bed-biofilm-reactor-mbbr-for-decentralized-grey-water-trea/

    Main equations
    --------------
    All flows normalized to treated water output = 1 [m³/hr]:

    Feedwater input:
        Q_in(t) = Q_out(t) / efficiency                  [m³/hr]

    Electricity input:
        E(t)    = SEC * Q_out(t)                         [kWh/hr]
        where SEC = specific_energy_consumption          [kWh/m³ treated]

    Sludge output (explicit solids-based factor, not hydraulic residual):
        S(t)    = SSP * Q_out(t)                         [m³ sludge/hr]
        where SSP = sludge_specific_production           [m³ sludge / m³ treated]

    Optional chemical input:
        C(t)    = CSC * Q_out(t)                         [unit chemical/hr]
        where CSC = chemical_specific_consumption        [unit / m³ treated]

    Optional biogas output:
        B(t)    = BSY * Q_out(t)                         [m³ biogas/hr]
        where BSY = biogas_specific_yield                [m³ biogas / m³ treated]

    Optional nutrient product output:
        N(t)    = NSY * Q_out(t)                         [unit nutrient/hr]
        where NSY = nutrient_specific_yield              [unit / m³ treated]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum treated water output, representing the design
      hydraulic capacity of the treatment plant.
    - Sludge is modeled via an explicit specific production factor (von Sperling), independent of hydraulic recovery.
      This separates solids generation from hydraulic throughput.
    - Influent quality and removal-efficiency fields are stored as metadata for scenario documentation and plausibility
      checks. They are not enforced as hard optimization constraints.
    - All optional side streams (chemical, biogas, nutrient) are linear and normalized to treated water output,
      preserving full MIMO compatibility.
    - No input grouping is applied; each bus occupies its own MIMO group. MIMO's _unify_groups handles this automatically.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "decentralized_WWTP"
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
    water_in_bus: Bus = None        # m³ influent wastewater
    water_out_bus: Bus = None       # m³ treated water (PRIMARY)
    sludge_out_bus: Bus = None      # m³ sludge

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    chemical_bus: Optional[Bus] = None    # unit chemical (coagulant, disinfectant)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    biogas_out_bus: Optional[Bus] = None    # m³ biogas
    nutrient_out_bus: Optional[Bus] = None  # unit nutrient product

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 2.8    # kWh / m³ treated water [Hamza]
    efficiency: float = 0.975                   # m³ treated / m³ influent (hydraulic recovery)[Metcalf & Eddy, von Sperling]
    sludge_specific_production: float = 0.025   # m³ sludge / m³ treated [Metcalf & Eddy, von Sperling]
    chemical_specific_consumption: float = 0.0  # unit chemical / m³ treated
    biogas_specific_yield: float = 0.0          # m³ biogas / m³ treated
    nutrient_specific_yield: float = 0.0        # unit nutrient / m³ treated

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0          # USD/m³ treated water
    carrier_cost: float = 0.0           # USD/kWh influent water
    sludge_disposal_cost: float = 0.0   # USD/m³ sludge
    chemical_cost: float = 0.0          # USD/unit chemical
    biogas_revenue: float = 0.0         # USD/m³ biogas  (applied as negative cost)
    nutrient_revenue: float = 0.0       # USD/unit nutrient product (negative cost)

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints)
    # Based on Metcalf & Eddy (2014) / Sasse (1998) / ATV-DVWK-A 131E (2000)
    # ------------------------------------------------------------------
    process_type: str = "generic_decentralized"
    quality_class: str = ""
    influent_cod: float = None            # mg/L
    influent_bod: float = None            # mg/L
    influent_tss: float = None            # mg/L
    cod_removal_efficiency: float = None  # fraction
    bod_removal_efficiency: float = None  # fraction
    tss_removal_efficiency: float = None  # fraction

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
        self.sludge_out_bus = attributes.pop("sludge_out_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.chemical_bus = attributes.pop("chemical_bus", None)
        self.biogas_out_bus = attributes.pop("biogas_out_bus", None)
        self.nutrient_out_bus = attributes.pop("nutrient_out_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop(
            "efficiency", self.efficiency
        )
        self.sludge_specific_production = attributes.pop(
            "sludge_specific_production", self.sludge_specific_production
        )
        self.chemical_specific_consumption = attributes.pop(
            "chemical_specific_consumption", self.chemical_specific_consumption
        )
        self.biogas_specific_yield = attributes.pop(
            "biogas_specific_yield", self.biogas_specific_yield
        )
        self.nutrient_specific_yield = attributes.pop(
            "nutrient_specific_yield", self.nutrient_specific_yield
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.sludge_disposal_cost = attributes.pop("sludge_disposal_cost", self.sludge_disposal_cost)
        self.chemical_cost = attributes.pop("chemical_cost", self.chemical_cost)
        self.biogas_revenue = attributes.pop("biogas_revenue", self.biogas_revenue)
        self.nutrient_revenue = attributes.pop("nutrient_revenue", self.nutrient_revenue)
        self.expandable = attributes.pop("expandable", self.expandable)
        self.capacity = attributes.pop("capacity", self.capacity)
        self.capacity_cost = attributes.pop("capacity_cost", self.capacity_cost)
        self.capacity_minimum = attributes.pop("capacity_minimum", self.capacity_minimum)
        self.capacity_potential = attributes.pop("capacity_potential", self.capacity_potential)

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
        self.process_type = attributes.pop("process_type", self.process_type)
        self.quality_class = attributes.pop("quality_class", self.quality_class)
        self.influent_cod = attributes.pop("influent_cod", self.influent_cod)
        self.influent_bod = attributes.pop("influent_bod", self.influent_bod)
        self.influent_tss = attributes.pop("influent_tss", self.influent_tss)
        self.cod_removal_efficiency = attributes.pop(
            "cod_removal_efficiency", self.cod_removal_efficiency
        )
        self.bod_removal_efficiency = attributes.pop(
            "bod_removal_efficiency", self.bod_removal_efficiency
        )
        self.tss_removal_efficiency = attributes.pop(
            "tss_removal_efficiency", self.tss_removal_efficiency
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # feedwater ratio: m³ influent per m³ treated water output
        # --------------------------------------------------------------
        self._feedwater_per_output = 1.0 / self.efficiency

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.specific_energy_consumption  # kWh/m³ — electricity per treated water
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output  # m³/m³ — influent per treated water
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.sludge_out_bus.label}"] = sequence(
            self.sludge_specific_production  # m³/m³ — sludge per treated water
        )

        if self.chemical_bus is not None:
            attributes[f"conversion_factor_{self.chemical_bus.label}"] = sequence(
                max(self.chemical_specific_consumption, 1e-9)
            )

        if self.biogas_out_bus is not None:
            attributes[f"conversion_factor_{self.biogas_out_bus.label}"] = sequence(
                max(self.biogas_specific_yield, 1e-9)
            )

        if self.nutrient_out_bus is not None:
            attributes[f"conversion_factor_{self.nutrient_out_bus.label}"] = sequence(
                max(self.nutrient_specific_yield, 1e-9)
            )

        # --------------------------------------------------------------
        # primary bus label resolution
        # --------------------------------------------------------------
        if self.primary == "water_out_bus":
            primary_label = self.water_out_bus.label
        elif self.primary == "water_in_bus":
            primary_label = self.water_in_bus.label
        elif self.primary == "sludge_out_bus":
            primary_label = self.sludge_out_bus.label
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
            to_bus_1=self.sludge_out_bus,
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

        if self.chemical_bus is not None and self.chemical_bus in self.inputs:
            self.inputs[self.chemical_bus].variable_costs = sequence(
                self.chemical_cost
            )

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

        if self.sludge_out_bus in self.outputs:
            self.outputs[self.sludge_out_bus].variable_costs = sequence(
                self.sludge_disposal_cost
            )

        if self.biogas_out_bus is not None and self.biogas_out_bus in self.outputs:
            self.outputs[self.biogas_out_bus].variable_costs = sequence(
                -abs(self.biogas_revenue)
            )

        if self.nutrient_out_bus is not None and self.nutrient_out_bus in self.outputs:
            self.outputs[self.nutrient_out_bus].variable_costs = sequence(
                -abs(self.nutrient_revenue)
            )

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 2
        # inputs
        for bus in [
            self.chemical_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.biogas_out_bus,
            self.nutrient_out_bus,
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1].")
        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")
        if self.sludge_specific_production < 0:
            raise ValueError("sludge_specific_production must be >= 0.")
        if self.chemical_specific_consumption < 0:
            raise ValueError("chemical_specific_consumption must be >= 0.")
        if self.biogas_specific_yield < 0:
            raise ValueError("biogas_specific_yield must be >= 0.")
        if self.nutrient_specific_yield < 0:
            raise ValueError("nutrient_specific_yield must be >= 0.")

        if self.chemical_bus is None and self.chemical_specific_consumption > 0:
            warnings.warn(
                "chemical_specific_consumption > 0 but chemical_bus is None. "
                "Chemical consumption will be ignored.",
                UserWarning,
            )
        if self.biogas_out_bus is None and self.biogas_specific_yield > 0:
            warnings.warn(
                "biogas_specific_yield > 0 but biogas_out_bus is None. "
                "Biogas yield will be ignored.",
                UserWarning,
            )
        if self.nutrient_out_bus is None and self.nutrient_specific_yield > 0:
            warnings.warn(
                "nutrient_specific_yield > 0 but nutrient_out_bus is None. "
                "Nutrient yield will be ignored.",
                UserWarning,
            )

        # ----------------------------------------------------------
        # soft checks on documentation / calibration fields
        # ----------------------------------------------------------
        for name, value in [
            ("cod_removal_efficiency", self.cod_removal_efficiency),
            ("bod_removal_efficiency", self.bod_removal_efficiency),
            ("tss_removal_efficiency", self.tss_removal_efficiency),
        ]:
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1] when provided.")

        for name, value in [
            ("influent_cod", self.influent_cod),
            ("influent_bod", self.influent_bod),
            ("influent_tss", self.influent_tss),
        ]:
            if value is not None and value < 0:
                warnings.warn(
                    f"{name} is negative; check units and data source.",
                    UserWarning,
                )

        # ----------------------------------------------------------
        # process-type plausibility checks (soft warnings, not hard failures)
        # Ranges from Sasse (1998), von Sperling (2007), Hamza (2022)
        # ----------------------------------------------------------
        _ranges = {
            "generic_decentralized": {
                "efficiency": (0.97, 0.995),
                "specific_energy_consumption": (1.50, 4.00),
                "sludge_specific_production": (0.013, 0.036),
            },
            "septic": {
                "efficiency": (0.97, 0.995),
                "specific_energy_consumption": (0.00, 0.05),
                "sludge_specific_production": (0.01, 0.06),
            },
            "abr": {
                "efficiency": (0.97, 0.995),
                "specific_energy_consumption": (0.00, 0.05),
                "sludge_specific_production": (0.01, 0.06),
            },
            "mbbr": {
                "efficiency": (0.97, 0.995),
                "specific_energy_consumption": (0.05, 0.30),
                "sludge_specific_production": (0.015, 0.04),
            },
            "sbr": {
                "efficiency": (0.97, 0.995),
                "specific_energy_consumption": (7.00, 12.00),
                "sludge_specific_production": (0.013, 0.036),
            },
            "mbr": {
                "efficiency": (0.97, 0.995),
                "specific_energy_consumption": (1.00, 8.00),
                "sludge_specific_production": (0.01, 0.03),
            },
        }

        if self.process_type not in _ranges:
            warnings.warn(
                f"Unknown process_type '{self.process_type}'. "
                "Skipping process-specific plausibility checks.",
                UserWarning,
            )
            return

        checks = _ranges[self.process_type]
        for param, attr in [
            ("efficiency", self.efficiency),
            ("specific_energy_consumption", self.specific_energy_consumption),
            ("sludge_specific_production", self.sludge_specific_production),
        ]:
            lo, hi = checks[param]
            if not lo <= attr <= hi:
                warnings.warn(
                    f"{param}={attr} is outside the typical range for "
                    f"process_type='{self.process_type}' ({lo}..{hi}).",
                    UserWarning,
                )