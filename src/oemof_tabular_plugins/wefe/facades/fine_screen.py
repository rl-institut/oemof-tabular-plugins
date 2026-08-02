import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class FineScreen(MIMO):
    """
    Literature-informed fine screen facade based on MIMO.

    Purpose
    -------
    Linear aggregate representation of a fine screening / headworks unit.
    The model is designed as a bookkeeping/process-yield unit representing
    a fine screen as a preliminary treatment step for retained-solids removal
    and downstream process protection. It is not a hydraulic screen design
    calculator.

    Core references
    ---------------
    1. Process role and unit-boundary justification for fine screening as a preliminary/headworks treatment unit.
       Metcalf & Eddy, Inc., Tchobanoglous, G., Stensel, H. F., Tsuchihashi, R., & Burton, F. L. (2014). Wastewater
       engineering: Treatment and resource recovery (5th ed.). McGraw-Hill Education.
       https://www.mheducation.com/highered/product/wastewater-engineering-treatment-and-resource-recovery-metcalf-and-eddy.html?viewOption=student
    2. Screen types, opening-size effects, screenings generation, washing/compaction, and hydraulic design guidance.
       HUBER Technology. Fine Screens - Basics and Applications.
       https://www.tpomag.com/uploads/downloads/Screening-Considerations_Guide_Huber_180119_133914.pdf
    3. Operational realism for screenings handling, washing/compaction, and capture rates relative to conventional bar screens.
       LEE + RO / Inland Empire Utilities Agency (IEUA). RP-4 Influent Screen Replacement: Headworks fine screen case study.
       https://www.lee-ro.com/solutions/ieua-rp-4
    4. Supports explicit modeling of screenings as a recoverable or disposable side-stream with resource and energy relevance.
       Lemonidis, I., Banti, D. C., Tzenos, C. A., Kalamaras, S. D., Kotsopoulos, T. A., & Samaras, P. (2022). Energy
       valorization of fine screenings from a municipal wastewater treatment plant. Energies, 15(21), 8236.
       https://doi.org/10.3390/en15218236

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Electricity demand:
        E(t) = specific_energy_consumption * Q_out(t)

    Raw water requirement:
        Q_in(t) = Q_out(t) / efficiency

    Screenings generation (optional):
        S(t) = screenings_yield * Q_out(t)

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum treated-water throughput of the screen unit.
    - screenings_out_bus is optional. If not provided, the facade behaves as a  2-input/1-output unit.
    - Documentation metadata fields are stored for scenario documentation and reporting only. They are not enforced as
      hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "fine_screen"
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
    water_out_bus: Bus = None           # m³ (PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    screenings_out_bus: Optional[Bus] = None  # m³ or kg equivalent

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.04           # kWh / m³ treated water [1, 2]
    efficiency: float = 0.995                           # m³ treated water / m³ raw water (water recovery) [1]
    screenings_yield: float = 0.005                     # m³ screenings / m³ treated water [3, 4]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                  # USD/m³ treated water
    carrier_cost: float = 0.0                   # USD/m³ raw water
    screenings_disposal_cost: float = 0.0       # USD/unit screenings output

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
    screen_type: str = ""                               # e.g. "drum", "band", "step" [2]
    screen_opening_mm: Optional[float] = None           # aperture size [mm] [2]
    approach_velocity_ms: Optional[float] = None        # upstream channel velocity [m/s] [2]
    washing_compaction: bool = False                    # screenings washed and compacted on-site [2, 3]
    cod_capture_ratio: Optional[float] = None           # fraction of COD retained [0, 1] [3, 4]
    tss_capture_ratio: Optional[float] = None           # fraction of TSS retained [0, 1] [3]
    valorization_route: str = ""                        # e.g. "landfill", "biogas", "compost" [4]

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
        self.screenings_out_bus = attributes.pop("screenings_out_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop(
            "efficiency", self.efficiency
        )
        self.screenings_yield = attributes.pop(
            "screenings_yield", self.screenings_yield
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.screenings_disposal_cost = attributes.pop(
            "screenings_disposal_cost", self.screenings_disposal_cost
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
        self.screen_type = attributes.pop("screen_type", self.screen_type)
        self.screen_opening_mm = attributes.pop(
            "screen_opening_mm", self.screen_opening_mm
        )
        self.approach_velocity_ms = attributes.pop(
            "approach_velocity_ms", self.approach_velocity_ms
        )
        self.washing_compaction = attributes.pop(
            "washing_compaction", self.washing_compaction
        )
        self.cod_capture_ratio = attributes.pop(
            "cod_capture_ratio", self.cod_capture_ratio
        )
        self.tss_capture_ratio = attributes.pop(
            "tss_capture_ratio", self.tss_capture_ratio
        )
        self.valorization_route = attributes.pop(
            "valorization_route", self.valorization_route
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._feedwater_per_treated_water = 1.0 / self.efficiency
        self._screenings_per_treated_water = self.screenings_yield

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
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.screenings_out_bus is not None:
            attributes[f"conversion_factor_{self.screenings_out_bus.label}"] = sequence(
                self._screenings_per_treated_water
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

        if self.screenings_out_bus is not None and self.screenings_out_bus in self.outputs:
            self.outputs[self.screenings_out_bus].variable_costs = sequence(
                self.screenings_disposal_cost
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
            self.screenings_out_bus
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

        if self.screenings_yield < 0:
            raise ValueError(f"screenings_yield must be >= 0, got {self.screenings_yield!r}.")

        if self.screen_opening_mm is not None and self.screen_opening_mm <= 0:
            raise ValueError(f"screen_opening_mm must be > 0, got {self.screen_opening_mm!r}.")

        if self.approach_velocity_ms is not None and self.approach_velocity_ms <= 0:
            raise ValueError(f"approach_velocity_ms must be > 0, got {self.approach_velocity_ms!r}.")

        for name, value in (
                ("cod_capture_ratio", self.cod_capture_ratio),
                ("tss_capture_ratio", self.tss_capture_ratio),
        ):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(
                    f"{name} must be in [0, 1], got {value!r}."
                )

        if (
                self.screenings_out_bus is None
                and self.screenings_yield not in (None, 0, 0.0)
        ):
            warnings.warn(
                f"screenings_yield={self.screenings_yield!r} is set but "
                "no screenings_out_bus is provided. Screenings generation "
                "will not be represented explicitly in the optimization.",
                UserWarning,
            )