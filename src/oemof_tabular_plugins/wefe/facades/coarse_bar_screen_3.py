import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class CoarseBarScreen(MIMO):
    """
    Literature-informed coarse bar screen facade based on MIMO.

    Purpose
    -------
    Preliminary-treatment unit representing a coarse bar screen at WWTP
    headworks. The model is designed as a proportional-flow bookkeeping
    unit representing a coarse screen as a debris-removal intervention
    protecting downstream equipment. It is not a full hydraulic or
    clogging model.

    Core references
    ---------------
    1. Process role, coarse-screen type classification, opening-size ranges (>= 6 mm), and headworks placement guidance.
       U.S. Environmental Protection Agency. (2003). Wastewater technology fact sheet: Screening and grit removal (EPA 832-F-03-011).
       https://nepis.epa.gov/Exe/ZyNET.exe/P1000S7N.txt?ZyActionD=ZyDocument&Client=EPA&Index=2016%20Thru%202020%7C1991%20Thru%201994%7C2011%20Thru%202015%7C1986%20Thru%201990%7C2006%20Thru%202010%7C1981%20Thru%201985%7C2000%20Thru%202005%7C1976%20Thru%201980%7C1995%20Thru%201999%7CPrior%20to%201976&Docs=&Query=832F03011&Time=&EndTime=&SearchMethod=2&TocRestrict=n&Toc=&TocEntry=&QField=&QFieldYear=&QFieldMonth=&QFieldDay=&UseQField=&IntQFieldOp=0&ExtQFieldOp=0&XmlQuery=&File=D%3A%5CZYFILES%5CINDEX%20DATA%5C00THRU05%5CTXT%5C00000015%5CP1000S7N.txt&User=ANONYMOUS&Password=anonymous&SortMethod=h%7C-&MaximumDocuments=15&FuzzyDegree=0&ImageQuality=r85g16/r85g16/x150y150g16/i500&Display=hpfr&DefSeekPage=&SearchBack=ZyActionL&Back=ZyActionS&BackDesc=Results%20page&MaximumPages=1&ZyEntry=1&SeekPage=
    2. Coarse-screen classification, opening-size categories, and engineering design framing.
       Water Environment Federation, & Environmental and Water Resources Institute of ASCE. (2010). Design of municipal
       wastewater treatment plants: WEF Manual of Practice No. 8 / ASCE Manuals and Reports on Engineering Practice No. 76
       (5th ed., Vol. 2, Chapter 11). WEF Press / McGraw-Hill Education.
    3. Preliminary-treatment role, downstream-protection function, and screening conceptual basis.
       Metcalf & Eddy, Inc., Tchobanoglous, G., Stensel, H. F., Tsuchihashi, R., & Burton, F. L. (2014). Wastewater
       engineering: Treatment and resource recovery (5th ed., Chap. 5). McGraw-Hill Education.
       https://www.mheducation.com/highered/product/wastewater-engineering-treatment-and-resource-recovery-metcalf-and-eddy.html?viewOption=student
    4. Supports a separation-performance parameter instead of a generic efficiency term, from CFD particle-tracking
       simulation of bar-screen retention.
       En-Nabety, L., & Boudi, E. M. (2024). Separation efficiency of a wastewater bar screen based on a 3D computational
       fluid dynamics modeling. Water Environment Research.
       https://www.sciencedirect.com/science/article/pii/S1944398624202988

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Feedwater per unit treated water output:
        feedwater_per_output = 1 / efficiency
        [m³ influent / m³ treated]

    Electricity demand per unit treated water output:
        electricity_per_output = specific_energy_consumption
        [kWh / m³ treated]

    Screenings per unit treated water output (if screenings_out_bus given):
        screenings_per_output = (1 - efficiency) / efficiency
        [m³ removed / m³ treated]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum treated-water throughput of the screen.
    - Detailed headloss buildup, clogging, velocity-dependent capture, and cleaning-cycle dynamics are not modeled here;
      these are stored as documentation / engineering metadata fields only.
    - screen_type, screen_opening_mm, approach_velocity_m_per_s, headloss_clean_m, headloss_dirty_m, and bar_angle_deg are
      documentation / QA fields and are not enforced as hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "coarse_bar_screen"
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
    screenings_out_bus: Optional[Bus] = None  # m³ removed screenings

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.04           # kWh / m³ treated water [2, 3]
    efficiency: float = 0.98                # m³ treated / m³ influent (treated water fraction) [3]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                  # USD/m³ treated water
    carrier_cost: float = 0.0                   # USD/m³ influent

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
    screen_type: str = "mechanical_bar_screen"     # manual, mechanical_bar_screen, trash_rack [1, 2]
    screen_opening_mm: float = 20.0                # [mm] coarse screens >= 6 mm [1]
    approach_velocity_m_per_s: float = None        # [m/s] hydraulic sizing [2]
    headloss_clean_m: float = None                 # [m] clean-screen head loss [4]
    headloss_dirty_m: float = None                 # [m] clogged-screen head loss [4]
    bar_angle_deg: float = None                    # [°] bar inclination angle [4]

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
        self.output_parameters = attributes.pop("output_parameters", {})

        # --------------------------------------------------------------
        # documentation / calibration defaults
        # --------------------------------------------------------------
        self.screen_type = attributes.pop("screen_type", self.screen_type)
        self.screen_opening_mm = attributes.pop(
            "screen_opening_mm", self.screen_opening_mm
        )
        self.approach_velocity_m_per_s = attributes.pop(
            "approach_velocity_m_per_s", self.approach_velocity_m_per_s
        )
        self.headloss_clean_m = attributes.pop(
            "headloss_clean_m", self.headloss_clean_m
        )
        self.headloss_dirty_m = attributes.pop(
            "headloss_dirty_m", self.headloss_dirty_m
        )
        self.bar_angle_deg = attributes.pop("bar_angle_deg", self.bar_angle_deg)

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self.feedwater_per_output = 1.0 / self.efficiency
        self.electricity_per_output = self.specific_energy_consumption
        self.screenings_per_output = (
            (1.0 - self.efficiency) / self.efficiency
            if self.screenings_out_bus is not None
            else None
        )

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.electricity_per_output
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self.feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.screenings_out_bus is not None:
            attributes[f"conversion_factor_{self.screenings_out_bus.label}"] = sequence(
                self.screenings_per_output
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
            raise ValueError("efficiency must be in (0, 1].")

        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        allowed_screen_types = {"manual", "mechanical_bar_screen", "trash_rack"}
        if self.screen_type not in allowed_screen_types:
            raise ValueError(
                f"screen_type must be one of {sorted(allowed_screen_types)}."
            )

        if self.screen_opening_mm is not None and self.screen_opening_mm < 6:
            raise ValueError(
                "screen_opening_mm must be >= 6 mm for coarse-screen "
                "classification."
            )
