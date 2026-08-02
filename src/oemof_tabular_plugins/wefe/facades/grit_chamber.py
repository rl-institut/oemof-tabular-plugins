import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class GritChamber(MIMO):
    """
    Literature-informed grit chamber facade based on MIMO.

    Purpose
    -------
    Hydraulically bounded pretreatment unit representing a grit chamber as
    two mandatory inputs (electricity, raw water) and one mandatory output
    (treated water), with optional grit-solids output and optional washwater
    input for grit handling. The model enforces proportional flow relations.
    It is not a full particle-settling simulator.

    Core references
    ---------------
    1. Horizontal-flow chamber design theory (L/H = v/Vo relation), target particle size (0.2 mm), and settling-velocity
       basis for grit capture.
       NPTEL, IIT Kharagpur. Wastewater Management (Course 105105048), Module 15: Grit Chamber, Lecture 19 (and Lecture 20,
       continued design equations for the same module).
       https://archive.nptel.ac.in/content/storage2/courses/105105048/M15L19.pdf
    2. Standard design ranges for detention time and velocity across chamber types: horizontal 45-90 s (typical 60 s),
       0.8-1.3 ft/s (typical 1.0 ft/s); aerated 2-5 min; chamber-type classification.
       U.S. EPA. (2023). Preliminary Wastewater Treatment: NPDES Operator Webinar Series.
       https://www.epa.gov/system/files/documents/2023-10/tawebinar_preliminarywastewatertreatment_230725.pdf
    3. Velocity-control device design (Parshall/Venturi flume, proportional weir), and aerated-chamber air-rate and detention
       -time defaults, adapted from the U.S. Unified Facilities Criteria (public domain).
       Guyer, J. P. An Introduction to Preliminary Wastewater Treatment (Course No. C02-033/C454). Continuing Education
       and Development, Inc.
       https://www.cedengineering.com/userfiles/C02-033%20-%20An%20Introduction%20to%20Preliminary%20Wastewater%20Treatment%20-%20US.pdf
    4. Typical measured influent/removed grit quantities (4-200 mL/m3, higher in combined sewers).
       Water Environment Federation, Municipal Resource Recovery Design Committee. (2017). Liquid Stream Fundamentals:
       Grit Removal (WSEC-2017-FS-021).
       https://www.resourcerecoverydata.org/WEFfactsheets/wsec-2017-fs-021-mrrdc-lsf-grit-removal_final.pdf

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Raw water requirement:
        Q_in(t) = Q_out(t) / efficiency

    Electricity demand:
        E(t) = specific_energy_consumption * Q_out(t)

    Grit output (optional, active only if grit_out_bus is provided):
        S_grit(t) = grit_influent_concentration
                    * (1 / efficiency)
                    * grit_capture_ratio
                    * Q_out(t)

    Washwater input (optional, requires grit_out_bus):
        Q_wash(t) = wash_volume_per_m3_grit * S_grit(t)
                  = wash_volume_per_m3_grit
                    * grit_influent_concentration
                    * (1 / efficiency)
                    * grit_capture_ratio
                    * Q_out(t)

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum treated-water throughput of the grit chamber unit.
    - grit_out_bus and washwater_bus are fully optional. If neither is provided, the facade behaves as a 2-input/1-output unit.
    - washwater_bus requires grit_out_bus to be set simultaneously, because washwater demand is physically proportional
      to grit output volume.
    - Documentation metadata fields (detention_time, design_velocity, target_particle_size_mm, flow_control_device, air_rate,
      aeration_specific_energy) trigger validation warnings but are not enforced as hard optimization constraints.
    - Chamber-type defaults are applied automatically in _validate_parameters() when the user does not specify values.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "grit_chamber"
    name: str = ""
    tech: str = "water-treatment"
    carrier: str = "water"
    chamber_type: str = "horizontal"    # "horizontal", "aerated", "vortex"
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
    water_in_bus: Bus = None                # m³ (feedwater)
    water_out_bus: Bus = None               # m³ (PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    washwater_bus: Optional[Bus] = None     # m³ (washwater for grit)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    grit_out_bus: Optional[Bus] = None      # m³ (wet grit)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.015      # kWh / m³ treated water [2, 3]
    efficiency: float = 0.98                        # treated water / influent water [1, 2]
    grit_capture_ratio: float = 0.90                # fraction of influent grit captured [1]
    grit_influent_concentration: float = 0.00015    # m³ grit / m³ influent water [4]
    wash_volume_per_m3_grit: float = 0.5            # m³ washwater / m³ wet grit
                                                    # HUBER RoSF G4E: < 2 m³/h washwater at up to 3 m³/h grit capacity
                                                    # → ratio < 0.67 m³/m³ (HUBER Technology, RoSF G4E datasheet, 2021)

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                      # USD/m³ treated water
    carrier_cost: float = 0.0                       # USD/m³ feed
    grit_disposal_cost: float = 0.0                 # USD/m³ wet grit

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
    detention_time: float = None                    # s; horizontal default: 60 s; aerated: 180 s [1, 2]
    design_velocity: float = None                   # m/s; horizontal default: 0.3 m/s, range 0.24-0.4 [1, 2]
    target_particle_size_mm: float = 0.2            # mm; 0.2 mm design target [1]
    flow_control_device: str = "proportional_weir"  # maintains v_h under varying Q [3]
    air_rate: float = None                          # m³ air / m³ wastewater; aerated chambers only [2, 3]
    aeration_specific_energy: float = None          # kWh / m³ [2, 3]

    def __init__(self, **attributes):
        # --------------------------------------------------------------
        # identity
        # --------------------------------------------------------------
        self.type = attributes.pop("type", self.type)
        self.name = attributes.pop("name", self.name)
        self.tech = attributes.pop("tech", self.tech)
        self.carrier = attributes.pop("carrier", self.carrier)
        self.chamber_type = attributes.pop("chamber_type", self.chamber_type)
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
        self.washwater_bus = attributes.pop("washwater_bus", None)
        self.grit_out_bus = attributes.pop("grit_out_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop("efficiency", self.efficiency)
        self.grit_capture_ratio = attributes.pop(
            "grit_capture_ratio", self.grit_capture_ratio
        )
        self.grit_influent_concentration = attributes.pop(
            "grit_influent_concentration", self.grit_influent_concentration
        )
        self.wash_volume_per_m3_grit = attributes.pop(
            "wash_volume_per_m3_grit", self.wash_volume_per_m3_grit
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.grit_disposal_cost = attributes.pop(
            "grit_disposal_cost", self.grit_disposal_cost
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
        self.detention_time = attributes.pop("detention_time", self.detention_time)
        self.design_velocity = attributes.pop("design_velocity", self.design_velocity)
        self.target_particle_size_mm = attributes.pop(
            "target_particle_size_mm", self.target_particle_size_mm
        )
        self.flow_control_device = attributes.pop(
            "flow_control_device", self.flow_control_device
        )
        self.air_rate = attributes.pop("air_rate", self.air_rate)
        self.aeration_specific_energy = attributes.pop(
            "aeration_specific_energy", self.aeration_specific_energy
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._feedwater_per_treated_water = 1.0 / self.efficiency
        self._grit_per_treated_water = (
                self.grit_influent_concentration
                * self._feedwater_per_treated_water
                * self.grit_capture_ratio
        )
        self._wash_per_treated_water = (
                self.wash_volume_per_m3_grit * self._grit_per_treated_water
        )

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

        if self.grit_out_bus is not None:
            attributes[f"conversion_factor_{self.grit_out_bus.label}"] = sequence(
                self._grit_per_treated_water
            )

        if self.washwater_bus is not None:
            attributes[f"conversion_factor_{self.washwater_bus.label}"] = sequence(
                self._wash_per_treated_water
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

        if self.grit_out_bus is not None and self.grit_out_bus in self.outputs:
            self.outputs[self.grit_out_bus].variable_costs = sequence(
                self.grit_disposal_cost
            )

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 1
        # inputs
        for bus in [
            self.washwater_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.grit_out_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        allowed_types = {"horizontal", "aerated", "vortex"}
        if self.chamber_type not in allowed_types:
            raise ValueError(f"chamber_type must be one of {allowed_types}, got {self.chamber_type!r}.")

        if not 0 < self.efficiency <= 1:
            raise ValueError(f"efficiency must be in (0, 1], got {self.efficiency!r}.")

        if self.specific_energy_consumption < 0:
            raise ValueError(f"specific_energy_consumption must be >= 0, got {self.specific_energy_consumption!r}.")

        if not 0 < self.grit_capture_ratio <= 1:
            raise ValueError(f"grit_capture_ratio must be in (0, 1], got {self.grit_capture_ratio!r}.")

        if self.grit_influent_concentration <= 0:
            raise ValueError(f"grit_influent_concentration must be > 0, got {self.grit_influent_concentration!r}.")

        if self.wash_volume_per_m3_grit < 0:
            raise ValueError(f"wash_volume_per_m3_grit must be >= 0, got {self.wash_volume_per_m3_grit!r}.")

        if self.washwater_bus is not None and self.grit_out_bus is None:
            raise ValueError(
                "washwater_bus requires grit_out_bus to be set, "
                "because washwater demand is proportional to grit output."
            )

        if self.grit_disposal_cost < 0:
            raise ValueError(f"grit_disposal_cost must be >= 0, got {self.grit_disposal_cost!r}.")

        if (
                self.grit_disposal_cost not in (None, 0, 0.0)
                and self.grit_out_bus is None
        ):
            warnings.warn(
                f"grit_disposal_cost={self.grit_disposal_cost!r} is set but "
                "no grit_out_bus is provided.",
                UserWarning,
            )

        if (
                self.target_particle_size_mm is not None
                and self.target_particle_size_mm <= 0
        ):
            raise ValueError(f"target_particle_size_mm must be > 0 if provided, got {self.target_particle_size_mm!r}.")

        if (
                self.grit_out_bus is None
                and self.grit_capture_ratio not in (None, 0, 0.0)
        ):
            warnings.warn(
                f"grit_capture_ratio={self.grit_capture_ratio!r} is set but "
                "no grit_out_bus is provided. Grit removal will not be "
                "represented explicitly in the optimization.",
                UserWarning,
            )

        # apply literature-informed defaults per chamber type
        if self.chamber_type == "horizontal":
            if self.detention_time is None:
                self.detention_time = 60.0
            if self.design_velocity is None:
                self.design_velocity = 0.3

        elif self.chamber_type == "aerated":
            if self.detention_time is None:
                self.detention_time = 180.0

        # validation warnings from literature ranges
        if self.chamber_type == "horizontal":
            if self.detention_time is not None and not (
                    30 <= self.detention_time <= 90
            ):
                warnings.warn(
                    f"Horizontal grit chamber detention_time="
                    f"{self.detention_time!r} s is outside the common "
                    "literature range",
                    UserWarning,
                )
            if self.design_velocity is not None and not (
                    0.24 <= self.design_velocity <= 0.4
            ):
                warnings.warn(
                    f"Horizontal grit chamber design_velocity="
                    f"{self.design_velocity!r} m/s is outside the common "
                    "literature range",
                    UserWarning,
                )

        if self.chamber_type == "aerated":
            if self.detention_time is not None and not (
                    120 <= self.detention_time <= 300
            ):
                warnings.warn(
                    f"Aerated grit chamber detention_time="
                    f"{self.detention_time!r} s is outside the common "
                    "literature range",
                    UserWarning,
                )