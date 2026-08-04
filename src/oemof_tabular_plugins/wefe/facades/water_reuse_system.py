import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO

@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class WaterReuseSystem(MIMO):
    """
    Literature-informed generic water reuse system facade based on MIMO.

    Purpose
    -------
    Generic membrane-based water reuse treatment train facade for planning-level
    WEFE nexus modeling. The model is designed as a bookkeeping / flow-split
    unit representing a complete reuse treatment train normalized to treated
    water output. It is not a full mechanistic membrane transport model or
    RO design tool.

    Core references
    ---------------
    1. Recovery-ratio operating envelope (70-85%), gross specific energy consumption range (0.3-1.5 kWh/m3), and
       fouling's impact on achievable capacity for membrane-based water reuse trains.
       American Water Works Association. (2018). M62: Membrane applications for water reuse. American Water Works Association.
       https://store.awwa.org/M62-Membrane-Applications-for-Water-Reuse
    2. Operating limits (recovery <=80%), fouling effects on flux and specific energy consumption, and
       cleaning-frequency/maintenance realism for RO trains in continuous operation.
       DuPont Water Solutions. (n.d.). RO Operations Advisor [Online platform and user documentation]. DuPont de Nemours, Inc.
       https://www.dupont.com/water/resources/ro-operations-advisor.html
    3. Reuse-target differentiation (potable / industrial / agricultural), techno-economic comparison of specific energy
       consumption and net cost by reuse application, and concentrate/brine management cost framing.
       Kehrein, P., Jafari, M., Slagt, M., Cornelissen, E., Osseweijer, P., Posada, J., & van Loosdrecht, M. (2021).
       A techno-economic analysis of membrane-based advanced treatment processes for the reuse of municipal
       wastewater. Water Reuse, 11(4), 705-725. https://doi.org/10.2166/wrd.2021.016
    4. Advanced membrane reuse energy fundamentals, fouling-energy coupling, and reverse osmosis as the standard
       technology underpinning advanced potable reuse trains.
       Tang, C. Y., Yang, Z., Guo, H., Wen, J. J., Nghiem, L. D., & Cornelissen, E. (2018). Potable water reuse through
       advanced membrane technology. Environmental Science & Technology, 52(18), 10215-10223. https://doi.org/10.1021/acs.est.8b00562
    5. Train-level specific energy consumption abstraction and energy-recovery-device feasibility across real potable
       reuse schemes (1.2-2.1 kWh/m3 for full direct/indirect schemes).
       Tow, E. W., Hartman, A. L., Jaworowski, A., Zucker, I., Kum, S., AzadiAghdam, M., Blatchley, E. R., Achilli, A.,
       Gu, H., Urper, G. M., & Warsinger, D. M. (2021). Modeling the energy consumption of potable water reuse schemes.
       Water Research X, 13, 100126. https://doi.org/10.1016/j.wroa.2021.100126

    Main equations
    --------------
    All flows normalized to treated water output = 1 [m³/hr]:

    Feedwater input per unit treated water:
        f_feed(t) = f_product(t) / efficiency
        [m³/hr]     [m³/hr]        [-]

    Reject output per unit treated water (only when reject_bus is set):
        f_reject(t) = f_product(t) * (1 - efficiency) / efficiency
        [m³/hr]        [m³/hr]        [-]

    Effective specific energy consumption:
        SEC_eff = specific_energy_consumption
                  * fouling_factor
                  * (1 - energy_recovery_factor)
        [kWh/m³]   [kWh/m³]              [-]           [-]

    Electricity input per unit treated water:
        f_elec(t) = SEC_eff * f_product(t)
        [kWh/hr]    [kWh/m³]  [m³/hr]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains maximum treated water production of the treatment train.
    - reject_bus is optional: when provided, concentrate/brine becomes a first-class output.
    - fouling_factor and energy_recovery_factor are reduced-order surrogates for operational deterioration and energy
      integration benefits. They are not mechanistic fouling or pressure models.
    - Characterization values (SEC reference ranges, recovery envelope) are stored as metadata for scenario documentation.
      They are enforced only as soft warnings, not as hard optimization constraints.
    - reuse_mode provides optional initialization presets for agricultural, industrial, and potable reuse targets (Kehrein et al., 2021).
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "water_reuse_system"
    name: str = ""
    tech: str = "wastewater-treatment"
    carrier: str = "water"
    reuse_mode: Optional[str] = None  # "agricultural", "industrial", "potable"
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
    water_in_bus: Bus = None        # m³  (secondary effluent / feedwater)
    water_out_bus: Bus = None       # m³  (product / reclaimed water) (PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension, e.g. heat_bus, chemical_bus

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    reject_bus: Optional[Bus] = None  # m³  (concentrate / brine)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.90      # kWh / m³ treated water (gross, baseline) [1, 4]
    efficiency: float = 0.80                       # m³ treated water / m³ feedwater [-], (0, 1) (recovery ratio) [1, 2]
    fouling_factor: float = 1.00                   # multiplier on SEC [-], must be > 0 [2, 1]
    energy_recovery_factor: float = 0.00           # fractional SEC reduction [-], [0, 1) [5, 4]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0      # USD/m³ treated water
    carrier_cost: float = 0.0       # USD/m³ influent water
    chemical_cost: float = 0.0      # USD/m³ treated water (OPEX proxy)
    reject_cost: float = 0.0        # USD/m³ reject water (disposal / brine handling)

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints)
    # Based on AWWA M62 (2018) [1] / DuPont [2] / Kehrein et al. (2021) [3] / Tow et al. (2021) [5]
    # ------------------------------------------------------------------
    sec_reference_potable: float = 1.20         # kWh/m³ [3, 5]
    sec_reference_industrial: float = 0.80      # kWh/m³ [3, 5]
    sec_reference_agricultural: float = 0.50    # kWh/m³ [3, 5]
    recovery_recommended_min: float = 0.50      # [-] [1, 2]
    recovery_recommended_max: float = 0.85      # [-] [1, 2]
    recovery_warning_threshold: float = 0.85    # [-] [1, 2]

    def __init__(self, **attributes):
        # --------------------------------------------------------------
        # identity
        # --------------------------------------------------------------
        self.type = attributes.pop("type", self.type)
        self.name = attributes.pop("name", self.name)
        self.tech = attributes.pop("tech", self.tech)
        self.carrier = attributes.pop("carrier", self.carrier)
        self.reuse_mode = attributes.pop("reuse_mode", self.reuse_mode)
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
        self.reject_bus = attributes.pop("reject_bus", None)

        # --------------------------------------------------------------
        # optional reuse-mode presets (Kehrein et al. (2021) [3])
        # applied before physics parameters so explicit user values override
        # --------------------------------------------------------------
        preset_map = {
            "agricultural": {
                "efficiency": 0.85,
                "specific_energy_consumption": 0.50,
            },
            "industrial": {
                "efficiency": 0.80,
                "specific_energy_consumption": 0.80,
            },
            "potable": {
                "efficiency": 0.75,
                "specific_energy_consumption": 1.20,
            },
        }
        preset = {}
        if self.reuse_mode is not None:
            if self.reuse_mode not in preset_map:
                raise ValueError(
                    "reuse_mode must be one of {'agricultural', 'industrial', 'potable'}."
                )
            preset = preset_map[self.reuse_mode]

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", preset.get("specific_energy_consumption", self.specific_energy_consumption),
        )
        self.efficiency = attributes.pop(
            "efficiency", preset.get("efficiency", self.efficiency),
        )
        self.fouling_factor = attributes.pop("fouling_factor", self.fouling_factor)
        self.energy_recovery_factor = attributes.pop(
            "energy_recovery_factor", self.energy_recovery_factor
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.chemical_cost = attributes.pop("chemical_cost", self.chemical_cost)
        self.reject_cost = attributes.pop("reject_cost", self.reject_cost)
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
        self.sec_reference_potable = attributes.pop(
            "sec_reference_potable", self.sec_reference_potable
        )
        self.sec_reference_industrial = attributes.pop(
            "sec_reference_industrial", self.sec_reference_industrial
        )
        self.sec_reference_agricultural = attributes.pop(
            "sec_reference_agricultural", self.sec_reference_agricultural
        )
        self.recovery_recommended_min = attributes.pop(
            "recovery_recommended_min", self.recovery_recommended_min
        )
        self.recovery_recommended_max = attributes.pop(
            "recovery_recommended_max", self.recovery_recommended_max
        )
        self.recovery_warning_threshold = attributes.pop(
            "recovery_warning_threshold", self.recovery_warning_threshold
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived conversion quantities
        # feedwater ratio: m³ influent per m³ treated water output
        # --------------------------------------------------------------
        self._feedwater_per_output = 1.0 / self.efficiency
        self._reject_per_output = (1.0 - self.efficiency) / self.efficiency
        self._effective_sec = (
                self.specific_energy_consumption
                * self.fouling_factor
                * (1.0 - self.energy_recovery_factor)
        )

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._effective_sec
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.reject_bus is not None:
            attributes[f"conversion_factor_{self.reject_bus.label}"] = sequence(
                self._reject_per_output
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
        total_marginal_cost = self.marginal_cost + self.chemical_cost

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

        if self.reject_bus is not None and self.reject_bus in self.outputs:
            self.outputs[self.reject_bus].variable_costs = sequence(
                self.reject_cost
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
            self.reject_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.efficiency < 1:
            raise ValueError("efficiency must be in (0, 1).")

        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        if self.fouling_factor <= 0:
            raise ValueError("fouling_factor must be > 0.")

        if not 0 <= self.energy_recovery_factor < 1:
            raise ValueError("energy_recovery_factor must be in [0, 1).")

        for param_name, value in {
            "reject_cost": self.reject_cost,
            "chemical_cost": self.chemical_cost,
        }.items():
            if value < 0:
                raise ValueError(f"{param_name} must be >= 0.")

        if self.primary == "reject_bus" and self.reject_bus is None:
            raise ValueError(
                "primary='reject_bus' requires reject_bus to be provided."
            )

        # ----------------------------------------------------------
        # soft warnings
        # ----------------------------------------------------------
        if self.efficiency > self.recovery_warning_threshold:
            warnings.warn(
                f"efficiency={self.efficiency:.2f} exceeds the recommended "
                f"threshold of {self.recovery_warning_threshold:.2f}. "
                "High-recovery operation increases fouling and scaling risk "
                "(DuPont RO Operations Advisor; AWWA M62).",
                UserWarning,
            )

        if (
                self.reuse_mode == "potable"
                and self.specific_energy_consumption < 0.50
        ):
            warnings.warn(
                f"specific_energy_consumption={self.specific_energy_consumption:.2f} kWh/m³ "
                "is unusually low for a potable reuse train. Full advanced treatment "
                "trains are typically 1.1–1.4 kWh/m³ "
                "(Tang et al. 2018; Tow et al. 2021).",
                UserWarning,
            )