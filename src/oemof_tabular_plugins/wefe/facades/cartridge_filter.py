import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class CartridgeFilter(MIMO):
    """
    Literature-informed cartridge filter facade based on MIMO.

    Purpose
    -------
    Linear pre-treatment / particle-removal facade representing a cartridge
    filter as a water-treatment unit that reduces turbidity and suspended
    solids to protect downstream processes (e.g. UV disinfection, membranes).
    The model is a flow-yield bookkeeping unit and is not a mechanistic
    pore-scale filtration model.

    Core references
    ---------------
    1. Pre-treatment role, filtration objectives, turbidity targets, cartridge filter as upstream barrier for UV/disinfection,
       critical control parameters, and log-removal credit framing.
       Environmental Protection Agency (Ireland). (2020). Water treatment manual: Filtration. EPA.
       https://www.epa.ie/publications/compliance--enforcement/drinking-water/advice--guidance/EPA-Water-Filtration-Manual.pdf
    2. Filtration mechanisms, headloss/run-length behaviour, solids loading, cleaning logic, and justification for simplified
       linear operational representation.
       Vigneswaran, S., Kandasamy, J., & Rogerson, M. (2009). Filtration Technologies in Wastewater Treatment. In Encyclopedia
       of Life Support Systems (EOLSS), Water and Wastewater Treatment Technologies. Eolss Publishers.
       https://www.eolss.net/sample-chapters/c07/E6-144-02.pdf
    3. Full-scale, quantitative cartridge filter selection and replacement guidance: replacement thresholds relative to
       pressure drop, fouling proxy, consumable/energy cost trade-offs, and practical operating envelopes.
       Farhat, N. M., Christodoulou, C., Placotas, P., Blankert, B., Sallangos, O., & Vrouwenvelder, J. S. (2020).
       Cartridge filter selection and replacement: Optimization of produced water quantity, quality, and cost.
       Desalination, 473, 114172. https://doi.org/10.1016/j.desal.2019.114172

    Main equations
    --------------
    All flows normalized to 1 m³ filtered water output (primary):

    Feedwater requirement:
        f_water_in(t) = 1 / efficiency                [m³ feed / m³ filtered]

    Electricity demand:
        f_electricity(t) = SEC(t)                   [kWh / m³ filtered]

    Optional reject stream:
        f_reject(t) = (1 / efficiency) - 1            [m³ reject / m³ filtered]

    Optional cartridge consumable input:
        f_cartridge(t) = SCC(t)                     [units / m³ filtered]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum filtered-water throughput of the unit.
    - reject_water_bus is optional. If absent, water loss from recovery < 1 is treated as an implicit system loss without a dedicated bus.
    - cartridge_bus is optional. If absent, consumable cost may be folded into marginal_cost via cartridge_cost.
    - specific_energy_consumption may be a scalar or time series to approximate fouling-induced headloss increase over a filter run.
    - Turbidity targets, log-removal credits, and headloss limits are stored as metadata for scenario documentation. They 
      are not enforced as hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "cartridge_filter"
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
    electricity_bus: Bus = None     # kWh
    water_in_bus: Bus = None        # m³ (raw / pretreated feed)
    water_out_bus: Bus = None       # m³ (filtered water - PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    cartridge_bus: Optional[Bus] = None     # cartridge / filter-media units

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    reject_water_bus: Optional[Bus] = None  # m³ (residual / solids-laden reject)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    efficiency: float = 0.97                                                # filtered water / feed water (recovery) [1, 2]
    specific_energy_consumption: Union[float, Sequence[float]] = 0.05       # kWh/m³  [3]
    specific_cartridge_consumption: Union[float, Sequence[float]] = 0.0     # Unit cartridge / m³ filtered water [3]
    availability: Union[float, Sequence[float]] = None                      # average uptime / net production factor [-] [1, 2]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0      # USD/m³ filtered water
    carrier_cost: float = 0.0       # USD/m³ feed
    cartridge_cost: float = 0.0     # USD/m³ filtered water; folded into marginal_cost when cartridge_bus is absent
                                    # set 0.0 when cartridge_bus is active to avoid double counting with bus source

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
    target_turbidity_in: float = None           # NTU — design feed-quality assumption      [1]
    target_turbidity_out: float = None          # NTU — claimed treated-water target        [1]
    log_removal_credit_claim: float = None      # log — barrier contribution                [1]
    max_headloss_m: float = None                # m — design operational headloss limit     [2, 3]
    replacement_interval_h: float = None        # h — consumable preprocessing assumption   [3]

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
        self.cartridge_bus = attributes.pop("cartridge_bus", None)
        self.reject_water_bus = attributes.pop("reject_water_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.efficiency = attributes.pop("efficiency", self.efficiency)
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.specific_cartridge_consumption = attributes.pop(
            "specific_cartridge_consumption", self.specific_cartridge_consumption
        )
        self.availability = attributes.pop("availability", self.availability)

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.cartridge_cost = attributes.pop("cartridge_cost", self.cartridge_cost)
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
        self.target_turbidity_in = attributes.pop(
            "target_turbidity_in", self.target_turbidity_in
        )
        self.target_turbidity_out = attributes.pop(
            "target_turbidity_out", self.target_turbidity_out
        )
        self.log_removal_credit_claim = attributes.pop(
            "log_removal_credit_claim", self.log_removal_credit_claim
        )
        self.max_headloss_m = attributes.pop(
            "max_headloss_m", self.max_headloss_m
        )
        self.replacement_interval_h = attributes.pop(
            "replacement_interval_h", self.replacement_interval_h
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._feedwater_per_permeate = 1.0 / self.efficiency
        self._reject_per_permeate = (1.0 / self.efficiency) - 1.0

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_permeate
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.reject_water_bus is not None:
            attributes[f"conversion_factor_{self.reject_water_bus.label}"] = sequence(
                self._reject_per_permeate
            )

        if self.cartridge_bus is not None:
            attributes[f"conversion_factor_{self.cartridge_bus.label}"] = sequence(
                self.specific_cartridge_consumption
            )

        # --------------------------------------------------------------
        # availability as activity bound
        # --------------------------------------------------------------
        if self.availability is not None:
            attributes["activity_bound_max"] = sequence(self.availability * self.capacity)

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
        total_marginal_cost = self.marginal_cost + self.cartridge_cost

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

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 1
        # inputs
        for bus in [
            self.cartridge_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.reject_water_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1].")

        if self.cartridge_cost < 0:
            raise ValueError("cartridge_cost must be >= 0.")

        # Validate scalar availability; sequences are left to the user
        if self.availability is not None:
            try:
                if not 0 <= self.availability <= 1:
                    raise ValueError("availability must be between 0 and 1.")
            except TypeError:
                pass

        # Validate scalar specific_cartridge_consumption
        try:
            if self.specific_cartridge_consumption < 0:
                raise ValueError("specific_cartridge_consumption must be >= 0.")
        except TypeError:
            pass

        # Validate scalar specific_energy_consumption
        try:
            if self.specific_energy_consumption < 0:
                raise ValueError("specific_energy_consumption must be >= 0.")
        except TypeError:
            pass

        if self.cartridge_bus is not None and self.cartridge_cost not in (0, 0.0, None):
            self.cartridge_cost = 0.0
            warnings.warn(
                "Both 'cartridge_bus' and 'cartridge_cost' are set. "
                "'cartridge_cost' will be ignored because consumable usage "
                "is already modeled through 'cartridge_bus'.",
                UserWarning,
            )