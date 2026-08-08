import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class IntakeStructure(MIMO):
    """
    Literature-informed surface-water intake structure facade based on MIMO.

    Purpose
    -------
    Lumped linear abstraction model for canal or river-type surface-water intake
    structures. The model represents an intake as a water-abstraction and
    screening unit that converts electricity and raw water into usable abstracted
    water. It is not a detailed hydraulic design model or sediment-transport
    simulation.

    Core references
    ---------------
    1. Hydrology, abstraction availability, and seasonal source constraints for planning intake structures.
       Lauterjung, H., & Schmidt, G. (1989). Planning of water intake structures for irrigation or hydropower.
       Deutsche Gesellschaft fuer Technische Zusammenarbeit (GTZ) GmbH.
       https://www.ircwash.org/sites/default/files/Lauterjung-1989-Planning.pdf
    2. Sediment as a core intake-performance issue - basis for the lumped sediment_rejection_factor.
       Scheuerlein, H., & Mtalo, F. (2009). Sediment exclusion at river intakes. In Fresh Surface Water (Vol. III).
       Encyclopedia of Life Support Systems (EOLSS), UNESCO.
       http://www.eolss.net/sample-chapters/c07/e2-07-05-04.pdf
    3. Operational risk from sediment build-up, vegetation, blockage, and source-quality events - basis for availability/
       activity bound modeling.
       World Health Organization. Sanitary inspection package: Surface water source and intake.
       https://cdn.who.int/media/docs/default-source/wash-documents/water-safety-and-quality/water-safety-planning/sanitary-inspection-packages/7.-surface-water-source-and-intake_web.pdf

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Total specific energy consumption:
        SEC_tot = SEC_main + SEC_aux
        [kWh/m³]  [kWh/m³]  [kWh/m³]

    Effective abstraction efficiency:
        eta_eff = eta_abs * (1 - phi_sed)
        where:
            eta_abs  = efficiency    [-]
            phi_sed  = sediment_rejection_factor [-]

    Raw water required per unit usable-water output:
        w_in = 1 / eta_eff
        [m³_raw / m³_useful]

    Optional sediment reject stream:
        w_sed = w_in - 1.0
        [m³_reject / m³_useful]

    Availability-bound on usable-water output:
        Q_out(t) <= availability(t) * capacity
        Implemented via activity_bound_max.

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum usable-water output of the intake,
      representing the design abstraction capacity of the structure.
    - Sediment effects are represented as a lumped effective abstraction efficiency derating. If sediment_out_bus is provided,
      the reject volume appears as a separate output stream; otherwise losses are folded silently into eta_eff.
    - Operational interruptions from hydrology, blockage, vegetation, or source-quality events are represented via a
      time-varying availability profile applied as an activity upper bound on the output.
    - Detailed siting, intake elevation, screen geometry, flood resilience, and hydraulic design checks remain outside
      the optimization model and must be handled during data preparation and engineering design.
    - Documentation-only fields (intake_type, source_type, screen_type, siting_comment) are stored as metadata for scenario
      documentation. They are not enforced as hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "intake_structure"
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
    water_in_bus: Bus = None        # m³ (raw surface water)
    water_out_bus: Bus = None       # m³ (usable abstracted water - PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    sediment_out_bus: Optional[Bus] = None  # m³ or kg (sediment reject, future use)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.008              # kWh/m³ [1]
    auxiliary_energy_consumption: float = 0.0               # kWh/m³ [1]
    efficiency: float = 0.98                                # [-]  scalar only [1]
    sediment_rejection_factor: float = 0.0                  # [-]  scalar only [2]
    availability: Union[float, Sequence[float]] = None      # [-] None = no bound [3]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0  # USD/m³ filtered water
    carrier_cost: float = 0.0   # USD/m³ raw surface water

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
    intake_type: str = ""               # e.g. "canal", "river", "reservoir", "floating" [1]
    source_type: str = ""               # e.g. "river", "canal", "lake" [1, 3]
    screen_type: str = ""               # e.g. "coarse", "fine", "Coanda" [2]
    siting_comment: str = ""            # free-text note on siting assumptions [1, 3]

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
        self.sediment_out_bus = attributes.pop("sediment_out_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.auxiliary_energy_consumption = attributes.pop(
            "auxiliary_energy_consumption", self.auxiliary_energy_consumption
        )
        self.efficiency = attributes.pop(
            "efficiency", self.efficiency
        )
        self.sediment_rejection_factor = attributes.pop(
            "sediment_rejection_factor", self.sediment_rejection_factor
        )
        self.availability = attributes.pop("availability", self.availability)

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
        self.intake_type = attributes.pop("intake_type", self.intake_type)
        self.source_type = attributes.pop("source_type", self.source_type)
        self.screen_type = attributes.pop("screen_type", self.screen_type)
        self.siting_comment = attributes.pop("siting_comment", self.siting_comment)

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # (Scheuerlein & Mtalo [2]; Lauterjung & Schmidt [1])
        # --------------------------------------------------------------
        self._eta_eff = self.efficiency * (1.0 - self.sediment_rejection_factor)
        self._w_in_per_out = 1.0 / self._eta_eff            # m³_raw / m³_useful
        self._w_sed_per_out = self._w_in_per_out - 1.0      # m³_reject / m³_useful
        self._total_sec = (
                self.specific_energy_consumption + self.auxiliary_energy_consumption
        )                                                   # kWh / m³_useful

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._total_sec
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._w_in_per_out
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.sediment_out_bus is not None:
            attributes[f"conversion_factor_{self.sediment_out_bus.label}"] = sequence(
                self._w_sed_per_out
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
            self.sediment_out_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1].")

        if not 0 <= self.sediment_rejection_factor < 1:
            raise ValueError("sediment_rejection_factor must be in [0, 1).")

        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        if self.auxiliary_energy_consumption < 0:
            raise ValueError("auxiliary_energy_consumption must be >= 0.")

            # Validate scalar availability; sequences are left to the user
        if self.availability is not None:
            try:
                if not 0 <= self.availability <= 1:
                    raise ValueError("availability must be between 0 and 1.")
            except TypeError:
                pass

        if self.sediment_out_bus is None and self.sediment_rejection_factor != 0.0:
            warnings.warn(
                "sediment_rejection_factor > 0 but no sediment_out_bus provided. "
                "Sediment losses are folded into effective abstraction efficiency "
                "and do not appear as a separate material stream.",
                UserWarning,
            )