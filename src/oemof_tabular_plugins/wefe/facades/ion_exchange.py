import dataclasses
import warnings
from typing import Sequence, Union, Optional
import numpy as np
from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class IonExchange(MIMO):
    """
    Literature-informed ion exchange water treatment facade based on MIMO.

    Purpose
    -------
    Generic water-treatment facade for ion exchange (IX) systems used in
    desalination pre-treatment, softening, nitrate removal, and similar
    applications. The model is designed as a bookkeeping/process-yield unit
    representing an IX unit as a water-treatment intervention. It is not a
    full breakthrough-cycle or contaminant-transport model.

    Core references
    ---------------
    1. Steady-state model variables, cost structure, resin capacity and selectivity as key sensitivities.
       Sitterley, K. A., & Dudchenko, A. (2023). WaterTAP technical brief: Ion exchange model demonstration and optimization
       (NREL/TP-5700-86512). National Renewable Energy Laboratory. https://doi.org/10.2172/2005544
    2. Regeneration steps, exhaustion behavior, practical operating limits, and cost relevance of regenerant disposal.
       Veolia Water Technologies. (n.d.). Handbook of industrial water treatment, Chapter 8 — Ion exchange, water demineralization
       & resin testing. Veolia. https://www.watertechnologies.com/handbook/chapter-08-ion-exchange
    3. Service/breakthrough/regeneration logic and parallel-vessel engineering simplifications.
       Ludwigson, M. (2023). Ion exchange for water treatment (Course 454). PDH Academy.
       https://pdhacademy.com/wp-content/uploads/2023/09/454-Ion-Exchange-for-Water-Treatment.pdf
    4. Gold-standard design reference for operating modes, resin selection, and system design.
       Wachinski, A. M. (2004). Ion exchange treatment for water. American Water Works Association.
       https://www.abebooks.com/9781583213223/Ion-Exchange-Treatment-Water-Wachinski-1583213228/plp

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Feedwater requirement:
        feedwater_per_output = 1 / efficiency       [m³_feed / m³_product]

    Electricity demand:
        electricity_per_output = SEC                    [kWh / m³_product]

    Regenerant demand:
        regenerant_per_output = regenerant_dose_kg_per_m3_product
                                                        [kg_chem / m³_product]

    Brine reject output:
        brine_per_output = 1 / efficiency - 1.0     [m³_brine / m³_product]

    Output costs (placed on flow edges, not lumped into marginal_cost):
        water_out_bus:   resin_replacement_cost_per_m3_product  [€/m³]
        waste_brine_bus: waste_disposal_cost_per_m3_product     [€/m³]


    Availability derating:
        Q_net(t) <= availability * Q_cap                [m³/hr]

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum treated-water throughput of the IX unit.
    - regenerant_bus is the correct optional INPUT for chemical supply (HCl / NaOH / NaCl).
    - waste_brine_bus is the correct optional OUTPUT for spent regenerant / brine reject.
    - availability is implemented as an activity_bound_max constraint, not as a hidden multiplier inside SEC or efficiency,
      to keep cost interpretation unambiguous.
    - Resin capacity, selectivity, breakthrough, and service-loading fields are kept for scenario documentation and future extension.
    - Characterization values (resin capacity, selectivity, service loading, breakthrough metrics) are stored as 
      documentation/calibration defaults. They are not enforced as hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "ion_exchange"
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

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    regenerant_bus: Optional[Bus] = None  # kg  (HCl / NaOH / NaCl supply)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    waste_brine_bus: Optional[Bus] = None  # m³  (spent regenerant / brine reject)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.06           # kWh / m³ treated water          [1]
    efficiency: float = 0.96                            # m³ treated / m³ feed (water recovery) [2, 3]
    availability: float = 1.0                           # fraction of productive operating time [3]
    regenerant_dose_kg_per_m3_product: float = 0.0      # kg chemical / m³ treated water   [2, 3]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                           # USD/m³ treated water
    carrier_cost: float = 0.0                            # USD/m³ feed
    waste_disposal_cost_per_m3_product: float = 0.0      # USD/m³ brine
    resin_replacement_cost_per_m3_product: float = 0.0   # USD/m³ treated water

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
    sec_typical_min: float = 0.03                       # kWh/m³ literature lower bound                 [1]
    sec_typical_max: float = 0.10                       # kWh/m³ literature upper bound                 [1]
    target_ion: str = ""                                # e.g. "nitrate", "hardness", "fluoride"        [4]
    influent_concentration_mgL: float = None            # mg/L feed contaminant concentration           [1]
    resin_capacity_eq_per_m3_resin: float = None        # total exchange capacity                       [1, 4]
    resin_selectivity: float = None                     # selectivity coefficient                       [1, 4]
    service_flow_rate_bvph: float = None                # BV/hr in service mode                         [3, 4]
    breakthrough_fraction: float = None                 # C/C₀ at exhaustion                            [3]
    breakthrough_time_h: float = None                   # time to breakthrough                          [3]
    regenerant_type: str = ""                           # e.g. "HCl", "NaOH", "NaCl"                    [2, 4]
    regenerant_dose_relative_to_stoich: float = None    # excess factor                                 [2]
    bed_volumes_per_cycle: float = None                 # service BV between regens                     [2, 3]
    parallel_trains: int = None                         # number of parallel IX vessels  [3]

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
        self.regenerant_bus = attributes.pop("regenerant_bus", None)
        self.waste_brine_bus = attributes.pop("waste_brine_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop("efficiency", self.efficiency)

        self.availability = attributes.pop("availability", self.availability)
        self.regenerant_dose_kg_per_m3_product = attributes.pop(
            "regenerant_dose_kg_per_m3_product", self.regenerant_dose_kg_per_m3_product
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.waste_disposal_cost_per_m3_product = attributes.pop(
            "waste_disposal_cost_per_m3_product", self.waste_disposal_cost_per_m3_product
        )
        self.resin_replacement_cost_per_m3_product = attributes.pop(
            "resin_replacement_cost_per_m3_product", self.resin_replacement_cost_per_m3_product
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
        self.sec_typical_min = attributes.pop("sec_typical_min", self.sec_typical_min)
        self.sec_typical_max = attributes.pop("sec_typical_max", self.sec_typical_max)
        self.target_ion = attributes.pop("target_ion", self.target_ion)
        self.influent_concentration_mgL = attributes.pop(
            "influent_concentration_mgL", self.influent_concentration_mgL
        )
        self.resin_capacity_eq_per_m3_resin = attributes.pop(
            "resin_capacity_eq_per_m3_resin", self.resin_capacity_eq_per_m3_resin
        )
        self.resin_selectivity = attributes.pop(
            "resin_selectivity", self.resin_selectivity
        )
        self.service_flow_rate_bvph = attributes.pop(
            "service_flow_rate_bvph", self.service_flow_rate_bvph
        )
        self.breakthrough_fraction = attributes.pop(
            "breakthrough_fraction", self.breakthrough_fraction
        )
        self.breakthrough_time_h = attributes.pop(
            "breakthrough_time_h", self.breakthrough_time_h
        )
        self.regenerant_type = attributes.pop(
            "regenerant_type", self.regenerant_type
        )
        self.regenerant_dose_relative_to_stoich = attributes.pop(
            "regenerant_dose_relative_to_stoich",
            self.regenerant_dose_relative_to_stoich,
        )
        self.bed_volumes_per_cycle = attributes.pop(
            "bed_volumes_per_cycle", self.bed_volumes_per_cycle
        )
        self.parallel_trains = attributes.pop(
            "parallel_trains", self.parallel_trains
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._feedwater_per_output = 1.0 / self.efficiency
        self._brine_per_output = self._feedwater_per_output - 1.0

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.regenerant_bus is not None:
            attributes[f"conversion_factor_{self.regenerant_bus.label}"] = sequence(
                self.regenerant_dose_kg_per_m3_product
            )

        if self.waste_brine_bus is not None:
            attributes[f"conversion_factor_{self.waste_brine_bus.label}"] = sequence(
                self._brine_per_output
            )

        # --------------------------------------------------------------
        # availability as activity bound
        # --------------------------------------------------------------
        if self.availability < 1.0:
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
        total_marginal_cost = np.add(self.marginal_cost, self.resin_replacement_cost_per_m3_product)

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

        if self.waste_brine_bus is not None and self.waste_brine_bus in self.outputs:
            self.outputs[self.waste_brine_bus].variable_costs = sequence(
                self.waste_disposal_cost_per_m3_product
            )

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 1
        # inputs
        for bus in [
            self.regenerant_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.waste_brine_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1].")
        if not 0 < self.availability <= 1:
            raise ValueError("availability must be in (0, 1].")
        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")
        if self.regenerant_dose_kg_per_m3_product < 0:
            raise ValueError("regenerant_dose_kg_per_m3_product must be >= 0.")

        if self.regenerant_bus is not None and self.regenerant_dose_kg_per_m3_product == 0:
            warnings.warn(
                "regenerant_bus connected but regenerant_dose_kg_per_m3_product=0 "
                "— regenerant flow will be zero.",
                UserWarning,
            )
        if self.regenerant_dose_kg_per_m3_product > 0 and self.regenerant_bus is None:
            warnings.warn(
                "regenerant_dose_kg_per_m3_product > 0 but no regenerant_bus — "
                "chemical consumption not tracked in the energy system graph.",
                UserWarning,
            )

        if self.breakthrough_fraction is not None and not 0 < self.breakthrough_fraction <= 1:
            raise ValueError("breakthrough_fraction must be in (0, 1].")
        if self.resin_capacity_eq_per_m3_resin is not None and self.resin_capacity_eq_per_m3_resin <= 0:
            raise ValueError("resin_capacity_eq_per_m3_resin must be > 0.")
