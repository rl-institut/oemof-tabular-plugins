import dataclasses
import warnings
from typing import Sequence, Union, Optional
import numpy as np
from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class BiologicalDenitrification(MIMO):
    """
    Literature-informed biological denitrification facade based on MIMO.

    Purpose
    -------
    Generic anoxic nitrate-removal facade for biological denitrification
    units used in water treatment systems. The model is designed as a
    bookkeeping/process-yield unit representing a denitrification reactor
    as a nitrogen-removal intervention. It is not a full mechanistic
    biokinetic reactor model.

    Core references
    ---------------
    1. Theoretical stoichiometric derivation of biological denitrification reactions (based on McCarty's half-reaction
       theory), yielding the COD-per-nitrate-removed ratio.
       Zhou, S. Q. (2001). Theoretical stoichiometry of biological denitrifications. Environmental Technology, 22(8), 869–880.
       https://doi.org/10.1080/09593332208618223
    2. Design parameters, carbon-source dosing strategy, and effluent standards for nitrogen and phosphorus control
       retrofits at municipal WWTPs.
       Hertzler, P., Dufresne, L., Randall, C., Barnard, J., Stensel, D., & Brown, J. (2010). Nutrient control design manual
       (EPA/600/R-10/100). U.S. Environmental Protection Agency, Office of Research and Development.
       https://www.epa.gov/sites/default/files/2019-02/documents/nutrient-control-design-manual.pdf
    3. Practical operational guidance stating approximately 4 mg/L of BOD is required per 1 mg/L of nitrate removed.
       Minnesota Pollution Control Agency. (2024). Denitrification (Document No. wq-wwtp8-30). Minnesota Pollution Control Agency.
       https://www.pca.state.mn.us/sites/default/files/wq-wwtp8-30.pdf

    Main equations
    --------------
    All flows normalized to 1 m³ net treated water (primary output):

    Effective removal:
        eta_eff = target_removal_efficiency * anoxic_factor        [-]

    NO3-N removed per unit treated water:
        R_NO3 = Cin_no3n * eta_eff                                 [g N / m³]

    Effluent concentration:
        Cout = Cin_no3n * (1 - eta_eff)                            [g N / m³]

    N2 production (conversion factor on N2_gas_bus):
        f_N2 = R_NO3 * n2_yield_per_no3n_removed                  [g N2 / m³]

    N2O byproduct (conversion factor on n2o_bus, optional):
        f_N2O = R_NO3 * n2o_emission_factor                        [g N2O / m³]

    Carbon demand — two modes (mutually exclusive):
        a) Empirical override (carbon_source_dose is not None):
            C_demand = carbon_source_dose * 1e-3                   [kg COD / m³]
        b) Stoichiometric (carbon_source_dose = None):
            C_demand = R_NO3 * 1e-3 * cod_demand_per_no3n_removed  [kg COD / m³]

    Carbon cost routing:
        - carbon_source_bus present:  cost charged by upstream supply node.
        - carbon_source_bus absent:   C_demand * carbon_source_cost folded
                                      into output variable_costs.

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum treated-water throughput of the unit.
    - anoxic_factor derates the nominal removal efficiency for imperfect anoxic conditions (e.g. dissolved oxygen intrusion).
      Set to 1.0 for ideal anoxic operation.
    - carbon_source_dose overrides stoichiometric dosing when set. A warning is raised so the modeler is aware of the override.
    - N2O emissions are only tracked in the optimization model if n2o_bus is provided. If n2o_emission_factor > 0 without
      n2o_bus, a warning is raised.
    - Characterization values (sec_typical_min/max, retention_time_days, temperature_c) are stored as metadata for
      scenario documentation. They are not enforced as hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "biological_denitrification"
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
    water_in_bus: Bus = None                # m³ untreated water
    water_out_bus: Bus = None               # m³ treated water (PRIMARY)
    N2_gas_bus: Bus = None                  # kg N2 (or kg N-equivalent)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    carbon_source_bus: Optional[Bus] = None  # kg COD — physical carbon source input

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    n2o_bus: Optional[Bus] = None  # kg N2O — incomplete denitrification byproduct

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.008          # kWh / m³ treated water [2, 3]
    Cin_no3n: float = 30.0                              # g N / m³ influent [2]
    target_removal_efficiency: float = 0.90             # 0..1 [2]
    anoxic_factor: float = 1.0                          # 0..1; derates removal for imperfect anoxic conditions [2]
    cod_demand_per_no3n_removed: float = 2.86           # kg COD / kg NO3-N removed [1]
    n2_yield_per_no3n_removed: float = 1.0              # kg N2 / kg NO3-N removed [1]
    n2o_emission_factor: float = 0.0                    # kg N2O / kg NO3-N removed [2]
    carbon_source_dose: Optional[float] = 90.0          # g/m³ = mg/L [3]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                          # USD/m³ net treated water
    carrier_cost: float = 0.0                           # USD/m³ feedwater
    carbon_source_cost: float = 0.40                    # USD/kg COD-equivalent

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
    sec_typical_min: float = 0.005                  # kWh/m³, lower bound from literature [3]
    sec_typical_max: float = 0.05                   # kWh/m³, upper bound from literature [3]
    retention_time_days: Optional[float] = None     # days; documentation only [2]
    temperature_c: Optional[float] = None           # °C; documentation only [2]

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
        self.N2_gas_bus = attributes.pop("N2_gas_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.carbon_source_bus = attributes.pop("carbon_source_bus", None)
        self.n2o_bus = attributes.pop("n2o_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.Cin_no3n = attributes.pop("Cin_no3n", self.Cin_no3n)
        self.target_removal_efficiency = attributes.pop(
            "target_removal_efficiency", self.target_removal_efficiency
        )
        self.anoxic_factor = attributes.pop("anoxic_factor", self.anoxic_factor)
        self.cod_demand_per_no3n_removed = attributes.pop(
            "cod_demand_per_no3n_removed", self.cod_demand_per_no3n_removed
        )
        self.n2_yield_per_no3n_removed = attributes.pop(
            "n2_yield_per_no3n_removed", self.n2_yield_per_no3n_removed
        )
        self.n2o_emission_factor = attributes.pop(
            "n2o_emission_factor", self.n2o_emission_factor
        )
        self.carbon_source_dose = attributes.pop(
            "carbon_source_dose", self.carbon_source_dose
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.carbon_source_cost = attributes.pop(
            "carbon_source_cost", self.carbon_source_cost
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
        self.retention_time_days = attributes.pop(
            "retention_time_days", self.retention_time_days
        )
        self.temperature_c = attributes.pop("temperature_c", self.temperature_c)

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        effective_removal_efficiency = (
                self.target_removal_efficiency * self.anoxic_factor
        )
        no3n_removed_per_m3 = self.Cin_no3n * effective_removal_efficiency
        Cout_no3n = self.Cin_no3n * (1.0 - effective_removal_efficiency)

        n2_production_per_m3 = no3n_removed_per_m3 * self.n2_yield_per_no3n_removed * 1e-3

        n2o_production_per_m3 = no3n_removed_per_m3 * self.n2o_emission_factor * 1e-3

        if self.carbon_source_dose is not None:
            warnings.warn(
                "carbon_source_dose used instead of stoichiometric dosing. "
                "Set carbon_source_dose=None to use cod_demand_per_no3n_removed.",
                UserWarning,
            )
            carbon_demand_per_m3 = self.carbon_source_dose * 1e-3
        else:
            no3n_removed_kg_per_m3 = no3n_removed_per_m3 * 1e-3
            carbon_demand_per_m3 = (
                    no3n_removed_kg_per_m3 * self.cod_demand_per_no3n_removed
            )

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self.specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.N2_gas_bus.label}"] = sequence(
            n2_production_per_m3
        )

        if self.carbon_source_bus is not None:
            attributes[f"conversion_factor_{self.carbon_source_bus.label}"] = sequence(
                carbon_demand_per_m3
            )

        if self.n2o_bus is not None:
            attributes[f"conversion_factor_{self.n2o_bus.label}"] = sequence(
                n2o_production_per_m3
            )

        # --------------------------------------------------------------
        # output-specific variable costs
        # --------------------------------------------------------------
        self.residual_output_variable_costs = 0.0
        if self.carbon_source_bus is None:
            self.residual_output_variable_costs = (
                    carbon_demand_per_m3 * self.carbon_source_cost
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
        elif self.primary == "N2_gas_bus":
            primary_label = self.N2_gas_bus.label
        else:
            primary_label = self.primary

        # --------------------------------------------------------------
        # initialize base MIMO facade
        # --------------------------------------------------------------
        super().__init__(
            from_bus_0=self.electricity_bus,
            from_bus_1=self.water_in_bus,
            to_bus_0=self.water_out_bus,
            to_bus_1=self.N2_gas_bus,
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
        total_marginal_cost = np.add(self.marginal_cost, self.residual_output_variable_costs)

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
        idx_out = 2
        # inputs
        for bus in [
            self.carbon_source_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.n2o_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if self.Cin_no3n < 0:
            raise ValueError("Cin_no3n must be >= 0.")
        if not 0 <= self.target_removal_efficiency <= 1:
            raise ValueError("target_removal_efficiency must be in [0, 1].")
        if not 0 <= self.anoxic_factor <= 1:
            raise ValueError("anoxic_factor must be in [0, 1].")
        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        for param_name, value in {
            "cod_demand_per_no3n_removed": self.cod_demand_per_no3n_removed,
            "n2_yield_per_no3n_removed": self.n2_yield_per_no3n_removed,
            "n2o_emission_factor": self.n2o_emission_factor,
            "carbon_source_cost": self.carbon_source_cost,
            "carrier_cost": self.carrier_cost,
        }.items():
            if value < 0:
                raise ValueError(f"{param_name} must be >= 0.")

        if self.carbon_source_dose is not None and self.carbon_source_dose < 0:
            raise ValueError("carbon_source_dose must be >= 0 when provided.")

        if self.n2o_emission_factor > 0 and self.n2o_bus is None:
            warnings.warn(
                "n2o_emission_factor > 0 but no n2o_bus provided. "
                "N2O production is tracked in custom_attributes only and "
                "has no effect on the optimization model.",
                UserWarning,
            )

        if self.carbon_source_bus is not None and self.carbon_source_cost > 0:
            warnings.warn(
                f"carbon_source_bus is provided and carbon_source_cost="
                f"{self.carbon_source_cost} USD/kg is set on this facade, "
                f"but it will NOT be applied here. Ensure the upstream supply "
                f"node on '{self.carbon_source_bus.label}' carries the "
                f"procurement cost as marginal_cost.",
                UserWarning,
            )

        if (
                self.sec_typical_min is not None
                and self.sec_typical_max is not None
                and not (
                self.sec_typical_min
                <= self.specific_energy_consumption
                <= self.sec_typical_max
        )
        ):
            warnings.warn(
                f"specific_energy_consumption={self.specific_energy_consumption} "
                f"kWh/m³ is outside the typical literature range "
                f"[{self.sec_typical_min}, {self.sec_typical_max}] kWh/m³.",
                UserWarning,
            )