import dataclasses
import warnings
from typing import Sequence, Union, Optional

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
    biokinetic reactor model (e.g. ASM1/ASM2d).

    Core references
    ---------------
    1. Zhou et al. (2001), Wat. Res.: specific energy and carbon-source
       stoichiometry for biological denitrification at full scale.
    2. EPA (2010), Nutrient Control Design Manual (EPA/600/R-10/100):
       design parameters, carbon-source dosing, and effluent standards.
    3. Minnesota PCA Denitrification Guidance: operational ranges for
       SEC and carbon demand used to set sec_typical_min/max defaults.

    Main equations
    --------------
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
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum
      treated-water throughput of the unit.
    - anoxic_factor derates the nominal removal efficiency for imperfect
      anoxic conditions (e.g. dissolved oxygen intrusion). Set to 1.0 for
      ideal anoxic operation.
    - carbon_source_dose overrides stoichiometric dosing when set. A warning
      is raised so the modeler is aware of the override. Set to None to use
      the cod_demand_per_no3n_removed stoichiometry instead.
    - N2O emissions are only tracked in the optimization model if n2o_bus
      is provided. If n2o_emission_factor > 0 without n2o_bus, a warning is
      raised and N2O is stored in custom_attributes only.
    - Effluent quality (Cout_no3n_mg_per_L) is always written to
      custom_attributes on output_parameters for post-processing, regardless
      of whether variable costs are non-zero.
    - Characterization values (sec_typical_min/max, retention_time_days,
      temperature_c) are stored as metadata for scenario documentation.
      They are not enforced as hard optimization constraints in v3.0.
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
    # Zhou et al., 2001 / Minnesota PCA / EPA, 2010
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.008          # kWh / m³ treated water
    Cin_no3n: float = 30.0                              # g N / m³ influent
    target_removal_efficiency: float = 0.90             # 0..1
    anoxic_factor: float = 1.0                          # 0..1; derates removal for imperfect anoxic conditions
    cod_demand_per_no3n_removed: float = 2.86           # kg COD / kg NO3-N removed
    n2_yield_per_no3n_removed: float = 1.0              # kg N2 / kg NO3-N removed
    n2o_emission_factor: float = 0.0                    # kg N2O / kg NO3-N removed
    carbon_source_dose: Optional[float] = 90.0          # g/m³ = mg/L

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                          # € / m³ treated water
    carrier_cost: float = 0.0                           # € / kWh electricity
    carbon_source_cost: float = 0.40                    # € / kg COD-equivalent

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # Zhou et al. (2001) / EPA (2010) style characterization fields
    # ------------------------------------------------------------------
    sec_typical_min: float = 0.005  # kWh/m³, lower bound from literature
    sec_typical_max: float = 0.05  # kWh/m³, upper bound from literature
    retention_time_days: Optional[float] = None  # days; documentation only
    temperature_c: Optional[float] = None  # °C; documentation only

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

        n2_production_per_m3 = no3n_removed_per_m3 * self.n2_yield_per_no3n_removed

        n2o_production_per_m3 = no3n_removed_per_m3 * self.n2o_emission_factor

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
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # --------------------------------------------------------------
        residual_output_variable_costs = 0.0
        if self.carbon_source_bus is None:
            residual_output_variable_costs += (
                    carbon_demand_per_m3 * self.carbon_source_cost
            )

        attributes.setdefault("output_parameters", {})
        attributes["output_parameters"]["custom_attributes"] = {
            "Cout_no3n_mg_per_L": Cout_no3n,
        }

        if residual_output_variable_costs > 0:
            attributes["output_parameters"]["variable_costs"] = (
                residual_output_variable_costs
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
                f"{self.carbon_source_cost} €/kg is set on this facade, "
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