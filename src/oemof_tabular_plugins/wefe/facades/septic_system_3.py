import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO

@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class SepticSystem(MIMO):
    """
    Literature-informed septic system facade based on MIMO.

    Purpose
    -------
    Linear planning surrogate for a conventional septic tank / onsite primary
    treatment unit. Models hydraulic split between clarified liquid effluent and
    sludge/scum residual, optional electricity demand for pumped variants, and
    an optional methane output for environmental accounting. It is not a full
    mechanistic biological reactor model.

    Core references
    ---------------
    1. Septic tank sizing, HRT 24-48 h, desludging intervals 2-3 yr, BOD removal 30-50 %, TSS removal 40-60 %.
       Shrestha, R. (2020). Septic tank design manual. Environment and Public Health Organization (ENPHO).
       https://enpho.org/wp-content/uploads/2021/10/SepticTankManualEnpho-Book.pdf
    2. Wastewater characteristics, effluent/septage quality ranges, and conventional septic tank role in primary treatment.
       Otis, R., Kreissl, J. F., Frederick, R., Goo, R., Casey, P., & Tonning, B. (2002).
       Onsite wastewater treatment systems manual (EPA/625/R-00/008). U.S. Environmental Protection Agency.
       https://www.epa.gov/sites/default/files/2015-06/documents/2004_07_07_septics_septic_2002_osdm_all.pdf
    3. Sludge accumulation and methane conversion in septic tanks; HRT, COD, and temperature are governing variables.
       Elmitwalli, T. (2013). Sludge accumulation and conversion to methane in a septic tank treating domestic
       wastewater or black water. Water Science and Technology, 68(4), 956–964. https://doi.org/10.2166/wst.2013.337
    4. Desludging interval affects BOD removal and CH4 yield; supports documentation-only checks for desludging_interval_years.
       Moonkawin, J., Huynh, L. T., & Schneider, M. Y. (2023). Challenges to accurate estimation
       of methane emission from septic tanks with long emptying intervals. Environmental Science & Technology.
       https://doi.org/10.1021/acs.est.3c05724 (Free full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC10621000/)
    5. Specific energy consumption
       Chesley, J. (2025, August 8). Beyond the grid: Exploring decentralized wastewater treatment. Ecologix Systems.
       https://ecologixsystems.com/articles/beyond-the-grid-decentralized-wastewater-treatment

    Main equations
    --------------
    Hydraulic split (normalized to treated liquid output = 1):

        feedwater_per_output = 1 / efficiency
        sludge_per_output    = sludge_yield_per_m3_in / efficiency

    If sludge_yield_per_m3_in == (1 - efficiency):
        Q_in(t) = Q_out(t) + Q_sludge(t)

    Electricity consumption:
        E(t) = specific_energy_consumption * Q_out(t)

    Optional methane proxy:
        Q_CH4(t) = methane_emission_factor_per_m3_in / efficiency * Q_out(t)

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains maximum treated liquid throughput.
    - Electricity input is optional. Set specific_energy_consumption=0.0 and omit electricity_bus for passive gravity-fed septic tanks.
    - CH4 output is optional and linear. If ch4_out_bus is not provided, methane_emission_factor_per_m3_in is stored as metadata only.
    - BOD/TSS/COD removal, HRT, and desludging interval are validated and documented but NOT enforced as dispatch constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "septic_system"
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
    water_in_bus: Bus = None    # m³
    water_out_bus: Bus = None   # m³  (PRIMARY)
    sludge_out_bus: Bus = None  # m³

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    electricity_bus: Optional[Bus] = None  # kWh — activate for pumped/aerated variants

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    ch4_out_bus: Optional[Bus] = None  # proxy unit — activate for detailed GHG accounting

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 0.13       # kWh/m³; 0.0 for passive gravity flow [Ecologix]
    efficiency: float = 0.85                        # m³ effluent / m³ influent (liquid recovery) [ENPHO]
    sludge_yield_per_m3_in: float = None            # defaults to 1 - efficiency
    methane_emission_factor_per_m3_in: float = 0.0  # m³ CH4 / m³ influent [Elmitwalli]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0          # USD/m³ treated water
    carrier_cost: float = 0.0           # USD/m³ grey water
    sludge_disposal_cost: float = 0.0   # USD/m³ sludge
    ghg_cost: float = 0.0               # USD/ proxy unit CH4 output

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints)
    # Based on ENPHO (2020) and CH4 paper (ACS ES&T, 2023)
    # ------------------------------------------------------------------
    bod_removal: float = 0.40                         # fraction  [ENPHO: 30-50%]
    tss_removal: float = 0.50                         # fraction  [ENPHO: 40-60%]
    cod_removal: Optional[float] = None               # fraction  [ENPHO]
    minimum_hydraulic_retention_time_h: float = 24.0  # hr  [ENPHO]
    desludging_interval_years: float = 2.5            # yr  [ENPHO; CH4 paper]

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
        self.water_in_bus = attributes.pop("water_in_bus")
        self.water_out_bus = attributes.pop("water_out_bus")
        self.sludge_out_bus = attributes.pop("sludge_out_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.electricity_bus = attributes.pop("electricity_bus", None)
        self.ch4_out_bus = attributes.pop("ch4_out_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop(
            "efficiency", self.efficiency
        )
        self.sludge_yield_per_m3_in = attributes.pop(
            "sludge_yield_per_m3_in", self.sludge_yield_per_m3_in
        )
        self.methane_emission_factor_per_m3_in = attributes.pop(
            "methane_emission_factor_per_m3_in",
            self.methane_emission_factor_per_m3_in,
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.sludge_disposal_cost = attributes.pop("sludge_disposal_cost", self.sludge_disposal_cost)
        self.ghg_cost = attributes.pop("ghg_cost", self.ghg_cost)
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
        self.bod_removal = attributes.pop("bod_removal", self.bod_removal)
        self.tss_removal = attributes.pop("tss_removal", self.tss_removal)
        self.cod_removal = attributes.pop("cod_removal", self.cod_removal)
        self.minimum_hydraulic_retention_time_h = attributes.pop(
            "minimum_hydraulic_retention_time_h",
            self.minimum_hydraulic_retention_time_h,
        )
        self.desludging_interval_years = attributes.pop(
            "desludging_interval_years", self.desludging_interval_years
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived hydraulic split ratios
        # normalized to treated liquid output [m³/hr] = 1
        # If sludge_yield_per_m3_in is None → defaults to (1 - efficiency)
        # giving the strict identity: Q_in = Q_out + Q_sludge
        # If set explicitly → independent empirical yield factor.
        # --------------------------------------------------------------
        if self.sludge_yield_per_m3_in is None:
            self.sludge_yield_per_m3_in = 1.0 - self.efficiency

        self._feedwater_per_output = 1.0 / self.efficiency
        self._sludge_per_output = self.sludge_yield_per_m3_in / self.efficiency
        self._electricity_per_output = self.specific_energy_consumption
        self._methane_per_output = (
                self.methane_emission_factor_per_m3_in / self.efficiency
        )

        # --------------------------------------------------------------
        # conversion factors
        # normalized to treated liquid output [m³/hr] = 1
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(self._feedwater_per_output)
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.sludge_out_bus.label}"] = sequence(self._sludge_per_output)
        if self.electricity_bus is not None:
            attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
                self._electricity_per_output
            )
        if self.ch4_out_bus is not None:
            attributes[f"conversion_factor_{self.ch4_out_bus.label}"] = sequence(
                self._methane_per_output
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
        else:
            primary_label = self.primary

        # --------------------------------------------------------------
        # initialize base MIMO facade
        # --------------------------------------------------------------
        super().__init__(
            from_bus_0=self.water_in_bus,
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

        if self.ch4_out_bus is not None and self.ch4_out_bus in self.outputs:
            self.outputs[self.ch4_out_bus].variable_costs = sequence(self.ghg_cost)


    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 1
        idx_out = 2
        # inputs
        for bus in [
            self.electricity_bus
        ]:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.ch4_out_bus
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
        if self.methane_emission_factor_per_m3_in < 0:
            raise ValueError("methane_emission_factor_per_m3_in must be >= 0.")
        if self.sludge_yield_per_m3_in is not None:
            if not 0 <= self.sludge_yield_per_m3_in <= 1:
                raise ValueError("sludge_yield_per_m3_in must be in [0, 1].")
        bounded = {
            "bod_removal": self.bod_removal,
            "tss_removal": self.tss_removal,
        }
        for name, value in bounded.items():
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        if self.cod_removal is not None and not 0 <= self.cod_removal <= 1:
            raise ValueError("cod_removal must be in [0, 1] when provided.")
        if self.minimum_hydraulic_retention_time_h <= 0:
            raise ValueError("minimum_hydraulic_retention_time_h must be > 0.")
        if self.desludging_interval_years <= 0:
            raise ValueError("desludging_interval_years must be > 0.")
        if self.ch4_out_bus is None and self.methane_emission_factor_per_m3_in > 0:
            warnings.warn(
                "methane_emission_factor_per_m3_in > 0 but no ch4_out_bus provided. "
                "Methane factor is stored as metadata only and will not be modeled.",
                UserWarning,
            )