import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class ConstructedWetland(MIMO):
    """
    Literature-informed constructed wetland facade based on MIMO.

    Purpose
    -------
    Generic water treatment facade representing a constructed wetland as a
    low-energy biological treatment unit for rural and peri-urban wastewater.
    The model is designed as a process-yield unit using fixed hydraulic
    conversion ratios. It is not a full mechanistic biological reactor model.

    Core references
    ---------------
    1. Core process concept, wetland types, treatment mechanisms, and performance/modelling boundaries.
       Brown, D. S., Kreissl, J. F., Gearhart, R. A., et al. (2000). Constructed wetlands treatment of municipal wastewaters
       (EPA/625/R-99/010). U.S. Environmental Protection Agency, Office of Research and Development.
       https://nepis.epa.gov/Exe/ZyNET.exe/30004TBD.txt?Client=EPA&Index=1995%20Thru%201999
    2. Design-level fields and hydraulic design variables (HRT, HLR, hydrology, substrate, vegetation, water balance).
       Davis, L. (1995). A handbook of constructed wetlands: A guide to creating wetlands for agricultural wastewater,
       domestic wastewater, coal mine drainage, stormwater in the Mid-Atlantic region. USDA Natural Resources Conservation
       Service & U.S. EPA Region III. https://www.epa.gov/sites/default/files/2015-10/documents/constructed-wetlands-handbook.pdf.
    3. Parameter selection and validation logic for design-level fields, including macrophyte species selection tied to
       pollutant-removal performance.
       Rahman, M. E., Bin Halmi, M. I. E., Bin Abd Samad, M. Y., Uddin, M. K., Mahmud, K., Abd Shukor, M. Y., Sheikh
       Abdullah, S. R., & Shamsuzzaman, S. M. (2020). Design, operation and optimization of constructed wetland for removal
       of pollutant. International Journal of Environmental Research and Public Health, 17(22), 8339.
       https://doi.org/10.3390/ijerph17228339
    4. Evidence-backed design envelopes and performance ranges from a quantitative meta-analysis of 55 studies / 163 real
       pilot and full-scale systems (bootstrap confidence intervals on HLR, HRT, and removal performance).
       Lam, V. S., Tran, T. C. P., Vo, T. D., Nguyen, D. D., & Nguyen, X. C. (2024). Meta-analysis review for pilot and
       large-scale constructed wetlands: Design parameters, treatment performance, and influencing factors. The Science
       of the total environment, 927, 172140. https://doi.org/10.1016/j.scitotenv.2024.172140
    5. Measured direct CH4/N2O and other gaseous emission factors from a systematic review and meta-analysis of constructed wetland studies.
       Hu, S., Zhu, H., Bañuelos, G., Shutes, B., Wang, X., Hou, S., & Yan, B. (2023). Factors influencing gaseous
       emissions in constructed wetlands: A meta-analysis and systematic review. International Journal of Environmental
       Research and Public Health, 20(5), 3876. https://doi.org/10.3390/ijerph20053876

    Main equations
    --------------
    All flows normalized to treated water output = 1 [m³/hr]:

    Core water recovery:
        water_in(t) = water_out(t) / efficiency
        [m³/hr]       [m³/hr]         [-]

    Electricity demand:
        electricity_in(t) = water_out(t) × specific_energy_consumption
        [kWh/hr]            [m³/hr]         [kWh/m³]

    Optional water loss side-stream (evapotranspiration / seepage)
        water_loss(t) = water_loss_fraction × water_out(t)

    Optional biomass side-stream (harvested macrophytes / retained solids)
        biomass_out(t) = biomass_yield_factor × water_out(t)

    Optional proxy greenhouse gas emissions (tied to treated water output)
        CH4_out(t) = ch4_emission_factor × water_out(t)
        N2O_out(t) = n2o_emission_factor × water_out(t)

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum treated water throughput of the unit.
    - Hydraulic design variables (HLR, HRT, water depth, substrate, vegetation, wetland type) are stored as documentation
      /validation fields. They are not enforced as hard optimization constraints.
    - specific_energy_consumption defaults to 0.07 kWh/m³ for pumped systems. Set to 0.0 for gravity-fed configurations.
    - CH4 and N2O are modelled as proportional outputs via conversion_factor, coupled to the main activity chain.
      Their factors must be > 0 when the corresponding bus is active (enforced in _validate_parameters).
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "constructed_wetland"
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
    electricity_bus: Bus = None     # kWh
    water_in_bus: Bus = None        # m³
    water_out_bus: Bus = None       # m³ (PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    water_loss_bus: Optional[Bus] = None        # m³ evapotranspiration / seepage / unrecovered water
    biomass_bus: Optional[Bus] = None           # kg harvested wetland biomass / retained solids
    ch4_bus: Optional[Bus] = None               # kg methane emissions
    n2o_bus: Optional[Bus] = None               # kg nitrous oxide emissions

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    efficiency: float = 0.90                    # m³ treated water / m³ influent water (recovery rate) [1, 3]
    specific_energy_consumption: float = 0.07   # kWh / m³ treated water ; 0.0 for passive gravity flow [1, 4]
    water_loss_fraction: float = 0.0            # m³ water loss / m³ treated water [2, 3]
    biomass_yield_factor: float = 0.0           # kg biomass / m³ treated water [3, 4]
    ch4_emission_factor: float = 0.0            # kgCO2e CH4 / m³ treated water [5]
    n2o_emission_factor: float = 0.0            # kgCO2e N2O / m³ treated water [5]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0              # USD/m³ treated water
    carrier_cost: float = 0.0               # USD/m³ grey water

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / validation fields (not hard constraints)
    # Based on the core literature references
    # ------------------------------------------------------------------
    wetland_type: str = "subsurface_horizontal"         # free_water_surface | subsurface_horizontal | subsurface_vertical | hybrid [1, 2]
    hydraulic_loading_rate: Optional[float] = None      # m³/m²/day — design check only [2, 3]
    hydraulic_retention_time: Optional[float] = None    # days — design check only [2, 3]
    water_depth: Optional[float] = None                 # m — design check only [2]
    substrate_type: str = ""                            # e.g. gravel, sand, zeolite [2]
    vegetation_type: str = ""                           # e.g. Phragmites australis [2]
    pretreatment_required: bool = True                  # safeguard against solids clogging [1]
    climate_zone_note: str = ""                         # seasonal sensitivity documentation [4]
    clogging_risk_note: str = ""                        # operational risk [3]

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
        self.water_loss_bus = attributes.pop("water_loss_bus", None)
        self.biomass_bus = attributes.pop("biomass_bus", None)
        self.ch4_bus = attributes.pop("ch4_bus", None)
        self.n2o_bus = attributes.pop("n2o_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.efficiency = attributes.pop(
            "efficiency", self.efficiency
        )

        self.water_loss_fraction = attributes.pop(
            "water_loss_fraction", self.water_loss_fraction
        )
        self.biomass_yield_factor = attributes.pop(
            "biomass_yield_factor", self.biomass_yield_factor
        )
        self.ch4_emission_factor = attributes.pop(
            "ch4_emission_factor", self.ch4_emission_factor
        )
        self.n2o_emission_factor = attributes.pop(
            "n2o_emission_factor", self.n2o_emission_factor
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
        # documentation / validation fields
        # --------------------------------------------------------------
        self.wetland_type = attributes.pop("wetland_type", self.wetland_type)
        self.hydraulic_loading_rate = attributes.pop(
            "hydraulic_loading_rate", self.hydraulic_loading_rate
        )
        self.hydraulic_retention_time = attributes.pop(
            "hydraulic_retention_time", self.hydraulic_retention_time
        )
        self.water_depth = attributes.pop("water_depth", self.water_depth)
        self.substrate_type = attributes.pop("substrate_type", self.substrate_type)
        self.vegetation_type = attributes.pop("vegetation_type", self.vegetation_type)
        self.pretreatment_required = attributes.pop(
            "pretreatment_required", self.pretreatment_required
        )
        self.climate_zone_note = attributes.pop(
            "climate_zone_note", self.climate_zone_note
        )
        self.clogging_risk_note = attributes.pop(
            "clogging_risk_note", self.clogging_risk_note
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

        if self.water_loss_bus is not None:
            attributes[f"conversion_factor_{self.water_loss_bus.label}"] = sequence(
                self.water_loss_fraction
            )

        if self.biomass_bus is not None:
            attributes[f"conversion_factor_{self.biomass_bus.label}"] = sequence(
                self.biomass_yield_factor
            )

        # --------------------------------------------------------------
        # optional direct emissions conversion factors
        # --------------------------------------------------------------
        if self.ch4_bus is not None:
            attributes[
                f"conversion_factor_{self.ch4_bus.label}"] = sequence(
                self.ch4_emission_factor
            )

        if self.n2o_bus is not None:
            attributes[
                f"conversion_factor_{self.n2o_bus.label}"] = sequence(
                self.n2o_emission_factor
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
            self.water_loss_bus,
            self.biomass_bus,
            self.ch4_bus,
            self.n2o_bus,
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        valid_wetland_types = {
            "free_water_surface",
            "subsurface_horizontal",
            "subsurface_vertical",
            "hybrid",
        }
        if self.wetland_type not in valid_wetland_types:
            raise ValueError(
                f"wetland_type must be one of {sorted(valid_wetland_types)}."
            )

        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1].")

        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        if self.water_loss_bus is not None and self.water_loss_fraction <= 0:
            raise ValueError(
                "water_loss_fraction must be > 0 when water_loss_bus is provided."
            )
        if self.biomass_bus is not None and self.biomass_yield_factor <= 0:
            raise ValueError(
                "biomass_yield_factor must be > 0 when biomass_bus is provided."
            )
        if self.ch4_bus is not None and self.ch4_emission_factor <= 0:
            raise ValueError(
                "ch4_emission_factor must be > 0 when ch4_bus is provided."
            )
        if self.n2o_bus is not None and self.n2o_emission_factor <= 0:
            raise ValueError(
                "n2o_emission_factor must be > 0 when n2o_bus is provided."
            )

        for name, value in {
            "water_loss_fraction": self.water_loss_fraction,
            "biomass_yield_factor": self.biomass_yield_factor,
            "ch4_emission_factor": self.ch4_emission_factor,
            "n2o_emission_factor": self.n2o_emission_factor,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")

        if (
                self.hydraulic_loading_rate is not None
                and self.hydraulic_loading_rate <= 0
        ):
            raise ValueError("hydraulic_loading_rate must be > 0 if provided.")

        if (
                self.hydraulic_retention_time is not None
                and self.hydraulic_retention_time <= 0
        ):
            raise ValueError("hydraulic_retention_time must be > 0 if provided.")

        if self.water_depth is not None and self.water_depth <= 0:
            raise ValueError("water_depth must be > 0 if provided.")

