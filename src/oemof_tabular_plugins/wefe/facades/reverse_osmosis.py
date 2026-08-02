import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class ReverseOsmosis(MIMO):
    """
    Literature-informed reverse osmosis (RO) facade based on MIMO.

    Purpose
    -------
    Generic pressure-driven membrane desalination facade for reverse osmosis
    units used in water treatment systems. The model is designed as a
    bookkeeping/process-yield unit representing an RO membrane as a
    water-treatment intervention. It is not a full mechanistic membrane
    transport model.

    Core references
    ---------------
    1. Recovery, permeate flux, concentrate concentration, concentration polarization, and net driving pressure (NDP) equations.
       Salinas-Rodríguez, S. G., Kennedy, M. D., & Schippers, J. C. (2021). Process design of reverse osmosis systems.
       In S. G. Salinas-Rodríguez, J. C. Schippers, G. L. Amy, I. S. Kim, & M. D. Kennedy (Eds.), Seawater reverse osmosis
       desalination: Assessment and pre-treatment of fouling and scaling (pp. 243-264). IWA Publishing.
       https://doi.org/10.2166/9781780409863_0243
    2. Operating limits, salt-rejection trends, and design-interpretation guidance for FilmTec RO/NF elements.
       DuPont Water Solutions. (2024). FilmTec reverse osmosis membranes technical manual (Form No. 45-D01504-en).
       https://www.dupont.com/content/dam/water/amer/us/en/water/public/documents/en/RO-NF-FilmTec-Manual-45-D01504-en.pdf
    3. Engineering design rules and recommended flux/recovery ranges for spiral-wound RO elements.
       LANXESS AG. (2024). Guidelines for the design of reverse osmosis membrane systems.
       https://kh.aquaenergyexpo.com/wp-content/uploads/2024/01/Guideline-for-the-design-of-reverse-osmosis-membrane-systems.pdf
    4. Specific energy consumption and net driving pressure calculation chains, including energy-recovery-device-adjusted SEC.
       DuPont Water Solutions. FilmTec design equations (Form No. 609-02057-604).
       https://www.lenntech.com/Data-sheets/Filmtec-Design-Equations-L.pdf
    5. Membrane fouling mechanisms and pretreatment-linked cleaning frequency.
       Salinas-Rodríguez, S. G., Kennedy, M. D., & Schippers, J. C. (2021). Fouling and pre-treatment. In S. G. Salinas-Rodríguez,
       J. C. Schippers, G. L. Amy, I. S. Kim, & M. D. Kennedy (Eds.), Seawater reverse osmosis desalination: Assessment
       and pre-treatment of fouling and scaling (pp. 59-83). https://doi.org/10.2166/9781780409863_0059

    Main equations
    --------------
    All flows normalized to treated water output = 1 [m³/hr]:

    Feedwater requirement:
        feedwater_per_output = 1 / efficiency         [m³_feed / m³_permeate]

    Brine / concentrate output:
        brine_per_output = 1/efficiency - 1             [m³_brine / m³_permeate]

    Net specific energy consumption [kWh / m³ permeate]:
        net_SEC = SEC_gross * (1 - energy_recovery_efficiency)

    CIP cleaning waste (time-averaged over cleaning cycles):
        cip_waste_per_output = cleaning_waste_ratio   [m³_cip / m³_permeate]

    Concentration factor (dimensionless, reporting only):
        CF = 1 / (1 - efficiency)

    Permeate TDS proxy (reporting only, requires tds_in):
        TDS_permeate = tds_in * (1 - salt_rejection)

    Brine TDS proxy (reporting only, requires tds_in):
        TDS_brine = tds_in * (1 - efficiency * salt_rejection) / (1 - efficiency)

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr]. Capacity constrains the maximum permeate output of the unit.
    - cleaning_waste_bus is an optional output representing the CIP (Clean-In-Place) effluent produced when the RO
      membrane is chemically cleaned. If the bus is omitted, CIP waste is implicitly absorbed into the brine stream.
    - Water-quality surrogates (estimated_permeate_tds, estimated_brine_tds, estimated_concentration_factor) are stored
      as reporting-only values. They are not enforced as hard optimization constraints.
    - Characterization values (typical SEC range, salt_rejection, tds_in) are stored as documentation/calibration defaults.
      They are not enforced as hard optimization constraints.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "reverse_osmosis"
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
    water_in_bus: Bus = None                # m³ (pretreated feedwater)
    water_out_bus: Bus = None               # m³ (permeate — PRIMARY)
    brine_out_bus: Bus = None               # m³ (concentrate)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    # reserved for future extension (e.g. antiscalant / acid dosing carrier)

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    cleaning_waste_bus: Optional[Bus] = None  # m³ CIP effluent from membrane cleaning

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    specific_energy_consumption: float = 1.2                # kWh / m³ permeate (gross) [1, 4]
    efficiency: float = 0.55                                # m³ permeate / m³ feedwater (recovery) [1, 3]
    energy_recovery_efficiency: float = 0.0                 # dimensionless [0, 1) [4]
    salt_rejection: float = 0.99                            # dimensionless [0, 1] [2]
    cleaning_waste_ratio: float = 0.0                       # m³ CIP effluent / m³ permeate (time-averaged) [5]
    max_recovery: Optional[float] = None                    # design upper bound check [1, 3]
    concentration_factor_limit: Optional[float] = None      # design upper bound check [1]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0                  # USD/m³ permeate
    carrier_cost: float = 0.0                   # USD/m³ feedwater
    brine_disposal_cost: float = 0.0            # USD/m³ brine
    cleaning_waste_disposal_cost: float = 0.0   # USD/m³ CIP effluent

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
    tds_in: Optional[float] = None                              # kg/m³ or g/L feed TDS [1]
    sec_typical_min: float = 0.5                                # kWh/m³, lower bound from literature [2, 4]
    sec_typical_max: float = 3.0                                # kWh/m³, upper bound from literature [2, 4]
    design_flux_lmh: Optional[float] = None                     # L/m²/hr, design flux, documentation only [3]
    ndp_bar: Optional[float] = None                             # bar, net driving pressure, documentation only [1, 4]
    concentration_polarization_factor: Optional[float] = None   # dimensionless, documentation only [1]

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
        self.brine_out_bus = attributes.pop("brine_out_bus")

        # --------------------------------------------------------------
        # optional buses
        # --------------------------------------------------------------
        self.cleaning_waste_bus = attributes.pop("cleaning_waste_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.efficiency = attributes.pop("efficiency", self.efficiency)
        self.specific_energy_consumption = attributes.pop(
            "specific_energy_consumption", self.specific_energy_consumption
        )
        self.energy_recovery_efficiency = attributes.pop(
            "energy_recovery_efficiency", self.energy_recovery_efficiency
        )
        self.salt_rejection = attributes.pop("salt_rejection", self.salt_rejection)
        self.cleaning_waste_ratio = attributes.pop(
            "cleaning_waste_ratio", self.cleaning_waste_ratio
        )
        self.max_recovery = attributes.pop("max_recovery", self.max_recovery)
        self.concentration_factor_limit = attributes.pop(
            "concentration_factor_limit", self.concentration_factor_limit
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.brine_disposal_cost = attributes.pop(
            "brine_disposal_cost", self.brine_disposal_cost
        )
        self.cleaning_waste_disposal_cost = attributes.pop(
            "cleaning_waste_disposal_cost", self.cleaning_waste_disposal_cost
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
        self.tds_in = attributes.pop("tds_in", self.tds_in)
        self.sec_typical_min = attributes.pop("sec_typical_min", self.sec_typical_min)
        self.sec_typical_max = attributes.pop("sec_typical_max", self.sec_typical_max)
        self.design_flux_lmh = attributes.pop("design_flux_lmh", self.design_flux_lmh)
        self.ndp_bar = attributes.pop("ndp_bar", self.ndp_bar)
        self.concentration_polarization_factor = attributes.pop(
            "concentration_polarization_factor", self.concentration_polarization_factor
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # (Salinas-Rodriguez et al., 2021 [1]; DuPont FilmTec Design Equations [4])
        # --------------------------------------------------------------
        self._feedwater_per_output = 1.0 / self.efficiency
        self._brine_per_output = self._feedwater_per_output - 1.0

        self._net_specific_energy_consumption = (
                self.specific_energy_consumption
                * (1.0 - self.energy_recovery_efficiency)
        )

        self._estimated_concentration_factor = 1.0 / (1.0 - self.efficiency)

        if self.tds_in is not None:
            self._estimated_permeate_tds = self.tds_in * (1.0 - self.salt_rejection)
            self._estimated_brine_tds = (
                    self.tds_in
                    * (1.0 - self.efficiency * self.salt_rejection)
                    / (1.0 - self.efficiency)
            )
        else:
            self._estimated_permeate_tds = None
            self._estimated_brine_tds = None

        # --------------------------------------------------------------
        # conversion factors
        # All normalized to treated water output = 1 [m³/hr].
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._net_specific_energy_consumption
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_output
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)
        attributes[f"conversion_factor_{self.brine_out_bus.label}"] = sequence(
            self._brine_per_output
        )
        if self.cleaning_waste_bus is not None:
            attributes[f"conversion_factor_{self.cleaning_waste_bus.label}"] = sequence(
            self.cleaning_waste_ratio
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
        elif self.primary == "brine_out_bus":
            primary_label = self.brine_out_bus.label
        else:
            primary_label = self.primary

        # --------------------------------------------------------------
        # initialize base MIMO facade
        # --------------------------------------------------------------
        super().__init__(
            from_bus_0=self.electricity_bus,
            from_bus_1=self.water_in_bus,
            to_bus_0=self.water_out_bus,
            to_bus_1=self.brine_out_bus,
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

        if self.brine_out_bus in self.outputs:
            self.outputs[self.brine_out_bus].variable_costs = sequence(
                self.brine_disposal_cost
            )

        if self.cleaning_waste_bus is not None and self.cleaning_waste_bus in self.outputs:
            self.outputs[self.cleaning_waste_bus].variable_costs = sequence(
                self.cleaning_waste_disposal_cost
            )

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 2
        # inputs
        for bus in []:
            if bus is not None:
                kwargs[f"from_bus_{idx_in}"] = bus
                idx_in += 1
        # outputs
        for bus in [
            self.cleaning_waste_bus
        ]:
            if bus is not None:
                kwargs[f"to_bus_{idx_out}"] = bus
                idx_out += 1

        return kwargs

    def _validate_parameters(self):
        if not 0 < self.efficiency < 1:
            raise ValueError("efficiency must be in (0, 1).")

        if not 0 <= self.energy_recovery_efficiency < 1:
            raise ValueError("energy_recovery_efficiency must be in [0, 1).")

        if not 0 <= self.salt_rejection <= 1:
            raise ValueError("salt_rejection must be in [0, 1].")

        if self.specific_energy_consumption < 0:
            raise ValueError("specific_energy_consumption must be >= 0.")

        if self.cleaning_waste_ratio < 0:
            raise ValueError("cleaning_waste_ratio must be >= 0.")

        if self.max_recovery is not None and self.efficiency > self.max_recovery:
            raise ValueError(
                f"Configured efficiency ({self.efficiency}) exceeds max_recovery ({self.max_recovery}).")

        if self.concentration_factor_limit is not None:
            cf = 1.0 / (1.0 - self.efficiency)
            if cf > self.concentration_factor_limit:
                raise ValueError(f"Estimated concentration factor ({cf:.3f}) exceeds concentration_factor_limit ({self.concentration_factor_limit}).")

        if self.cleaning_waste_bus is not None and self.cleaning_waste_ratio <= 0:
            warnings.warn(
                "cleaning_waste_bus is set but cleaning_waste_ratio is 0. "
                "No CIP waste flow will be enforced.",
                UserWarning,
            )