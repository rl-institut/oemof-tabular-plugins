import dataclasses
import warnings
from typing import Sequence, Union, Optional

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class CeramicFilter(MIMO):
    """
    Literature-informed ceramic membrane filtration facade based on MIMO.

    Purpose
    -------
    Process-unit facade for dead-end ceramic microfiltration in a WEFE
    mini-grid context. The model represents a ceramic membrane as a
    feedwater treatment unit producing permeate, with optional explicit
    reject stream modeling. Electricity demand is proportional to permeate
    output. It is not a full mechanistic membrane reactor model.

    Core references
    ---------------
    1. Gitis & Rothenberg (2016): Ceramic Membranes — New Opportunities
       and Practical Applications. Wiley-VCH.
       Technology framing, application context, and process-unit model
       justification.
    2. Viegas et al. (2015): Water reclamation with hybrid coagulation–
       ceramic microfiltration. J. Water Reuse Desalin. 5(4), 550–562.
       Pilot-scale calibration: recovery, flux, filtration cycle,
       CEB frequency, TMP, and stable operating windows.
    3. State-of-the-art review on ceramic membranes for water treatment.
       Fouling-control, backwash, and cleaning realism. Justifies simplified
       operational penalty (backwash_sec, availability_factor) rather than
       nonlinear fouling physics.

    Main equations
    --------------
    All flows normalized to 1 m³ of permeate output:

    Feedwater input:
        Q_feed(t) = (1 / recovery) * Q_permeate(t)
        [m³/hr]      [-]              [m³/hr]

    Electricity input:
        E(t) = (filtration_sec + backwash_sec) * Q_permeate(t)
        [kWh/hr]   [kWh/m³]         [kWh/m³]     [m³/hr]

    Optional reject output:
        Q_reject(t) = ((1 - recovery) / recovery) * Q_permeate(t)
        [m³/hr]         [-]                           [m³/hr]

    Availability derating (optional linear cleaning/downtime surrogate):
        Q_permeate(t) <= availability_factor * capacity

    Notes
    -----
    - Primary flow is water_out_bus [m³/hr] (permeate). Capacity constrains
      the maximum permeate throughput of the unit.
    - reject_bus is optional. If not provided, reject is not explicitly
      represented as a system flow in v3.0.
    - Each bus is in its own auto-group (no additive mixing). MIMO chaining
      equates all group flows through their conversion factors, normalizing
      all inputs and outputs to the common permeate activity basis.
    - Flux, TMP, filtration cycle time, CEB/day, and coagulant dose are
      stored as metadata for calibration and scenario documentation. They
      are not enforced as hard optimization constraints in v3.0.
    - carrier_cost applies to the electricity input only.
    - marginal_cost and chemical_cleaning_cost apply to the permeate output.
    - reject_disposal_cost applies to the reject output only if reject_bus
      is provided.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "ceramic_filter"
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
    water_in_bus: Bus = None                # m³ (feedwater)
    water_out_bus: Bus = None               # m³ (permeate / PRIMARY)

    # ------------------------------------------------------------------
    # optional input buses
    # ------------------------------------------------------------------
    backwash_water_bus: Optional[Bus] = None  # m³ — external backwash water source

    # ------------------------------------------------------------------
    # optional output buses
    # ------------------------------------------------------------------
    reject_bus: Optional[Bus] = None  # m³ (concentrate / reject)

    # ------------------------------------------------------------------
    # active physical parameters (used in constraints / split logic)
    # ------------------------------------------------------------------
    recovery: float = 0.97              # permeate / feedwater [-]
    filtration_sec: float = 0.02        # kWh / m³ permeate
    backwash_sec: float = 0.0           # kWh / m³ permeate
    availability_factor: float = 1.0    # average uptime / net production factor [-]

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0             # €/m³ permeate
    carrier_cost: float = 0.0              # €/kWh electricity
    reject_disposal_cost: float = 0.0      # €/m³ reject
    chemical_cleaning_cost: float = 0.0    # €/m³ permeate — folded periodic OPEX term

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

    # ------------------------------------------------------------------
    # documentation / calibration defaults (not hard constraints in v3.0)
    # ------------------------------------------------------------------
    filtration_time_min: float = None
    ceb_per_day: float = None
    tmp_reference_bar: float = None
    flux_reference_lmh: float = None
    backwash_water_ratio: float = 0.0
    membrane_material: str = "Al2O3"
    pore_size_um: float = None
    coagulant_dose_mg_per_l: float = None

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
        self.backwash_water_bus = attributes.pop("backwash_water_bus", None)
        self.reject_bus = attributes.pop("reject_bus", None)

        # --------------------------------------------------------------
        # active physical parameters
        # --------------------------------------------------------------
        self.recovery = attributes.pop("recovery", self.recovery)
        self.filtration_sec = attributes.pop("filtration_sec", self.filtration_sec)
        self.backwash_sec = attributes.pop("backwash_sec", self.backwash_sec)
        self.availability_factor = attributes.pop(
            "availability_factor", self.availability_factor
        )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.carrier_cost = attributes.pop("carrier_cost", self.carrier_cost)
        self.reject_disposal_cost = attributes.pop(
            "reject_disposal_cost", self.reject_disposal_cost
        )
        self.chemical_cleaning_cost = attributes.pop(
            "chemical_cleaning_cost", self.chemical_cleaning_cost
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
        self.filtration_time_min = attributes.pop(
            "filtration_time_min", self.filtration_time_min
        )
        self.ceb_per_day = attributes.pop("ceb_per_day", self.ceb_per_day)
        self.tmp_reference_bar = attributes.pop(
            "tmp_reference_bar", self.tmp_reference_bar
        )
        self.flux_reference_lmh = attributes.pop(
            "flux_reference_lmh", self.flux_reference_lmh
        )
        self.backwash_water_ratio = attributes.pop(
            "backwash_water_ratio", self.backwash_water_ratio
        )
        self.membrane_material = attributes.pop(
            "membrane_material", self.membrane_material
        )
        self.pore_size_um = attributes.pop("pore_size_um", self.pore_size_um)
        self.coagulant_dose_mg_per_l = attributes.pop(
            "coagulant_dose_mg_per_l", self.coagulant_dose_mg_per_l
        )

        # --------------------------------------------------------------
        # validate parameters
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # derived constants
        # --------------------------------------------------------------
        self._total_sec = self.filtration_sec + self.backwash_sec
        self._feedwater_per_permeate = 1.0 / self.recovery
        self._reject_per_permeate = (1.0 - self.recovery) / self.recovery

        # --------------------------------------------------------------
        # conversion factors
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.electricity_bus.label}"] = sequence(
            self._total_sec
        )
        attributes[f"conversion_factor_{self.water_in_bus.label}"] = sequence(
            self._feedwater_per_permeate
        )
        attributes[f"conversion_factor_{self.water_out_bus.label}"] = sequence(1.0)

        if self.reject_bus is not None:
            attributes[f"conversion_factor_{self.reject_bus.label}"] = sequence(
                self._reject_per_permeate
            )

        if self.backwash_water_bus is not None:
            attributes[f"conversion_factor_{self.backwash_water_bus.label}"] = sequence(
                max(self.backwash_water_ratio, 1e-9)
            )

        # --------------------------------------------------------------
        # output-specific variable costs/ revenue / output parameters / reporting metadata
        # --------------------------------------------------------------
        output_parameters = attributes.pop("output_parameters", {})
        output_parameters.setdefault(
            "variable_costs", self.marginal_cost + self.chemical_cleaning_cost
        )
        attributes["output_parameters"] = output_parameters

        if self.reject_bus is not None:
            reject_output_parameters = attributes.pop("reject_output_parameters", {})
            reject_output_parameters.setdefault(
                "variable_costs", self.reject_disposal_cost
            )
            attributes["reject_output_parameters"] = reject_output_parameters

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

    def _optional_bus_kwargs(self):
        kwargs = {}
        idx_in = 2
        idx_out = 1
        # inputs
        for bus in [
            self.backwash_water_bus
        ]:
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
        if not 0 < self.recovery <= 1:
            raise ValueError("recovery must be in (0, 1].")

        non_negative = {
            "filtration_sec": self.filtration_sec,
            "backwash_sec": self.backwash_sec,
            "backwash_water_ratio": self.backwash_water_ratio,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0.")

        if not 0 < self.availability_factor <= 1:
            raise ValueError("availability_factor must be in (0, 1].")

        if self.availability_factor < 1.0:
            warnings.warn(
                f"availability_factor={self.availability_factor} < 1.0. "
                "This is a linearized derating surrogate for cleaning downtime. "
                "Scale the effective capacity by this factor or apply "
                "activity_bound_max externally in the model builder.",
                UserWarning,
            )

        if (
                self.capacity_minimum is not None
                and self.capacity_potential is not None
                and self.capacity_minimum > self.capacity_potential
        ):
            raise ValueError(
                "capacity_minimum cannot be larger than capacity_potential."
            )