import dataclasses
from typing import Sequence, Union
import json

from oemof.solph.buses import Bus
from oemof.solph._plumbing import sequence
from oemof_tabular_plugins.wefe.facades import MIMO


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class Mixer(MIMO):
    """
    Generic 4-into-1 stream mixer based on MIMO.

    Purpose
    -------
    Combines up to four input streams into a single output stream via a
    simple additive mass/volume balance. Optionally supports flow
    shares to fix or bound the contribution of individual input buses
    relative to the group total.

    Main equation
    -------------
    Single input group ("in_main"), single output group ("out_main"):
        GROUP_FLOW_in_main(t)  = sum_i flow_i(t) / cf_i(t)
        GROUP_FLOW_out_main(t) = flow_out(t) / cf_out(t)
        GROUP_FLOW_in_main(t) == GROUP_FLOW_out_main(t)

    i.e. inputs are SUMMED (not pairwise-equalized) onto the common output
    bus, weighted by their conversion factors. Ratio between inputs is free
    unless constrained via input_flow_share_config.

    Flow shares
    -----------
    input_flow_share_config is a dict keyed by input role
    ("input_bus_0".."input_bus_3"), each valued by a dict of
    {"min"/"max"/"fix": value}. A share is always relative to the group
    total, not to another bus directly:
        flow_i(t) / cf_i(t)  <op>  GROUP_FLOW_in_main(t) * share(t)
    To pin a ratio between two buses, fix both against the group total
    (e.g. 0.3 and 0.7) — their sum should not exceed 1, and any unshared
    buses will be implicitly squeezed toward zero if the fixed shares
    already claim the full group total. min/max on two buses bounds each
    independently and does not by itself lock their ratio to each other.

    Notes
    -----
    - All four input buses are mandatory (pass a zero-flow/negligible source
      upstream if a stream is unused — do not omit).
    - conversion_factor_<bus> defaults to 1.0 for any bus not given an
      explicit density/scaling factor.
    """

    # ------------------------------------------------------------------
    # tabular identity
    # ------------------------------------------------------------------
    type: str = "mixer"
    name: str = ""
    tech: str = "mixer"
    carrier: str = "water"
    primary: str = "output_bus"

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
    input_bus_0: Bus = None
    input_bus_1: Bus = None
    input_bus_2: Bus = None
    input_bus_3: Bus = None
    output_bus: Bus = None

    # ------------------------------------------------------------------
    # conversion factors (input side) — default 1.0 each if not given
    # ------------------------------------------------------------------
    conversion_factor_input_0: float = 1.0
    conversion_factor_input_1: float = 1.0
    conversion_factor_input_2: float = 1.0
    conversion_factor_input_3: float = 1.0
    conversion_factor_output: float = 1.0

    # ------------------------------------------------------------------
    # flow shares between inputs (optional)
    # dict: {"input_bus_0": {"fix": 0.3}, "input_bus_1": {"fix": 0.7}, ...}
    # ------------------------------------------------------------------
    input_flow_share_config: dict = None

    # ------------------------------------------------------------------
    # economics
    # ------------------------------------------------------------------
    marginal_cost: float = 0.0
    max_mixer_load: Union[float, Sequence[float]] = None  # activity bound cap

    # ------------------------------------------------------------------
    # multiperiod
    # ------------------------------------------------------------------
    lifetime: int = None
    age: int = 0
    fixed_costs: Union[float, Sequence[float]] = None

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
        self.input_bus_0 = attributes.pop("input_bus_0")
        self.input_bus_1 = attributes.pop("input_bus_1")
        self.input_bus_2 = attributes.pop("input_bus_2")
        self.input_bus_3 = attributes.pop("input_bus_3")
        self.output_bus = attributes.pop("output_bus")

        # --------------------------------------------------------------
        # conversion factors
        # --------------------------------------------------------------
        self.conversion_factor_input_0 = attributes.pop(
            "conversion_factor_input_0", self.conversion_factor_input_0
        )
        self.conversion_factor_input_1 = attributes.pop(
            "conversion_factor_input_1", self.conversion_factor_input_1
        )
        self.conversion_factor_input_2 = attributes.pop(
            "conversion_factor_input_2", self.conversion_factor_input_2
        )
        self.conversion_factor_input_3 = attributes.pop(
            "conversion_factor_input_3", self.conversion_factor_input_3
        )
        self.conversion_factor_output = attributes.pop(
            "conversion_factor_output", self.conversion_factor_output
        )

        # --------------------------------------------------------------
        # flow shares
        # --------------------------------------------------------------
        self.input_flow_share_config = attributes.pop(
            "input_flow_share_config", {}
        )

        if isinstance(self.input_flow_share_config, str):
            self.input_flow_share_config = json.loads(
                self.input_flow_share_config
            )

        # --------------------------------------------------------------
        # economics / investment
        # --------------------------------------------------------------
        self.marginal_cost = attributes.pop("marginal_cost", self.marginal_cost)
        self.expandable = attributes.pop("expandable", self.expandable)
        self.capacity = attributes.pop("capacity", self.capacity)
        self.capacity_cost = attributes.pop("capacity_cost", self.capacity_cost)
        self.capacity_minimum = attributes.pop(
            "capacity_minimum", self.capacity_minimum
        )
        self.capacity_potential = attributes.pop(
            "capacity_potential", self.capacity_potential
        )
        self.max_mixer_load = attributes.pop("max_mixer_load", None)

        # --------------------------------------------------------------
        # multiperiod
        # --------------------------------------------------------------
        self.lifetime = attributes.pop("lifetime", self.lifetime)
        self.age = attributes.pop("age", self.age)
        self.fixed_costs = attributes.pop("fixed_costs", self.fixed_costs)
        self.output_parameters = attributes.pop("output_parameters", {})

        # --------------------------------------------------------------
        # validate
        # --------------------------------------------------------------
        self._validate_parameters()

        # --------------------------------------------------------------
        # bus role lookup, used in several places below
        # --------------------------------------------------------------
        self._input_bus_by_role = {
            "input_bus_0": self.input_bus_0,
            "input_bus_1": self.input_bus_1,
            "input_bus_2": self.input_bus_2,
            "input_bus_3": self.input_bus_3,
        }

        # --------------------------------------------------------------
        # groups: 4-into-1 additive balance
        # --------------------------------------------------------------
        groups = {
            "in_main": [
                self.input_bus_0.label,
                self.input_bus_1.label,
                self.input_bus_2.label,
                self.input_bus_3.label,
            ],
            "out_main": [self.output_bus.label],
        }
        attributes["groups"] = groups

        # --------------------------------------------------------------
        # conversion factors (division semantic GROUP_FLOW = sum flow_i/cf_i)
        # --------------------------------------------------------------
        attributes[f"conversion_factor_{self.input_bus_0.label}"] = sequence(
            self.conversion_factor_input_0
        )
        attributes[f"conversion_factor_{self.input_bus_1.label}"] = sequence(
            self.conversion_factor_input_1
        )
        attributes[f"conversion_factor_{self.input_bus_2.label}"] = sequence(
            self.conversion_factor_input_2
        )
        attributes[f"conversion_factor_{self.input_bus_3.label}"] = sequence(
            self.conversion_factor_input_3
        )
        attributes[f"conversion_factor_{self.output_bus.label}"] = sequence(
            self.conversion_factor_output
        )
        attributes["conversion_factor_in_main"] = sequence(1.0)
        attributes["conversion_factor_out_main"] = sequence(1.0)

        # --------------------------------------------------------------
        # flow shares between input buses (relative to in_main group total)
        # --------------------------------------------------------------
        if self.input_flow_share_config:
            for role, shares in self.input_flow_share_config.items():
                bus = self._input_bus_by_role.get(role)
                if bus is None:
                    raise ValueError(
                        f"Unknown input role '{role}' in "
                        "input_flow_share_config — must be one of "
                        f"{list(self._input_bus_by_role)}."
                    )
                for share_type, value in shares.items():
                    if share_type not in ("min", "max", "fix"):
                        raise ValueError(
                            f"Invalid flow share type '{share_type}' for "
                            f"'{role}' — must be 'min', 'max', or 'fix'."
                        )
                    attributes[
                        f"flow_share_{share_type}_{bus.label}"
                    ] = sequence(value)

        # --------------------------------------------------------------
        # activity bound - optional cap on total mixed throughput
        # --------------------------------------------------------------
        if self.max_mixer_load is not None:
            attributes["activity_bound_max"] = sequence(self.max_mixer_load)

        # --------------------------------------------------------------
        # primary bus resolution
        # --------------------------------------------------------------
        bus_by_key = dict(self._input_bus_by_role, output_bus=self.output_bus)
        primary_label = (
            bus_by_key[self.primary].label
            if self.primary in bus_by_key
            else self.primary
        )

        # --------------------------------------------------------------
        # init base MIMO facade
        # --------------------------------------------------------------
        super().__init__(
            from_bus_0=self.input_bus_0,
            from_bus_1=self.input_bus_1,
            from_bus_2=self.input_bus_2,
            from_bus_3=self.input_bus_3,
            to_bus_0=self.output_bus,
            primary=primary_label,
            marginal_cost=self.marginal_cost,
            expandable=self.expandable,
            capacity=self.capacity,
            capacity_cost=self.capacity_cost,
            capacity_minimum=self.capacity_minimum,
            capacity_potential=self.capacity_potential,
            lifetime=self.lifetime,
            age=self.age,
            fixed_costs=self.fixed_costs,
            **attributes,
        )

        # ------------------------------------------------------------
        # PATCH: MIMO/mimo_converter.py gaps.
        # ------------------------------------------------------------
        self._apply_flow_parameters()
        if self.input_flow_share_config:
            self._validate_flow_share_wiring()

    def _apply_flow_parameters(self):
        if self.output_bus in self.outputs:
            out_flow = self.outputs[self.output_bus]
            out_flow.variable_costs = sequence(self.marginal_cost)
            if not self.expandable and self.capacity is not None:
                out_flow.nominal_value = self.capacity
            custom_attrs = (getattr(self, "output_parameters", None) or {}).get(
                "custom_attributes"
            )
            if custom_attrs:
                for attribute, value in custom_attrs.items():
                    setattr(out_flow, attribute, value)

    def _validate_flow_share_wiring(self):
        """
        Confirms the flow_share_<type>_<bus> kwargs generated from
        input_flow_share_config were correctly parsed by
        MIMO._init_facade_flow_shares / MultiInputMultiOutputConverter.
        _init_flow_shares (mimo_converter.py) into self.input_flow_shares
        (set by super().__init__(); resolved form: {type: {Bus: sequence}}).
        That parsing path resolves bus labels via string matching, which
        could in principle mis-resolve or drop a share silently on a label
        collision. Remove once upstream raises on ambiguity itself.
        """
        for role, shares in self.input_flow_share_config.items():
            bus = self._input_bus_by_role[role]
            for share_type, expected_value in shares.items():
                resolved_for_type = self.input_flow_shares.get(share_type, {})
                if bus not in resolved_for_type:
                    raise RuntimeError(
                        f"Mixer: flow share '{share_type}' for bus "
                        f"'{bus.label}' was not found in the resolved "
                        "input flow shares after construction — check for "
                        "label collisions in flow share key parsing."
                    )
                got = resolved_for_type[bus][0]  # sequence(...) at t=0
                if got != expected_value:
                    raise RuntimeError(
                        f"Mixer: flow share '{share_type}' for bus "
                        f"'{bus.label}' resolved to {got}, expected "
                        f"{expected_value}."
                    )

    def _validate_parameters(self):
        for name, value in {
            "conversion_factor_input_0": self.conversion_factor_input_0,
            "conversion_factor_input_1": self.conversion_factor_input_1,
            "conversion_factor_input_2": self.conversion_factor_input_2,
            "conversion_factor_input_3": self.conversion_factor_input_3,
            "conversion_factor_output": self.conversion_factor_output,
        }.items():
            if value is None or value <= 0:
                raise ValueError(f"{name} must be > 0.")
        if self.marginal_cost is not None and self.marginal_cost < 0:
            raise ValueError("marginal_cost must be >= 0.")
        if self.max_mixer_load is not None:
            if isinstance(self.max_mixer_load, (int, float)) and self.max_mixer_load <= 0:
                raise ValueError("max_mixer_load must be > 0.")
        if self.input_flow_share_config:
            valid_roles = {
                "input_bus_0", "input_bus_1", "input_bus_2", "input_bus_3"
            }
            for role, shares in self.input_flow_share_config.items():
                if role not in valid_roles:
                    raise ValueError(
                        f"Unknown input role '{role}' in "
                        f"input_flow_share_config — must be one of "
                        f"{sorted(valid_roles)}."
                    )
                if "fix" in shares and (
                    "min" in shares or "max" in shares
                ):
                    raise ValueError(
                        f"'{role}': cannot combine 'fix' with 'min'/'max' "
                        "on the same bus."
                    )
            fixed_total = sum(
                v["fix"]
                for v in self.input_flow_share_config.values()
                if "fix" in v
            )
            if fixed_total > 1.0 + 1e-9:
                raise ValueError(
                    f"Sum of fixed input flow shares ({fixed_total}) "
                    "exceeds 1 — infeasible."
                )
