from dataclasses import field, dataclass

from typing import Sequence, Union

from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import dataclass_facade, Facade

from oemof_tabular_plugins.wefe.facades import MIMO


@dataclass_facade
class WaterFiltration(Converter, Facade):
    r"""WaterFiltration unit with two inputs and one output.

    Parameters
    ----------
    electricity_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its low temperature input.
    water_in_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its water input.
    water_out_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its water output.
    capacity: numeric
        The thermal capacity (high temperature output side) of the unit.
    carrier_cost: numeric
        Carrier cost for one unit of used input. Default: 0
    capacity_cost: numeric
        Investment costs per unit of output capacity.
        If capacity is not set, this value will be used for optimizing the
        conversion output capacity.
    expandable: boolean or numeric (binary)
        True, if capacity can be expanded within optimization. Default: False.
    lifetime: int (optional)
        Lifetime of the component in years. Necessary for multi-period
        investment optimization.
        Note: Only applicable for a multi-period model. Default: None.
    age : int (optional)
        The initial age of a flow (usually given in years);
        once it reaches its lifetime (considering also
        an initial age), the flow is forced to 0.
        Note: Only applicable for a multi-period model. Default: 0.
    fixed_costs : numeric (iterable or scalar) (optional)
        The fixed costs associated with a flow.
        Note: Only applicable for a multi-period model. Default: None.
    capacity_potential: numeric
        Maximum invest capacity in unit of output capacity. Default: +inf.
    input_parameters: dict (optional)
        Set parameters on the input edge of the conversion unit
         (see oemof.solph for more information on possible parameters)
    output_parameters: dict (optional)
        Set parameters on the output edge of the conversion unit
         (see oemof.solph for more information on possible parameters)
    """

    electricity_bus: Bus

    water_in_bus: Bus

    water_out_bus: Bus

    tech: str

    carrier: str = ""

    capacity: float = None

    marginal_cost: float = 0

    carrier_cost: float = 0

    capacity_cost: float = None

    expandable: bool = False

    lifetime: int = None

    age: int = 0

    fixed_costs: Union[float, Sequence[float]] = None

    capacity_potential: float = float("+inf")

    input_parameters: dict = field(default_factory=dict)

    output_parameters: dict = field(default_factory=dict)

    def build_solph_components(self):
        """TODO change efficiencies here"""
        self.conversion_factors.update(
            {
                self.electricity_bus: sequence(1),
                self.water_in_bus: sequence(1),
                self.water_out_bus: sequence(1),
            }
        )

        self.inputs.update(
            {
                self.electricity_bus: Flow(
                    variable_costs=self.carrier_cost, **self.input_parameters
                ),
                self.water_in_bus: Flow(),
            }
        )

        self.outputs.update(
            {
                self.water_out_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost,
                    investment=self._investment(),
                )
            }
        )


@dataclass_facade
class WaterPump(Converter, Facade):
    r"""WaterPump unit with two inputs and one output.

    Parameters
    ----------
    electricity_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its low temperature input.
    water_in_bus: oemof.solph.Bus
        An oemof bus instance where unit is connected to with
        its water input.
    water_out_bus: oemof.solph.Bus
        An oemof bus instance where the unit is connected to with
        its water output.
    capacity: numeric
        The thermal capacity (high temperature output side) of the unit.
    carrier_cost: numeric
        Carrier cost for one unit of used input. Default: 0
    capacity_cost: numeric
        Investment costs per unit of output capacity.
        If capacity is not set, this value will be used for optimizing the
        conversion output capacity.
    expandable: boolean or numeric (binary)
        True, if capacity can be expanded within optimization. Default: False.
    lifetime: int (optional)
        Lifetime of the component in years. Necessary for multi-period
        investment optimization.
        Note: Only applicable for a multi-period model. Default: None.
    age : int (optional)
        The initial age of a flow (usually given in years);
        once it reaches its lifetime (considering also
        an initial age), the flow is forced to 0.
        Note: Only applicable for a multi-period model. Default: 0.
    fixed_costs : numeric (iterable or scalar) (optional)
        The fixed costs associated with a flow.
        Note: Only applicable for a multi-period model. Default: None.
    capacity_potential: numeric
        Maximum invest capacity in unit of output capacity. Default: +inf.
    input_parameters: dict (optional)
        Set parameters on the input edge of the conversion unit
         (see oemof.solph for more information on possible parameters)
    output_parameters: dict (optional)
        Set parameters on the output edge of the conversion unit
         (see oemof.solph for more information on possible parameters)
    """

    electricity_bus: Bus

    water_in_bus: Bus

    water_out_bus: Bus

    tech: str

    carrier: str = ""

    capacity: float = None

    marginal_cost: float = 0

    carrier_cost: float = 0

    capacity_cost: float = None

    expandable: bool = False

    lifetime: int = None

    age: int = 0

    fixed_costs: Union[float, Sequence[float]] = None

    capacity_potential: float = float("+inf")

    input_parameters: dict = field(default_factory=dict)

    output_parameters: dict = field(default_factory=dict)

    def build_solph_components(self):
        """TODO change efficiencies here"""
        self.conversion_factors.update(
            {
                self.electricity_bus: sequence(1),
                self.water_in_bus: sequence(1),
                self.water_out_bus: sequence(1),
            }
        )

        self.inputs.update(
            {
                self.electricity_bus: Flow(
                    variable_costs=self.carrier_cost, **self.input_parameters
                ),
                self.water_in_bus: Flow(),
            }
        )

        self.outputs.update(
            {
                self.water_out_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost,
                    investment=self._investment(),
                )
            }
        )


@dataclass(unsafe_hash=False, frozen=False, eq=False)
class ReverseOsmosis(MIMO):
    r"""
    Crop Model Converter with 3 inputs (irradiation, precipitation, irrigation)
    and 2 outputs (crop harvest, remaining biomass), based on MultipleInputMultipleOutputConverter.

    Functions designed according to 3 publications, please note the abbreviations in the function descriptions:
    FAO56       ISBN: 978-92-5-104219-9
    SIMPLE      https://doi.org/10.1016/j.eja.2019.01.009
    ARID        https://doi.org/10.2134/agronj2011.0286

    Parameters
    ----------
    type: str
        will link MimoCrop to TYPEMAP, currently named 'mimo-crop'
    name: str
        a remarkable name for your component
    tech: str
        whatever, I just specified it as 'mimo'
    carrier: str
        carrier of the primary bus, should be overwritten by carriers directly assigned to buses
    primary: str
        primary bus object to which investments etc. are assigned
    expandable: boolean or numeric (binary)
        True, if capacity can be expanded within optimization. Default: False.
    capacity: numeric
        The capacity of crop. It is expressed in cultivated area [m²]
    capacity_minimum: numeric
        Minimum invest capacity in unit of output capacity.
    capacity_potential: numeric   (seems not to work as it should)
        Maximum invest capacity in unit of output capacity.
    capacity_cost: numeric
        Investment costs per unit of output capacity. If capacity is not set,
        this value will be used for optimizing the conversion output capacity.
    crop_type: str
        the name of crop as defined in global_specs/crop_specs.py
    solar_energy_bus: oemof.solph.Bus
        An oemof bus instance where the MimoCrop unit is connected to with
        its input, it is expected to provide W/m² irradiance.
    precipitation_bus: oemof.solph.Bus
        An oemof bus instance where the MimoCrop unit is connected to with
        its input, it is expected to provide mm precipitation.
    irrigation_bus: oemof.solph.Bus
        An oemof bus instance where the MimoCrop unit is connected to with
        its input, it is expected to provide mm irrigation.
    crop_bus: oemof.solph.Bus
        An oemof bus instance where the MimoCrop unit is connected to with
        its crop-harvest output. The unit is kg.
    biomass_bus: oemof.solph.Bus
        An oemof bus instance where the MimoCrop is connected with its
        non-edible, remaining (residual) biomass output. The unit is kg.
    time_profile: time series
        time profile separate from time index
    ghi_profile: time series
        global horizontal solar irradiance in W/m²
    tp_profile: time series
        total precipitation in mm
    t_air_profile: time series
        ambient air tempearture in °C
    t_dp_profile: time series
        dew point temperature in °C
    windspeed_profile: time series
        windspeed in m/s
    elevation: float
        elevation above sea level of the location
    crop_type: str
        the name of crop as defined in global_specs/crop_specs.py
    sowing_date: str
        date when cultivation period starts, MM-DD format
    harvest_date:
        date when cultivation period ends, MM-DD format
    has_irrigation: bool
        if set to False, irrigation profile will be set to 0

    Other optional parameters that could be added:
    ---------------------------------------------
    marginal_cost: numeric
        Marginal cost for one unit of produced output. Default: 0
    carrier_cost: numeric
        Carrier cost for one unit of used input. Default: 0
    input_parameters: dict (optional)
        Set parameters on the input edge of the conversion unit
        (see oemof.solph for more information on possible parameters)
    output_parameters: dict (optional)
        Set parameters on the output edge of the conversion unit
        (see oemof.solph for more information on possible parameters)

    Notes:
    ---------------------------------------------
    One input (eg. ghi) has to be volatile with a fixed capacity, other 2 dispatchable.
    MimoCrop capacity_potential seems to not work correctly.
    Two options to successfully run MimoCrop:
    1) Expandable = True and ignore capacity, use capacity of volatile source instead.
    2) Expandable = False, set capacity equal to that of the volatile source.

    In case you experience a weird error, for example "TypeError: can't multiply sequence by non-int of type 'float'",
    this is likely due to empty rows at the end of your input csv files.
    Please remove them and try again.
    """

    type: str = "mimo-crop"

    name: str = ""

    tech: str = "mimo"

    carrier: str = ""

    primary: str = ""

    expandable: bool = False

    capacity: float = 0

    capacity_minimum: float = None

    capacity_potential: float = None

    capacity_cost: float = 0

    solar_energy_bus: Bus = None

    precipitation_bus: Bus = None

    irrigation_bus: Bus = None

    crop_bus: Bus = None

    biomass_bus: Bus = None

    time_profile: Union[float, Sequence[float]] = None

    ghi_profile: Union[float, Sequence[float]] = None

    tp_profile: Union[float, Sequence[float]] = None

    t_air_profile: Union[float, Sequence[float]] = None

    t_dp_profile: Union[float, Sequence[float]] = None

    windspeed_profile: Union[float, Sequence[float]] = None

    elevation: float = 0

    crop_type: str = ""  # according to crop_dict

    sowing_date: str = ""  # MM-DD format

    harvest_date: str = ""  # MM-DD format

    has_irrigation: bool = False
