# TODO: include water (buses, balance, ARID factor)
# TODO: fix documentation

from dataclasses import field
from typing import Sequence, Union

from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import Facade, dataclass_facade
from oemof_industry.mimo_converter import MIMO

from src.oemof_tabular_plugins.wefe.global_specs.crops import crop_dict
from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, Piecewise, Constraint, maximize
import numpy as np
import pandas as pd
import json
from scipy.interpolate import interp1d
from datetime import datetime
import pytz

import os

# TODO: user input latitude and crop specifications?
# TODO: user input weather data -> adjust weatherpath (WEFESiteAnalyst input data)


# -------------- RELEVANT PATHS --------------
currentpath = os.path.abspath(__file__)
geometrypath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(currentpath))),
    "wefe", "global_specs", "geometry.json")
weatherpath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(currentpath))))),
    "examples", "scenarios", "wefe_apv_system_mimo", "data", "sequences")


# -------------- FACADE FOR APV SYSTEM --------------
@dataclass_facade
class APVSystem(MIMO):
    r"""APV System unit with multiple inputs (2) and multiple outputs (3).
        Inputs: Solar, water
        Outputs: Electricity, biomass, water

        Parameters
        ----------
        from_bus_0: oemof.solph.Bus
            An oemof bus instance where the APV system unit is connected to with
            its first input (solar)
        electricity_bus: oemof.solph.Bus
            An oemof bus instance where the APV system unit is connected to with
            its electrical output.
        biomass_bus: oemof.solph.Bus
            An oemof bus instance where the APV system unit is connected to with
            its biomass output.
        water_bus: oemof.solph.Bus
            An oemof bus instance where the APV system unit is connected to with
            its water output.
        capacity: numeric
            The conversion capacity (output side) of the unit.
        marginal_cost: numeric
            Marginal cost for one unit of produced output. Default: 0
        carrier_cost: numeric
            Carrier cost for one unit of used input. Default: 0
        capacity_cost: numeric
            Investment costs per unit of output capacity.
            If capacity is not set, this value will be used for optimizing the
            conversion output capacity.
        expandable: boolean or numeric (binary)
            True, if capacity can be expanded within optimization. Default: False.
        capacity_potential: numeric
            Maximum invest capacity in unit of output capacity.
        capacity_minimum: numeric
            Minimum invest capacity in unit of output capacity.
        input_parameters: dict (optional)
            Set parameters on the input edge of the conversion unit
            (see oemof.solph for more information on possible parameters)
        output_parameters: dict (optional)
            Set parameters on the output edge of the conversion unit
             (see oemof.solph for more information on possible parameters)
        t_air: array-like
            Ambient air temperature
        ghi: array-like
            Global horizontal irradiance
        lat: numeric
            Latitude
        crop_type: string
            Choose crop type from Dictionary in wefe/global_specs/crops.py
        sowing_date: string
            Set sowing date in the format yyyy-mm-dd
        minimal_crop_yield: float
            Minimal crop yield relative to open field, has to be in interval [0, 1]

        SHOULD INCLUDE FUNCTIONS AND EXAMPLE HERE

        """

    # type: str
    #
    # name: str
    #
    # tech: str
    #
    # carrier: str
    #
    # primary: str

    solar_bus_in: Bus

    water_bus_in: Bus

    elec_bus_out: Bus

    bio_bus_out: Bus

    water_bus_out: Bus

    # ghi: Union[float, Sequence[float]]
    #
    # t_air: Union[float, Sequence[float]]
    #
    # t_dp: Union[float, Sequence[float]]
    #
    # tp: Union[float, Sequence[float]]
    #
    # windspeed: Union[float, Sequence[float]]

    # capacity: float = None

    # marginal_cost: float = 0
    #
    # carrier_cost: float = 0

    # capacity_cost: float = None
    #
    # expandable: bool = False
    #
    # capacity_potential: float = float("+inf")
    #
    # capacity_minimum: float = None
    #
    # capex: float = None
    #
    # opex_fix: float = None
    #
    # lifetime: float = None
    #
    # input_parameters: dict = field(default_factory=dict)
    #
    # output_parameters: dict = field(default_factory=dict)

    lat: float = 16.6

    crop_type: str = 'cassava'

    sowing_date: str = '2022-01-01'

    minimal_crop_yield: float = 0.8

    def __init__(self, **kwargs):
        """ """
        # Perform methods to obtain conversion factors as class attributes
        self._apv_geometry()
        self._apv_production()

        conversion_factors = {
            self.solar_bus_in: sequence(1),
            self.water_bus_in: sequence(self.water_in_efficiency),
            self.elec_bus_out: sequence(self.pv_efficiency),
            self.bio_bus_out: sequence(self.biomass_efficiency),
            self.water_bus_out: sequence(self.water_out_efficiency)
        }

        super().__init__(from_bus_0=kwargs.pop("solar_bus_in"),
                         from_bus_1=kwargs.pop("water_bus_in"),
                         to_bus_0=kwargs.pop("elec_bus_out"),
                         to_bus_1=kwargs.pop("bio_bus_out"),
                         to_bus_2=kwargs.pop("water_bus_out"),
                         conversion_factors=conversion_factors, **kwargs)

    def _apv_geometry(self):
        """ """
        # Location coordinates to obtain geometry parameters
        lat = self.lat

        # Location-specific, fixed geometry parameters
        # module size from bifacial_radiance 'test_module'
        y = 1.74  # module length (y = N/S)
        x = 1.036  # module width (x = E/W)
        # IBC minimum slope (by means of PV: tilt) for proper rainwater runoff
        min_slope = 0.25 / 12
        min_tilt = np.ceil(np.degrees(np.arctan(min_slope)))
        # tilt should ideally be close to latitude, but allow for rainwater runoff
        tilt = max(round(abs(lat)), min_tilt)
        rad_tilt = np.radians(tilt)
        # minimum solar noon altitude (solar angle at solstice when sun is straight south (lat>0) or north (lat<0)
        min_solar_angle = 90 - round(abs(lat)) - 23.5
        rad_min_solar_angle = np.radians(min_solar_angle)
        # minimum distance to prevent the panels from shading each other
        min_ygap = y * np.sin(rad_tilt) / np.tan(rad_min_solar_angle)
        # define pitch as distance from edge of one module across row up to the edge of the next module
        pitch = round(y * np.cos(rad_tilt) + min_ygap, 2)

        r"""
        Obtain shading and bifaciality factors from global_specs/geometry.json, 
        interpolate for given latitude
        """

        # Load the geometry data
        with open(geometrypath, 'r') as f:
            geometry = json.load(f)

        # Convert lists back to numpy arrays and ensure numerical types
        def convert_from_serializable(obj):
            if isinstance(obj, list):
                return np.array(obj)
            if isinstance(obj, str) and obj.replace('.', '', 1).isdigit():
                return float(obj) if '.' in obj else int(obj)
            return obj

        geometry = {float(lat): {k: convert_from_serializable(v)
                                 for k, v in lat_results.items()} for lat, lat_results in geometry.items()}

        def interpolate_values(lat, geometry):
            # Extracting data
            lats = list(geometry.keys())

            # Interpolating fbifacial and fshading for each xgap
            xgaps = geometry[lats[0]]['xgaps'].tolist() \
                if isinstance(geometry[lats[0]]['xgaps'], np.ndarray) \
                else geometry[lats[0]]['xgaps']
            fbifacial_interp_funcs = []
            fshading_interp_funcs = []

            for xgap in xgaps:
                fbifacials = [geometry[lat]['fbifacials'][xgap] for lat in lats]
                fshadings = [geometry[lat]['fshadings'][xgap] for lat in lats]
                fbifacial_interp_funcs.append(interp1d(lats, fbifacials, kind='linear'))
                fshading_interp_funcs.append(interp1d(lats, fshadings, kind='linear'))

            # Calculating new values for the given latitude, convert back from numpy array
            new_fbifacials = [interp_func(lat).item() for interp_func in fbifacial_interp_funcs]
            new_fshadings = [interp_func(lat).item() for interp_func in fshading_interp_funcs]

            return {
                'xgaps': xgaps,
                'fbifacials': new_fbifacials,
                'fshadings': new_fshadings,
            }

        geo_params = interpolate_values(lat, geometry)

        xgaps = geo_params['xgaps']
        fbifacials = geo_params['fbifacials']
        fshadings = geo_params['fshadings']

        r"""
        Calculate evapotranspiration for every xgap, 
        calculate rainwater harvesting potential for every xgap,
        set up water balance,
        calculate ARID for every xgap
        """

        r""" 
        Optimize geometry based on analysis results

        Optimization objective is the LER (land equivalent ratio = relative electricity yield + relative crop yield)
        based on the results for bifaciality factors (fbifacials) and shading factors (fshadings) 
        generated by iterating the different gaps between modules (xgaps)

        Area requirement per PV module is a conclusive result.
        """

        # convert the non-linear term for relative electricity yield per area to a piecewise linear function 'farea'
        fareas = [(1 + fbifacials[i]) * x / ((1 + fbifacials[0]) * (x + xgaps[i])) for i in range(len(xgaps))]

        # Define the optimization model
        opti_model = ConcreteModel()

        # Decision variable: xgap
        opti_model.xgap = Var(bounds=(xgaps[0], xgaps[-1]))

        # Set up the piecewise linear functions for shading and area terms
        opti_model.fshading = Var()
        opti_model.fshading_pieces = Piecewise(opti_model.fshading, opti_model.xgap,
                                               pw_pts=xgaps,
                                               f_rule=fshadings,
                                               pw_constr_type='EQ')

        opti_model.farea = Var()
        opti_model.farea_pieces = Piecewise(opti_model.farea, opti_model.xgap,
                                            pw_pts=xgaps,
                                            f_rule=fareas,
                                            pw_constr_type='EQ')

        # Minimum shading constraint
        min_fshading = self.minimal_crop_yield
        opti_model.min_shading_constraint = Constraint(expr=opti_model.fshading >= min_fshading)

        # Objective: Maximize LER
        def objective_rule(model):
            return model.farea + model.fshading

        opti_model.objective = Objective(rule=objective_rule, sense=maximize)

        # Solver
        solver = SolverFactory('cbc')
        result = solver.solve(opti_model)

        # Output results
        xgap_optimal = opti_model.xgap.value
        fshading_optimal = opti_model.fshading.value
        farea_optimal = opti_model.farea.value
        fbifacial_optimal = ((farea_optimal * (1 + fbifacials[0]) * (xgap_optimal + x)) / x) - 1
        ler_optimal = farea_optimal + fshading_optimal

        # Assign important results as instance attributes
        self.area_module = x * y * np.cos(rad_tilt)
        self.area_apv = (xgap_optimal + x) * pitch
        self.fshading = fshading_optimal
        self.fbifacial = fbifacial_optimal
        self.ler = ler_optimal

        return self

    def _apv_production(self):
        r"""
        Calculate biomass and PV efficiency as full-year hourly conversion factor series
        """

        # Obtain weather data
        df = pd.read_csv(os.path.join(weatherpath, "apv_mimo_profile.csv"), index_col=0, parse_dates=True)

        # Extract crop specific growth parameters from dictionary
        crop_type = self.crop_type
        t_opt = crop_dict[crop_type]['t_opt']
        t_base = crop_dict[crop_type]['t_base']
        t_heat = crop_dict[crop_type]['t_max']
        t_extreme = crop_dict[crop_type]['t_ext']
        t_sum = crop_dict[crop_type]['t_sum']
        i50a = crop_dict[crop_type]['i50a']
        i50b = crop_dict[crop_type]['i50b']
        f_solar_max = crop_dict[crop_type]['f_solar_max']
        rue = crop_dict[crop_type]['rue']
        # Convert Radiation Use Efficiency from W/m² to MJ/(m²*h)
        rue *= 3.6e-3
        # Transfer sowing date into Timestamp object
        sowing_date = datetime.strptime(self.sowing_date, '%Y-%m-%d').replace(tzinfo=pytz.UTC)
        #sowing_date = pd.to_datetime(self.sowing_date + str(" 01:00:00"))

        # Define PV parameters
        p_rated = 270  # [Wp]
        rad_ref = 1000  # [W/m²]
        t_ref = 25  # [°C]
        noct = 48  # [°C]

        # Define and apply functions for modelling biomass generation
        def temp(t_air, t_opt, t_base):
            if t_air < t_base:
                f_temp = 0
            elif t_base <= t_air < t_opt:
                f_temp = (t_air - t_base) / (t_opt - t_base)
            elif t_air >= t_opt:
                f_temp = 1
            return f_temp

        df['f_temp'] = df['t_air'].apply(
            lambda t_air: temp(t_air, t_opt, t_base))

        def heat(t_air, t_heat, t_extreme):
            if t_air <= t_heat:
                f_heat = 1
            elif t_heat < t_air <= t_extreme:
                f_heat = 1 - (t_air - t_heat) / (t_extreme - t_heat)
            elif t_air > t_extreme:
                f_heat = 0
            return f_heat

        df['f_heat'] = df['t_air'].apply(
            lambda t_air: heat(t_air, t_heat, t_extreme))

        def development(date, sowing_date, t_air, t_base):
            if date < sowing_date:
                delta_cum_temp = 0
            elif t_air > t_base:
                delta_cum_temp = (t_air - t_base) / 24  # SIMPLE crop model has daily temp values, convert to hourly
            else:
                delta_cum_temp = 0
            return delta_cum_temp

        df['cum_temp'] = df.apply(
            lambda row: development(row.name, sowing_date, row['t_air'], t_base), axis=1).cumsum()

        def solar(cum_temp, t_sum, i50a, i50b, f_solar_max):
            cum_temp_to_reach_f_solar_max = i50a - 100 * np.log(
                1 / 999)  # derived from f_solar = 0.999f_solar_max
            if cum_temp < 1:
                f_solar = 0
            elif cum_temp < cum_temp_to_reach_f_solar_max:
                f_solar = f_solar_max / (1 + np.exp(-0.01 * (cum_temp - i50a)))
            elif cum_temp < t_sum:
                f_solar = f_solar_max / (1 + np.exp(0.01 * (cum_temp - (t_sum - i50b))))
            else:
                f_solar = 0
            return f_solar

        df['f_solar'] = df['cum_temp'].apply(
            lambda cum_temp: solar(cum_temp, t_sum, i50a, i50b, f_solar_max))

        df['biomass_efficiency'] = self.fshading * rue * df['f_solar'] * df['f_temp'] * df['f_heat']

        # Define and apply function for modelling electricity generation
        def power(p_rated, rad, rad_ref, t_air, t_ref, noct):
            f_temp = 1 - 3.7e-3 * (t_air + ((noct - 20) / 800) * rad - t_ref)
            p = p_rated * 1 / rad_ref * f_temp
            return p

        df['pv_efficiency'] = df.apply(
            lambda row: power(p_rated, row['ghi'], rad_ref, row['t_air'], t_ref, noct), axis=1)

        df['pv_efficiency'] *= (1 + self.fbifacial)

        self.biomass_efficiency = df['biomass_efficiency']
        self.pv_efficiency = df['pv_efficiency']

        # ------ Arbitrary water input and output time series to check MIMO compatibility ------#
        df['water_in'] = 0.3
        df['water_out'] = 0.1
        self.water_in_efficiency = df['water_in']
        self.water_out_efficiency = df['water_out']

        return self
