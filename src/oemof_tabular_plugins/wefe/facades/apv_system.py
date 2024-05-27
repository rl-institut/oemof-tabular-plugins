# TODO: assign water_bus output independent of solar_bus (absolute output instead of conversion factor)
# TODO: install bifacial radiance (rather get rid of it anyway)
# TODO: fix documentation

from dataclasses import field
from typing import Sequence, Union

from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import Facade, dataclass_facade

from src.oemof_tabular_plugins.wefe.global_specs.crops import crop_dict
from bifacial_radiance import *
from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, Piecewise, Constraint, maximize
import numpy as np
import pandas as pd
import os
from pathlib import Path


@dataclass_facade
class APVSystem(Converter, Facade):
    r"""APV System unit with 1 input and 2 outputs.

        Parameters
        ----------
        solar_bus: oemof.solph.Bus
            An oemof bus instance where the APV system unit is connected to with
            its input.
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
        lon: numeric
            Longitude
        reference_year: numeric
            Reference year for radiance simulation, if None: average over multiple years
        simulation_name: string
            Custom name for radiance simulation, if None: generic name based on lat and lon
        crop_type: string
            Choose crop type from Dictionary in ...
        sowing_date: string
            Set sowing date in the format yyyy-mm-dd
        minimal_crop_yield: float
            Minimal crop yield relative to open field, has to be in interval [0, 1]

        SHOULD INCLUDE FUNCTIONS AND EXAMPLE HERE

        """
    solar_bus: Bus

    electricity_bus: Bus

    biomass_bus: Bus

    water_bus: Bus

    carrier: str

    tech: str

    t_air: Union[float, Sequence[float]]

    ghi: Union[float, Sequence[float]]

    capacity: float = None

    marginal_cost: float = 0

    carrier_cost: float = 0

    capacity_cost: float = None

    expandable: bool = False

    capacity_potential: float = float("+inf")

    capacity_minimum: float = None

    input_parameters: dict = field(default_factory=dict)

    output_parameters: dict = field(default_factory=dict)

    lat: float = 0

    lon: float = 0

    reference_year: int = None

    simulation_name: str = None

    crop_type: str = 'cassava'

    sowing_date: str = '2022-01-01 '

    minimal_crop_yield: float = 0.8

    def build_solph_components(self):
        """ """

        def apv_geometry(instance):
            # TODO: sensitivity, steps and step_width as variable arguments?

            r"""
             ***** Part 1: Define general geometry parameters and initiate radiance model *****
             """
            # Location coordinates and reference year for radiance simulation
            lat = instance.lat
            lon = instance.lon
            year = instance.reference_year

            # Simulation name
            if instance.simulation_name != None:
                sim_name = instance.simulation_name
            else:
                sim_name = "lat" + str(lat) + "lon" + str(lon)

            # Create directory within project for radiation analysis
            folder = Path().resolve() / 'bifacial_radiance' / 'TEMP' / sim_name
            print(f"Your simulation will be stored in {folder}")

            if not os.path.exists(folder):
                os.makedirs(folder)

            # Create Radiance object
            rad_model = RadianceObj(name=sim_name, path=str(folder))

            # Get weather data
            epwfile = rad_model.getEPW(lat=lat, lon=lon)
            metdata = rad_model.readWeatherFile(weatherFile=epwfile, coerce_year=year)

            # Ground albedo and sky
            albedo = 0.23  # FAO56
            rad_model.setGround(material=albedo)  # must be run before for .gencumsky or .gendaylit
            rad_model.genCumSky()

            # Module and scene
            moduletype = 'test_module'
            # all parameters derived from module.json, x=1.036 and y=1.74, portrait mode, numpanels=1
            y = 1.74  # module length (y = N/S)
            x = 1.036  # module width (x = E/W)

            # panel facing south if lat >= 0, else facing north
            azimuth = 180 if lat >= 0 else 0
            # IBC minimum slope (by means of PV: tilt) for proper rainwater runoff
            min_slope = 0.25 / 12
            min_tilt = np.ceil(np.degrees(np.atan(min_slope)))
            # tilt should ideally be close to latitude, but allow for rainwater runoff
            tilt = max(round(abs(lat)), min_tilt)
            rad_tilt = np.radians(tilt)
            # minimum solar noon altitude (solar angle at solstice when sun is straight south (lat>0) or north (lat<0)
            min_solar_angle = 90 - round(abs(lat)) - 23.5
            rad_min_solar_angle = np.radians(min_solar_angle)
            # minimum distance to prevent the panels from shading each other
            min_ygap = round(y * np.sin(rad_tilt) / np.tan(rad_min_solar_angle), 2)
            # define pitch as distance from edge of one module across row up to the edge of the next module
            pitch = y * np.cos(rad_tilt) + min_ygap
            # clearance height fixed at 3 m (no significant change afterwards; see paper Sánchez, Meza, Dittmann; own iteration)
            clearance_height = 3
            # numer of Rows and Modules per row fixed at 5 each, no significant change afterwards; own iteration results)
            n = 15
            # analysis sensitivity (number of sensors in x and y direction on panel and ground)
            sensors = 10
            # iteration steps and width
            steps = 5
            step_width = 1

            # Fill SceneDict with constant geometry parameters
            sceneDict = {'tilt': tilt,
                         'pitch': pitch,
                         'clearance_height': clearance_height,
                         'azimuth': azimuth,
                         'nMods': n,
                         'nRows': n
                         }

            r"""
            Part 2: Analyze radiance model for varying gaps between PV modules
            """

            # Create lists for analysis results:
            xgaps = []
            fbifacials = []
            fshadings = []

            # Iterate through xgaps and calculate shading and bifaciality factors
            for step in range(steps):
                xgap = step * step_width
                # Create module based on xgap; create scene based on module
                module = rad_model.makeModule(name=moduletype, x=x, y=y, xgap=xgap)
                scene = rad_model.makeScene(module=module, sceneDict=sceneDict)

                # Combine everything in .oct file for analysis
                oct = rad_model.makeOct()

                # Create analysis object
                analysis = AnalysisObj(octfile=oct, name=rad_model.basename)

                # Create scans on the panel (automatic sensor positioning)
                frontscan, backscan = analysis.moduleAnalysis(scene=scene, sensorsy=sensors, sensorsx=sensors)

                # Copy panel scans and adjust for the ground
                groundfrontscan = frontscan.copy()

                # setting height to 12 cm from the ground (FAO56), constant height
                groundfrontscan['zstart'] = 0.12
                groundfrontscan['zinc'] = 0
                # keep first x, increase x spacing, so it covers 1 module and gap to next module
                groundfrontscan['xinc'] = (x + xgap) / (sensors - 1)
                # keep first y, increase y spacing, so it covers 1 pitch
                groundfrontscan['yinc'] = pitch / (sensors - 1)

                groundbackscan = groundfrontscan.copy()

                # Panel analysis
                analysis.analysis(octfile=oct, name=rad_model.basename + f"_panelscan_{step}",
                                  frontscan=frontscan, backscan=backscan)
                fbifacial = np.mean(analysis.backRatio)
                # fshadingpv = np.mean(analysis.Wm2Front) / metdata.ghi.sum()

                # Ground analysis
                analysis.analysis(octfile=oct, name=rad_model.basename + f"_groundscan_{step}",
                                  frontscan=groundfrontscan, backscan=groundbackscan)
                fshading = np.mean(analysis.Wm2Front) / metdata.ghi.sum()

                # Store results in lists
                xgaps.append(xgap)
                fbifacials.append(fbifacial)
                fshadings.append(fshading)

            r""" 
            Part 3: Optimize geometry based on analysis results

            Optimization objective is the LER (land equivalent ratio = relative electricity yield + relative crop yield)
            based on the results for bifaciality factors (fbifacials) and shading factors (fshadings) 
            generated by iterating through the different gaps between modules (xgaps)

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
            min_fshading = instance.minimal_crop_yield
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
            instance.area_module = x * y * np.cos(rad_tilt)
            instance.area_apv = (xgap_optimal + x) * pitch
            instance.f_shading = fshading_optimal
            instance.f_bifacial = fbifacial_optimal
            instance.ler = ler_optimal

            return instance

        apv_geometry(self)

        def apv_production(instance):
            r"""

            :param sowing_date:
            :param crop_type:
            :param xgaps:
            :param fshadings:
            :param fbifacials:
            :param rad:
            :param t_air:
            :return:
            """

            # create date-time-indexed DataFrame with ghi and t_air as columns
            date_time_index = pd.date_range(
                "1/1/2022", periods=8760, freq="H")
            df = pd.DataFrame(index=date_time_index)
            df['ghi'] = instance.ghi
            df['t_air'] = instance.t_air

            # Extract crop specific growth parameters from dictionary
            crop_type = instance.crop_type
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
            sowing_date = pd.to_datetime(instance.sowing_date + str(" 01:00:00"))

            # Extract PV parameters
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

            df['biomass_efficiency'] = rue * df['f_solar'] * df['f_temp'] * df['f_heat']

            # Define and apply function for modelling electricity generation
            def power(p_rated, rad, rad_ref, t_air, t_ref, noct):
                f_temp = 1 - 3.7e-3 * (t_air + ((noct - 20) / 800) * rad - t_ref)
                p = p_rated * 1 / rad_ref * f_temp
                return p

            df['pv_efficiency'] = df.apply(
                lambda row: power(p_rated, row['ghi'], rad_ref, row['t_air'], t_ref, noct), axis=1)

            instance.biomass_efficiency = df['biomass_efficiency']
            instance.pv_efficiency = df['pv_efficiency']
            # instance.water

        apv_production(self)

        self.conversion_factors.update(
            {
                self.solar_bus: sequence(1),
                self.electricity_bus: sequence(self.pv_efficiency),
                self.biomass_bus: sequence(self.biomass_efficiency)
            }
        )

        self.inputs.update(
            {
                self.from_bus: Flow(
                    variable_costs=self.carrier_cost, **self.input_parameters
                )
            }
        )

        self.outputs.update(
            {
                self.electricity_bus: Flow(
                    nominal_value=self._nominal_value(),
                    variable_costs=self.marginal_cost,
                    investment=self._investment(),
                    **self.output_parameters,
                ),
                self.biomass_bus: Flow()
            }
        )
