r"""
VERSION 4 - not working
Only functions to create csv input for MIMO facade
solar in
water in
electricity out
biomass out
water out
"""

# TODO: include water (buses, balance, ARID factor)
# TODO: fix documentation
# TODO: check if weather_data and sowing_date have the same year

from dataclasses import field
from typing import Sequence, Union

from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

from oemof.tabular._facade import Facade, dataclass_facade
from oemof_industry.mimo_converter import MIMO

import logging
from oemof.tools import logger, economics

from src.oemof_tabular_plugins.wefe.global_specs.crops import crop_dict
from pyomo.environ import ConcreteModel, Var, Objective, SolverFactory, Piecewise, Constraint, maximize
import numpy as np
import pandas as pd
import json
from scipy.interpolate import interp1d
from datetime import datetime
import pytz
import os

import pdb

logger.define_logging()

# -------------- RELEVANT PATHS --------------
currentpath = os.path.abspath(__file__)
specspath = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(currentpath))),
    "wefe", "global_specs")


# elementspath = os.path.join(os.path.dirname(sequencepath), "elements")


# -------------- APV pre-processing --------------
def pre_processing_apv(scenariopath):
    def _apv_production(scenariopath, apv_dict):
        r"""
        Calculate biomass and PV efficiency as full-year hourly conversion factor series
        """

        # ----- Weather data from sequence csv -----
        weatherpath = os.path.join(scenariopath, "data", "sequences", "apv_mimo_profile.csv")
        df = pd.read_csv(weatherpath, index_col=0, parse_dates=True)

        # ---- Crop growth data from element csv -----
        crop_type = apv_dict['crop_type']
        minimal_crop_yield = apv_dict['minimal_crop_yield']
        # Transfer sowing date into Timestamp object matching the df index
        index_timezone = df.index.tz
        sowing_date = pd.Timestamp(apv_dict['sowing_date']).tz_localize(index_timezone)
        z = apv_dict['elevation']

        # ----- Crop specific growth parameters from crop_dict -----
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

        # ----- Soil parameters -----
        z = 259
        rzd = 400  # zeta   ## root zone depth [mm]
        wuc = 0.096  # alpha  ## water uptake constant
        awc = 0.13  # theta_m   ## water holding capacity
        rcn = 65  # eta     ## runoff curve number
        ddc = 0.55  # beta      ## deep drainage coefficient

        # ----- PV parameters -----
        p_rated = 270  # [Wp]
        rad_ref = 1000  # [W/m²]
        t_ref = 25  # [°C]
        noct = 48  # [°C]

        # ----- Geometry -----
        def apv_geometry(lat):
            """
            Obtain geometry parameters and bifacial_radiance simulation results from geometry.json,
            interpolate for given latitude and return geometry_params
            """

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
            geometrypath = os.path.join(specspath, "geometry.json")
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

            def latitude_interpolation(lat, geometry):
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

            interpolated = latitude_interpolation(lat, geometry)

            return {
                'x': x,
                'y': y,
                'tilt': tilt,
                'pitch': pitch,
                'xgaps': interpolated['xgaps'],
                'fbifacials': interpolated['fbifacials'],
                'fshadings': interpolated['fshadings'],
            }

        latitude = apv_dict['lat']
        geo_params = apv_geometry(latitude)

        x = geo_params['x']
        y = geo_params['y']
        tilt = geo_params['tilt']
        pitch = geo_params['pitch']
        xgaps = geo_params.pop('xgaps')
        fbifacials = geo_params.pop('fbifacials')
        fshadings = geo_params.pop('fshadings')

        area_pv = x * y * np.cos(tilt)
        geo_params['area_pv'] = area_pv

        # ----- General functions for modelling water, biomass and electricity -----
        def development(date, sowing_date, t_air, t_base):
            """ Cumulative temperature experienced by plant as measure for plant development (SIMPLE) [K] """
            if date < sowing_date:
                delta_cum_temp = 0
            elif t_air > t_base:
                delta_cum_temp = (t_air - t_base) / 24  # SIMPLE crop model has daily temp values, convert to hourly
            else:
                delta_cum_temp = 0
            return delta_cum_temp

        def temp(t_air, t_opt, t_base):
            """ Temperature effect on plant growth (SIMPLE) [-] """
            if t_air < t_base:
                f_temp = 0
            elif t_base <= t_air < t_opt:
                f_temp = (t_air - t_base) / (t_opt - t_base)
            else:
                f_temp = 1
            return f_temp

        def heat(t_air, t_heat, t_extreme):
            """ Heat stress effect on plant growth (SIMPLE) [-] """
            if t_air <= t_heat:
                f_heat = 1
            elif t_heat < t_air <= t_extreme:
                f_heat = 1 - (t_air - t_heat) / (t_extreme - t_heat)
            else:
                f_heat = 0
            return f_heat

        def power(p_rated, rad, rad_ref, t_air, t_ref, noct):
            """ Hourly PV power output in relation to incoming radiation """
            f_temp = 1 - 3.7e-3 * (t_air + ((noct - 20) / 800) * rad - t_ref)
            p = p_rated * rad / rad_ref * f_temp
            return p

        df['cum_temp'] = df.apply(
            lambda row: development(row.name, sowing_date, row['t_air'], t_base),
            axis=1
        ).cumsum()

        df['f_temp'] = df['t_air'].apply(
            lambda t_air: temp(t_air, t_opt, t_base)
        )

        df['f_heat'] = df['t_air'].apply(
            lambda t_air: heat(t_air, t_heat, t_extreme)
        )

        df['pv_power'] = df.apply(
            lambda row: power(p_rated, row['ghi'], rad_ref, row['t_air'], t_ref, noct),
            axis=1
        )

        # ----- Geometry-dependent functions for modelling water, biomass and electricity -----
        def soil_heat_flux(ghi, irr_w):
            """ Soil heat flux as fraction of incoming radiation (FAO56) [W/m²] """
            if ghi > 0:
                g = 0.1 * irr_w
            else:
                g = 0.5 * irr_w
            return g

        def potential_evapotranspiration(z, t_air, t_dp, w10, irr_w, g_w):
            """ Potential evapotranspiration for reference grass (FAO56) [mm/h] """
            cp_air = 1.013e-3  # specific heat at constant pressure [MJ/(kg °C)]
            epsilon = 0.622  # ratio molecular weight of water vapour/dry air [-]
            h_vap = 2.45  # latent heat of vaporization [MJ/kg]
            rho_w = 1000  # density of water [kg/m³]
            k_c = 1  # crop coefficient (FAO) [-]

            irr = 86.4e-3 * irr_w  # W/m² to MJ/m²*day
            g = 3.6e-3 * g_w  # W/m² to MJ/m²*day
            w2 = w10 * 4.87 / np.log(672.58)  # wind speed at 2m above ground (FAO56) [m/s]
            p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26  # atmospheric pressure [kPa]
            gamma = cp_air * p / (h_vap * epsilon)  # psychrometric constant
            delta = 4098 * (0.6108 * np.exp(17.27 * t_air / (t_air + 237.3))) / (
                    t_air + 237.3) ** 2  # slope of sat vap press curve

            def vap_pressure(t):
                """ Water vapor saturation pressure at specific temperature (t)"""
                e = 0.6108 * np.exp(17.27 * t / (t + 237.3))
                return e

            e_s = vap_pressure(t_air)  # saturation vapour pressure [kPa]
            e_a = vap_pressure(t_dp)  # actual vapour pressure (sat vap press at dewpoint temp) [kPa]

            et_0 = (0.408 * delta * (irr - g) + gamma * 900 / (t_air + 273) * w2 * (e_s - e_a)) / (
                    delta + gamma * 1.34 * w2)  # [mm/m²*day]
            et_p = k_c * et_0 / 24  # [mm/m²*h]
            # q_et = et * rho_w * h_vap / 1000  # [W/m²]
            return et_p

        def runoff(p, rcn):
            """ Surface runoff of precipitation (ARID) [mm/h] """
            s = 25400 / rcn - 254
            i_a = 0.2 * s
            if p > i_a:
                r = (p - i_a) ** 2 / (p + i_a - s) / 24
            else:
                r = 0
            return r

        def soil_water_balance(df):
            """
            Soil water content (swc) for timestep i based on
            precipitation (tp) and surface runoff for timestep i as well as
            soil water content, deep drainage and actual evapotranspiration (et_a) for timestep i-1
            (ARID) [-]
            """
            df['deep_drain'] = 0
            df['et_a'] = 0
            df['swc'] = 0

            def deep_drainage(ddc, rzd, water_cap, water_con_bd):
                """ Deep drainage of soil water (ARID) [mm/h] """
                if water_con_bd > water_cap:
                    d = ddc * rzd * (water_con_bd - water_cap) / 24
                else:
                    d = 0
                return d

            swc_cache = awc
            df.loc[df.index[0], 'swc'] = swc_cache
            for index, row in df.iloc[1:].iterrows():
                df.loc[index, 'deep_drain'] = deep_drainage(ddc=ddc, rzd=rzd, water_cap=awc, water_con_bd=swc_cache)
                df.loc[index, 'et_a'] = min(wuc * rzd * swc_cache / 24, df.loc[index, 'et_p'])
                df.loc[index, 'swc'] = swc_cache + (row['tp']  # + row['irrigation']
                                                    - df.loc[index, 'et_a']
                                                    - row['runoff']
                                                    - df.loc[index, 'deep_drain']
                                                    ) / rzd
                swc_cache = df.loc[index, 'swc']

        def drought(et_p, et_a):
            """ Drought stress effect on plant growth (SIMPLE) [-] """
            f_drought = min(et_p, et_a) / et_p
            return f_drought

        # adapt i50b to drought and heat stress

        def solar_interception(cum_temp, t_sum, i50a, i50b, f_solar_max):
            """ Interception of incoming radiation according to development stage (SIMPLE) [-] """
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

        def biomass_generation(df, frt):
            """ Biomass generation including the radiation transmission factor (frt) """
            df['g'] = df.apply(
                lambda row: soil_heat_flux(row['ghi'], frt * row['ghi']),
                axis=1
            )

            df['et_p'] = df.apply(
                lambda row: potential_evapotranspiration(
                    z, row['t_air'], row['t_dp'], row['windspeed'], frt * row['ghi'], row['g']
                ),
                axis=1
            )

            df['runoff'] = df['tp'].apply(
                lambda tp: runoff(tp, rcn)
            )

            soil_water_balance(df)

            df['f_drought'] = df.apply(
                lambda row: drought(row['et_p'], row['et_a']),
                axis=1
            )

            df['f_solar'] = df['cum_temp'].apply(
                lambda cum_temp: solar_interception(cum_temp, t_sum, i50a, i50b, f_solar_max)
            )

            df['biomass_gen'] = frt * df['ghi'] * df['f_solar'] * rue * df['f_temp'] * df[['f_heat', 'f_drought']].min(
                axis=1)

            return df['biomass_gen']

        def electricity_generation(df, frb):
            """ Electricity generation including the radiation bifaciality factor (frb) """
            df['electricity_gen'] = (1 + frb) * df['pv_power']
            return df['electricity_gen']

        # ------- Geometry optimization: Maximize LER -------
        biomass_open = biomass_generation(df, frt=1).sum()
        electricity_rel = [(1 + fbifacials[i]) * x / ((1 + fbifacials[0]) * (x + xgaps[i])) for i in range(3)]
        biomass_rel = [biomass_generation(df, frt=fshadings[i]).sum() / biomass_open for i in range(3)]

        # Define the optimization model
        opti_model = ConcreteModel()
        opti_model.xgap = Var(bounds=(min(xgaps), max(xgaps)))

        # Set up the piecewise linear functions for biomass and electricity terms
        opti_model.bio_rel = Var()
        opti_model.bio_rel_pieces = Piecewise(opti_model.bio_rel, opti_model.xgap,
                                              pw_pts=xgaps[:3],
                                              f_rule=biomass_rel,
                                              pw_constr_type='EQ')

        opti_model.elec_rel = Var()
        opti_model.elec_rel_pieces = Piecewise(opti_model.elec_rel, opti_model.xgap,
                                               pw_pts=xgaps[:3],
                                               f_rule=electricity_rel,
                                               pw_constr_type='EQ')

        # Minimum relative crop yield constraint
        min_bio_rel = minimal_crop_yield
        opti_model.min_shading_constraint = Constraint(expr=opti_model.bio_rel >= min_bio_rel)

        # Optimization
        def objective_rule(model):
            return model.elec_rel + model.bio_rel

        opti_model.objective = Objective(rule=objective_rule, sense=maximize)
        solver = SolverFactory('cbc')
        solver.solve(opti_model)
        xgap_optimal = opti_model.xgap.value
        ler = opti_model.elec_rel.value + opti_model.bio_rel.value

        # ----- Results processing -----
        interp_frb = interp1d(xgaps, fbifacials, kind='linear')
        interp_frt = interp1d(xgaps, fshadings, kind='linear')
        fbifacial_optimal = interp_frb(xgap_optimal)
        fshading_optimal = interp_frt(xgap_optimal)

        electricity_gen = electricity_generation(df, frb=fbifacial_optimal)
        biomass_gen = biomass_generation(df, frt=fshading_optimal)
        area_apv = (x + xgap_optimal) * pitch

        # ----- Arbitrary water time series -----
        df['water_in'] = 2
        df['water_out'] = 1

        # ----- No nulls in conversion factor lists -----
        df.replace(0, 0.00001, inplace=True)


        apv_dict.update(
            {
                **geo_params,
                'area_apv': area_apv,
                'ler': ler,
                'solar': df['ghi'].tolist(),
                'electricity': electricity_gen.tolist(),
                'biomass': biomass_gen.tolist(),
                'water_in': df['water_in'].tolist(),
                'water_out': df['water_out'].tolist()
            }
        )

        return apv_dict

    def _update_apv_element(directory):
        """ """
        has_apv = False
        elements_path = os.path.join(
            directory, 'data', 'elements'
        )
        for element in os.listdir(elements_path):
            if element.endswith('.csv'):
                # Read in csv file, check for row with 'apv-system' in column 'name'
                element_path = os.path.join(elements_path, element)
                element_df = pd.read_csv(element_path, sep=';')
                apv_row = element_df[element_df['name'] == 'apv-system']

                # Check if a matching row was found
                if not apv_row.empty:
                    has_apv = True
                    # Drop apv_row from Dataframe and convert to dictionary
                    element_df = element_df.drop(apv_row.index)
                    apv_dict = apv_row.iloc[0].to_dict()


                    # Update the dictionary
                    apv_dict = _apv_production(scenariopath, apv_dict)

                    # following part has to be more robust (eg. INFO if buses are missing)
                    bus_dict = {key: value
                                for key, value in apv_dict.items()
                                if 'bus' in key}
                    for key, value in bus_dict.items():
                        if key == 'from_bus_0':
                            apv_dict[f'conversion_factor_{value}'] = apv_dict.pop('solar')
                        if key == 'from_bus_1':
                            apv_dict[f'conversion_factor_{value}'] = apv_dict.pop('water_in')
                        if key == 'to_bus_0':
                            apv_dict[f'conversion_factor_{value}'] = apv_dict.pop('electricity')
                        if key == 'to_bus_1':
                            apv_dict[f'conversion_factor_{value}'] = apv_dict.pop('biomass')
                        if key == 'to_bus_2':
                            apv_dict[f'conversion_factor_{value}'] = apv_dict.pop('water_out')

                    # Convert updated dictionary back to DataFrame and concat
                    new_apv_row = pd.DataFrame([apv_dict])
                    element_df = pd.concat([element_df, new_apv_row], ignore_index=True)
                    element_df.to_csv(element_path, sep=';', index=False)
                    logger.info("APV system element discovered and updated")

        if not has_apv:
            logger.info("No element found with 'apv-system' in column 'name'.")

    _update_apv_element(scenariopath)
