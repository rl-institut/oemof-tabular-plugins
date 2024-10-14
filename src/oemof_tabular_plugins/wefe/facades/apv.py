from dataclasses import field
from typing import Sequence, Union

import numpy as np
import pandas as pd

from pyomo.environ import (
    ConcreteModel,
    Var,
    Objective,
    SolverFactory,
    Piecewise,
    Constraint,
    maximize,
)

from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus
from oemof.solph.components import Converter
from oemof.solph.flows import Flow

import dataclasses
from oemof.tabular._facade import dataclass_facade, Facade
from oemof.tabular.facades import Conversion
from oemof_tabular_plugins.wefe.facades import MIMO
from oemof_tabular_plugins.wefe.facades import functions as f
from oemof_tabular_plugins.wefe.global_specs import (
    crop_dict,
    soil_dict,
    geo_dict,
    pv_dict,
)


@dataclasses.dataclass(unsafe_hash=False, frozen=False, eq=False)
class APV(MIMO):
    """
    APV model converter with 3 inputs (irradiation, precipitation, irrigation)
    and 4 outputs (electricity, crop harvest, remaining biomass, rainwater harvest),
    based on MultipleInputMultipleOutputConverter.

    Parameters
    ----------
    type: str
        will link APV to TYPEMAP, currently named 'mimo-crop'
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
        An oemof bus instance where the APV unit is connected to with
        its input, it is expected to provide W/m² irradiance.
    precipitation_bus: oemof.solph.Bus
        An oemof bus instance where the APV unit is connected to with
        its input, it is expected to provide mm precipitation.
    irrigation_bus: oemof.solph.Bus
        An oemof bus instance where the APV unit is connected to with
        its input, it is expected to provide mm irrigation.
    dc_electricity_bus: oemof.solph.Bus
        An oemof bus instance where the APV unit is connected to with
        its electricity output. The unit is kWh.
    crop_bus: oemof.solph.Bus
        An oemof bus instance where the APV unit is connected to with
        its crop-harvest output. The unit is kg.
    biomass_bus: oemof.solph.Bus
        An oemof bus instance where the APV unit is connected with its
        non-edible, remaining (residual) biomass output. The unit is kg.
    water_harvest_bus: oemof.solph.Bus
        An oemof bus instance where the APV unit is connected to with
        its rainwater-harvest-output output. The unit is mm.
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
    latitude: numeric
        latitude of the location
    elevation: numeric
        elevation above sea level of the location
    pv_type: str
        the name of the PV module as defined in global_specs/pv_modules.py
    crop_type: str
        the name of crop as defined in global_specs/crop_specs.py
    min_bio_rel: numeric
        minimum constraint for relative biomass yield in [0,1],
        if it is too close to 1 the optimization may be infeasible
    sowing_date: str
        date when cultivation period starts, MM-DD format
    harvest_date: str
        date when cultivation period ends, MM-DD format
    has_irrigation: bool
        if set to False, irrigation profile will be set to 0
    has_rainwater_harvesting: bool
        if set to False, rainwater harvest profile will be set to 0

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
    MIMO capacity_potential seems to not work correctly.
    Two options to successfully run APV:
    1) Expandable = True and ignore capacity, use capacity of volatile source instead.
    2) Expandable = False, set capacity equal to that of the volatile source.

    In case you experience a weird error, for example "TypeError: can't multiply sequence by non-int of type 'float'",
    this is likely due to empty rows at the end of your input csv files.
    Please remove them and try again.
    """

    type: str = "apv"

    name: str = ""

    tech: str = "mimo"

    primary: str = ""

    carrier: str = ""

    expandable: bool = False

    capacity: float = None

    capacity_minimum: float = None

    capacity_potential: float = None

    capacity_cost: float = None

    solar_energy_bus: Bus = None

    precipitation_bus: Bus = None

    irrigation_bus: Bus = None

    dc_electricity_bus: Bus = None

    crop_bus: Bus = None

    biomass_bus: Bus = None

    water_harvest_bus: Bus = None

    time_profile: Union[float, Sequence[float]] = None

    ghi_profile: Union[float, Sequence[float]] = None

    tp_profile: Union[float, Sequence[float]] = None

    t_air_profile: Union[float, Sequence[float]] = None

    t_dp_profile: Union[float, Sequence[float]] = None

    windspeed_profile: Union[float, Sequence[float]] = None

    latitude: float = 0.0

    elevation: float = 0.0

    pv_type: str = ""

    crop_type: str = ""

    min_bio_rel: float = 0.0

    sowing_date: str = ""

    harvest_date: str = ""

    has_irrigation: bool = False

    has_rainwater_harvesting: bool = False

    def __init__(self, **attributes):
        """
        Preprocessing of input data to calculate conversion_factors
        and assign these together with the correct busses and
        other attributes to MIMO parent class
        """

        def apv_geometry(latitude, y, **kwargs):
            """
            Calculate tilt and pitch based on latitude and module length (y).
            Obtain bifacial_radiance simulation results from geometry.json (xgaps, frts, frbs),
            interpolate for given latitude.

            Parameters
            ----------
            latitude: numeric
                latitude of the location where APV is modelled
            y: numeric
                module length [m] in North-South orientation (portrait mode required)

            Returns
            -------
            Dict(
                tilt: numeric
                    PV panel tilt angle in degrees
                pitch: numeric
                    distance between two PV panel arrays in N-S orientation [m]
                xgaps: list(numeric)
                    pre-defined xgap values (E-W spacing between panels on one array) [m]
                frbs: list(numeric)
                    radiation bifaciality factors in [0,1] for every xgap for the given latitude
                frts: list(numeric)
                    radiation transmission factors in [0,1] for every xgap for the given latitude
                )
            """
            # IBC minimum slope (by means of PV: tilt) for proper rainwater runoff
            # Source: https://iibec.org/asce-7-standard-low-slope-roof/
            min_slope = 0.25 / 12
            min_tilt = np.ceil(np.degrees(np.arctan(min_slope)))
            # tilt should ideally be close to latitude, but allow for rainwater runoff
            tilt = max(round(abs(latitude)), min_tilt)
            # minimum solar noon altitude (solar angle at solstice when sun is straight south (lat>0) or north (lat<0)
            # source: https://doi.org/10.1016/B978-0-12-397270-5.00002-9
            min_solar_angle = 90 - round(abs(latitude)) - 23.5
            # minimum distance between the PV arrays to prevent the panels from shading each other
            min_arraygap = (
                y * np.sin(np.radians(tilt)) / np.tan(np.radians(min_solar_angle))
            )
            # define pitch as distance from edge of one module across row up to the edge of the next module
            pitch = round(y * np.cos(np.radians(tilt)) + min_arraygap, 2)

            # get lats
            lats = list(geo_dict.keys())
            xgaps = geo_dict[lats[0]]["xgaps"]

            frbs_for_given_lat = []
            frts_for_given_lat = []
            for xgap in xgaps:
                frbs = [geo_dict[lat]["fbifacials"][xgap] for lat in lats]
                frts = [geo_dict[lat]["fshadings"][xgap] for lat in lats]
                frbs_for_given_lat.append(np.interp(latitude, lats, frbs))
                frts_for_given_lat.append(np.interp(latitude, lats, frts))

            return {
                "tilt": tilt,
                "pitch": pitch,
                "xgaps": xgaps,
                "frbs": frbs_for_given_lat,
                "frts": frts_for_given_lat,
            }

        def electricity_relative_output(frb0, frb, xgap, x, has_bifaciality, **kwargs):
            """
            Relative electricity output [-] only for LER calculation

            Parameters
            ----------
            frb0: numeric
                radiation bifaciality factor for xgap=0
            frb: numeric
                radiation bifaciality factor for any xgap value
            xgap: numeric
                xgap value (E-W panel spacing)
            x: numeric
                PV panel width (portrait mode: x = short side, E-W orientation)
            has_bifaciality: bool
                decides if frb and frb0 will be used

            Returns
            -------
            relative_output: numeric in [0,1]
                electricity output of an APV system with
                E-W spacing of value 'xgap' and respective value 'frb'
                in relation to full-density PV with
                E-W spacing of value 0 and respective value 'frb0'
            """
            frb, frb0 = (0, 0) if not has_bifaciality else (frb, frb0)
            relative_output = (1 + frb) * x / ((1 + frb0) * (x + xgap))
            return relative_output

        def geometry_optimization(
            df,
            min_bio_rel,
            capacity,
            x,
            y,
            tilt,
            pitch,
            xgaps,
            frbs,
            frts,
            p_rated,
            **kwargs,
        ):
            """
            Optimizes xgap (E-W spacing of PV panels on one array) to maximize LER
            LER = relative biomass yield in APV compared to open field cultivation
                + relative electricity yield in APV compared to full-density PV

            Parameters
            ----------
            df: DataFrame object
                required columns are ["ghi", "t_air", "t_dp", "tp", "windspeed", "cum_temp", "f_temp", "f_heat"]
            min_bio_rel: numeric
                minimum constraint for relative biomass yield in [0,1],
                if it is too close to 1 the optimization may be infeasible
            capacity: numeric
                capacity of the APV system [m²]
            x: numeric
                PV panel width (E-W orientation, short side) [m]
            y: numeric
                PV panel length (N-S orientation) [m]
            tilt: numeric
                    PV panel tilt angle in degrees
            pitch: numeric
                distance between two PV panel arrays in N-S orientation [m]
            xgaps: list(numeric)
                pre-defined xgap values (E-W spacing between panels on one array) [m]
            frbs: list(numeric)
                radiation bifaciality factors in [0,1] for every xgap for the given latitude
            frts: list(numeric)
                radiation transmission factors in [0,1] for every xgap for the given latitude
            p_rated: numeric
                nominal PV power [W]

            Returns
            -------
            Dict(
                xgap: numeric
                    optimized E-W spacing between panels on one array [m]
                frb: numeric in [0,1]
                    radiation bifaciality factor corresponding to xgap
                frt: numeric in [0,1]
                    radiation transmission factor corresponding to xgap
                area_apv: numeric
                    area of one PV panel in the APV system incl. spacing to other panels [m²]
                numpanels: numeric
                    number of panels for the given capacity (area)
                kwp: numeric
                    peak PV capacity
                gcr: numeric
                    ground coverage ratio, fraction of capacity (area) covered by PV panels
                ler: numeric
                    land equivalent ratio
                )
            """
            # Shadow area of the PV panel
            area_pv_shadow = x * y * np.cos(np.radians(tilt))
            # Ground coverage ratio list
            gcrs = [
                area_pv_shadow / ((x + xgaps[i]) * pitch) for i in range(len(xgaps))
            ]

            # Relative electricity yield for every xgap compared to a full density PV installation with xgap=0
            electricity_rel = [
                electricity_relative_output(
                    frb0=frbs[0], frb=frbs[i], xgap=xgaps[i], **pv_params
                )
                for i in range(len(xgaps))
            ]

            # Total biomass in the open field
            df = f.calc_f_water(
                df, frt=1, gcr=0, **attributes, **soil_params, **crop_params
            )
            df = f.calc_f_solar(df, **crop_params)
            biomass_open = f.calc_biomass(df, frt=1, **crop_params).sum()

            # Relative biomass for every xgap compared to the open field
            biomass_rel = []
            for i in range(len(xgaps)):
                df = f.calc_f_water(
                    df,
                    frt=frts[i],
                    gcr=gcrs[i],
                    **attributes,
                    **soil_params,
                    **crop_params,
                )
                df = f.calc_f_solar(df, **crop_params)
                biomass_rel.append(
                    f.calc_biomass(df, frt=frts[i], **crop_params).sum() / biomass_open
                )

            # Define the optimization model
            # TODO: for monotonous functions like electricity_rel and biomass_rel, simple interpolation could do
            opti_model = ConcreteModel()
            opti_model.xgap = Var(bounds=(min(xgaps), max(xgaps)))

            # Set up the piecewise linear functions for biomass and electricity terms
            opti_model.bio_rel = Var()
            opti_model.bio_rel_pieces = Piecewise(
                opti_model.bio_rel,
                opti_model.xgap,
                pw_pts=xgaps,
                f_rule=biomass_rel,
                pw_constr_type="EQ",
            )

            opti_model.elec_rel = Var()
            opti_model.elec_rel_pieces = Piecewise(
                opti_model.elec_rel,
                opti_model.xgap,
                pw_pts=xgaps,
                f_rule=electricity_rel,
                pw_constr_type="EQ",
            )

            # Minimum relative crop yield constraint
            opti_model.min_shading_constraint = Constraint(
                expr=opti_model.bio_rel >= min_bio_rel
            )

            # Optimization
            opti_model.objective = Objective(
                rule=opti_model.elec_rel + opti_model.bio_rel, sense=maximize
            )
            solver = SolverFactory("cbc")
            solver.solve(opti_model)

            # Results processing
            xgap_opti = opti_model.xgap.value
            frb_opti = np.interp(xgap_opti, xgaps, frbs)
            frt_opti = np.interp(xgap_opti, xgaps, frts)
            area_apv = (x + xgap_opti) * pitch
            numpanels = capacity / area_apv
            kwp = numpanels * p_rated * f.C_W_TO_KW
            gcr = area_pv_shadow / area_apv
            ler = opti_model.elec_rel.value + opti_model.bio_rel.value

            return {
                "xgap": xgap_opti,
                "frb": frb_opti,
                "frt": frt_opti,
                "area_apv": area_apv,
                "numpanels": numpanels,
                "kwp": kwp,
                "gcr": gcr,
                "ler": ler,
            }

        def calc_electricity(df, area_apv, frb, has_bifaciality, **kwargs):
            """
            Electricity generation including the radiation bifaciality factor (frb)

            Parameters
            ----------
            df: DataFrame object
                required columns are ['ghi', 't_air']
            area_apv: numeric
                area of one PV panel in the APV system incl. spacing to other panels [m²]
            frb: numeric in [0,1]
                radiation bifaciality factor
            has_bifaciality: bool
                decides if frb is used

            Returns
            -------
            df: DataFrame object
                additional columns are ['electricity']
            """
            df["pv_power"] = df.apply(
                lambda row: f.power(row["ghi"], row["t_air"], **pv_params), axis=1
            )
            frb = 0 if not has_bifaciality else frb
            panels_per_m2 = 1 / area_apv
            df["electricity"] = panels_per_m2 * (1 + frb) * df["pv_power"] * f.C_W_TO_KW
            df.drop(columns=["pv_power"])
            return df

        def calc_rainwater_harvest(df, gcr, has_rainwater_harvesting, **kwargs):
            """
            Rain water gained through catchments on PV panels

            Parameters
            ----------
            df: DataFrame object
                required columns are ['tp']
            gcr: numeric
                ground coverage ratio, fraction of capacity (area) covered by PV panels
            has_rainwater_harvesting: bool
                if False, rainwater harvest time series is 0

            Returns
            -------
            df: DataFrame object
                additional columns are ['water_harvest']
            """
            if not has_rainwater_harvesting:
                df["water_harvest"] = 0.0
            else:
                df["water_harvest"] = gcr * df["tp"]
            return df

        # --------------- Additional functions --------------------
        def convert_capacity_cost(p_rated, area_apv, **kwargs):
            """
            Converts the capacity_cost attribute from <currency>/kWp to <currency>/m²

            Parameters
            ----------
            capacity_cost: numeric
                capacity_cost of the APV system in <currency>/kWp
            p_rated: numeric
                nominal PV power [W]
            area_apv: numeric
                    area of one PV panel in the APV system incl. spacing to other panels [m²]

            Returns
            -------
            capacity_cost: numeric
                capacity_cost of the APV system in <currency>/m²
            """
            capacity_cost = (
                attributes.pop("capacity_cost") * p_rated * f.C_W_TO_KW / area_apv
            )
            return capacity_cost

        # Create DataFrame out of input profiles (time series), time_profile will be set as DatetimeIndex
        profiles_dict = {
            key.replace("_profile", ""): value
            for key, value in attributes.items()
            if key.endswith("profile")
        }
        time_index = profiles_dict.pop("time")
        profiles_df = pd.DataFrame(data=profiles_dict, index=time_index)

        # Get pv, crop and soil parameters from database, calculate cultivation parameters
        self.pv_type = attributes.pop("pv_type")
        self.crop_type = attributes.pop("crop_type")
        pv_params = pv_dict[self.pv_type]
        crop_params = crop_dict[self.crop_type]
        soil_params = soil_dict[self.crop_type]
        cultivation_params = f.specify_cultivation_parameters(
            profiles_df.index, **attributes
        )

        # Apply geometry-independent crop model functions to profiles
        f.calc_cumulative_temperature(profiles_df, **crop_params, **cultivation_params)
        f.calc_f_temp(profiles_df, **crop_params)
        f.calc_f_heat(profiles_df, **crop_params)

        # Update t_sum of crop_params if custom_harvest is True
        crop_params.update(
            f.custom_cultivation_period(
                profiles_df, **cultivation_params, **crop_params
            )
        )

        # Calculate geometry parameters for current location incl. frts and frbs
        geo_params = apv_geometry(**attributes, **pv_params)

        # Optimize xgap to maximize LER under min_bio_rel constraint; returns final frt, frb, xgap among others
        geo_params.update(
            geometry_optimization(profiles_df, **attributes, **geo_params, **pv_params)
        )

        # Calculate irrigation and total biomass yield
        f.calc_f_water(
            profiles_df, **attributes, **geo_params, **soil_params, **crop_params
        )
        f.calc_f_solar(profiles_df, **crop_params)
        f.adapt_irrigation(profiles_df)
        f.calc_biomass(profiles_df, **geo_params, **crop_params)

        # Calculate electricity yield and rainwater harvest
        calc_electricity(profiles_df, **pv_params, **geo_params)
        calc_rainwater_harvest(profiles_df, **geo_params, **attributes)

        # Format the profiles (no 0 allowed, no negative values, round to 10 decimal places)
        profiles_df = profiles_df.mask((profiles_df <= 1e-5), 1e-10)
        profiles_df = profiles_df.round(10)

        # Assign conversion factor profiles to correct busses
        conversion_factors = {}
        for key, value in attributes.items():
            if hasattr(value, "type") and value.type == "bus":
                if key.startswith("solar_energy_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["ghi"]
                    )
                if key.startswith("precipitation_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["tp"]
                    )
                if key.startswith("irrigation_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["irrigation"]
                    )
                if key.startswith("dc_electricity_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["electricity"]
                    )
                if key.startswith("crop_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        crop_params["hi"] * profiles_df["total_biomass"]  # crop harvest
                    )
                if key.startswith("biomass_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        (1 - crop_params["hi"])
                        * profiles_df["total_biomass"]  # residual biomass
                    )
                if key.startswith("water_harvest_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["water_harvest"]
                    )
        attributes.update(conversion_factors)

        # Assign attributes that are explicitly passed to parent class MIMO during init
        self.solar_energy_bus = attributes.pop("solar_energy_bus")
        self.precipitation_bus = attributes.pop("precipitation_bus")
        self.irrigation_bus = attributes.pop("irrigation_bus")
        self.dc_electricity_bus = attributes.pop("dc_electricity_bus")
        self.crop_bus = attributes.pop("crop_bus")
        self.biomass_bus = attributes.pop("biomass_bus")
        self.water_harvest_bus = attributes.pop("water_harvest_bus")
        self.capacity_cost = convert_capacity_cost(
            **attributes, **pv_params, **geo_params
        )

        # Initializes MIMO with correct busses, updated capacity_cost and conversion_factors
        super().__init__(
            from_bus_0=self.solar_energy_bus,
            from_bus_1=self.precipitation_bus,
            from_bus_2=self.irrigation_bus,
            to_bus_0=self.dc_electricity_bus,
            to_bus_1=self.crop_bus,
            to_bus_2=self.biomass_bus,
            to_bus_3=self.water_harvest_bus,
            capacity_cost=self.capacity_cost,
            **attributes,
        )

        # Assign other mandatory attributes after parent class init
        self.type = attributes.pop("type", None)
        self.name = attributes.pop("name", None)
        self.carrier = attributes.pop("carrier", None)
        self.tech = attributes.pop("tech", None)
        self.primary = attributes.pop("primary", None)
        self.expandable = attributes.pop("expandable", None)
        self.capacity = attributes.pop("capacity", None)
        self.capacity_minimum = attributes.pop("capacity_minimum", None)
        self.capacity_potential = attributes.pop("capacity_potential", None)
        self.capacity_cost = attributes.pop("capacity_cost", None)
