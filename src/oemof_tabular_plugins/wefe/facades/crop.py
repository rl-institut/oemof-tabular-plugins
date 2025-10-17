import logging
from typing import Sequence, Union

import numpy as np
import pandas as pd

from oemof.solph._plumbing import sequence
from oemof.solph.buses import Bus

from dataclasses import dataclass, field
from oemof_tabular_plugins.wefe.facades import MIMO
from oemof_tabular_plugins.wefe.facades import functions as f
from oemof_tabular_plugins.wefe.global_specs import crop_dict, soil_dict


@dataclass(unsafe_hash=False, frozen=False, eq=False)
class Crop(MIMO):
    """ """
    name: str = ""

    type: str = "crop"

    tech: str = "crop"

    carrier: str = ""

    primary: str = ""

    expandable: bool = False

    capacity: float = 0

    capacity_minimum: float = None

    capacity_potential: float = None

    capacity_cost: float = 0

    irrigation_bus: Bus = None

    crop_bus: Bus = None

    biomass_bus: Bus = None

    crop_time_profile: Union[float, Sequence[float]] = None

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

    def __init__(self, **attributes):
        """
        Preprocessing of crop input data to calculate conversion_factors
        and assign these together with the correct busses to MIMO parent class
        """
        # Create DataFrame out of input profiles (time series), time_profile will be set as DatetimeIndex
        profiles_dict = {
            key.replace("_profile", ""): value
            for key, value in attributes.items()
            if key.endswith("profile")
        }
        time_index = profiles_dict.pop("time")
        profiles_df = pd.DataFrame(data=profiles_dict, index=time_index)

        # Get crop and soil parameters from database, calculate cultivation parameters
        self.crop_type = attributes.pop("crop_type")
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

        # Calculate irrigation and total biomass yield
        f.calc_f_water(
            profiles_df,
            has_rainwater_harvesting=False,
            frt=1,
            gcr=0,
            **attributes,
            **soil_params,
            **crop_params,
        )
        f.calc_f_solar(profiles_df, **crop_params)
        f.adapt_irrigation(profiles_df)
        f.calc_biomass(profiles_df, frt=1, **crop_params)

        # Format the profiles (no 0 allowed, no negative values, round to 10 decimal places)
        profiles_df = profiles_df.mask((profiles_df <= 1e-5), 1e-10)
        profiles_df = profiles_df.round(10)

        # Update conversion factors for the different inputs and outputs
        conversion_factors = {}
        for key, value in attributes.items():
            if hasattr(value, "type") and value.type == "bus":
                if key.startswith("irrigation_bus"):
                    conversion_factors[f"conversion_factor_{value}"] = sequence(
                        profiles_df["irrigation"]
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
        attributes.update(conversion_factors)

        self.irrigation_bus = attributes.pop("irrigation_bus")
        self.crop_bus = attributes.pop("crop_bus")
        self.biomass_bus = attributes.pop("biomass_bus")

        # Initializes MIMO with the crop-specific buses and conversion_factors
        super().__init__(
            from_bus_0=self.irrigation_bus,
            to_bus_0=self.crop_bus,
            to_bus_1=self.biomass_bus,
            **attributes,
        )

        # Other mandatory arguments
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