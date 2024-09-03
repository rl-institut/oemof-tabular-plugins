"""
Source
------------
SIMPLE crop model: Crop and cultivar parameters (Table 1b)
https://doi.org/10.1016/j.eja.2019.01.009

If multiple values for different cultivars are given, the average is used.

Crops
------------
wheat
rice
maize
soybean
drybean
peanut
potato
cassava
tomato
sweetcorn
greenbean
carrot
cotton
banana

Parameters
-------------
t_sum:          cumulative temperature until maturity [K]
hi:             harvest index (percentage of biomass that is actual harvest eg. fruits of the plant) [-]
i50a:           cumulative temp. from sowing on to reach 50% of solar radiation interception during growth [K]
i50b:           cumulative temp. until maturity to fall back to 50% of solar rad. intercept. during decline [K]
t_base:         base temp. for plant growth [°C]
t_opt:          optimal temp. for plant growth [°C]
rue:            radiation use efficiency [g/MJ/m²]
i50maxh:        coefficient for heat tress impact on i50b [-]
i50maxw:        coefficient for water stress impact on i50b [-]
t_max:          temperature where plant growth starts to be impaired [°C]
t_ext:          temperature where plant growth stops due to heat stress [°C]
s_co2:          sensitivity to CO2 availability [-]
s_water:        sensitivity to water stress [-]
f_solar_max:    maximum solar radiation interception (fraction of sunlight that is reaching the plants) [-]
"""

crop_dict = {
    "banana": {
        "t_sum": 6600,
        "hi": 0.19,
        "i50a": 600,
        "i50b": 400,
        "t_base": 10,
        "t_opt": 25,
        "rue": 0.80,
        "i50maxh": 100,
        "i50maxw": 5,
        "t_max": 34,
        "t_ext": 45,
        "s_co2": 0.07,
        "s_water": 2.5,
        "f_solar_max": 0.95,
    },
    "cassava": {
        "t_sum": 5400,
        "hi": 0.65,
        "i50a": 650,
        "i50b": 300,
        "t_base": 12,
        "t_opt": 28,
        "rue": 1.10,
        "i50maxh": 100,
        "i50maxw": 15,
        "t_max": 38,
        "t_ext": 50,
        "s_co2": 0.07,
        "s_water": 1.0,
        "f_solar_max": 0.95,
    },
    "wheat": {
        "t_sum": 2150,
        "hi": 0.34,
        "i50a": 280,
        "i50b": 50,
        "t_base": 0,
        "t_opt": 15,
        "rue": 1.24,
        "i50maxh": 100,
        "i50maxw": 25,
        "t_max": 34,
        "t_ext": 45,
        "s_co2": 0.08,
        "s_water": 0.4,
        "f_solar_max": 0.95,
    },
    "tomato": {
        "t_sum": 2800,
        "hi": 0.68,
        "i50a": 520,
        "i50b": 400,
        "t_base": 6,
        "t_opt": 26,
        "rue": 1,
        "i50maxh": 100,
        "i50maxw": 5,
        "t_max": 32,
        "t_ext": 45,
        "s_co2": 0.07,
        "s_water": 2.5,
        "f_solar_max": 0.95,
    },
    "sweetcorn": {
        "t_sum": 1900,
        "hi": 0.40,
        "i50a": 500,
        "i50b": 250,
        "t_base": 8,
        "t_opt": 27,
        "rue": 1.7,
        "i50maxh": 100,
        "i50maxw": 5,
        "t_max": 34,
        "t_ext": 50,
        "s_co2": 0.01,
        "s_water": 2,
        "f_solar_max": 0.95,
    },
    "greenbean": {
        "t_sum": 1600,
        "hi": 0.45,
        "i50a": 370,
        "i50b": 500,
        "t_base": 5,
        "t_opt": 27,
        "rue": 0.86,
        "i50maxh": 100,
        "i50maxw": 10,
        "t_max": 32,
        "t_ext": 45,
        "s_co2": 0.07,
        "s_water": 0.4,
        "f_solar_max": 0.95,
    },
    "carrot": {
        "t_sum": 2450,
        "hi": 0.70,
        "i50a": 550,
        "i50b": 250,
        "t_base": 4,
        "t_opt": 22,
        "rue": 1,
        "i50maxh": 100,
        "i50maxw": 5,
        "t_max": 32,
        "t_ext": 45,
        "s_co2": 0.07,
        "s_water": 2,
        "f_solar_max": 0.95,
    },
    "cotton": {
        "t_sum": 4600,
        "hi": 0.40,
        "i50a": 680,
        "i50b": 200,
        "t_base": 11,
        "t_opt": 28,
        "rue": 0.85,
        "i50maxh": 40,
        "i50maxw": 10,
        "t_max": 35,
        "t_ext": 50,
        "s_co2": 0.09,
        "s_water": 1.2,
        "f_solar_max": 0.95,
    },
    "peanut": {
        "t_sum": 3100,
        "hi": 0.35,
        "i50a": 520,
        "i50b": 550,
        "t_base": 10,
        "t_opt": 28,
        "rue": 1.2,
        "i50maxh": 100,
        "i50maxw": 5,
        "t_max": 36,
        "t_ext": 50,
        "s_co2": 0.07,
        "s_water": 2,
        "f_solar_max": 0.95,
    },
    "rice": {
        "t_sum": 2300,
        "hi": 0.47,
        "i50a": 850,
        "i50b": 200,
        "t_base": 9,
        "t_opt": 26,
        "rue": 1.24,
        "i50maxh": 100,
        "i50maxw": 10,
        "t_max": 34,
        "t_ext": 50,
        "s_co2": 0.08,
        "s_water": 1,
        "f_solar_max": 0.95,
    },
    "maize": {
        "t_sum": 2050,
        "hi": 0.50,
        "i50a": 500,
        "i50b": 50,
        "t_base": 8,
        "t_opt": 28,
        "rue": 2.1,
        "i50maxh": 100,
        "i50maxw": 12,
        "t_max": 34,
        "t_ext": 50,
        "s_co2": 0.01,
        "s_water": 1.2,
        "f_solar_max": 0.95,
    },
    "soybean": {
        "t_sum": 2425,
        "hi": 0.375,
        "i50a": 640,
        "i50b": 250,
        "t_base": 6,
        "t_opt": 27,
        "rue": 0.86,
        "i50maxh": 120,
        "i50maxw": 20,
        "t_max": 36,
        "t_ext": 50,
        "s_co2": 0.07,
        "s_water": 0.9,
        "f_solar_max": 0.95,
    },
    "drybean": {
        "t_sum": 2700,
        "hi": 0.40,
        "i50a": 450,
        "i50b": 600,
        "t_base": 5,
        "t_opt": 27,
        "rue": 0.8,
        "i50maxh": 90,
        "i50maxw": 20,
        "t_max": 32,
        "t_ext": 45,
        "s_co2": 0.07,
        "s_water": 0.9,
        "f_solar_max": 0.95,
    },
    "potato": {
        "t_sum": 2467,
        "hi": 0.73,
        "i50a": 592,
        "i50b": 417,
        "t_base": 4,
        "t_opt": 22,
        "rue": 1.3,
        "i50maxh": 50,
        "i50maxw": 30,
        "t_max": 34,
        "t_ext": 45,
        "s_co2": 0.1,
        "s_water": 0.4,
    }
}
