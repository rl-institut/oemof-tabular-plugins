"""
Source
------------
SIMPLE crop model: Site information and related soil parameters (Table 1b)
https://doi.org/10.1016/j.eja.2019.01.009

If multiple values for different cultivars are given, the average is used.

Please note that these values are stored as global parameters for every crop,
even though they reflect specific locations which may lead to errors.

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
------------
awc:    average water holding capacity [-]
rcn:    runoff curve number [-]
ddc:    deep drainage coefficient [-]
rzd:    root zone depth [mm]
"""

soil_dict = {
    'banana': {
        'awc': 0.11,
        'rcn': 70,
        'ddc': 0.5,
        'rzd': 1500,
    },
    'cassava': {
        'awc': 0.13,
        'rcn': 85,
        'ddc': 0.4,
        'rzd': 1500,
    },
    'wheat': {
        'awc': 0.135,
        'rcn': 67.5,
        'ddc': 0.4,
        'rzd': 900,
    },
    'tomato': {
        'awc': 0.165,
        'rcn': 75.5,
        'ddc': 0.3,
        'rzd': 900,
    },
    'rice': {
        'awc': 0.12,
        'rcn': 70,
        'ddc': 0.3,
        'rzd': 400,
    },
    'maize': {
        'awc': 0.12,
        'rcn': 70,
        'ddc': 0.3,
        'rzd': 1500,
    },
    'soybean': {
        'awc': 0.13,
        'rcn': 70,
        'ddc': 0.57,
        'rzd': 1630,
    },
    'drybean': {
        'awc': 0.08,
        'rcn': 65,
        'ddc': 0.5,
        'rzd': 700,
    },
    'potato': {
        'awc': 0.11,
        'rcn': 67.25,
        'ddc': 0.625,
        'rzd': 825,
    },
    'sweetcorn': {
        'awc': 0.25,
        'rcn': 60,
        'ddc': 0.7,
        'rzd': 1500,
    },
    'green bean': {
        'awc': 0.08,
        'rcn': 65,
        'ddc': 0.5,
        'rzd': 1000,
    },
    'carrot': {
        'awc': 0.09,
        'rcn': 75,
        'ddc': 0.4,
        'rzd': 500,
    },
    'cotton': {
        'awc': 0.12,
        'rcn': 70,
        'ddc': 0.3,
        'rzd': 600,
    }
}
