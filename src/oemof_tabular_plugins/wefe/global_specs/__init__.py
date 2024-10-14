"""
This subpackage of wefe will contain the global global_specs that can be used across multiple projects
e.g. different PV panel global_specs or different crop global_specs (energy content etc)
ToDo: decide if these global_specs actually belong in oemof-tabular-plugins or if they should be stored somewhere else
"""

import json
import os

from .soil_specs import soil_dict
from .crop_specs import crop_dict
from .pv_modules import pv_dict

current_dir = os.path.dirname(os.path.abspath(__file__))
geometry_path = os.path.join(current_dir, "geometry.json")
with open(geometry_path, "r") as f:
    geo_dict = json.load(f)
