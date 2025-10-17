from oemof_tabular_plugins.wefe.constraints.constraint_facades import (
    CONSTRAINT_TYPE_MAP,
)
from oemof.tabular.facades import TYPEMAP

from oemof.tabular.facades import Volatile
from .facades import (
    PVPanel,
    WindTurbine,
    MIMO,
    APV,
    MimoCrop,
    SimpleCrop,
    Crop,
    WaterPump,
    WaterFiltration,
    Inverter,
    RRHydropower,
)

WEFE_TYPEMAP = {
    "water-pump": WaterPump,
    "water-filtration": WaterFiltration,
    "crop": Crop,
    "pv-panel": PVPanel,
    "mimo": MIMO,
    "apv": APV,
    "inverter": Inverter,
    "hydropower": RRHydropower,
    "river-flow": Volatile,
    "wind-turbine": WindTurbine
}

WEFE_TYPEMAP.update(TYPEMAP)
