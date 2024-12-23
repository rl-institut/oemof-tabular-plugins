from oemof_tabular_plugins.wefe.constraints.constraint_facades import (
    CONSTRAINT_TYPE_MAP,
)
from oemof.tabular.facades import TYPEMAP

from oemof.tabular.facades import (
    Volatile
)
from .facades import (
    PVPanel,
    MIMO,
    APV,
    MimoCrop,
    SimpleCrop,
    WaterPump,
    WaterFiltration,
    Inverter,
    RRHydropower
)

WEFE_TYPEMAP = {
    "water-pump": WaterPump,
    "water-filtration": WaterFiltration,
    "crop": SimpleCrop,
    "mimo-crop": MimoCrop,
    "pv-panel": PVPanel,
    "mimo": MIMO,
    "apv": APV,
    "inverter": Inverter,
    "hydropower": RRHydropower,
    "river-flow": Volatile,
}

WEFE_TYPEMAP.update(TYPEMAP)
