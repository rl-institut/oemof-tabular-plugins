from oemof_tabular_plugins.wefe.constraints.constraint_facades import (
    CONSTRAINT_TYPE_MAP,
)
from oemof.tabular.facades import TYPEMAP
from .facades import (
    PVPanel1,
    PVPanel2,
    PVPanel3,
    PVPanel4,
    PVPanel5,
    MIMO,
    APV,
    MimoCrop,
    SimpleCrop,
    WaterPump,
    WaterFiltration,
)

WEFE_TYPEMAP = {
    "water-pump": WaterPump,
    "water-filtration": WaterFiltration,
    "crop": SimpleCrop,
    "mimo-crop": MimoCrop,
    "pv-panel": PVPanel1,
    "pv-panel-2": PVPanel2,
    "pv-panel-3": PVPanel3,
    "pv-panel-4": PVPanel4,
    "pv-panel-5": PVPanel5,
    "mimo": MIMO,
    "apv": APV,
}

WEFE_TYPEMAP.update(TYPEMAP)
