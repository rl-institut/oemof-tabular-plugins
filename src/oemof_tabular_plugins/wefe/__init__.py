from oemof_tabular_plugins.wefe.constraints.constraint_facades import (
    CONSTRAINT_TYPE_MAP,
)
from oemof.tabular.facades import TYPEMAP
from .facades import WaterPump, WaterFiltration, SimpleCrop, PVPanel, MIMO

WEFE_TYPEMAP = {
    "water-pump": WaterPump,
    "water-filtration": WaterFiltration,
    "crop": SimpleCrop,
    "pv-panel": PVPanel,
    "mimo": MIMO,
}

WEFE_TYPEMAP = TYPEMAP.update(WEFE_TYPEMAP)
