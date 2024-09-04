from oemof_tabular_plugins.wefe.constraints.constraint_facades import (
    CONSTRAINT_TYPE_MAP,
)

from .facades import WaterPump, WaterFiltration, SimpleCrop, PVPanel, MIMO

TYPEMAP = {
    "water-pump": WaterPump,
    "water-filtration": WaterFiltration,
    "crop": SimpleCrop,
    "pv-panel": PVPanel,
    "mimo": MIMO,
}
