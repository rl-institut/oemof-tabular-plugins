from oemof_tabular_plugins.wefe.constraints.constraint_facades import (
    CONSTRAINT_TYPE_MAP,
)
from oemof_tabular_plugins.general import OTP_TYPEMAP
from .facades import (
    PVPanel,
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
    "pv-panel": PVPanel,
    "mimo": MIMO,
    "apv": APV,
}

WEFE_TYPEMAP.update(OTP_TYPEMAP)
