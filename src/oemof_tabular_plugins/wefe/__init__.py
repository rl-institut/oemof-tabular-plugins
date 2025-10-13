from oemof_tabular_plugins.wefe.constraints.constraint_facades import (
    CONSTRAINT_TYPE_MAP,
)
from oemof.tabular.facades import TYPEMAP

from oemof.tabular.facades import Volatile
from oemof_tabular_plugins.wefe.facades import UltraFiltration, NanoFiltration, MicroFiltration, BioFiltration, \
    ActivatedCarbonFilter, CeramicFilter, CartridgeFilter, SlowSandFilter, ElectrodialysisUnit, IonExchange, \
    UVDisinfection, Boiling, Distillation, MembraneDistillation, Ozonation, PhotocatalyticUnit, \
    BiologicalDenitrification, Adsorption, CoagulationFlocculation

from .facades import (
    PVPanel,
    WindTurbine,
    MIMO,
    APV,
    MimoCrop,
    SimpleCrop,
    WaterPump,
    WaterFiltration,
    Inverter,
    RRHydropower,
    ReverseOsmosis
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
    "wind-turbine": WindTurbine,
    "reverse_osmosis": ReverseOsmosis,
    "ultrafiltration": UltraFiltration,
    "nanofiltration": NanoFiltration,
    "microfiltration": MicroFiltration,
    "biofiltration": BioFiltration,
    "activated_carbon_filter": ActivatedCarbonFilter,
    "ceramic_filter": CeramicFilter,
    "cartridge_filter": CartridgeFilter,
    "slow_sand_filter": SlowSandFilter,
    "electrodialysis": ElectrodialysisUnit,
    "ion_exchange": IonExchange,
    "uv_disinfection": UVDisinfection,
    "boiling": Boiling,
    "distillation": Distillation,
    "membrane_distillation": MembraneDistillation,
    "ozonation": Ozonation,
    "photocatalysis": PhotocatalyticUnit,
    "biological_denitrification": BiologicalDenitrification,
    "adsorption": Adsorption,
    "coagulation_flocculation": CoagulationFlocculation
}

WEFE_TYPEMAP.update(TYPEMAP)
