"""
Run this script from the root directory of the datapackage to update or create meta-data in the
form of a json file (datapackage.json)
"""
from oemof.tabular.datapackage import building
from oemof_tabular_plugins.datapackage import building as otp_building

# this part is for testing only: It allows to pass
# the filename of inferred metadata other than the default.
if "kwargs" not in locals():
    kwargs = {}

# this automatically generates a datapackage.json file and the foreign keys should be
# updated depending on the energy system design (works for existing facades, but for
# new facades the additional 'add_foreign_keys_to_datapackage' function might be needed)
building.infer_metadata(
    package_name="wefe-pv-panel",
    foreign_keys={
        "bus": [
            "dispatchable",
            "load",
            "excess",
            "pv-panel"
        ],
        "profile": ["dispatchable", "load", "pv-panel"],
        "from_to_bus": ["pv-panel"],
    },
    **kwargs,
)

# this function adds any additional foreign keys to the datapackage.json file
otp_building.add_foreign_keys_to_datapackage(
    "datapackage.json",
    {
        "pv_panel": {
            "from_bus": {"resource": "bus", "fields": "name"},
            "to_bus": {"resource": "bus", "fields": "name"},
            "t_air": {"resource": "pv_panel_profile"},
            "ghi": {"resource": "pv_panel_profile"}
        }
    }
)
