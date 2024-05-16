"""
Run this script from the root directory of the datapackage to update
or create meta data.
"""
from oemof.tabular.datapackage import building

# This part is for testing only: It allows to pass
# the filename of inferred metadata other than the default.
if "kwargs" not in locals():
    kwargs = {}

# This automatically generates a datapackage.json file and the foreign keys should be
# updated depending on the energy system design
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

# run building.infer, reopen json file and ammend datapackage