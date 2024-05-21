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
    package_name="general-basic",
    foreign_keys={
        "bus": ["volatile", "dispatchable", "storage", "load", "shortage", "excess"],
        "profile": ["load", "volatile"],
        "chp": ["chp"],
        "from_to_bus": ["link", "conversion"],
    },
    **kwargs,
)
