"""
Run this script from the root directory of the datapackage to update or create meta-data in the
form of a json file (datapackage.json)
"""

# TODO this should be with from oemof.tabular.datapackage import building when https://github.com/oemof/oemof-tabular/pull/173 is merged
from oemof_tabular_plugins.datapackage import building as otp_building

# this part is for testing only: It allows to pass
# the filename of inferred metadata other than the default.
if "kwargs" not in locals():
    kwargs = {}

# this automatically generates a datapackage.json file
otp_building.infer_metadata_from_data(
    package_name="wefe-pv-panel",
)
