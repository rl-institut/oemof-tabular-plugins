import json
from oemof_tabular_plugins.general.pre_processing import logger


def add_foreign_keys_to_datapackage(datapackage_filename, foreign_keys):
    """Add additional/custom foreign keys to the datapackage for new facades (that are not
    already existing in oemof-tabular). To be implemented in your_scenario/scripts/infer.py after running
    the building.infer_metadata function which creates the datapackage.json file. This function
    allows for any missing foreign keys to be added (must be specified by user). The aim is to eventually
    integrate this function into oemof-tabular.

    Examples
    --------
    See examples/scenarios/wefe_pv_panel/scripts/infer.py to see this function in use

    :param datapackage_filename: datapackage json filename
    :param foreign_keys: dictionary specifying the foreign key references to be added
    """
    # load the existing datapackage.json file
    with open(datapackage_filename, "r") as file:
        datapackage = json.load(file)
    # loop through each resource in the datapackage
    for resource in datapackage["resources"]:
        resource_name = resource["name"]
        if resource_name in foreign_keys:
            # initialize foreignKeys if it doesn't exist
            if "foreignKeys" not in resource["schema"]:
                resource["schema"]["foreignKeys"] = []
            # add foreign key references based on the provided dictionary
            for field, reference in foreign_keys[resource_name].items():
                resource["schema"]["foreignKeys"].append(
                    {"fields": field, "reference": reference}
                )
            logger.info(f"Additional foreign keys added for {resource_name}")
    # save the modified datapackage.json file
    with open(datapackage_filename, "w") as file:
        json.dump(datapackage, file, indent=4)
