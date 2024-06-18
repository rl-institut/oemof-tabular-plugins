from oemof_industry.mimo_converter import MIMO


def validate_mimo_datapackage(cls, resource):
    """
    This datapackage validation method is necessary for the MIMO converter because the
    'primary' field automatically gets updated to the foreign keys in the datapackage.json
    by the infer_metadata_from_data, but it should not be interpreted as a foreign key.
    This method removes it.

    :param cls: class instance
    :param resource: the datapackage resource
    """
    # check if the 'foreignKeys' field exists in the schema
    if (
        "schema" in resource.descriptor
        and "foreignKeys" in resource.descriptor["schema"]
    ):
        # loop through each foreign key
        for foreign_key in resource.descriptor["schema"]["foreignKeys"]:
            if "primary" in foreign_key["fields"]:
                # remove the foreign_key regarding 'primary' from the resource
                resource.descriptor["schema"]["foreignKeys"].remove(foreign_key)
                break
    pass


def processing_mimo_raw_inputs(cls, resource, results_df):
    # function to apply on df from above (drop the thee columns (conversion_factor_ac-elec-bus,
    # conversion_factor_permeate-bus, conversion_factor_brine-bus) and turn them into one
    # conversion_factor column), then add the primary column

    return results_df


MIMO.validate_datapackage = classmethod(validate_mimo_datapackage)
MIMO.processing_raw_inputs = classmethod(processing_mimo_raw_inputs)
