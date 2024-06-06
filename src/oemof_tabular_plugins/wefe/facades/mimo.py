from oemof.industry import MimoFacade


def validate_mimo_datapackage(cls, resource):
    # modify the resource (datapackage.resource)
    # should it return the resource?
    pass


def processing_mimo_raw_inputs(cls, resource, results_df):
    # function to apply on df from above (drop the thee columns (conversion_factor_ac-elec-bus, conversion_factor_permeate-bus, conversion_factor_brine-bus) and turn them into one conversion_factor column), then add the primary column

    return results_df


MimoFacade.validate_datapackage = classmethod(validate_mimo_datapackage)
MimoFacade.processing_raw_inputs = classmethod(processing_mimo_raw_inputs)
