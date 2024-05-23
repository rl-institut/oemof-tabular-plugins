def pre_processing_moo():
    """This function will run the multi-objective optimization

    The outcome is that the main costs 'capacity_cost' will be replaced by an aggregated
    indicator representing the multi-objective optimization goals.
    This function will replace the pre_processing_costs function if moo is set to True

    Inputs
    Weights: defined by model-user e.g. percentages up to 1
        (in optiMG, the user defining this will be e.g. local prosumers)
    0.5 for costs 0.2 for emissions 0.2 for land requirement 0.1 for water dissipated
    Has to add up to 1, otherwise error

    Normalization: done with global values
    User sets costs are defined in the CSV e.g. CAPEX, OPEX fix, lifetime
        -> this cost is normalised based on global GDP
        -> value is calculated for proportion of cost to global GDP
    Same applies for total emissions
        -> the value is normalised based on total global annual GHG emissions
    Land requirements
        ->  the value is normalised based on the worlds surface area
    Water dissipated
        -> the value is normalised based on e.g. global availability
    Then these values are added together

    User includes specific cost, specific emission factor, specific land requirement, specific water footprint
    This function will take those, normalise based on normalisation data (global)
    One aggregated value will be calculated based on adding these values
    This value will be entered in the csv file under 'capacity_cost'
    The csv file will be updated

    :return:
    """
    return
