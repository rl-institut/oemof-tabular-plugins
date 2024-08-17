# from .pre_processing import calculate_annuity
from oemof.tools import logger, economics
# from .pre_processing import calculate_annuity


def calculate_annuity(capex, opex_fix, lifetime, wacc):
    """
    Calculates the total annuity for each component, including CAPEX and fixed OPEX.
    :param capex: CAPEX (currency/MW*) *or the unit you choose to use throughout the model e.g. kW/GW
    :param opex_fix: fixed OPEX (currency/MW*/year)
    :param lifetime: lifetime of the component (years)
    :param wacc: weighted average cost of capital (WACC) applied throughout the model (%)
    :return: total annuity (currency/MW*/year)
    """
    annuity_capex = economics.annuity(capex, lifetime, wacc)
    annuity_opex_fix = opex_fix
    annuity_total = round(annuity_capex + annuity_opex_fix, 2)
    return annuity_total

def pre_processing_moo(wacc, element, element_path, element_df):
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

    Applies pre-processing costs to the input CSV files, where the annuity ('capacity_cost') is either
    used directly if stated, or if left empty then calculated using the calculate_annuity function,
    or if all parameters are stated a choice is given.
    :param wacc: weighted average cost of capital (WACC) applied throughout the model (%)
    :param element: csv filename
    :param element_path: path of the csv file
    :param element_df: dataframe containing data from the csv file
    """

    # for every element other than storage, the MOO Indicator is 'capacity_cost'
    # for storage, the annuity cost parameter is 'storage_capacity_cost'
    if element != "storage.csv":
        moo_indicator = "capacity_cost"
    else:
        moo_indicator = "storage_capacity_cost"
        # loop through each entry in the csv file

    # ---------------- Possible SCENARIOS ----------------

    if element in ["bus.csv", "load.csv", "excess.csv"]:
        scenario = "no moo indicator"
    elif element in ["conversion.csv", "mimo.csv", "storage.csv", "volatile.csv"]:
        scenario = "moo indicator calculation"
    elif element in "dispatchable.csv":
        scenario = "dispatchable moo indicator calculation"
    else:
        scenario = "no moo indicator_raise error undefined tech"

    # ---------------- ACTIONS TAKEN FOR EACH SCENARIO ----------------
    for index, row in element_df.iterrows():
        # define the row name
        row_name = row["name"]
        if scenario == "moo indicator calculation":
            # store the parameters
            capex = row["capex"]
            opex_fix = row["opex_fix"]
            lifetime = row["lifetime"]
            capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
            element_df.at[index, moo_indicator] = float(capacity_cost)
            # log info message
            logger.info(
            f"the annuity ('{moo_indicator}') has been calculated and updated for"
            f" '{row_name}' in '{element}'.")

        elif scenario == "dispatchable moo indicator calculation":
            logger.info(
                f"'{element}' does not contain all '{moo_indicator}' parameter. Skipping for now, working on it later..."
            )

        elif scenario == "no moo indicator":
            logger.info(
                f"'{element}' does not contain '{moo_indicator}' parameter. Skipping..."
            )
    # save the updated dataframe to the csv file
    element_df.to_csv(element_path, sep=";", index=False)
    return
