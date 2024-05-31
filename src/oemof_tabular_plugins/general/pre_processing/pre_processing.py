import os
import pandas as pd
import logging
from oemof.tools import logger, economics
import json
from .pre_processing_moo import pre_processing_moo

logger.define_logging()


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


def pre_processing_costs(wacc, element, element_path, element_df):
    """
    Applies pre-processing costs to the input CSV files, where the annuity ('capacity_cost') is either
    used directly if stated, or if left empty then calculated using the calculate_annuity function,
    or if all parameters are stated a choice is given.
    :param wacc: weighted average cost of capital (WACC) applied throughout the model (%)
    :param element: csv filename
    :param element_path: path of the csv file
    :param element_df: dataframe containing data from the csv file
    """
    # for every element other than storage, the annuity cost parameter is 'capacity_cost'
    # for storage, the annuity cost parameter is 'storage_capacity_cost'
    if element != "storage.csv":
        annuity_cost = "capacity_cost"
    else:
        annuity_cost = "storage_capacity_cost"
    # check if any of the required columns are missing
    cost_columns = {"capex", "opex_fix", "lifetime"}
    missing_columns = cost_columns - set(element_df.columns)

    # ---------------- POSSIBLE SCENARIO GROUPS FOR PARAMETER ENTRIES ----------------
    # scenario group "annuity no cost params": annuity parameter is included and all of capex, opex fix
    # and lifetime parameters are not included in the csv file
    if annuity_cost in element_df.columns and cost_columns == missing_columns:
        scenario_group = "annuity no cost params"
    # scenario group "annuity partial cost params": annuity parameter is included and some but not all of
    # capex, opex fix and lifetime parameters are included in the csv file
    elif (
        annuity_cost in element_df.columns
        and missing_columns
        and cost_columns != missing_columns
    ):
        scenario_group = "annuity partial cost params"
    # scenario group "annuity all cost params": annuity parameter is included and all of capex, opex fix
    # and lifetime parameters are included in the csv file
    elif annuity_cost in element_df.columns and not missing_columns:
        scenario_group = "annuity all cost params"
    # scenario group "no annuity partial/all cost params": annuity parameter is not included and at least one
    # of capex, opex fix and lifetime parameters are included in the csv file
    elif annuity_cost not in element_df.columns and cost_columns != missing_columns:
        scenario_group = "no annuity partial/all cost params"
    # scenario group "no annuity no cost params": the annuity parameter is not included and neither are
    # capex, opex fix and lifetime parameters in the csv file
    elif annuity_cost not in element_df.columns and cost_columns == missing_columns:
        scenario_group = "no annuity no cost params"

    # ---------------- POSSIBLE SCENARIOS FOR EACH SCENARIO GROUP ----------------
    # loop through each entry in the csv file
    for index, row in element_df.iterrows():
        # define the row name
        row_name = row["name"]
        if scenario_group == "annuity no cost params":
            # scenario "annuity empty no cost params": the annuity parameter is left empty and the other cost
            # parameters have not been included
            if pd.isna(row[annuity_cost]):
                scenario = "annuity empty no cost params"
            # scenario "annuity defined no cost params": the annuity parameter is defined and the other cost
            # parameters have not been included
            else:
                scenario = "annuity defined no cost params"
        elif scenario_group == "annuity partial cost params":
            # scenario "annuity empty partial cost params": the annuity parameter is left empty and only
            # some other financial parameters are included
            if pd.isna(row[annuity_cost]):
                scenario = "annuity empty partial cost params"
            # scenario "annuity defined partial cost params": the annuity parameter is defined and only
            # some other financial parameters are included
            else:
                scenario = "annuity defined partial cost params"
        elif scenario_group == "annuity all cost params":
            # store the parameters
            capex = row["capex"]
            opex_fix = row["opex_fix"]
            lifetime = row["lifetime"]
            # scenario "annuity empty all cost params": the annuity parameter is left empty and all of the
            # other financial parameters are defined
            if (
                pd.isna(row[annuity_cost])
                and pd.notna(capex)
                and pd.notna(opex_fix)
                and pd.notna(lifetime)
            ):
                scenario = "annuity empty all cost params defined"
            # scenario "annuity all cost params some empty": the annuity parameter is either defined or empty,
            # but at least one of 'capex', 'opex_fix' and 'lifetime' is left empty
            elif pd.isna(capex) or pd.isna(opex_fix) or pd.isna(lifetime):
                scenario = "annuity all cost params some empty"
            # scenario "annuity defined all cost params defined": both the annuity parameter is defined and
            # all the other financial parameters are defined
            else:
                scenario = "annuity defined all cost params defined"
        elif scenario_group == "no annuity partial/all cost params":
            # store the parameters
            capex = row["capex"]
            opex_fix = row["opex_fix"]
            lifetime = row["lifetime"]
            # scenario "no annuity partial/all cost params empty": at least one of 'capex', 'opex_fix' and
            # 'lifetime' is left empty
            if pd.isna(capex) or pd.isna(opex_fix) or pd.isna(lifetime):
                scenario = "no annuity partial/all cost params empty"
            # scenario "no annuity all cost params defined": all financial parameters are defined
            else:
                scenario = "no annuity all cost params defined"
        elif scenario_group == "no annuity no cost params":
            # scenario "no annuity no cost params": neither the annuity or financial parameters are defined
            scenario = "no annuity no cost params"

        # ---------------- ACTIONS TAKEN FOR EACH SCENARIO ----------------
        if scenario == "annuity empty no cost params":
            # raise value error
            raise ValueError(
                f"'{annuity_cost}' (the annuity) has been left empty for '{row_name}' "
                f"in '{element}', and 'capex', 'opex_fix' and 'lifetime' have not "
                f" been included. \nEither the annuity ('{annuity_cost}') must be "
                f"directly stated or all of the other financial parameters must be stated "
                f"to calculate the annuity."
            )
        elif scenario == "annuity defined no cost params":
            # log info message
            logger.info(
                f"The annuity cost is directly used for '{row_name}' in '{element}'."
            )
        elif scenario == "annuity empty partial cost params":
            # raise value error
            raise ValueError(
                f"'{annuity_cost}' (the annuity) has been left empty for '{row_name}' "
                f"in '{element}', and not all of 'capex', 'opex_fix' and 'lifetime' have"
                f" been included. \nEither the annuity ('{annuity_cost}') must be "
                f"directly stated or all of the other financial parameters must be stated "
                f"to calculate the annuity."
            )
        elif scenario == "annuity defined partial cost params":
            # log warning message
            logging.warning(
                f"'{annuity_cost}' (the annuity) has been defined and some but not all "
                f"of 'capex', 'opex_fix' and 'lifetime' have been defined for {row_name} "
                f"in {element}. The annuity will be directly used but be aware that some "
                f"cost results will not be calculated."
            )
        elif scenario == "annuity empty all cost params defined":
            # calculate the annuity using the calculate_annuity function
            capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
            # update the dataframe
            element_df.at[index, annuity_cost] = float(capacity_cost)
            # log info message
            logger.info(
                f"the annuity ('{annuity_cost}') has been calculated and updated for"
                f" '{row_name}' in '{element}'."
            )
        elif scenario == "annuity all cost params some empty":
            # log warning message
            logging.warning(
                f"One or more of 'capex', 'opex_fix' and 'lifetime' have been left "
                f"empty for {row_name} in {element}. The annuity will be directly used "
                f"but be aware that some cost results will not be calculated."
            )
        elif scenario == "annuity defined all cost params defined":
            capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
            # update the dataframe
            element_df.at[index, annuity_cost] = float(capacity_cost)
            # if all parameters are defined, the user is asked if they want to calculate the annuity
            # from the capex, opex_fix and lifetime or use the annuity directly
            logger.info(
                f"All parameters ('capex', 'opex_fix', 'lifetime') and '{annuity_cost}' are "
                f"provided for '{row_name}' in '{element}'. \nThe defined annuity has been replaced with "
                f"the calculated value from capex, opex_fix and lifetime."
            )
        elif scenario == "no annuity partial/all cost params empty":
            # raise value error
            raise ValueError(
                f"One or more of 'capex', 'opex_fix' and 'lifetime' have been left "
                f"empty for {row_name} in {element}. Please enter values or remove the"
                f" parameters and include "
                f"the 'capacity_cost'."
            )
        elif scenario == "no annuity all cost params defined":
            # calculate the annuity using the calculate_annuity function
            capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
            # update the dataframe
            element_df["capacity_cost"] = float(capacity_cost)
            # log info message
            logger.info(
                f"the annuity ('{annuity_cost}') has been calculated and updated for"
                f" '{row_name}' in '{element}'."
            )
        elif scenario == "no annuity no cost params":
            logger.info(
                f"'{element}' does not contain '{annuity_cost}' parameter. Skipping..."
            )

    # save the updated dataframe to the csv file
    element_df.to_csv(element_path, sep=";", index=False)
    return


def pre_processing_custom_attributes(element_path, element_df, custom_attributes):
    # ToDo: confirm if this function is needed, whether the attributes need to be added to
    #  'output_parameters' or if this is not necessary
    """Updates the 'output_parameters' field in the CSV file for the specified element if custom
    attributes are defined.

    :param element_path: path of the csv file
    :param element_df: dataframe containing data from the csv file
    :param custom_attributes: list of custom attributes included in the model (defined in compute.py)
    """
    # iterate over each entry in the dataframe (from csv file)
    for index, row in element_df.iterrows():
        # create empty custom attributes dict
        custom_attributes_dict = {}
        # set boolean to false
        has_custom_attributes = False
        # check if custom_attributes is not none before iterating
        if custom_attributes is not None:
            # check if any of the custom attributes list are in the dataframe columns
            for attribute in custom_attributes:
                if attribute in element_df.columns:
                    value = row[attribute]
                    # add the attribute to the custom attributes dict
                    custom_attributes_dict[attribute] = value
                    # set boolean to true
                    has_custom_attributes = True
                # if custom attributes are found for this row, add them to 'output_parameters'
                if has_custom_attributes:
                    output_parameters_str = json.dumps(
                        {"custom_attributes": custom_attributes_dict}
                    )
                    element_df.at[index, "output_parameters"] = output_parameters_str
                else:
                    # no custom attributes found, do not update 'output_parameters'
                    continue
    # write the updated dataframe back to the csv file
    element_df.to_csv(element_path, sep=";", index=False)
    return


def pre_processing(scenario_dir, wacc, custom_attributes=None, moo=False):
    """Performs pre-processing of input scenario data before running the model.

    :param scenario_dir: scenario directory path
    :param wacc: weighted average cost of capital (WACC) applied throughout the model (%)
    :param custom_attributes: list of custom attributes included in the model (defined in compute.py), default is None
    :param moo: whether the multi-objective optimization is activated, default is False
    """
    if moo is False:
        logger.info(f"Optimization activated for only costs")
    elif moo is True:
        logger.info(f"Multi-objective optimization activated")
    logger.info("Pre-processing activated")
    # locate the elements directory
    elements_dir = os.path.join(scenario_dir, "data", "elements")
    # raise error if the elements directory is not found in the scenario directory
    if not os.path.exists(elements_dir):
        raise FileNotFoundError(f"No 'elements' directory found in {scenario_dir}.")
    # loop through each csv file in the elements directory
    for element in os.listdir(elements_dir):
        # only consider csv files
        if element.endswith(".csv"):
            # set the path of the considered csv file
            element_path = os.path.join(elements_dir, element)
            # read the csv file and save it as a pandas dataframe
            element_df = pd.read_csv(element_path, sep=";")
            if moo is False:
                # performs pre-processing of additional cost data (capex, opex_fix, lifetime)
                pre_processing_costs(wacc, element, element_path, element_df)
                # performs pre-processing for custom attributes (e.g. emission factor, renewable factor, land
                # requirement)
                pre_processing_custom_attributes(
                    element_path, element_df, custom_attributes
                )
            elif moo is True:
                pre_processing_moo()
    logger.info("Pre-processing completed")
    return
