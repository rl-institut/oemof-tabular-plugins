import os
import pandas as pd
import logging
from oemof.tools import logger, economics

logger.define_logging()


def calculate_annuity(capex, opex_fix, lifetime, wacc):
    """
    Calculates the total annuity for each component, including CAPEX and fixed OPEX.
    :param capex: CAPEX (currency/MW*) *or the unit you choose to use throughout the model e.g. kW/GW
    :param opex_fix: the fixed OPEX (currency/MW*/year)
    :param lifetime: the lifetime of the component (years)
    :param wacc: the weighted average cost of capital (WACC) applied throughout the model (%)
    :return: the total annuity (currency/MW*/year)
    """
    annuity_capex = economics.annuity(capex, lifetime, wacc)
    annuity_opex_fix = opex_fix
    annuity_total = round(annuity_capex + annuity_opex_fix, 2)
    return annuity_total


def pre_processing(scenario_dir, wacc):
    """
    Applies pre-processing to the input CSV files, where the annuity ('capacity_cost') is either
    used directly if stated, or if left empty then calculated using the calculate_annuity function,
    or if all parameters are stated a choice is given.
    :param scenario_dir: the scenario directory
    :param wacc: the weighted average cost of capital (WACC) applied throughout the model (%)
    :return: updated input CSV files
    """
    print("------------------------------------------------------"
          "\nPRE-PROCESSING ACTIVATED")
    # locate the elements directory
    elements_dir = os.path.join(scenario_dir, "data", "elements")
    # raise error if the elements directory is not found in the scenario directory
    if not os.path.exists(elements_dir):
        raise FileNotFoundError(f"No 'elements' directory found in {scenario_dir}.")
    # loop through each CSV file in the elements directory
    for element in os.listdir(elements_dir):
        print("------------------------------------------------------")
        # only consider CSV files
        if element.endswith(".csv"):
            # set the path of the considered CSV file
            element_path = os.path.join(elements_dir, element)
            # read the CSV file and save it as a pandas DataFrame
            element_df = pd.read_csv(element_path, sep=';')
            # for every element other than storage, the annuity cost parameter is 'capacity_cost'
            # for storage, the annuity cost parameter is 'storage_capacity_cost'
            if element != 'storage.csv':
                annuity_cost = 'capacity_cost'
            else:
                annuity_cost = 'storage_capacity_cost'
            # check if any of the required columns are missing
            cost_columns = {'capex', 'opex_fix', 'lifetime'}
            missing_columns = cost_columns - set(element_df.columns)

            # ---------------- POSSIBLE SCENARIO GROUPS FOR PARAMETER ENTRIES ----------------
            # scenario group "annuity no cost params": annuity parameter is included and all of capex, opex fix
            # and lifetime parameters are not included in the csv file
            if annuity_cost in element_df.columns and cost_columns == missing_columns:
                scenario_group = "annuity no cost params"
            # scenario group "annuity partial cost params": annuity parameter is included and some but not all of
            # capex, opex fix and lifetime parameters are included in the csv file
            elif annuity_cost in element_df.columns and missing_columns and cost_columns != missing_columns:
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
                row_name = row['name']
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
                    # some of the other financial parameters are included
                    if pd.isna(row[annuity_cost]):
                        scenario = "annuity empty partial cost params"
                    # scenario "annuity defined partial cost params": the annuity parameter is defined and only
                    # some of the other financial parameters are included
                    else:
                        scenario = "annuity defined partial cost params"
                elif scenario_group == "annuity all cost params":
                    # store the parameters
                    capex = row['capex']
                    opex_fix = row['opex_fix']
                    lifetime = row['lifetime']
                    # scenario "annuity empty all cost params": the annuity parameter is left empty and all of the
                    # other financial parameters are defined
                    if pd.isna(row[annuity_cost]) and pd.notna(capex) and pd.notna(opex_fix) and \
                            pd.notna(lifetime):
                        scenario = "annuity empty all cost params defined"
                    # scenario "annuity all cost params some empty": the annuity parameter is either defined or empty,
                    # but at least one of 'capex', 'opex_fix' and 'lifetime' is left empty
                    elif pd.isna(capex) or pd.isna(opex_fix) or pd.isna(lifetime):
                        scenario = "annuity all cost params some empty"
                    # scenario "annuity defined all cost params defined": both the annuity parameter is defined and
                    # all of the other financial parameters are defined
                    else:
                        scenario = "annuity defined all cost params defined"
                elif scenario_group == "no annuity partial/all cost params":
                    # store the parameters
                    capex = row['capex']
                    opex_fix = row['opex_fix']
                    lifetime = row['lifetime']
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
                    raise ValueError(f"'{annuity_cost}' (the annuity) has been left empty for '{row_name}' "
                                     f"in '{element}', and 'capex', 'opex_fix' and 'lifetime' have not "
                                     f" been included. \nEither the annuity ('{annuity_cost}') must be "
                                     f"directly stated or all of the other financial parameters must be stated "
                                     f"to calculate the annuity.")
                elif scenario == "annuity defined no cost params":
                    # log info message
                    logger.info(f"The annuity cost is directly used for '{row_name}' in '{element}'.")
                elif scenario == "annuity empty partial cost params":
                    # raise value error
                    raise ValueError(f"'{annuity_cost}' (the annuity) has been left empty for '{row_name}' "
                                     f"in '{element}', and not all of 'capex', 'opex_fix' and 'lifetime' have"
                                     f" been included. \nEither the annuity ('{annuity_cost}') must be "
                                     f"directly stated or all of the other financial parameters must be stated "
                                     f"to calculate the annuity.")
                elif scenario == "annuity defined partial cost params":
                    # raise value error
                    raise ValueError(f"'{annuity_cost}' is defined and some but not all of 'capex', 'opex_fix'"
                                     f"and 'lifetime' have been included. Please either only state '{annuity_cost}'"
                                     f"or include all other financial parameters to calculate the annuity.")
                elif scenario == "annuity empty all cost params defined":
                    # calculate the annuity using the calculate_annuity function
                    capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
                    # update the dataframe
                    element_df.at[index, annuity_cost] = float(capacity_cost)
                    # log info message
                    logger.info(f"the annuity ('{annuity_cost}') has been calculated and updated for"
                                f" '{row_name}' in '{element}'.")
                elif scenario == "annuity all cost params some empty":
                    # raise value error
                    raise ValueError(f"One or more of 'capex', 'opex_fix' and 'lifetime' have been left "
                                     f"empty. Please enter values or remove the parameters.")
                elif scenario == "annuity defined all cost params defined":
                    # if all parameters are defined, the user is asked if they want to calculate the annuity
                    # from the capex, opex_fix and lifetime or use the annuity directly
                    logger.info(f"All parameters ('capex', 'opex_fix', 'lifetime') and '{annuity_cost}' are "
                                f"provided for '{row_name}' in '{element}'.")
                    while True:
                        user_choice = input(
                            f"Do you want to calculate the annuity from 'capex', 'opex_fix' and 'lifetime' rather "
                            f"than use the annuity value provided in '{annuity_cost}'? (yes/no): ").lower()
                        # if the user chooses 'yes', the annuity cost parameter is replaced by the one calculated from
                        # the calculate_annuity function
                        if user_choice == 'yes':
                            capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
                            # update the dataframe
                            element_df.at[index, annuity_cost] = float(capacity_cost)
                            # log info message
                            logger.info(f"The annuity ('{annuity_cost}') has been calculated and updated for "
                                        f"'{row_name}' in '{element}'.")
                            # exit the loop
                            break
                        # if the user chooses 'no', the annuity cost parameter is used directly and the other parameters
                        # are ignored
                        if user_choice == 'no':
                            # log warning message
                            logging.warning('The annuity cost is used directly rather than calculating from other '
                                           f'parameters. This could lead to discrepancies in the results '
                                           f'- please check!')
                            # exit the loop
                            break
                        else:
                            # if the user enters something other than yes or no, they are asked to re-enter
                            # their answer
                            logger.info("Invalid choice. Please enter 'yes' or 'no'.")
                elif scenario == "no annuity partial/all cost params empty":
                    # raise value error
                    raise ValueError(f"One or more of 'capex', 'opex_fix' and 'lifetime' have been left "
                                     f"empty. Please enter values or remove the parameters and include "
                                     f"the 'capacity_cost'.")
                elif scenario == "no annuity all cost params defined":
                    # calculate the annuity using the calculate_annuity function
                    capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
                    # update the dataframe
                    element_df['capacity_cost'] = float(capacity_cost)
                    # log info message
                    logger.info(f"the annuity ('{annuity_cost}') has been calculated and updated for"
                                f" '{row_name}' in '{element}'.")
                elif scenario == "no annuity no cost params":
                    logger.info(f"'{element}' does not contain '{annuity_cost}' parameter. Skipping...")

            # save the updated dataframe to the csv file
            element_df.to_csv(element_path, sep=';', index=False)
        print("------------------------------------------------------")
    return
