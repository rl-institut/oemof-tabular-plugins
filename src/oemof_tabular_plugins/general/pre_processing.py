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
            # scenario group a: annuity parameter is included and all of CAPEX, OPEX fix and lifetime parameters
            # are not included in the csv file
            if annuity_cost in element_df.columns and cost_columns == missing_columns:
                scenario_group = "a"
            # scenario group b: annuity parameter is included and some but not all of CAPEX, OPEX fix and lifetime
            # parameters are included in the csv file
            elif annuity_cost in element_df.columns and missing_columns and cost_columns != missing_columns:
                scenario_group = "b"
            # scenario group c: annuity parameter is included and all of CAPEX, OPEX fix and lifetime parameters
            # are included in the csv file
            elif annuity_cost in element_df.columns and not missing_columns:
                scenario_group = "c"
            # scenario group d: annuity parameter is not included and all of CAPEX, OPEX fix and lifetime parameters
            # are included in the csv file
            elif annuity_cost not in element_df.columns:
                scenario_group = "d"
            # scenario group e: the annuity parameter is not included and neither are CAPEX, OPEX fix and lifetime
            # parameters in the csv file
            elif annuity_cost not in element_df.columns and cost_columns == missing_columns:
                scenario_group = "e"

            # print statement for testing, remove later
            print("scenario_group: ", scenario_group)

            # ---------------- POSSIBLE SCENARIOS FOR EACH SCENARIO GROUP ----------------
            # loop through each entry in the csv file
            for index, row in element_df.iterrows():
                # define the row name
                row_name = row['name']
                if scenario_group == "a":
                    # scenario a1: the annuity parameter is left empty and the other cost parameters have not been
                    # included
                    if pd.isna(row[annuity_cost]):
                        scenario = "a1"
                    # scenario a2: the annuity parameter is defined and the other cost parameters have not been
                    # included
                    else:
                        scenario = "a2"
                elif scenario_group == "b":
                    # scenario b1: the annuity parameter is left empty and only some of the other financial
                    # parameters are included
                    if pd.isna(row[annuity_cost]):
                        scenario = "b1"
                    # scenario b2: the annuity parameter is defined and only some of the other financial
                    # parameters are included
                    else:
                        scenario = "b2"
                elif scenario_group == "c":
                    # store the parameters
                    capex = row['capex']
                    opex_fix = row['opex_fix']
                    lifetime = row['lifetime']
                    # scenario c1: the annuity parameter is left empty and all of the other financial parameters
                    # are defined
                    if pd.isna(row[annuity_cost]) and pd.notna(capex) and pd.notna(opex_fix) and \
                            pd.notna(lifetime):
                        scenario = "c1"
                    # scenario c2: the annuity parameter is either defined or empty, but at least one of
                    # 'capex', 'opex_fix' and 'lifetime' is left empty
                    elif pd.isna(capex) or pd.isna(opex_fix) or pd.isna(lifetime):
                        scenario = "c2"
                    # scenario c3: both the annuity parameter is defined and all of the other financial parameters
                    # are defined
                    else:
                        scenario = "c3"

                # ---------------- ACTIONS TAKEN FOR EACH SCENARIO ----------------
                if scenario == "a1":
                    # raise value error
                    raise ValueError(f"'{annuity_cost}' (the annuity) has been left empty for '{row_name}' "
                                         f"in '{element}', and 'capex', 'opex_fix' and 'lifetime' have not "
                                         f" been included. \nEither the annuity ('{annuity_cost}') must be "
                                         f"directly stated or all of the other financial parameters must be stated "
                                         f"to calculate the annuity.")
                elif scenario == "a2":
                    # log info message
                    logger.info(f"The annuity cost is directly used for '{row_name}' in '{element}'.")
                elif scenario == "b1":
                    # raise value error
                    raise ValueError(f"'{annuity_cost}' (the annuity) has been left empty for '{row_name}' "
                                         f"in '{element}', and not all of 'capex', 'opex_fix' and 'lifetime' have"
                                         f" been included. \nEither the annuity ('{annuity_cost}') must be "
                                         f"directly stated or all of the other financial parameters must be stated "
                                         f"to calculate the annuity.")
                elif scenario == "b2":
                    # raise value error
                    raise ValueError(f"'{annuity_cost}' is defined and some but not all of 'capex', 'opex_fix'"
                                     f"and lifetime have been included. Please either only state '{annuity_cost}'"
                                     f"or include all other financial parameters to calculate the annuity.")
                elif scenario == "c1":
                    # calculate the annuity using the calculate_annuity function
                    capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
                    # update the dataframe
                    element_df.at[index, annuity_cost] = float(capacity_cost)
                    # log info message
                    logger.info(f"the annuity ('{annuity_cost}') has been calculated and updated for"
                                f" '{row_name}' in '{element}'.")
                elif scenario == "c2":
                    # raise value error
                    raise ValueError(f"One or more of 'capex', 'opex_fix' and 'lifetime' have been left "
                                     f"empty. Please enter values or remove the parameters.")
                elif scenario == "c3":
                    # if all parameters are defined, the user is asked if they want to calculate the annuity
                    # from the capex, opex_fix and lifetime or use the annuity directly
                    logger.info(f"All parameters ('capex', 'opex_fix', 'lifetime') and '{annuity_cost}' are "
                                f"provided for '{row_name}' in '{element}'.")
                    user_choice = input(
                        f"Do you want to use the annuity value provided in '{annuity_cost}' rather than "
                        "calculating the annuity from 'capex', "
                        "'opex_fix' and 'lifetime'? (yes/no): ").lower()
                    if user_choice == 'yes':
                        capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
                        # update the DataFrame
                        element_df.at[index, annuity_cost] = float(capacity_cost)
                        print(
                            f"INFO: The annuity ('{annuity_cost}') has been calculated and updated for "
                            f"'{row_name}' in '{element}'.")

                    # if the user chooses 'no', the annuity cost parameter is used directly and the other parameters
                    # are ignored
                    if user_choice == 'no':
                        pass
                        logger.warning('The annuity cost is used directly rather than calculating from other '
                              f'parameters. This could lead to discrepancies in the results - please check!')
                    else:
                        print("Invalid choice. Please enter 'yes' or 'no'.")

                if param_scenario == "b":
                    raise ValueError(f"'{annuity_cost}' (the annuity) has been left empty for '{row_name}' "
                                         f"in '{element}', and not all of 'capex', 'opex_fix' and 'lifetime' have"
                                         f" been included. \nEither the annuity ('{annuity_cost}') must be "
                                         f"directly stated or all of the other financial parameters must be stated "
                                         f"to calculate the annuity.")
                if param_scenario == "c":
                    # store the parameters
                    capex = row['capex']
                    opex_fix = row['opex_fix']
                    lifetime = row['lifetime']
                    # if the annuity cost value is empty and 'capex', 'opex_fix' and 'lifetime' are stated, the
                    # annuity is calculated using the calculate_annuity function
                    if pd.isna(row[annuity_cost]) and pd.notna(capex) and pd.notna(opex_fix) and pd.notna(lifetime):
                        capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
                        # update the DataFrame
                        element_df.at[index, annuity_cost] = float(capacity_cost)
                        logger.info(f"the annuity ('{annuity_cost}') has been calculated and updated for"
                              f" '{row_name}' in '{element}'.")
                    # if any of 'capex', 'opex_fix' and 'lifetime' parameters are not stated, raise an error
                    elif pd.isna(capex) or pd.isna(opex_fix) or pd.isna(lifetime):
                        raise ValueError(f"One or more of 'capex', 'opex_fix' and 'lifetime' have been left "
                                         f"empty. Please enter values or remove the parameters.")
                    # if all parameters are defined, the user is asked if they want to calculate the annuity
                    # from the capex, opex_fix and lifetime or use the annuity directly
                    else:
                        logger.info(f"All parameters ('capex', 'opex_fix', 'lifetime') and '{annuity_cost}' are "
                                    f"provided for '{row_name}' in '{element}'.")
                        user_choice = input(
                        f"Do you want to calculate the annuity from 'capex', 'opex_fix' and 'lifetime' rather "
                        f"than use the annuity value provided in '{annuity_cost}'? (yes/no): ").lower()


                # if all inputs have been provided, the user is asked whether to take the annuity_cost value directly
                # or calculate it from the other parameters
                elif not pd.isna(row[annuity_cost]) and not pd.isna(row['capex']) and not pd.isna(
                        row['opex_fix']) and not pd.isna(row['lifetime']):
                    print(
                        f"INFO: All parameters ('capex', 'opex_fix', 'lifetime') and '{annuity_cost}' are provided for '{row_name}' in '{element}'.")
                    user_choice = input(
                        f"Do you want to use the annuity value provided in '{annuity_cost}' rather than "
                        "calculating the annuity from 'capex', "
                        "'opex_fix' and 'lifetime'? (yes/no): ").lower()
                    # if the user chooses 'yes', the annuity cost parameter is used directly and the other parameters
                    # are ignored
                    if user_choice == 'yes':
                        pass
                        print(f'WARNING: The annuity cost is used directly rather than calculating from other '
                              f'parameters. This could lead to discrepancies in the results - please check!')
                    # if the user chooses 'no', the annuity cost parameter is replaced by the one calculated from
                    # the calculate_annuity function
                    elif user_choice == 'no':
                        capex = row['capex']
                        opex_fix = row['opex_fix']
                        lifetime = row['lifetime']
                        capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
                        # update the DataFrame
                        element_df.at[index, annuity_cost] = float(capacity_cost)
                        print(
                            f"INFO: The annuity ('{annuity_cost}') has been calculated and updated for "
                            f"'{row_name}' in '{element}'.")
                    else:
                        print("Invalid choice. Please enter 'yes' or 'no'.")



            # if the annuity cost parameter is not included in the CSV file, this file will be ignored
            # NOTE: this is to skip files such as no_annuity_cost.csv, load.csv etc. that do not need to calculate an annuity
            # NOTE: this means you must include this parameter in your CSV file if you want it to be considered!
            if annuity_cost not in element_df.columns:
                print(f"INFO: '{element}' does not contain '{annuity_cost}' parameter. Skipping...")
                continue
                # if any of 'capex', 'opex_fix' or 'lifetime' are missing but the annuity cost value is defined, any
                # of 'capex', 'opex_fix' or 'lifetime' that is included will be removed from the CSV file
                if not pd.isna(element_df.at[0, annuity_cost]):
                    print(f"WARNING: '{element}' is missing columns {missing_columns} but '{annuity_cost}' is defined. "
                          f"This means the parameters {cost_columns} will be ignored (and deleted if applicable), "
                          f"also for post-processing.")
                    # drop any remaining columns in cost_columns
                    element_df = element_df.drop(columns=cost_columns, errors='ignore')
                else:
                    # raise error if annuity cost parameter value is not stated, and not all of 'capex', 'opex_fix'
                    # and lifetime have been stated
                    raise ValueError(f"'{element}' is missing required columns {missing_columns}."
                                     f" '{annuity_cost}' value is also empty. Please check the inputs ")
                # loop through each entry in the CSV file
            for index, row in element_df.iterrows():
                # define the row name
                row_name = row['name']
                # if the annuity cost value is empty, check to see if the other parameters have been stated
                if pd.isna(row[annuity_cost]):
                    print(f"INFO: '{annuity_cost}' has been left empty for '{row_name}' in '{element}': checking to "
                          f"see if 'capex', 'opex_fix' and 'lifetime' have been included.")
                    # raise error if the annuity cost value is empty and one or more of the other parameters is
                    # also left empty
                    if pd.isna(row['capex']) or pd.isna(row['opex_fix']) or pd.isna(row['lifetime']):
                        raise ValueError(f"'{annuity_cost}' has been left empty for '{row_name}' in '{element}'"
                                         f" and one or more of 'capex',opex_fix' and 'lifetime' are also empty."
                                         f"\n Either the annuity ('{annuity_cost}') must be directly stated or the"
                                         f" other financial parameters ('capex', 'opex_fix', 'lifetime') must be "
                                         f"stated to calculate the annuity.")
                    else:
                        # if the annuity cost value is empty and 'capex', 'opex_fix' and 'lifetime' are stated, the
                        # annuity is calculated using the calculate_annuity function
                        capex = row['capex']
                        opex_fix = row['opex_fix']
                        lifetime = row['lifetime']
                        capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
                        # update the DataFrame
                        element_df.at[index, annuity_cost] = float(capacity_cost)
                        print(f"INFO: the annuity ('{annuity_cost}') has been calculated and updated for"
                              f" '{row_name}' in '{element}'.")
                elif not pd.isna(row[annuity_cost]) and not all(
                        column in element_df.columns for column in cost_columns):
                    continue
                # if all inputs have been provided, the user is asked whether to take the annuity_cost value directly
                # or calculate it from the other parameters
                elif not pd.isna(row[annuity_cost]) and not pd.isna(row['capex']) and not pd.isna(
                        row['opex_fix']) and not pd.isna(row['lifetime']):
                    print(
                        f"INFO: All parameters ('capex', 'opex_fix', 'lifetime') and '{annuity_cost}' are provided for '{row_name}' in '{element}'.")
                    user_choice = input(
                        f"Do you want to use the annuity value provided in '{annuity_cost}' rather than "
                        "calculating the annuity from 'capex', "
                        "'opex_fix' and 'lifetime'? (yes/no): ").lower()
                    # if the user chooses 'yes', the annuity cost parameter is used directly and the other parameters
                    # are ignored
                    if user_choice == 'yes':
                        pass
                        print(f'WARNING: The annuity cost is used directly rather than calculating from other '
                              f'parameters. This could lead to discrepancies in the results - please check!')
                    # if the user chooses 'no', the annuity cost parameter is replaced by the one calculated from
                    # the calculate_annuity function
                    elif user_choice == 'no':
                        capex = row['capex']
                        opex_fix = row['opex_fix']
                        lifetime = row['lifetime']
                        capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
                        # update the DataFrame
                        element_df.at[index, annuity_cost] = float(capacity_cost)
                        print(
                            f"INFO: The annuity ('{annuity_cost}') has been calculated and updated for "
                            f"'{row_name}' in '{element}'.")
                    else:
                        print("Invalid choice. Please enter 'yes' or 'no'.")

                # save the updated DataFrame to the CSV file
            element_df.to_csv(element_path, sep=';', index=False)
        print("------------------------------------------------------")
    return
