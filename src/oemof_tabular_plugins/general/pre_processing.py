import os
import pandas as pd


def calculate_annuity(capex, opex_fix, lifetime, wacc):
    """
    Calculates the total annuity for each component, including CAPEX and fixed OPEX.
    :param capex: CAPEX (currency/MW*) *or the unit you choose to use throughout the model e.g. kW/GW
    :param opex_fix: the fixed OPEX (currency/MW*/year)
    :param lifetime: the lifetime of the component (years)
    :param wacc: the weighted average cost of capital (WACC) applied throughout the model (%)
    :return: the total annuity (currency/MW*/year)
    """
    annuity_capex = (capex) * (wacc * (1 + wacc) ** lifetime) / (((1 + wacc) ** lifetime) - 1)
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
            # if the annuity cost parameter is not included in the CSV file, this file will be ignored
            # NOTE: this is to skip files such as bus.csv, load.csv etc. that do not need to calculate an annuity
            # NOTE: this means you must include this parameter in your CSV file if you want it to be considered!
            if annuity_cost not in element_df.columns:
                print(f"INFO: '{element}' does not contain '{annuity_cost}' parameter. Skipping...")
                continue

            # check if any of the required columns are missing
            cost_columns = {'capex', 'opex_fix', 'lifetime'}
            missing_columns = cost_columns - set(element_df.columns)
            if missing_columns:
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
                elif not pd.isna(row[annuity_cost]) and not pd.isna(row['capex']) and not pd.isna(row['opex_fix']) and not pd.isna(row['lifetime']):
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
