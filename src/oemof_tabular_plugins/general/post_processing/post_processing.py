import logging
import os
import warnings

import pandas as pd
from oemof.tabular.postprocessing import calculations as clc, naming
from oemof.tabular.postprocessing.core import Calculator

from oemof_tabular_plugins.datapackage.post_processing import (
    construct_dataframe_from_results,
    process_raw_results,
    process_raw_inputs,
    apply_calculations,
    apply_kpi_calculations,
)

# ------ New post-processing to create tables ------
# This dictionary contains groups of columns that should be extracted from the df_results to generate a clearer overview


# TODO add a column for planned capacity (not optimizable but including costs) in capacities if it gets properly
#  implemented (planned capacity can be set by setting capacity_minimum == capacity_potential and dispatchable = True
RESULT_TABLE_COLUMNS = {
    "costs": ["upfront_investment_costs", "total_annuity", "total_variable_costs"],
    "capacities": [
        "capacity",
        "storage_capacity",
        "capacity_minimum",
        "capacity_potential",
        "storage_capacity_potential",
        "expandable",
        "investments",
        "total_capacity",
    ],
}


def extract_table_from_results(df_results, columns):
    """Extracts a set of columns from the df_results DataFrame. The lists of columns to generate these tables can be
    defined in RESULT_TABLE_COLUMNS
    :param df_results: multiindex results dataframe with additional columns (OTPCalculator.df_results)
    :param columns: list of columns to generate the sub-table (should be defined in RESULT_TABLE_COLUMNS)
    :return: dataframe containing the columns specified in columns, if present in df_results
    """
    missing_columns = []
    for col in columns:
        try:
            df_results[col]
        except KeyError:
            # If the key is not in the df_results dataframe, log it as a warning and pop the column from the list
            logging.warning(
                f"The column {col} was not found in the results DataFrame, will be skipped in the subtable"
            )
            missing_columns.append(col)

    columns = [col for col in columns if col not in missing_columns]
    results_table = df_results[columns].copy()
    # TODO some of these names may be confusing because they are just the columns, maybe there should be a
    #  verbose parameter in CALCULATED OUTPUTS that we can then also use here
    results_table.columns = [
        col.title().replace("_", " ") for col in results_table.columns
    ]
    return results_table


def save_table_to_csv(table, results_path, filename):
    """Saves a DataFrame to a .csv file"""
    filepath = os.path.join(results_path, filename)
    table.to_csv(filepath)


# TODO figure out the best table/display for storage results


# --------------------------------------------------
class OTPCalculator(Calculator):
    def __init__(self, input_parameters, energy_system, dp_path):

        self.df_results = construct_dataframe_from_results(energy_system)
        self.df_results = process_raw_results(self.df_results)
        self.df_results = process_raw_inputs(self.df_results, dp_path)
        apply_calculations(self.df_results)
        self.kpis = apply_kpi_calculations(self.df_results)

        super().__init__(input_parameters, energy_system.results)


def post_processing(params, es, results_path, dp_path):
    # ToDo: adapt this function after multi-index dataframe is implemented to make it more concise / cleaner
    # ToDo: params can be accessed in results so will not need to be a separate argument
    """
    The main post-processing function extracts various scalar and timeseries data and stores it in CSV files.
    :param params: energy system parameters
    :param es: oemof energy_system with results in it, ie es.results = processing.results(m) has been performed
    :param results_path: results directory path
    :param dp_path: path to the datapackage.json file
    """
    # initiate calculator for post-processing
    calculator = OTPCalculator(params, es, dp_path)
    # print(calculator.df_results)
    results = es.results
    results_by_flow = calculator.df_results
    results_by_flow.to_csv(results_path + "/all_results_by_flow.csv", index=True)
    kpis = calculator.kpis
    kpis.to_csv(results_path + "/kpis.csv", index=True)

    # get sub-tables from results dataframe
    cost_table = extract_table_from_results(
        calculator.df_results, RESULT_TABLE_COLUMNS["costs"]
    )
    capacities_table = extract_table_from_results(
        calculator.df_results, RESULT_TABLE_COLUMNS["capacities"]
    )

    # save tables to csv files
    tables_to_save = {"costs.csv": cost_table, "capacities.csv": capacities_table}
    for filename, table in tables_to_save.items():
        save_table_to_csv(table, results_path, filename)

    # ----- OLD POST-PROCESSING - TO BE DELETED ONCE CERTAIN -----

    # calculate scalars using functions from clc module
    aggregated_flows = clc.AggregatedFlows(calculator).result
    storage_losses = clc.StorageLosses(calculator).result
    transmission_losses = clc.TransmissionLosses(calculator).result
    invested_capacity = clc.InvestedCapacity(calculator).result
    invested_storage_capacity = clc.InvestedStorageCapacity(calculator).result
    invested_capacity_costs = clc.InvestedCapacityCosts(calculator).result
    invested_storage_capacity_costs = clc.InvestedStorageCapacityCosts(
        calculator
    ).result
    summed_carrier_costs = clc.SummedCarrierCosts(calculator).result
    summed_marginal_costs = clc.SummedMarginalCosts(calculator).result
    total_system_costs = clc.TotalSystemCosts(calculator).result

    # combine all results into a single dataframe
    all_scalars = [
        aggregated_flows,
        storage_losses,
        transmission_losses,
        invested_capacity,
        invested_storage_capacity,
        invested_capacity_costs,
        invested_storage_capacity_costs,
        summed_carrier_costs,
        summed_marginal_costs,
    ]
    # map variable names and add component information
    all_scalars = pd.concat(all_scalars, axis=0)
    all_scalars = naming.map_var_names(
        all_scalars,
        calculator.scalar_params,
        calculator.busses,
        calculator.links,
    )
    all_scalars = naming.add_component_info(all_scalars, calculator.scalar_params)
    print("Total System Cost", total_system_costs)
    total_system_costs.index.names = ("name", "var_name")
    all_scalars = pd.concat([all_scalars, total_system_costs], axis=0)
    all_scalars = all_scalars.sort_values(by=["carrier", "tech", "var_name"])
    # save all scalar results to a csv file
    filepath_name_all_scalars = os.path.join(results_path, "all_scalars.csv")
    all_scalars.to_csv(filepath_name_all_scalars)

    # saves all hourly timeseries as a dataframe (see test_postprocessing.py in oemof-tabular/tests
    # for example if wanting to filter particular nodes)
    all_sequences = clc.AggregatedFlows(calculator, resample_mode="H")
    # save all timeseries (sequence) results to a csv file
    filepath_name_all_sequences = os.path.join(results_path, "all_sequences.csv")
    all_sequences.sequences.to_csv(filepath_name_all_sequences)

    return
