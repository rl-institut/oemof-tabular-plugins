import logging
import os
import pandas as pd
import warnings
from oemof.tabular.postprocessing.core import Calculator
from oemof.tabular.postprocessing import calculations as clc, naming


def excess_generation(all_scalars):
    # assuming your DataFrame has a MultiIndex with levels ("name", "var_name")
    excess_rows = all_scalars[all_scalars.index.get_level_values("name").str.contains('excess')]
    # convert the excess_rows DataFrame to a dictionary
    excess_dict = excess_rows['var_value'].to_dict()
    # extract only the first part of the MultiIndex ('name') and use it as the key
    excess_dict = {(key[0]): value for key, value in excess_dict.items()}

    return excess_dict


def specific_system_costs(all_scalars, total_system_costs):
    """
    Calculates the specific system costs based on total system costs (this might change) and total demand in
    MWh (including demands from all sectors)
    :return:
    """
    # conditionally extract values based on the 'type' column
    demand_values = all_scalars.loc[all_scalars['type'] == 'load', 'var_value'].tolist()
    demand_values_sum = sum(demand_values)
    # extract total_system_cost value from dataframe
    total_system_cost = total_system_costs['var_value'].iloc[0]
    # calculate specific system costs (currency/kWh)
    specific_system_cost = total_system_cost / demand_values_sum / 1000

    return specific_system_cost


def calculate_renewable_share(results):
    """
    Calculates the renewable share of generation based on the renewable factor set in the inputs.
    ToDo: proper testing and appropriate warnings/logging info
    :param results: oemof model results
    :return: renewable share value
    """
    # initiate renewable_generation and nonrenewable_generation values
    renewable_generation = 0
    nonrenewable_generation = 0

    # loop through the results dict
    for entry_key, entry_value in results.items():
        # store the 'sequences' value for each oemof object tuple in results dict
        sequences = entry_value.get('sequences', None)
        # check if the oemof object tuple has the 'output_parameters' attribute
        if hasattr(entry_key[0], 'output_parameters'):
            # store the 'output_parameters' dict as output_param_dict
            output_param_dict = entry_key[0].output_parameters
            # retrieve the 'renewable_factor' value
            renewable_factor = output_param_dict.get('custom_attributes', {}).get('renewable_factor')
            # if the renewable factor is 0, add the sum of flows to nonrenewable_generation
            if renewable_factor == 0:
                nonrenewable_generation += sequences.sum().sum()
            # if the renewable factor is 1, add the sum of flows to renewable_generation
            elif renewable_factor == 1:
                renewable_generation += sequences.sum().sum()
        else:
            # if the oemof object tuple does not have the 'output_parameters' attribute, set the flows to 0
            nonrenewable_generation += 0
            renewable_generation += 0

    # calculate the total generation
    total_generation = renewable_generation + nonrenewable_generation
    # if total generation is 0, return 0 to avoid division by 0
    if total_generation == 0:
        warning_message = "Total generation is 0. This may be because there is no generation or the" \
                          " renewable factor is not defined in the output parameters of the inputs."
        warnings.warn(warning_message, UserWarning)
    return 0
    # calculate the renewable share (rounded to 2dp)
    renewable_share = round(renewable_generation / total_generation, 2)

    return renewable_share


def calculate_total_emissions(results):
    """

    :param results:
    :return:
    """
    # initiate total emissions value
    total_emissions = 0

    # loop through the results dict
    for entry_key, entry_value in results.items():
        # store the 'sequences' value for each oemof object tuple in results dict
        sequences = entry_value.get('sequences', None)
        # check if the oemof object tuple has the 'output_parameters' attribute
        if hasattr(entry_key[0], 'output_parameters'):
            # store the 'output_parameters' dict as output_param_dict
            output_param_dict = entry_key[0].output_parameters
            # retrieve the 'specific_emission' value if it exists
            specific_emission = output_param_dict.get('custom_attributes', {}).get('specific_emission')
            if specific_emission is not None:
                total_emissions += specific_emission * sequences.sum().sum()
                logging.info(f"Specific emissions recorded for {entry_key}")
    # round the total emissions to 2dp
    total_emissions = round(total_emissions, 2)

    return total_emissions


def post_processing(params, results, results_path):
    # initiate calculator for post-processing
    calculator = Calculator(params, results)

    # calculate scalars using functions from clc module
    aggregated_flows = clc.AggregatedFlows(calculator).result
    storage_losses = clc.StorageLosses(calculator).result
    transmission_losses = clc.TransmissionLosses(calculator).result
    invested_capacity = clc.InvestedCapacity(calculator).result
    invested_storage_capacity = clc.InvestedStorageCapacity(calculator).result
    invested_capacity_costs = clc.InvestedCapacityCosts(calculator).result
    invested_storage_capacity_costs = clc.InvestedStorageCapacityCosts(calculator).result
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
    all_scalars = naming.add_component_info(
        all_scalars, calculator.scalar_params
    )
    print('Total System Cost', total_system_costs)
    total_system_costs.index.names = ("name", "var_name")
    all_scalars = pd.concat([all_scalars, total_system_costs], axis=0)
    all_scalars = all_scalars.sort_values(by=["carrier", "tech", "var_name"])
    # save all scalar results to a csv file
    filepath_name_all_scalars = os.path.join(results_path, 'all_scalars.csv')
    all_scalars.to_csv(filepath_name_all_scalars)

    # saves all hourly timeseries as a dataframe (see test_postprocessing.py in oemof-tabular/tests
    # for example if wanting to filter particular nodes)
    all_sequences = clc.AggregatedFlows(calculator, resample_mode="H")
    # save all timeseries (sequence) results to a csv file
    filepath_name_all_sequences = os.path.join(results_path, 'all_sequences.csv')
    all_sequences.sequences.to_csv(filepath_name_all_sequences)

    # store the relevant KPI variables
    specific_system_cost = round(specific_system_costs(all_scalars, total_system_costs), 3)
    renewable_share = calculate_renewable_share(results)
    excess_gen = excess_generation(all_scalars)
    total_emissions = calculate_total_emissions(results)

    # create a dataframe with the KPI variables
    kpi_data = {'Variable': ['specific_system_cost', 'renewable_share', 'total_emissions'],
            'Value': [specific_system_cost, renewable_share, total_emissions]}
    # store KPI data as a dataframe
    kpi_df = pd.DataFrame(kpi_data)
    for key, value in excess_gen.items():
        kpi_df = kpi_df._append({'Variable': key, 'Value': value}, ignore_index=True)
        # replace any parameters with '-' in the name with '_' for uniformity
        kpi_df['Variable'] = kpi_df['Variable'].str.replace('-', '_')
    # save all KPI results to a csv file
    filepath_name_kpis = os.path.join(results_path, 'kpis.csv')
    # save the DataFrame to a CSV file
    kpi_df.to_csv(filepath_name_kpis, index=False)

    return