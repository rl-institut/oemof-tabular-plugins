import logging
import os
import pandas as pd
import numpy as np
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


def create_capacities_table(all_scalars, results):
    # ToDo: maybe there is a way to make this function cleaner/shorter
    # set columns of the capacities dataframe
    capacities_df = pd.DataFrame(
        columns=['Component', 'Type', 'Carrier', 'Existing Capacity', 'Capacity Potential', 'Optimizable',
                 'Optimized Capacity'])
    # create an empty set for the component names
    component_names = set()
    # iterate over the index and row of the dataframe
    for idx, row in all_scalars.iterrows():
        # store variables to be included in the dataframe
        component_name = idx[0]
        component_variable = idx[1]
        component_type = row['type']
        component_carrier = row['carrier']
        # only include component names that haven't already been included to avoid repetition,
        # and don't include system in the dataframe because this refers to the system costs (nothing with capacities)
        # and don't include the storage components because these are stored in a different table
        if component_name not in component_names and component_name != 'system' and 'storage' not in component_name:
            component_names.add(component_name)
            # add component name and corresponding type and carrier to the dataframe
            capacities_df = capacities_df._append({'Component': component_name,
                                                   'Type': component_type,
                                                   'Carrier': component_carrier}, ignore_index=True)

        # check if 'invest_out' is in the component_variable
        if 'invest_out' in component_variable:
            # if it is, get the corresponding 'var_value' for the optimized capacity value
            component_opt_capacity = row['var_value']
            # if the value is -0.0, adapt this to 0.0
            if component_opt_capacity == -0.0:
                component_opt_capacity = 0.0
            # add or update 'Optimized Capacity' for the component_name with the optimized capacity value
            capacities_df.loc[
                capacities_df['Component'] == component_name, 'Optimized Capacity'] = component_opt_capacity

    # loop through the results dict
    for entry_key, entry_value in results.items():
        # check if the oemof object tuple has the 'capacity' attribute
        if hasattr(entry_key[0], 'capacity'):
            # store the existing capacity as a variable
            existing_capacity = entry_key[0].capacity
            # convert entry_key[0] to string
            component_name_str = str(entry_key[0])
            # check if component_name_str is in capacities_df['Component']
            if any(component_name_str in val for val in capacities_df['Component'].values):
                # update the existing capacity value in capacities_df
                capacities_df.loc[
                    capacities_df['Component'] == component_name_str, 'Existing Capacity'] = existing_capacity
        # check if the oemof object tuple has the 'expandable' attribute
        if hasattr(entry_key[0], 'expandable'):
            # store the expandable boolean as a variable
            expandable = entry_key[0].expandable
            # convert entry_key[0] to string
            expandable_name_str = str(entry_key[0])
            # Check if expandable_name_str is in capacities_df['Component']
            if any(expandable_name_str in val for val in capacities_df['Component'].values):
                # Update the existing expandable value in capacities_df
                capacities_df.loc[
                    capacities_df['Component'] == expandable_name_str, 'Optimizable'] = expandable
        # check if the oemof object tuple has the 'capacity_potential' attribute
        if hasattr(entry_key[0], 'capacity_potential'):
            # store the capacity potential as a variable
            capacity_potential = entry_key[0].capacity_potential
            # convert entry_key[0] to string
            cp_name_str = str(entry_key[0])
            # Check if cp_name_str is in capacities_df['Component']
            if any(cp_name_str in val for val in capacities_df['Component'].values):
                # Update the existing expandable value in capacities_df
                capacities_df.loc[
                    capacities_df['Component'] == cp_name_str, 'Capacity Potential'] = capacity_potential
    return capacities_df


def create_storage_capacities_table(all_scalars, results):
    # ToDo: this function requires the naming of storage components to have 'storage' in them, there is
    #  probably a cleaner way of doing it
    # ToDo: this is a bit of a repetition of the above function, maybe there is a better way to do this?
    # ToDo: note that for storages, storage capacity is the capacity in e.g. MWh and capacity is the
    #  max input/output in e.g. MW
    # set columns of the capacities dataframe
    storage_capacities_df = pd.DataFrame(columns=['Component', 'Type', 'Carrier', 'Existing Storage Capacity',
                                                  'Existing Max Input/Output', 'Storage Capacity Potential',
                                                  'Max Input/Output Potential', 'Optimizable',
                                                  'Optimized Storage Capacity', 'Optimized Max Input/Output'])
    # create an empty set for the component names
    component_names = set()
    # iterate over the index and row of the dataframe
    for idx, row in all_scalars.iterrows():
        # store variables to be included in the dataframe
        component_name = idx[0]
        component_variable = idx[1]
        component_type = row['type']
        component_carrier = row['carrier']
        # only include component names that haven't already been included to avoid repetition,
        # and only include component names that have 'storage' in
        if component_name not in component_names and 'storage' in component_name:
            component_names.add(component_name)
            # add component name and corresponding type and carrier to the dataframe
            storage_capacities_df = storage_capacities_df._append({'Component': component_name,
                                                   'Type': component_type,
                                                   'Carrier': component_carrier}, ignore_index=True)
        # check if 'invest' is equal to the component_variable
        if 'invest' == component_variable:
            # if it is, get the corresponding 'var_value' for the optimized capacity value
            component_opt_capacity = row['var_value']
            # if the value is -0.0, adapt this to 0.0
            if component_opt_capacity == -0.0:
                component_opt_capacity = 0.0
            # add or update 'Optimized Storage Capacity' for the component_name with the optimized capacity value
            storage_capacities_df.loc[
                storage_capacities_df['Component'] == component_name,
                'Optimized Storage Capacity'] = component_opt_capacity
        # check if 'invest_out' is in the component_variable
        if 'invest_out' in component_variable:
            # if it is, get the corresponding 'var_value' for the optimized capacity value
            component_opt_capacity = row['var_value']
            # if the value is -0.0, adapt this to 0.0
            if component_opt_capacity == -0.0:
                component_opt_capacity = 0.0
            # add or update 'Optimized Max Input/Output' for the component_name with the optimized capacity value
            storage_capacities_df.loc[
                storage_capacities_df['Component'] == component_name,
                'Optimized Max Input/Output'] = component_opt_capacity

    # loop through the results dict
    for entry_key, entry_value in results.items():
        # check if the oemof object tuple has the 'storage_capacity' attribute
        if hasattr(entry_key[0], 'storage_capacity'):
            # store the existing capacity as a variable
            existing_storage_capacity = entry_key[0].storage_capacity
            # convert entry_key[0] to string
            component_name_str = str(entry_key[0])
            # check if component_name_str is in storage_capacities_df['Component']
            if any(component_name_str in val for val in storage_capacities_df['Component'].values):
                # update the existing capacity value in capacities_df
                storage_capacities_df.loc[
                    storage_capacities_df[
                        'Component'] == component_name_str, 'Existing Storage Capacity'] = existing_storage_capacity
        # check if the oemof object tuple has the 'capacity' attribute
        if hasattr(entry_key[0], 'capacity'):
            # store the existing capacity as a variable
            existing_capacity = entry_key[0].capacity
            # convert entry_key[0] to string
            component_name_str = str(entry_key[0])
            # check if component_name_str is in storage_capacities_df['Component']
            if any(component_name_str in val for val in storage_capacities_df['Component'].values):
                # update the existing capacity value in capacities_df
                storage_capacities_df.loc[
                    storage_capacities_df[
                        'Component'] == component_name_str, 'Existing Max Input/Output'] = existing_capacity
        # check if the oemof object tuple has the 'expandable' attribute
        if hasattr(entry_key[0], 'expandable'):
            # store the expandable boolean as a variable
            expandable = entry_key[0].expandable
            # convert entry_key[0] to string
            expandable_name_str = str(entry_key[0])
            # Check if expandable_name_str is in storage_capacities_df['Component']
            if any(expandable_name_str in val for val in storage_capacities_df['Component'].values):
                # Update the existing expandable value in storage_capacities_df
                storage_capacities_df.loc[
                    storage_capacities_df['Component'] == expandable_name_str, 'Optimizable'] = expandable
        # check if the oemof object tuple has the 'storage_capacity_potential' attribute
        if hasattr(entry_key[0], 'storage_capacity_potential'):
            # store the storage capacity potential as a variable
            storage_capacity_potential = entry_key[0].storage_capacity_potential
            # convert entry_key[0] to string
            cp_name_str = str(entry_key[0])
            # Check if cp_name_str is in storage_capacities_df['Component']
            if any(cp_name_str in val for val in storage_capacities_df['Component'].values):
                # Update the existing expandable value in storage_capacities_df
                storage_capacities_df.loc[
                    storage_capacities_df['Component'] == cp_name_str, 'Storage Capacity Potential'] = storage_capacity_potential
        # check if the oemof object tuple has the 'capacity_potential' attribute
        if hasattr(entry_key[0], 'capacity_potential'):
            # store the capacity potential as a variable
            capacity_potential = entry_key[0].capacity_potential
            # convert entry_key[0] to string
            cp_name_str = str(entry_key[0])
            # Check if cp_name_str is in storage_capacities_df['Component']
            if any(cp_name_str in val for val in storage_capacities_df['Component'].values):
                # Update the existing expandable value in storage_capacities_df
                storage_capacities_df.loc[
                    storage_capacities_df['Component'] == cp_name_str, 'Max Input/Output Potential'] = capacity_potential

    return storage_capacities_df


def create_aggregated_flows_table(aggregated_flows):
    # Create an empty DataFrame to store the flows
    flows_df = pd.DataFrame(columns=['From', 'To', 'Aggregated Flow'])

    # Iterate over the items of the Series
    for idx, value in aggregated_flows.items():
        # Extract the source, target, and var_name from the index
        from_, to, _ = idx

        # Append a row to the DataFrame
        flows_df = flows_df._append({'From': from_, 'To': to, 'Aggregated Flow': float(value)}, ignore_index=True)

    return flows_df


def create_costs_table(all_scalars, results):
    # create an empty dataframe
    costs_df = pd.DataFrame(columns=['Component', 'Upfront Investment Cost',
                                                  'Annuity (CAPEX + Fixed O&M)', 'Variable Costs (In)',
                                                  'Variable Costs (Out)'])
    # create an empty set for the component names
    component_names = set()
    # iterate over the index and row of the dataframe
    for idx, row in all_scalars.iterrows():
        # store variables to be included in the dataframe
        component_name = idx[0]
        component_variable = idx[1]
        # only include component names that haven't already been included to avoid repetition,
        # and don't include system in the dataframe because this refers to the total system costs (include elsewhere)
        # and don't include the storage components because these are stored in a different table
        if component_name not in component_names and component_name != 'system' and 'storage' not in component_name:
            component_names.add(component_name)
            # add component name and corresponding type and carrier to the dataframe
            costs_df = costs_df._append({'Component': component_name}, ignore_index=True)

        # check if 'invest_costs_out' is in the component_variable
        if 'invest_costs_out' in component_variable:
            # if it is, get the corresponding 'var_value' for the investment cost value
            invest_costs_out = row['var_value']
            # if the value is -0.0, adapt this to 0.0
            if invest_costs_out == -0.0:
                invest_costs_out = 0.0
            # add or update 'Annuity (CAPEX + Fixed O&M)' for the component_name with the optimized capacity value
            costs_df.loc[
                costs_df['Component'] == component_name, 'Annuity (CAPEX + Fixed O&M)'] = invest_costs_out
        # check if 'variable_costs_in' is in the component_variable
        if 'variable_costs_in' in component_variable:
            # if it is, get the corresponding 'var_value' for the variable cost in value
            variable_costs_in = row['var_value']
            # if the value is -0.0, adapt this to 0.0
            if variable_costs_in == -0.0:
                variable_costs_in = 0.0
            # add or update 'Variable Costs (In)' for the component_name with the optimized capacity value
            costs_df.loc[
                costs_df['Component'] == component_name, 'Variable Costs (In)'] = variable_costs_in
        # check if 'variable_costs_out' is in the component_variable
        if 'variable_costs_out' in component_variable:
            # if it is, get the corresponding 'var_value' for the variable cost out value
            variable_costs_out = row['var_value']
            # if the value is -0.0, adapt this to 0.0
            if variable_costs_out == -0.0:
                variable_costs_out = 0.0
            # add or update 'Variable Costs (Out)' for the component_name with the optimized capacity value
            costs_df.loc[
                costs_df['Component'] == component_name, 'Variable Costs (Out)'] = variable_costs_out

    return costs_df


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

    capacities_df = create_capacities_table(all_scalars, results)
    storage_capacities_df = create_storage_capacities_table(all_scalars, results)
    flows_df = create_aggregated_flows_table(aggregated_flows)
    costs_df = create_costs_table(all_scalars, results)

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
    # save all capacities to a csv file
    filepath_name_capacities = os.path.join(results_path, 'capacities.csv')
    # save the DataFrame to a CSV file
    capacities_df.to_csv(filepath_name_capacities, index=False)
    # save all storage capacities to a csv file
    filepath_name_stor_capacities = os.path.join(results_path, 'storage_capacities.csv')
    # save the DataFrame to a CSV file
    storage_capacities_df.to_csv(filepath_name_stor_capacities, index=False)
    # save all flows to a csv file
    filepath_name_flows = os.path.join(results_path, 'flows.csv')
    # save the DataFrame to a CSV file
    flows_df.to_csv(filepath_name_flows, index=False)
    # save all costs to a csv file
    filepath_name_costs = os.path.join(results_path, 'costs.csv')
    # save the DataFrame to a CSV file
    costs_df.to_csv(filepath_name_costs, index=False)

    return all_scalars
