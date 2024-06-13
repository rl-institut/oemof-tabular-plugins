import logging

import pandas as pd
from datapackage import Package
import oemof.solph as solph
import numpy as np

# ToDo: check to see if the storage optimized input/output (invest_out) and
#  optimized capacity (invest) are saved correctly
# ToDo: is another raw output from the results is investment costs? or does this have to be calculated?
RAW_OUTPUTS = ["investments"]
PROCESSED_RAW_OUTPUTS = ["flow_min", "flow_max", "aggregated_flow"]
RAW_INPUTS = [
    "marginal_cost",
    "carrier_cost",
    "capacity_cost",
    "storage_capacity_cost",
    "capacity",
    "capacity_potential",
    "expandable",
    "storage_capacity",
    "storage_capacity_potential",
    "min_capacity",
    "max_capacity",
    "efficiency",
    "capex",
    "opex_fix",
    "lifetime",
    "renewable_factor",
    "emission_factor",
    "land_requirement_factor",
    "water_footprint_factor",
]


def compute_total_capacity(results_df):
    # ToDo: check for storage where there is both capacity and storage capacity
    """Calculates total capacity by adding existing capacity (capacity) to optimized capacity (investments)"""
    return results_df.capacity + results_df.investments


def compute_total_annuity(results_df):
    """Calculates total capacity by adding existing capacity (capacity) to optimized capacity (investments)"""
    # TODO fix this to use storage_capacity_cost for the storage (or fix on the storage side)
    return results_df.capacity_cost + results_df.investments


def compute_upfront_investment_costs(results_df):
    # ToDo: check for storage if investments is based on correct parameter
    """Calculates investment costs by multiplying capex with optimized capacity (investments)"""
    if "capex" not in results_df.index:
        return None
    else:
        return results_df.capex * results_df.investments


def compute_opex_fix_costs(results_df):
    """Calculates yearly opex costs by multiplying opex with optimized capacity (investments)"""
    if "opex_fix" not in results_df.index:
        return None
    else:
        return results_df.opex_fix * results_df.investments


def compute_variable_costs(results_df):
    """Calculates variable costs by multiplying the marginal cost by the aggregated flow if the direction is out,
     and by the carrier cost if the direction is in. The total marginal costs for each asset correspond to the sum
     of the marginal costs for the in- and output flows"""
    if results_df.name[1] == "out":
        if "marginal_cost" not in results_df.index:
            return None
        return results_df.marginal_cost * results_df.aggregated_flow
    elif results_df.name[1] == "in":
        if "carrier_cost" not in results_df.index:
            return None
        return results_df.carrier_cost * results_df.aggregated_flow


def compute_renewable_generation(results_df):
    """Calculates renewable generation by multiplying aggregated flow by renewable factor"""
    if "renewable_factor" not in results_df.index:
        return None
    else:
        return results_df.aggregated_flow * results_df.renewable_factor


def compute_co2_emissions(results_df):
    """Calculates CO2 emissions by multiplying aggregated flow by emission factor"""
    if "emission_factor" not in results_df.index:
        return None
    else:
        return results_df.aggregated_flow * results_df.emission_factor


def compute_additional_land_requirement(results_df):
    """Calculates land requirement needed for optimized capacities"""
    if "land_requirement_factor" not in results_df.index:
        return None
    else:
        return results_df.investments * results_df.land_requirement_factor


def compute_total_land_requirement(results_df):
    """Calculates land requirement needed for total capacities"""
    if "land_requirement_factor" not in results_df.index:
        return None
    else:
        return results_df.total_capacity * results_df.land_requirement_factor


def compute_water_footprint(results_df):
    """Calculates water footprint by multiplying aggregated flow by water footprint factor"""
    if "water_footprint_factor" not in results_df.index:
        return None
    else:
        return results_df.aggregated_flow * results_df.water_footprint_factor


def _check_arguments(df, column_names, col_name):
    """Check that all required argument are present in the DataFrame columns"""
    for arg in column_names:
        if arg not in df.columns:
            raise AttributeError(
                f"The column {arg} is not present within the results DataFrame and is required to compute '{col_name}', listed in the calculations to be executed"
            )


# TODO turn the dict into a class simular to the one of Calculation of oemof.tabular
CALCULATED_OUTPUTS = [
    {
        "column_name": "total_capacity",
        "operation": compute_total_capacity,
        "description": "The total capacity is calculated by adding the optimized capacity (investments) "
                       "to the existing capacity (capacity)",
        "argument_names": ["investments", "capacity"],
    },
    {
        "column_name": "total_annuity",
        "operation": compute_total_annuity,
        "description": "Total annuity is calculated by multiplying the optimized capacity "
                       "by the capacity cost (annuity considering CAPEX, OPEX and WACC)",
        "argument_names": ["investments", "capacity_cost"],
    },
    {
        "column_name": "upfront_investment_costs",
        "operation": compute_upfront_investment_costs,
        "description": "Upfront investment costs are calculated by multiplying the optimized capacity "
                       "by the CAPEX",
        "argument_names": ["investments", "capex"],
    },
    {
        "column_name": "total_opex_fix_costs",
        "operation": compute_opex_fix_costs,
        "description": "Operation and maintenance costs are calculated by multiplying the optimized capacity "
                       "by the OPEX",
        "argument_names": ["aggregated_flow", "marginal_cost", "carrier_cost"],
    },
    {
        "column_name": "total_variable_costs",
        "operation": compute_variable_costs,
        "description": "Variable costs are calculated by multiplying the total flow "
                       "by the marginal/carrier costs",
        "argument_names": ["aggregated_flow", "marginal_cost", "carrier_cost"],
    },
    {
        "column_name": "renewable_generation",
        "operation": compute_renewable_generation,
        "description": "The renewable generation for each component is computed from the flow and the "
                       "renewable factor.",
        "argument_names": [
            "aggregated_flow",
            "renewable_factor",
        ],
    },
    {
        "column_name": "cO2_emmissions",
        "operation": compute_co2_emissions,
        "description": "CO2 emissions are calculated from the flow and the emission factor.",
        "argument_names": ["aggregated_flow", "emission_factor"],
    },
    {
        "column_name": "additional_land_requirement",
        "operation": compute_additional_land_requirement,
        "description": "The additional land requirement calculates the land required for the optimized capacities.",
        "argument_names": ["investments", "emission_factor"],
    },
    {
        "column_name": "total_land_requirement",
        "operation": compute_total_land_requirement,
        "description": "The total land requirement calculates the land required for the total capacities.",
        "argument_names": ["total_capacity", "emission_factor"],
    },
]

# Add docstrings from function handles for documentation purposes
for calc in CALCULATED_OUTPUTS:
    func_handle = calc.get("operation", None)
    if callable(func_handle):
        calc["docstring"] = func_handle.__doc__
    else:
        calc["docstring"] = ""


def _validate_calculation(calculation):
    """Check if the parameters of a calculation are there and of the right format"""
    var_name = calculation.get("column_name", None)
    fhandle = calculation.get("operation", None)

    if var_name is None:
        raise ValueError(
            f"The 'column_name' under which the calculation should be saved in the results DataFrame is missing from the calculation dict: {calc}. Please check your input or look at help(apply_calculations) for the formatting of the calculation dict"
        )

    if not callable(fhandle):
        raise ValueError(
            f"The provided function handle for calculation of column '{var_name}' is not callable"
        )


def infer_busses_carrier(energy_system):
    """Loop through the nodes of an energy system and infer the carrier of busses from them

    Parameters
    ----------
    energy_system: oemof.solph.EnergySystem instance

    Returns
    -------
    dict mapping the busses labels to their carrier

    """

    busses_carrier = {}

    for node in energy_system.nodes:
        if hasattr(node, "carrier"):
            # quick fix to work for MIMO component
            # ToDo: assign carrier to busses instead of components to avoid problems
            for attribute in ("bus", "from_bus", "from_bus_0", "to_bus_1"):
                if hasattr(node, attribute):

                    bus_label = getattr(node, attribute).label
                    if bus_label in busses_carrier:
                        if busses_carrier[bus_label] != node.carrier:
                            print(
                                "busses carrier[bus label]", busses_carrier[bus_label]
                            )
                            print("node.carrier: ", node.carrier)
                            raise ValueError(
                                f"Two different carriers ({busses_carrier[bus_label]}, {node.carrier}) are associated to the same bus '{bus_label}'"
                            )
                    else:
                        busses_carrier[bus_label] = node.carrier

    busses = [node.label for node in energy_system.nodes if isinstance(node, solph.Bus)]

    for bus_label in busses:
        if bus_label not in busses_carrier:
            raise ValueError(
                f"Bus '{bus_label}' is missing from the busses carrier dict inferred from the EnergySystem instance"
            )

    return busses_carrier


def infer_asset_types(energy_system):
    """Loop through the nodes of an energy system and infer their types

    Parameters
    ----------
    energy_system: oemof.solph.EnergySystem instance

    Returns
    -------
    a dict mapping the asset (nodes which are not busses) labels to their type

    """
    asset_types = {}
    for node in energy_system.nodes:
        if isinstance(node, solph.Bus) is False:
            asset_types[node.label] = node.type
    return asset_types


def construct_multi_index_levels(flow_tuple, busses_info, assets_info=None):
    """Infer the index levels of the multi index dataframe sequence tuple and extra optional information

    Parameters
    ----------
    flow_tuple: tuple of bus label and asset label
        (A,B) means flow from A to B
    busses_info: either a list or a dict
        if not a list of busses labels, then a dict where busses labels are keys mapped to the bus carrier
    assets_info: dict
        mapping of asset labels to their type

    Returns
    -------
    a tuple with (bus label, direction of flow relative to asset, asset label, bus carrier (optional), asset type (optional)) direction is either 'in' or 'out'.

    The minimal tuple (b_elec, "in", demand) would read the flow goes from bus 'b_elec' '"in"' asset 'demand'

    """
    if isinstance(busses_info, dict):
        busses_labels = [bn for bn in busses_info]
    elif isinstance(busses_info, list):
        busses_labels = busses_info

    # infer which of the 2 nodes composing the flow is the bus
    bus_label = set(busses_labels).intersection(set(flow_tuple))
    if len(bus_label) == 1:
        bus_label = bus_label.pop()
    else:
        raise ValueError("Flow tuple does not contain only one bus node")
    # get position of bus node in the flow tuple
    idx_bus = flow_tuple.index(bus_label)
    answer = None

    # determine whether the flow goes from bus to asset or reverse
    if idx_bus == 0:
        # going from bus to asset, so the flow goes in to the asset
        asset_label = flow_tuple[1]
        answer = (bus_label, "in", asset_label)

    elif idx_bus == 1:
        asset_label = flow_tuple[0]
        # going from asset to bus, so the flow goes out of the asset
        answer = (bus_label, "out", asset_label)

    # add information of the bus carrier, if provided
    if isinstance(busses_info, dict):
        answer = answer + (busses_info[bus_label],)
    # add information of the asset type, if provided
    if assets_info is not None:
        answer = answer + (assets_info[asset_label],)
    return answer


def construct_dataframe_from_results(energy_system, bus_carrier=True, asset_type=True):
    """

    Parameters
    ----------
    energy_system: oemof.solph.EnergySystem instance
    bus_carrier: bool (opt)
        If set to true, the multi-index of the DataFrame will have a level about bus carrier
    asset_type: bool (opt)
        If set to true, the multi-index of the DataFrame will have a level about the asset type


    Returns
    -------
    Dataframe with oemof result sequences's timestamps as columns as well as investment and a multi-index built automatically, see construct_multi_index_levels for more information on the multi-index
    """
    mi_levels = [
        "bus",
        "direction",
        "asset",
    ]

    busses_info = infer_busses_carrier(energy_system)
    if bus_carrier is False:
        busses_info = list(busses_info.keys())
    else:
        mi_levels.append("carrier")

    if asset_type is True:
        assets_info = infer_asset_types(energy_system)
        mi_levels.append("facade_type")
    else:
        assets_info = None

    results = energy_system.results

    if isinstance(results, dict):
        ts = []
        investments = []
        flows = []
        for x, res in solph.views.convert_keys_to_strings(results).items():
            # filter out entries where the second element of the tuple is 'None' and ensure the
            # tuple has exactly two elements
            if x[1] != "None" and len(x) == 2:
                col_name = res["sequences"].columns[0]
                ts.append(
                    res["sequences"].rename(
                        columns={col_name: x, "variable_name": "timesteps"}
                    )
                )
                # here change this information for flow_tuple in ('mimo', 'in_group_0', '0')
                flows.append(
                    construct_multi_index_levels(
                        x, busses_info=busses_info, assets_info=assets_info
                    )
                )

                invest = None if res["scalars"].empty is True else res["scalars"].invest
                investments.append(invest)
        ts_df = pd.concat(ts, axis=1, join="inner")
        mindex = pd.MultiIndex.from_tuples(flows, names=mi_levels)

        df = pd.DataFrame(
            data=ts_df.T.to_dict(orient="split")["data"],
            index=mindex,
            columns=ts_df.index,
        )

        df["investments"] = investments
        df.sort_index(inplace=True)

    return df


def process_raw_results(df_results):
    """Compute the min, max and aggregated flows for each asset-bus pair

    Parameters
    ----------
    df_results: pandas DataFrame
        the outcome of construct_dataframe_from_results()

    Returns
    -------
    """
    temp = df_results[df_results.columns.difference(RAW_OUTPUTS)]
    df_results["flow_min"] = temp.min(axis=1)
    df_results["flow_max"] = temp.max(axis=1)
    df_results["aggregated_flow"] = temp.sum(axis=1)
    return df_results


def process_raw_inputs(df_results, dp_path, raw_inputs=RAW_INPUTS, typemap=None):
    """Find the input parameters from the datapackage.json file


    Parameters
    ----------
    df_results: pandas DataFrame
        the outcome of construct_dataframe_from_results()
    dp_path: string
        path to the datapackage.json file
    raw_inputs: list of string
        list of parameters from the datapackage one would like to collect for result post-processing

    Returns
    -------

    """
    if typemap is None:
        typemap = {}

    p = Package(dp_path)
    # initialise inputs_df with raw inputs as indexes
    inputs_df = pd.DataFrame(index=raw_inputs)
    # inputs_df = None
    for r in p.resources:
        if "elements" in r.descriptor["path"] and r.name != "bus":
            df = pd.DataFrame.from_records(r.read(keyed=True), index="name")
            resource_inputs = df[list(set(raw_inputs).intersection(set(df.columns)))].T
            if inputs_df is None:
                if not resource_inputs.empty:
                    inputs_df = resource_inputs
            else:
                inputs_df = inputs_df.join(resource_inputs)
            # if r.name in typemap:
            # TODO here test if facade_type has the method 'validate_datapackage'
            #   inputs_df = typemap[r.name].processing_raw_inputs(r, inputs_df)

    # kick out the lines where all values are NaN
    inputs_df = inputs_df.dropna(how="all")
    # append the inputs of the datapackage to the results DataFrame
    inputs_df.T.index.name = "asset"
    return df_results.join(inputs_df.T.apply(pd.to_numeric, downcast="float"))


def apply_calculations(results_df, calculations=CALCULATED_OUTPUTS):
    """Apply calculation and populate the columns of the results_df

    Parameters
    ----------
    df_results: pandas DataFrame
        the outcome of process_raw_input()
    calculations: list of dict
        each dict should contain
            "column_name" (the name of the new column within results_df),
            "operation" (handle of a function which will be applied row-wise to results_df),
            "description" (a string for documentation purposes)
            and "argument_names" (list of columns needed within results_df)

    Returns
    -------

    """
    for calc in calculations:
        _validate_calculation(calc)
        var_name = calc.get("column_name")
        argument_names = calc.get("argument_names", [])
        func_handle = calc.get("operation")
        try:
            _check_arguments(results_df, column_names=argument_names, col_name=var_name)
        except AttributeError as e:
            logging.warning(e)
            continue

        results_df[var_name] = results_df.apply(
            func_handle,
            axis=1,
        )
        # check if the new column contains all None values and remove it if so
        if results_df[var_name].isna().all():
            results_df.drop(columns=[var_name], inplace=True)
            logging.info(
                f"Removed column '{var_name}' because it contains all None values."
            )
