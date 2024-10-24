import logging

import pandas as pd
from datapackage import Package
import oemof.solph as solph
import numpy as np
from oemof_tabular_plugins.datapackage.building import infer_busses_carrier

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
    "capacity_minimum",
    "expandable",
    "storage_capacity",
    "storage_capacity_potential",
    "efficiency",
    "capex",
    "opex_fix",
    "lifetime",
    "renewable_factor",
    "emission_factor",
    "ghg_emission_factor",
    "land_requirement_factor",
    "water_footprint_factor",
]


# Functions for results per component
def compute_capacity_total(results_df):
    # ToDo: how it is here is that now total capacity is considering the storage capacity (in MWh) for storage.
    #  Check how the storage capacities should be displayed in the results to make it not confusing for the user. Maybe
    #  the storage components need two total capacity results (one for power and one for energy)?
    """Calculates total capacity by adding existing capacity (capacity) to optimized capacity (investments)"""
    investments = results_df.investments
    if investments is None:
        investments = 0

    if "storage" in results_df.name:
        return results_df.storage_capacity + investments
    else:
        return results_df.capacity + investments


def compute_capacity_added(results_df):
    """Calculates duplicate optimized capacity (investments) into a column with a better name"""
    investments = results_df.investments
    if investments is None:
        investments = 0
    return investments


def compute_annuity_total(results_df):
    """Calculates total annuity by multiplying the annuity by the optimized capacity"""
    # ToDo: now storage_capacity_cost is used for the annuity if the component is storage.
    #  Check that this is correctly applied for storage components or if two different costs should
    #  be calculated (one for power and one for energy)

    investments = results_df.investments
    if investments is None:
        investments = 0

    if "storage" in results_df.name:
        return results_df.storage_capacity_cost * investments
    else:
        return results_df.capacity_cost * investments


def compute_upfront_investment_costs(results_df):
    # ToDo: check for storage if investments is based on correct parameter
    """Calculates investment costs by multiplying capex with optimized capacity (investments)"""
    if "capex" not in results_df.index:
        return None
    else:
        investments = results_df.investments
        if investments is None:
            investments = 0

        return results_df.capex * investments


def compute_opex_fix_costs(results_df):
    """Calculates yearly opex costs by multiplying opex with optimized capacity (investments)"""
    if "opex_fix" not in results_df.index:
        return None
    else:
        investments = results_df.investments
        if investments is None:
            investments = 0
        return results_df.opex_fix * investments


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


def compute_ghg_emissions(results_df):
    """Calculates ghg emissions by multiplying aggregated flow by emission factor"""
    if "ghg_emission_factor" not in results_df.index:
        return None
    else:
        return results_df.aggregated_flow * results_df.ghg_emission_factor


def compute_land_requirement_additional(results_df):
    """Calculates land requirement needed for optimized capacities"""
    if "land_requirement_factor" not in results_df.index:
        return None
    else:
        investments = results_df.investments
        if investments is None:
            investments = 0
        return investments * results_df.land_requirement_factor


def compute_land_requirement_total(results_df):
    """Calculates land requirement needed for total capacities"""
    if "land_requirement_factor" not in results_df.index:
        return None
    else:
        return results_df.capacity_total * results_df.land_requirement_factor


def compute_water_footprint(results_df):
    """Calculates water footprint by multiplying aggregated flow by water footprint factor"""
    if "water_footprint_factor" not in results_df.index:
        return None
    else:
        return results_df.aggregated_flow * results_df.water_footprint_factor


# Functions for whole system results
def compute_system_annuity_total(results_df):
    """Calculates system total annuity by summing the total annuity for each component"""
    # ToDo: this method looks through each component and if it is mentioned twice, the annuity only
    #  gets considered once e.g. for storage, except for a MIMO because the costs only get considered once already.
    #  This is a quick fix and I didn't have time to figure out how this should be done in the cleanest way
    seen_components = set()
    annuity_total = 0
    # get the position of the component label in the MultiIndex
    idx_component = results_df.index.names.index("asset")
    for index, row in results_df.iterrows():
        component_name = index[idx_component]
        # check if the component has been included before
        if component_name not in seen_components:
            annuity_value = row["annuity_total"]
            if pd.isna(annuity_value):
                annuity_value = 0
            annuity_total += annuity_value
            seen_components.add(component_name)
    return annuity_total


def compute_system_variable_costs_total(results_df):
    """Calculates the total variable costs by summing the variable costs for each component flow"""
    # This function has not been implemented the same as above because here we want to consider the variable
    # costs attached to each flow instead of each component
    variable_costs_total = results_df["variable_costs_total"].sum()
    return variable_costs_total


def compute_system_cost_total(results_df):
    """Calculates the total system cost by summing the total annuity and total variable costs
    for each component"""
    # ToDo: quick fix - I didn't have time but rather than repeating the functions above, it would be good to calculate
    #  this from the kpis dataframe instead of results_df (then can use the annuity_total and variable_costs_total
    #  directly. To do this though, the apply_kpi_calculations function has to be adapted
    seen_components = set()
    annuity_total = 0
    # get the position of the component label in the MultiIndex
    idx_component = results_df.index.names.index("asset")
    for index, row in results_df.iterrows():
        component_name = index[idx_component]
        # check if the component has been included before
        if component_name not in seen_components:
            annuity_value = row["annuity_total"]
            if pd.isna(annuity_value):
                annuity_value = 0
            annuity_total += annuity_value
            seen_components.add(component_name)
    variable_costs_total = results_df["variable_costs_total"].sum()
    system_cost_total = annuity_total + variable_costs_total
    return system_cost_total


def compute_system_upfront_investments_total(results_df):
    """Calculates the total upfront investments by summing the upfront investments for each component"""
    # ToDo: this method looks through each component and if it is mentioned twice, the annuity only
    #  gets considered once e.g. for storage, except for a MIMO because the costs only get considered once already.
    #  This is a quick fix and I didn't have time to figure out how this should be done in the cleanest way
    seen_components = set()
    upfront_investments_total = 0
    # get the position of the component label in the MultiIndex
    idx_component = results_df.index.names.index("asset")
    for index, row in results_df.iterrows():
        component_name = index[idx_component]
        # check if the component has been included before
        if component_name not in seen_components:
            upfront_investment = row["upfront_investment_costs"]
            if pd.isna(upfront_investment):
                upfront_investment = 0
            upfront_investments_total += upfront_investment
            seen_components.add(component_name)
    return upfront_investments_total


def compute_system_co2_emissions_total(results_df):
    """Calculates the total CO2 emissions by summing up the CO2 emissions on each component flow"""
    # ToDo: so far these are simply summed for each flow, but should check this is correct in every case
    emissions_total = results_df["co2_emissions"].sum()
    return emissions_total


def compute_system_opex_total(results_df):
    """Calculates the total OPEX by summing up the opex from each component"""
    opex_total = results_df["opex_fix_costs_total"].sum()
    return opex_total


def compute_system_land_requirement_additional(results_df):
    """Calculates the additional land requirement from optimized capacities by summing the additional
    land requirement for each component"""
    # ToDo: this method looks through each component and if it is mentioned twice, the annuity only
    #  gets considered once e.g. for storage, except for a MIMO because the costs only get considered once already.
    #  This is a quick fix and I didn't have time to figure out how this should be done in the cleanest way
    seen_components = set()
    add_land_requirement_total = 0
    # get the position of the component label in the MultiIndex
    idx_component = results_df.index.names.index("asset")
    for index, row in results_df.iterrows():
        component_name = index[idx_component]
        # check if the component has been included before
        if component_name not in seen_components:
            add_land_requirement = row["land_requirement_additional"]
            if pd.isna(add_land_requirement):
                add_land_requirement = 0
            add_land_requirement_total += add_land_requirement
            seen_components.add(component_name)
    return add_land_requirement_total


def compute_system_land_requirement_total(results_df):
    """Calculates the total land requirement by summing the total land requirement for each component"""
    # ToDo: this method looks through each component and if it is mentioned twice, the annuity only
    #  gets considered once e.g. for storage, except for a MIMO because the costs only get considered once already.
    #  This is a quick fix and I didn't have time to figure out how this should be done in the cleanest way
    seen_components = set()
    land_requirement_total = 0
    # get the position of the component label in the MultiIndex
    idx_component = results_df.index.names.index("asset")
    for index, row in results_df.iterrows():
        component_name = index[idx_component]
        # check if the component has been included before
        if component_name not in seen_components:
            land_requirement = row["land_requirement_total"]
            if pd.isna(land_requirement):
                land_requirement = 0
            land_requirement_total += land_requirement
            seen_components.add(component_name)
    return land_requirement_total


def compute_water_footprint_total(results_df):
    """Calculates the total water footprint by summing the total water footprint for each component"""
    # ToDo: so far these are simply summed for each flow, but should check this is correct in every case
    water_footprint_total = results_df["water_footprint"].sum()
    return water_footprint_total


def compute_ghg_emissions_total(results_df):
    """Calculates the total ghg emissions by summing the total ghg emissions for each component"""
    ghg_emission_total = results_df["ghg_emissions"].sum()
    return ghg_emission_total


def compute_specific_system_cost(results_df):
    """Calculates the total upfront investments by summing the upfront investments for each component"""
    # ToDo: will need to be adapted when non-energetic loads are included - for now only electricity is
    #  considered but this is not correct
    # ToDo: need to decide how this should be calculated for energy systems with multiple carriers
    #  (both energetic and non-energetic)
    total_load = 0
    total_system_cost = (
        results_df["annuity_total"].sum() + results_df["variable_costs_total"].sum()
    )
    # get the position of the facade label in the MultiIndex
    idx_facade = results_df.index.names.index("facade_type")
    # get the position of the carrier label in the MultiIndex, if it exists
    if "carrier" in results_df.index.names:
        idx_carrier = results_df.index.names.index("carrier")
    else:
        idx_carrier = None
    for index, row in results_df.iterrows():
        # This is a quick fix to not include water - need to talk to Julian about how other demands should
        # be considered
        if idx_carrier is None:
            if index[idx_facade] == "load":
                total_load += row.get("aggregated_flow", 0)
        else:
            if index[idx_facade] == "load" and index[idx_carrier] == "electricity":
                total_load += row.get("aggregated_flow", 0)
    specific_system_cost = total_system_cost / total_load
    return specific_system_cost


def compute_renewable_share(results_df):
    """Calculates the renewable share based on the renewable generation of each flow and the
    total aggregated flow of any component where the renewable factor is set (should be only set on sources)
    """
    # ToDo: this might need to be reconsidered when the renewable share is set on a non-source component
    #  e.g. if the PV panel is a transformer component and the renewable share is on the output. It might still
    #  work but definitely needs to be checked
    renewable_generation_total = 0
    generation_total = 0
    for index, row in results_df.iterrows():
        if not pd.isna(row["renewable_factor"]):
            generation = row["aggregated_flow"]
            renewable_generation = row["aggregated_flow"] * row["renewable_factor"]
            generation_total += generation
            renewable_generation_total += renewable_generation
    if generation_total == 0:
        renewable_share = 0
    else:
        renewable_share = renewable_generation_total / generation_total
    return renewable_share


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
        "column_name": "capacity_total",
        "operation": compute_capacity_total,
        "description": "The total capacity is calculated by adding the optimized capacity (investments) "
        "to the existing capacity (capacity)",
        "argument_names": ["investments", "capacity"],
    },
    {
        "column_name": "capacity_added",
        "operation": compute_capacity_added,
        "description": "The optimized capacity column is duplicated with a better name than 'investments'",
        "argument_names": ["investments"],
    },
    {
        "column_name": "annuity_total",
        "operation": compute_annuity_total,
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
        "column_name": "opex_fix_costs_total",
        "operation": compute_opex_fix_costs,
        "description": "Operation and maintenance costs are calculated by multiplying the optimized capacity "
        "by the OPEX",
        "argument_names": ["aggregated_flow", "opex_fix"],
    },
    {
        "column_name": "variable_costs_total",
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
        "column_name": "co2_emissions",
        "operation": compute_co2_emissions,
        "description": "CO2 emissions are calculated from the flow and the emission factor",
        "argument_names": ["aggregated_flow", "emission_factor"],
    },
    {
        "column_name": "ghg_emissions",
        "operation": compute_ghg_emissions,
        "description": "GHG emissions are calculated from the flow and the emission factor",
        "argument_names": ["aggregated_flow", "ghg_emission_factor"],
    },
    # TODO this is specific to APV, should not be added here but rather in wefe/post_processing
    {
        "column_name": "land_requirement_additional",
        "operation": compute_land_requirement_additional,
        "description": "The additional land requirement calculates the land required for the optimized capacities",
        "argument_names": ["investments", "land_requirement_factor"],
    },
    {
        "column_name": "land_requirement_total",
        "operation": compute_land_requirement_total,
        "description": "The total land requirement calculates the land required for the total capacities",
        "argument_names": ["capacity_total", "land_requirement_factor"],
    },
    {
        "column_name": "water_footprint",
        "operation": compute_water_footprint,
        "description": "The water footprint calculates the water footprint for the aggregated flows of each component",
        "argument_names": ["aggregated_flow", "water_footprint_factor"],
    },
]

# ToDo: turn dict into a class (see CALCULATED_OUTPUTS) and decide where this belongs - either here or in processing
#  or maybe these can be joined with CALCULATED_OUTPUTS and there is another parameter that defines if it is a calculation
#  per component (to be added to df_results) or a calculation for the whole system (e.g. LCOE, total emissions etc).
#  Probably this should be included with the other CALCULATED_OUTPUTS eventually, but should ask PF
CALCULATED_KPIS = [
    {
        "column_name": "annuity_total",
        "operation": compute_system_annuity_total,
        "description": "The system total annuity is calculated by summing up the total annuity for each component",
        "argument_names": ["annuity_total"],
    },
    {
        "column_name": "variable_costs_total",
        "operation": compute_system_variable_costs_total,
        "description": "The system total variable costs is calculated by summing up the total variable costs for "
        "each component flow",
        "argument_names": ["variable_costs_total"],
    },
    {
        "column_name": "system_cost_total",
        "operation": compute_system_cost_total,
        "description": "The total system cost is calculated by adding the total annuity to the total variable costs",
        "argument_names": ["annuity_total", "variable_costs_total"],
    },
    {
        "column_name": "total_upfront_investments",
        "operation": compute_system_upfront_investments_total,
        "description": "The total upfront investments value is calculated by summing the upfront investment"
        "costs for each component",
        "argument_names": ["upfront_investment_costs"],
    },
    {
        "column_name": "specific_system_cost",
        "operation": compute_specific_system_cost,
        "description": "T",
        "argument_names": ["aggregated_flow", "annuity_total", "variable_costs_total"],
    },
    {
        "column_name": "co2_emissions_total",
        "operation": compute_system_co2_emissions_total,
        "description": "The total emissions is calculated by summing the c02 emissions "
        "for each component",
        "argument_names": ["co2_emissions"],
    },
    {
        "column_name": "land_requirement_additional",
        "operation": compute_system_land_requirement_additional,
        "description": "The total additional land requirement is calculated by summing the additional land requirement "
        "for each component",
        "argument_names": ["land_requirement_additional"],
    },
    {
        "column_name": "land_requirement_total",
        "operation": compute_system_land_requirement_total,
        "description": "The total land requirement is calculated by summing the total land requirement "
        "for each component",
        "argument_names": ["land_requirement_total"],
    },
    {
        "column_name": "total_water_footprint",
        "operation": compute_water_footprint_total,
        "description": "The total water footprint is calculated by summing the water footprint required "
        "for each component",
        "argument_names": ["water_footprint"],
    },
    {
        "column_name": "renewable_share",
        "operation": compute_renewable_share,
        "description": "The renewable share is calculated by dividing the renewable generation by the total "
        "generation",
        "argument_names": ["renewable_factor", "aggregated_flow"],
    },
    {
        "column_name": "ghg_emissions_total",
        "operation": compute_ghg_emissions_total,
        "description": "",
        "argument_names": ["ghg_emissions"],
    },
    {
        "column_name": "system_opex_total",
        "operation": compute_system_opex_total,
        "description": "",
        "argument_names": ["opex_fix_costs_total"],
    },
]

# Add docstrings from function handles for documentation purposes
for calc in CALCULATED_OUTPUTS + CALCULATED_KPIS:
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
            if hasattr(node, "type"):
                asset_type = node.type
            else:
                asset_type = type(node)
            if asset_type is None:
                asset_type = node.tech
            asset_types[node.label] = asset_type
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


def construct_dataframe_from_results(
    energy_system, dp_path=None, infer_bus_carrier=True, asset_type=True
):
    """

    Parameters
    ----------
    energy_system: oemof.solph.EnergySystem instance
    bus_carrier: dict (opt) mapping the bus name to its carrier
        If not None the multi-index of the DataFrame will have a level about bus carrier
    infer_bus_carrier: bool (opt)
        if True and bus_carrier is none, the bus-carrier mapping will be inferred
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
    # lookup bus-carrier mapping in the dp_path, if existing
    p = Package(dp_path)
    bus_carrier = infer_busses_carrier(p, infer_from_component=False)

    if bus_carrier is None:
        if infer_bus_carrier is True:
            busses_info = infer_busses_carrier(p)
            mi_levels.append("carrier")
            logging.warning(
                "No carrier column found in 'elements/bus.csv' file of datapackage, the bus-carrier mapping will be inferred from the component's carrier"
            )
        else:
            busses_info = [
                node.label
                for node in energy_system.nodes
                if isinstance(node, solph.Bus)
            ]
            logging.info(
                "No bus-carrier mapping found and infer_bus_carrier set to 'False'. Result dataframe will not contain 'carrier' in its MultiIndex levels."
            )

    else:
        busses_info = bus_carrier
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


def apply_calculations(results_df, calculations=None):
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
    if calculations is None:
        calculations = []

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
        # ToDo: I've commented this out for now but decide if this or some form should be kept in
        # # check if the new column contains all None values and remove it if so
        # if results_df[var_name].isna().all():
        #     results_df.drop(columns=[var_name], inplace=True)
        #     logging.info(
        #         f"Removed column '{var_name}' because it contains all None values."
        #     )


def apply_kpi_calculations(results_df, calculations=None):
    """Apply calculation and return a new DataFrame with the KPIs.

    Parameters
    ----------
    results_df : pd.DataFrame
        The input DataFrame with raw data.
    calculations : list of dict
        List of calculations to be applied. Each calculation is a dictionary
        with keys: "column_name", "argument_names", and "operation".

    Returns
    -------
    pd.DataFrame
        A new DataFrame containing the calculated KPI values with var_name as the index.
    """

    if calculations is None:
        calculations = []

    kpis = []

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

        kpi_value = func_handle(results_df)
        kpis.append({"kpi": var_name, "value": kpi_value})

    if kpis:
        answer = pd.DataFrame(kpis).set_index("kpi")
    else:
        answer = None
    return answer
