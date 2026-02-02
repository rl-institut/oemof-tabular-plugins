import logging
import os
from datapackage import Package
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
    get_raw_inputs,
    RAW_OUTPUTS,
    PROCESSED_RAW_OUTPUTS,
    CALCULATED_OUTPUTS,
    CALCULATED_KPIS,
)

from .gui import prepare_app

# ------ New post-processing to create tables ------
# This dictionary contains groups of columns that should be extracted from the df_results to generate a clearer overview


# TODO add a column for planned capacity (not optimizable but including costs) in capacities if it gets properly
#  implemented (planned capacity can be set by setting capacity_minimum == capacity_potential and dispatchable = True)
RESULT_TABLE_COLUMNS = {
    "costs": ["upfront_investment_costs", "annuity_total", "variable_costs_total"],
    "capacities": [
        "capacity",
        "storage_capacity",
        "capacity_minimum",
        "capacity_potential",
        "storage_capacity_potential",
        "expandable",
        "investments",
        "capacity_total",
    ],
}

CAPACITIES_UNIT = {
    "electricity": {"default": "[kW]", "storage": "[kWh]"},
    "wind": {"default": "[m/s]", "storage": "[kWh]"},
    "irradiation": {"default": "[kWh/m²]", "storage": "[kWh]"},
    "crop": {"default": "[m²]", "storage": "[kg]"},
    "water": {"default": "[m³/h]", "storage": "[m³]"},
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
    # TODO: set index=False or do we need the additional index column in the csv files?
    """Saves a DataFrame to a .csv file"""
    filepath = os.path.join(results_path, filename)
    table.to_csv(filepath)


# TODO figure out the best table/display for storage results


# --------------------------------------------------
class OTPCalculator(Calculator):
    def __init__(
        self, input_parameters, energy_system, dp_path, infer_bus_carrier=True, moo=False
    ):

        self.moo = moo
        self.raw_inputs_list = get_raw_inputs(moo=moo)

        self.df_results = construct_dataframe_from_results(
            energy_system, dp_path=dp_path, infer_bus_carrier=infer_bus_carrier
        )
        self.n_timesteps = len(energy_system.timeindex)

        self.df_results = process_raw_results(self.df_results)
        self.df_results = process_raw_inputs(self.df_results, dp_path, moo=moo)

        # Load cf_aware from datapackage if MOO is active
        self.cf_aware = None
        if moo:
            try:
                import os
                from oemof_tabular_plugins.general.pre_processing.pre_processing_moo import get_moo_timeseries

                # Extract scenario directory from dp_path
                scenario_dir = os.path.dirname(os.path.dirname(dp_path))
                cf_aware_array = get_moo_timeseries(
                    scenario_dir, ts_name="cf_aware", resource_name="moo_profile"
                )
                # Take the mean value as cf_aware is typically constant for a location
                self.cf_aware = float(cf_aware_array.mean())

                # Add cf_aware as a column to df_results for use in KPI calculations
                self.df_results['cf_aware'] = self.cf_aware

                logging.info(f"Loaded cf_aware value: {self.cf_aware}")
            except Exception as e:
                cf_aware_default = 4.5
                logging.warning(f"Could not load cf_aware from moo_profile.csv: {str(e)}. Using default value of 4.5")
                self.cf_aware = cf_aware_default
                self.df_results['cf_aware'] = self.cf_aware

        self.kpis = None

        try:
            super().__init__(input_parameters, energy_system.results)
        except Exception as e:
            logging.error(
                f"Initialisation of Calculator did not succeed: {str(e)}. This should not impact oemof-tabular-plugins results."
            )

    def apply_calculations(self, calculations):
        apply_calculations(self.df_results, calculations=calculations)

    def apply_kpi_calculations(self, calculations):
        self.kpis = apply_kpi_calculations(self.df_results, calculations=calculations)

    def __scalars(self, scalar_category):
        """Ignore the flow data columns (by construction those are the first columns after the multi-index)"""
        scalars = self.df_results.iloc[:, self.n_timesteps :]
        answer = scalars
        if scalar_category == "raw_inputs":
            existing_cols = []
            for c in scalars.columns:
                if c in self.raw_inputs_list:
                    existing_cols.append(c)
            answer = scalars[existing_cols]
        elif scalar_category == "outputs":
            answer = scalars[scalars.columns.difference(self.raw_inputs_list)]
        return answer

    @property
    def raw_outputs(self):
        self.df_results.iloc[:, : self.n_timesteps]
        cols = self.df_results.iloc[:, : self.n_timesteps].columns.tolist()
        cols = cols + RAW_OUTPUTS + PROCESSED_RAW_OUTPUTS
        return self.df_results[cols]

    @property
    def raw_inputs(self):
        return self.__scalars("raw_inputs")

    @property
    def calculated_outputs(self):
        return self.__scalars("outputs")


def post_processing(
    params,
    es,
    results_path,
    dp_path,
    dash_app=False,
    parameters_units=None,
    infer_bus_carrier=True,
    calculations=None,
    kpi_calculations=None,
    moo=False,
):
    # ToDo: adapt this function after multi-index dataframe is implemented to make it more concise / cleaner
    # ToDo: params can be accessed in results so will not need to be a separate argument
    """
    The main post-processing function extracts various scalar and timeseries data and stores it in CSV files.
    :param params: energy system parameters
    :param es: oemof energy_system with results in it, ie es.results = processing.results(m) has been performed
    :param results_path: results directory path
    :param dp_path: path to the datapackage.json file
    :param moo: bool, if True MOO mode is active
    """

    if parameters_units is None:
        #  Units of Capacities and Kpis in Results
        parameters_units = {
            "drinking-water-storage": "[m³]",
            "rainwater-harvesting": "[m²]",
            "service-water-storage": "[m³]",
            "run-of-river-hydropower": "[kW]",
            "sw-ro": "[m³/h]",
            "seawater-reverse-osmosis": "[m³/h]",
            "electricity-grid": "[kWh]",
            "seawater": "[m³]",
            "seawater-source": "[m³]",
            "water-truck": "[m³]",
            "battery-storage": "[kWh]",
            "inverter": "[kW]",
            "water-filtration": "[m³/h]",
            "water-filtration-system": "[m³/h]",
            "water-pump": "[m³/h]",
            "river-water-uptake": "[m³/h]",
            "crop": "[m²]",
            "banana": "[m²]",
            "banana-production": "[kg/a]",
            "groundwater": "[m³]",
            "bottled-water": "[m³]",
            "diesel-generator": "[kW]",
            "photovoltaics": "[kWp]",
            "wind-turbine": "[kW]",
            "hydropower": "[kW]",
            "pv-panel": "[kW]",
            "water-storage": "[m³]",
            "mimo": "[m³/h]",
            "annuity_total": "[USD/a]",
            "variable_costs_total": "[USD/a]",
            "ghg_emission_total": "[kgCO2e/a]",
            "system_cost_total": "[USD/a]",
            "land_requirement_additional": "[m²]",
            "total_upfront_investments": "[USD]",
            "land_requirement_total": "[m²]",
            "total_water_consumption": "[m³/a]",
            "total_annual_cost_moo": "[USD/a]",
            "ghg_emissions_total": "[kgCO2e/a]",
            "total_water_footprint": "[m³]",
            "system_opex_total": "[USD/a]",
            "total_variable_cost_moo": "[USD/a]",
        }

    if calculations is None:
        calculations = CALCULATED_OUTPUTS
    if kpi_calculations is None:
        kpi_calculations = CALCULATED_KPIS
    # initiate calculator for post-processing
    calculator = OTPCalculator(params, es, dp_path, infer_bus_carrier=infer_bus_carrier, moo=moo)
    calculator.apply_calculations(calculations)
    calculator.apply_kpi_calculations(kpi_calculations)

    tables_to_save = {}

    results_by_flow = calculator.df_results

    result_tables = {}
    service_tables = {}

    if results_by_flow is not None:
        results_by_flow.to_csv(results_path / "all_results_by_flow.csv", index=True)
        # get sub-tables from results dataframe
        capacities_table = extract_table_from_results(
            results_by_flow, RESULT_TABLE_COLUMNS["capacities"]
        )

        # ignore the dispatchable sources optimized capacities
        capacities_table = capacities_table[
            capacities_table.index.get_level_values("facade_type") != "dispatchable"
        ]

        # Assign units to the capacities:
        def set_unit(carrier, facade_type):
            unit_choice = CAPACITIES_UNIT.get(carrier, {"default": "", "storage": ""})

            if facade_type in unit_choice:
                unit = unit_choice[facade_type]
            else:
                unit = unit_choice["default"]
            return unit

        capacities_table.reset_index(inplace=True)
        if "Capacity Potential" not in capacities_table.columns:
            capacities_table["Capacity Potential"] = None

        if "carrier" not in capacities_table.columns:
            capacities_table["carrier"] = None

        capacities_table["unit"] = capacities_table.apply(
            lambda x: set_unit(x.carrier, x.facade_type), axis=1
        )

        capacities_table.rename(
            columns={
                "asset": "Component name",
                "Investments": "Optimized Capacity",
                "Capacity Potential": "Maximum Capacity",
            },
            inplace=True,
        )

        # eliminate double occurences of same asset
        capacities_table = capacities_table.loc[
            (capacities_table["Capacity Total"] > 0),
            [
                "Component name",
                "Capacity",
                "Optimized Capacity",
                "Capacity Total",
                "Maximum Capacity",
                "unit",
            ],
        ]
        capacities_table = capacities_table.drop_duplicates(subset=["Component name"])

        result_tables.update({"capacities": capacities_table})

        cost_table = extract_table_from_results(
            calculator.df_results, RESULT_TABLE_COLUMNS["costs"]
        )

        # result_tables.update({"costs": cost_table})

        # IDEA define calculations for WEFE components in post_processing and make a merge within __init__ of WEFE
        # TODO list components (facades) automatically (low prio)
        df = results_by_flow.reset_index()
        if "carrier" not in df.columns:
            df["carrier"] = None
        service_busses = df.loc[df.facade_type == "load"].bus.tolist()
        service_busses += df.loc[
            (df.direction == "out") & (df.facade_type == "crop")
        ].bus.tolist()

        for bus in service_busses:
            df_bus = df.loc[df.bus == bus]

            service_tables[bus.replace("-bus", "")] = df_bus[
                ["asset", "direction", "aggregated_flow", "carrier", "facade_type"]
            ]

        # TODO: eventually add units (but kpis.csv also does not have units)
        service_flows = []
        service_flow_values = []
        for service, table in service_tables.items():
            table = table.loc[
                table["direction"] == "out",
                ["asset", "aggregated_flow", "carrier", "facade_type"],
            ]
            for row in table.itertuples(index=False):
                service_flows.append(f"{row.asset}_to_{service}")
                service_flow_values.append(row.aggregated_flow)

        service_flows_table = pd.DataFrame(
            {"flow": service_flows, "value": service_flow_values}
        )
        # TODO: Move this to tables_to_save BUT to do so, we have to get rid of extra index col
        if service_flows_table is not None:
            service_flows_table.to_csv(results_path / "service_flows.csv", index=False)

        # save tables to csv files
        tables_to_save.update(
            {"costs.csv": cost_table, "capacities.csv": capacities_table}
        )

    kpis = calculator.kpis
    if kpis is not None:
        kpis.to_csv(results_path / "kpis.csv", index=True)

        if "mimo" in results_by_flow.index.get_level_values("asset"):
            kpis.loc["total_water_produced"] = results_by_flow.loc[
                "permeate-bus", "in"
            ]["aggregated_flow"].sum()
            kpis.loc["total_brine_produced"] = results_by_flow.loc["brine-bus", "in"][
                "aggregated_flow"
            ].sum()
            kpis.loc["total_electricity_produced"] = results_by_flow.loc[
                "ac-elec-bus", "in"
            ]["aggregated_flow"].sum()

        result_tables.update({"kpis": kpis})

    for filename, table in tables_to_save.items():
        save_table_to_csv(table, results_path, filename)

    if dash_app is True:

        demo_app = prepare_app(
            es,
            dp_path=dp_path,
            tables=result_tables,
            services=service_tables,
            units=parameters_units,
        )
        demo_app.run(debug=False, port=8060)

    # ----- OLD POST-PROCESSING - TO BE DELETED ONCE CERTAIN -----
    if hasattr(calculator, "scalar_params"):
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

    return calculator
