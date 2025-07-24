import os

# from attr import attributes
# from oemof.solph import EnergySystem, Model
# from oemof.solph import processing
# from oemof.solph.processing import parameter_as_dict

import numpy as np
import pandas as pd
import shutil
from pathlib import Path

# TODO this should be with from oemof.tabular.datapackage import building when https://github.com/oemof/oemof-tabular/pull/173 is merged
from oemof_tabular_plugins.script import compute_scenario

# ---- imports to be used when the package has been installed ----
from oemof.tabular import datapackage  # noqa
from oemof_tabular_plugins.wefe import WEFE_TYPEMAP as TYPEMAP

parameters_units = {
    "drinking-water-storage": "[m³]",
    "total_annual_cost_moo": "[USD/a]",
    "rainwater-harvesting": "[m²]",
    "service-water-storage": "[m³]",
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
    "ghg_emissions_total": "[kgCO2e/a]",
    "system_cost_total": "[USD/a]",
    "land_requirement_additional": "[m²]",
    "total_upfront_investments": "[USD]",
    "land_requirement_total": "[m²]",
    "total_water_footprint": "[m³]",
    "system_opex_total": "[USD/a]",
    "total_variable_cost_moo": "[USD/a]",
    "total_water_consumption": "[m³/a]",
    "total_indirect_water_consumption": "[m³/a]",
    "ac-elec": "[kWh]",
    "water_scarcity_footprint": "[m³]",
}

# -------------- RELEVANT PATHS --------------
# get the project directory by navigating up one level from the current script file
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# -------------- USER INPUTS --------------
base_scenario = "arusi_8760"
element = "volatile"
component = "photovoltaics"
attribute = "capex"
increment = 0.2
steps = 10

# Regionalized Characterisation Factor for Available water remaining (AWARE) - might move later;
# this parameter is needed to calculate the regionalized water scarcity footprint in moo.
# weighted average cost of capital (WACC) - might move later
# this parameter is needed if CAPEX, OPEX fix and lifetime are included
wacc = 0.06

# -------------- SET UP SCENARIOS FOR SENSITIVITY ANALYSIS --------------
print(
    "Set up sensitivity analysis: create multiple, alternated copies of base scenario"
)

# Lists for the new scenarios created to vary an attribute and the respective values of the varied attribute
scenarios = []
attributes = []

# Turn increment and steps into continuous list
first_step = increment
last_step = increment * steps
step_list = np.round(np.linspace(first_step, last_step, steps), 2)

# Additional relevant paths
example_dir = Path(__file__).parent.parent
scenario_dir = example_dir / "scenarios"
base_scenario_dir = scenario_dir / base_scenario
result_dir = example_dir / "results"

# Create copies of base_scenario and change targeted attribute
for step in step_list:
    # Create new scenario path
    new_scenario = f"{base_scenario}_{component}_{attribute}_{step}"
    new_scenario_dir = scenario_dir / new_scenario

    # Create full copy of base scenario directory into new scenario path
    # (dirs_exist_ok=True enables overwriting, otherwise error if directory already exists)
    shutil.copytree(base_scenario_dir, new_scenario_dir, dirs_exist_ok=True)

    # Access new scenario directory and change targeted element-component-attribute based on original value and step
    element_path = new_scenario_dir / "data" / "elements" / f"{element}.csv"
    df = pd.read_csv(element_path, sep=";")
    original_attribute = df.loc[df["name"] == component, attribute].item()
    new_attribute = original_attribute * step
    df.loc[df["name"] == component, attribute] = new_attribute
    df.to_csv(element_path, index=False, sep=";")

    # Add new scenario to scenarios list, respective attribute to attributes list
    scenarios.append(new_scenario)
    attributes.append(new_attribute)

print("Scenarios created")

# -------------- ADDITIONAL FUNCTIONALITIES (OEMOF-TABULAR-PLUGINS) --------------
# include the custom attribute parameters to be included in the model
# this can be moved somewhere and included in a dict or something similar with all possible additional attributes
custom_attributes = [
    "ghg_emission_factor",
    "renewable_factor",
    "land_requirement_factor",
    "water_consumption_factor",
    "indirect_water_consumption_factor",
    "land_requirement",
    "water_footprint",
    "ghg_emissions",
    "resource_cost",
    "annuity",
]
# set whether the multi-objective optimization should be performed
moo = True

# -------------- RUNNING THE SCENARIOS --------------
for scenario in scenarios:
    print("Running scenario with datapackage {}".format(scenario))
    # set paths for scenario and result directories
    scenario_dir = os.path.join(project_dir, "scenarios", scenario)
    results_path = os.path.join(project_dir, "results", scenario, "output")

    calculator = compute_scenario(
        scenario_dir,
        results_path,
        wacc,
        scenario_name=scenario,
        custom_attributes=custom_attributes,
        typemap=TYPEMAP,
        moo=moo,
        dash_app=False,  # dash has to be deactivated if multiple scenarios are computed!
        parameters_units=parameters_units,
    )
    df = calculator.df_results
    print(df)
    print(calculator.raw_outputs)
    print(calculator.calculated_outputs)
    print(calculator.raw_inputs)

print("All scenarios computed")

# -------------- POST_PROCESSING: SENSITIVITY ANALYSIS --------------
results_path = (
    Path(__file__).parent / f"{base_scenario}_{component}_{attribute}_sensitivity.csv"
)
print(f"Obtain scenario results, combine and export to {results_path}")

# Set up DataFrame for sensitivity analysis with columns: base scenario, new scenario, varied attribute
sensitivity_df = pd.DataFrame(
    columns=["base_scenario", "component", "variation", attribute]
)

# Access results of every scenario and create one row in the DataFrame for each scenario
for i in range(len(scenarios)):
    # Set up new row for the data of this scenario with basic information
    new_row = {
        "base_scenario": base_scenario,
        "component": component,
        "variation": step_list[i],
        attribute: attributes[i],
    }

    # Append the new row to the sensitivity DataFrame
    sensitivity_df.loc[i] = new_row

    # Access kpis of the scenario
    kpis_path = result_dir / scenarios[i] / "output" / "kpis.csv"
    kpis_df = pd.read_csv(kpis_path, sep=",")

    # Transpose kpis to have only one row with all values, align row index with sensitivity DataFrame index
    kpis_df = kpis_df.set_index("kpi").T
    kpis_df.index = [i]

    # Add kpis to the new row of the sensitivity DataFrame, make sure columns exist and fill with nan if necessary
    for col in kpis_df.columns:
        if col not in sensitivity_df.columns:
            sensitivity_df[col] = pd.NA
        sensitivity_df.loc[i, col] = kpis_df.loc[i, col]

    # Access component capacities of the scenario
    capacities_path = result_dir / scenarios[i] / "output" / "capacities.csv"
    capacities_df = pd.read_csv(capacities_path, sep=",")

    # Access the different components (rows of the capacities DataFrame)
    for j in range(len(capacities_df)):
        # Get the row and the components name
        component_row = capacities_df.iloc[j]
        component_name = component_row["Component name"]

        # Convert DataFrame headers (e.g. Capacity -> <component>_capacity),
        # create Dict with the component's data and its new header
        component_data = {
            f"{component_name}_capacity": component_row["Capacity"],
            f"{component_name}_opt_capacity": component_row["Optimized Capacity"],
            f"{component_name}_tot_capacity": component_row["Capacity Total"],
            f"{component_name}_max_capacity": component_row["Maximum Capacity"],
            f"{component_name}_unit": component_row["unit"],
        }

        # Turn Dict into DataFrame and align index
        component_df = pd.DataFrame([component_data])
        component_df.index = [i]

        # Add component data to the new row
        for col in component_df.columns:
            if col not in sensitivity_df.columns:
                sensitivity_df[col] = pd.NA
            sensitivity_df.loc[i, col] = component_df.loc[i, col]


# Convert all numeric objects to float-type (ignore non-numeric objects), then round all float-type objects
try:
    sensitivity_df = sensitivity_df.apply(pd.to_numeric)
except ValueError or TypeError:
    pass
sensitivity_df = sensitivity_df.round(2)

# Save sensitivity analysis DataFrame as csv-file
sensitivity_df.to_csv(results_path)


print("done")
