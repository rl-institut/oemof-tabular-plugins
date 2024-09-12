import os

# TODO this should be with from oemof.tabular.datapackage import building when https://github.com/oemof/oemof-tabular/pull/173 is merged
from oemof_tabular_plugins.script import display_scenario_results

# ---- imports to be used when the package has been installed ----
from oemof.tabular import datapackage  # noqa


parameters_units = {
    "battery_storage": "[kWh]",
    "inverter": "[kW]",
    "pv-panel": "[kW]",
    "diesel-generator": "[kW]",
    "water-storage": "[m³]",
    "mimo": "[m³/h]",
    "annuity_total": "[$]",
    "variable_costs_total": "[$]",
    "system_cost_total": "[$]",
    "specific_system_cost": "[$]",
    "total_upfront_investments": "[$]",
    "banana-plantation": "[m²]",
    "land_requirement_total": "[m²]",
    "ghg_emissions_total": "[?kg?]",
    "land_requirement_additional": "[m²]",
    "total_water_footprint": "[m³]",
    "river-water-uptake": "[m³]",
    "water-filtration-system": "[m³]",
    "rainwater-harvesting": "[m³]",
    "water": "[m³]",
    "electricity": "[kWh]",
    "crop": "[kg]",
}


# -------------- RELEVANT PATHS --------------
# get the project directory by navigating up one level from the current script file
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# -------------- USER INPUTS --------------
# list of scenarios to be evaluated
scenarios = [
    # "general_add_cost_inputs",
    # "general_basic",
    # "general_constraints",
    # "general_custom_attributes",
    # "wefe_custom_attributes",
    # "wefe_pv_panel",
    # "wefe_reverse_osmosis",
    # "aiwa"
    # "arusi"
]


# -------------- RUNNING THE SCENARIOS --------------
for scenario in scenarios:
    print("Running scenario with datapackage {}".format(scenario))
    # set paths for scenario and result directories
    scenario_dir = os.path.join(project_dir, "scenarios", scenario)
    results_path = os.path.join(project_dir, "results", scenario, "output")

    calculator = display_scenario_results(
        scenario_dir,
        results_path,
        parameters_units=parameters_units,
    )

    df = calculator.df_results
    print(df)

print("done")
