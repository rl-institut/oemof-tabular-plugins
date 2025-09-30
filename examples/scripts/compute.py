import os
from oemof.solph import EnergySystem, Model
from oemof.solph import processing
from oemof.solph.processing import parameter_as_dict

# TODO this should be with from oemof.tabular.datapackage import building when https://github.com/oemof/oemof-tabular/pull/173 is merged
from oemof_tabular_plugins.script import compute_scenario

# ---- imports to be used when the package has been installed ----
from oemof.tabular import datapackage  # noqa


from oemof_tabular_plugins.wefe import WEFE_TYPEMAP as TYPEMAP

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
    "test_wind_volatile"
    # "test_pv_noload"
    # "test_crop_volopen_cropfix"
    # "test_crop_volpot_croppot"
    # "test_crop_volfix_cropfix_2"
    # "test_pv_load_volopen_pvpot"
    # "test_pv_load_volpot_pvfix_2"
    # "test_pv_load_volcombi_pvcombi"
    # "test_crop_volpot_croppot"
    # "test_pv_load_volpot_pvpot"
    # "test_pv_noload_volfix_pvfix"
    # "test_crop_volmin_cropmin"
    # "test_crop_volpot_croppot_2"
    # "test_crop_volcombi_cropcombi_mins"
    # "test_crop_volcombi_cropcombi"
    # "test_mimo_crop"
    # "general_add_cost_inputs",
    # "general_basic",
    # "general_constraints",
    # "general_custom_attributes",
    # "wefe_custom_attributes",
    # "wefe_pv_panel",
    # "wefe_reverse_osmosis",
]
# weighted average cost of capital (WACC) - might move later
# this parameter is needed if CAPEX, OPEX fix and lifetime are included
wacc = 0.06

# -------------- ADDITIONAL FUNCTIONALITIES (OEMOF-TABULAR-PLUGINS) --------------
# include the custom attribute parameters to be included in the model
# this can be moved somewhere and included in a dict or something similar with all possible additional attributes
custom_attributes = [
    "emission_factor",
    "renewable_factor",
    "land_requirement_factor",
    "water_footprint_factor",
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
        dash_app=True,
        parameters_units=parameters_units,
        skip_infer_datapackage_metadata=False
    )
    df = calculator.df_results
    print(df)
    print(calculator.raw_outputs)
    print(calculator.calculated_outputs)
    print(calculator.raw_inputs)
    import pdb

    pdb.set_trace()


print("done")
