import os
from oemof.solph import EnergySystem, Model
from oemof.solph import processing
from oemof.solph.processing import parameter_as_dict

# TODO this should be with from oemof.tabular.datapackage import building when https://github.com/oemof/oemof-tabular/pull/173 is merged
from oemof_tabular_plugins.script import compute_scenario

# ---- imports to be used when the package has been installed ----
from oemof.tabular import datapackage  # noqa
from oemof.tabular.facades import TYPEMAP


from oemof_tabular_plugins.wefe.facades import PVPanel, MIMO, SimpleCrop


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
      "arusi"
]
# weighted average cost of capital (WACC) - might move later
# this parameter is needed if CAPEX, OPEX fix and lifetime are included
wacc = 0.06

# -------------- ADDITIONAL FUNCTIONALITIES (OEMOF-TABULAR-PLUGINS) --------------
# include the custom attribute parameters to be included in the model
# this can be moved somewhere and included in a dict or something similar with all possible additional attributes
custom_attributes = [
    "ghg_emission_factor",
    "renewable_factor",
    "land_requirement_factor",
    "water_footprint_factor",
    "land_requirement",
    "water_footprint",
    "ghg_emissions",
    "resource_cost",
]
# set whether the multi-objective optimization should be performed
moo = True

# add PV Panel (from oemof-tabular-plugins) to facades type map (from oemof-tabular) - might move later
TYPEMAP["pv-panel"] = PVPanel
TYPEMAP["mimo"] = MIMO
TYPEMAP["crop"] = SimpleCrop

# -------------- RUNNING THE SCENARIOS --------------
for scenario in scenarios:
    print("Running scenario with datapackage {}".format(scenario))
    # set paths for scenario and result directories
    scenario_dir = os.path.join(project_dir, "scenarios", scenario)
    results_path = os.path.join(project_dir, "results", scenario, "output")

    compute_scenario(
        scenario_dir,
        results_path,
        wacc,
        scenario_name=scenario,
        custom_attributes=custom_attributes,
        typemap=TYPEMAP,
        moo=moo,
        dash_app=True,
    )


print("done")
