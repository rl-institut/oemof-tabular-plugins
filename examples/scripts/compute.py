import os
import pandas as pd
from oemof.solph import EnergySystem, Model
from oemof.solph.processing import parameter_as_dict

# ---- imports to be used when the package has been installed ----
from oemof.tabular import datapackage # noqa
from oemof.tabular.facades import TYPEMAP
# ---- imports from oemof-tabular-plugins package ----
# ToDo: adapt the way these imports are called once oemof-tabular-plugins has expanded a bit
from src.oemof_tabular_plugins.general.post_processing import post_processing
from src.oemof_tabular_plugins.general.constraint_facades import CONSTRAINT_TYPE_MAP
from src.oemof_tabular_plugins.general.pre_processing import pre_processing
from src.oemof_tabular_plugins.wefe.facades.pv_panel import PVPanel

# get the project directory by navigating up one level from the current script file
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# list of scenarios to be evaluated - manually updated by user!
scenarios = ["test_wefe_pvpanel"]
# weighted average cost of capital (WACC) - might move later
#wacc = 0.06
# add PV Panel (from oemof-tabular-plugins) to facades type map (from oemof-tabular) - might move later
TYPEMAP["pv-panel"] = PVPanel

for scenario in scenarios:
    print("Running scenario with datapackage {}".format(scenario))
    # set paths for scenario and result directories
    scenario_dir = os.path.join(project_dir, "scenarios", scenario)
    results_path = os.path.join(project_dir, "results", scenario, "output")
    # create results directory if it doesn't already exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # pre-processing to update input csv files based on
    #pre_processing(scenario_dir, wacc)

    # create energy system object from the datapackage
    es = EnergySystem.from_datapackage(
        os.path.join(scenario_dir, "datapackage.json"),
        attributemap={},
        typemap=TYPEMAP,
    )

    # create model from energy system (this is just oemof.solph)
    m = Model(es)

    # add constraints from datapackage to the model
    m.add_constraints_from_datapackage(
        os.path.join(scenario_dir, "datapackage.json"),
        constraint_type_map=CONSTRAINT_TYPE_MAP,
    )

    # if you want dual variables / shadow prices uncomment line below
    # m.receive_duals()

    # select solver 'gurobi', 'cplex', 'glpk' etc
    m.solve("cbc")

    # extract parameters and results
    params = parameter_as_dict(es)
    results = m.results()

    all_scalars = post_processing(params, results, results_path)


print('done')