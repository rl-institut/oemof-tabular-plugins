import os

import pandas as pd
from oemof.solph import EnergySystem, Model
from oemof.solph import processing
from oemof.solph.processing import parameter_as_dict

# TODO this should be with from oemof.tabular.datapackage import building when https://github.com/oemof/oemof-tabular/pull/173 is merged
from oemof_tabular_plugins.datapackage import building as otp_building

# ---- imports to be used when the package has been installed ----
from oemof.tabular import datapackage  # noqa
from oemof.tabular.facades import TYPEMAP

# ---- imports from oemof-tabular-plugins package ----
from oemof_tabular_plugins.general import (
    post_processing,
    CONSTRAINT_TYPE_MAP,
    pre_processing,
    logger,
)

from oemof_tabular_plugins.wefe import CONSTRAINT_TYPE_MAP as WEFE_CONSTRAINT_TYPE_MAP

CONSTRAINT_TYPE_MAP.update(WEFE_CONSTRAINT_TYPE_MAP)


def compute_scenario(
    scenario_dir,
    results_path,
    wacc,
    scenario_name=None,
    custom_attributes=None,
    typemap=None,
    moo=False,
    dash_app=False,
    parameters_units=None,
    infer_bus_carrier=True,
    skip_preprocessing=False,
    skip_infer_datapackage_metadata=False,
    save_raw_results=True,
):
    """

    Parameters
    ----------
    scenario_dir
    results_path
    wacc
    scenario_name
    custom_attributes
    typemap: default to oemof.tabular.facades.TYPEMAP
    moo
    skip_preprocessing: bool (opt)
        If True, the pre-processing to update input csv files based on cost parameters: CAPEX, OPEX fix, lifetime, WACC will not take place
        Default: False
    skip_infer_datapackage_metadata: bool (opt)
        If True, the file datapackage.json will not be updated with the possible changes in elements/*.csv and sequences/*.csv
        Default: False

    Returns
    -------

    """
    if scenario_name is None:
        scenario_name = os.path.basename(scenario_dir)

    if custom_attributes is None:
        custom_attributes = []

    if typemap is None:
        typemap = TYPEMAP

    # create results directory if it doesn't already exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if skip_preprocessing is False:
        # pre-processing to update input csv files based on cost parameters: CAPEX, OPEX fix, lifetime, WACC
        pre_processing(scenario_dir, wacc, custom_attributes, moo)

    if skip_infer_datapackage_metadata is False:
        otp_building.infer_metadata_from_data(
            package_name=scenario_name,
            path=scenario_dir,
            typemap=typemap,
        )

    # create energy system object from the datapackage
    es = EnergySystem.from_datapackage(
        os.path.join(scenario_dir, "datapackage.json"),
        attributemap={},
        typemap=typemap,
    )

    logger.info("Energy system created from datapackage")

    from oemof_visio import ESGraphRenderer

    gr = ESGraphRenderer(
        energy_system=es, filepath=os.path.join(results_path, scenario_name)
    )
    gr.render()

    # create model from energy system (this is just oemof.solph)
    m = Model(es)
    logger.info("Model created from energy system")
    # mimo = [n for n in es.nodes if "mimo" in n.label]
    # print([l.label for l in mimo])
    #
    # crop = mimo[0]
    # print("Investments on inputs")
    # print([f"{i.label}({str(f.investment)})" for i,f in crop.inputs.items()])
    # print("Nominal values on inputs")
    # print([f"{i.label}({str(f.nominal_value)})" for i,f in crop.inputs.items()])
    #
    # print("Investments on outputs")
    # print([f"{i.label}({str(f.investment)})" for i,f in crop.outputs.items()])
    # print("Nominal values on outputs")
    # print([f"{i.label}({str(f.nominal_value)})" for i,f in crop.outputs.items()])

    # add constraints from datapackage to the model
    m.add_constraints_from_datapackage(
        os.path.join(scenario_dir, "datapackage.json"),
        constraint_type_map=CONSTRAINT_TYPE_MAP,
    )
    logger.info("Constraints added to model")

    # if you want dual variables / shadow prices uncomment line below
    # m.receive_duals()

    # select solver 'gurobi', 'cplex', 'glpk' etc
    m.solve("cbc")

    # extract parameters and results
    try:
        params = parameter_as_dict(m.es)
    except ValueError:
        params = None
    es.results = processing.results(m)
    if save_raw_results is True:
        es.dump(dpath=results_path, filename="oemof_raw")

    return post_processing(
        params,
        es,
        results_path,
        dp_path=os.path.join(scenario_dir, "datapackage.json"),
        dash_app=dash_app,
        parameters_units=parameters_units,
        infer_bus_carrier=infer_bus_carrier,
    )


def display_scenario_results(
    scenario_dir,
    results_path,
    dash_app=True,
    parameters_units=None,
):
    """

    Parameters
    ----------
    scenario_dir
    results_path
    Returns
    -------

    """

    # create results directory if it doesn't already exist
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    oemof_dump_file = os.path.join(results_path, "oemof_raw")

    if os.path.exists(oemof_dump_file):
        es = EnergySystem()
        es.restore(dpath=results_path, filename="oemof_raw")

        params = None

        calculator = post_processing(
            params,
            es,
            results_path,
            dp_path=os.path.join(scenario_dir, "datapackage.json"),
            dash_app=dash_app,
            parameters_units=parameters_units,
            infer_bus_carrier=True,
        )
        return calculator
    else:
        raise FileNotFoundError(
            f"The file {oemof_dump_file} could not be found, you need to set the argument 'save_raw_results' to True when using the function 'compute_scenario'"
        )
