import os
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

    # create model from energy system (this is just oemof.solph)
    m = Model(es)
    logger.info("Model created from energy system")

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
    params = parameter_as_dict(es)
    es.results = processing.results(m)

    return post_processing(
        params,
        es,
        results_path,
        dp_path=os.path.join(scenario_dir, "datapackage.json"),
        dash_app=dash_app,
        parameters_units=parameters_units,
        infer_bus_carrier=infer_bus_carrier,
    )
