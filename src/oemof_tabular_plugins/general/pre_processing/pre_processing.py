import os
import pandas as pd
import logging
from oemof.tools import economics
import json
import datapackage as dp
import tableschema
from decimal import Decimal
from copy import deepcopy
from .pre_processing_moo import pre_processing_moo, get_moo_timeseries

logger = logging.getLogger(__name__)


def scenario_datapackage(scenario_dir):
    dp_json = os.path.join(scenario_dir, "datapackage.json")
    if os.path.exists(dp_json):
        answer = dp.Package(dp_json)
    else:
        answer = dp.Package(base_path=scenario_dir)
    return answer


def calculate_annuity(capex, opex_fix, lifetime, wacc):
    """
    Calculates the total annuity for each component, including CAPEX and fixed OPEX.
    :param capex: CAPEX (currency/MW*) *or the unit you choose to use throughout the model e.g. kW/GW
    :param opex_fix: fixed OPEX (currency/MW*/year)
    :param lifetime: lifetime of the component (years)
    :param wacc: weighted average cost of capital (WACC) applied throughout the model (%)
    :return: total annuity (currency/MW*/year)
    """
    annuity_capex = economics.annuity(capex, lifetime, wacc)
    annuity_opex_fix = opex_fix
    annuity_total = round(annuity_capex + annuity_opex_fix, 2)
    return annuity_total


def pre_processing_costs(wacc, element, element_path, element_df):
    """
    Applies pre-processing costs to the input CSV files, where the annuity ('capacity_cost') is either
    used directly if stated, or if left empty then calculated using the calculate_annuity function,
    or if all parameters are stated a choice is given.
    :param wacc: weighted average cost of capital (WACC) applied throughout the model (%)
    :param element: csv filename
    :param element_path: path of the csv file
    :param element_df: dataframe containing data from the csv file
    """
    # Every element that has the column 'storage_capacity_cost' is assumed to only contain components of type storage
    # where 'storage_capacity_cost' represents the cost parameter to be calculated.
    # All other elements require 'capacity_cost' as the cost parameter to be calculated.
    if "storage_capacity_cost" in element_df.columns:
        annuity_cost = "storage_capacity_cost"
    else:
        annuity_cost = "capacity_cost"

    # For consistency, 'annuity' (raw, economic annuity) is defined separately:
    # It will be the same as 'capacity_cost' for cost-optimization.
    # It will be different for MOO, as 'capacity_cost' takes additional weight factors into account
    annuity_cost_raw = "annuity"

    # Reset capacity_cost for rows that have cost parameters
    # removing potential artefacts of MOO runs and forcing recalculation
    if annuity_cost in element_df.columns:
        for index, row in element_df.iterrows():
            # Check if this row has the cost parameters to recalculate
            has_params = all([
                'capex' in element_df.columns and pd.notna(row.get('capex')),
                'opex_fix' in element_df.columns and pd.notna(row.get('opex_fix')),
                'lifetime' in element_df.columns and pd.notna(row.get('lifetime'))
            ])
            if has_params:
                # Clear capacity_cost to force recalculation
                element_df.at[index, annuity_cost] = None
        logger.info(f"Cleared '{annuity_cost}' for components with cost parameters in '{element}'")

    # check if any of the required columns are missing
    cost_columns = {"capex", "opex_fix", "lifetime"}
    missing_columns = cost_columns - set(element_df.columns)

    # ---------------- POSSIBLE SCENARIO GROUPS FOR PARAMETER ENTRIES ----------------
    # scenario group "annuity no cost params": annuity parameter is included and all of capex, opex fix
    # and lifetime parameters are not included in the csv file
    if annuity_cost in element_df.columns and cost_columns == missing_columns:
        scenario_group = "annuity no cost params"
    # scenario group "annuity partial cost params": annuity parameter is included and some but not all of
    # capex, opex fix and lifetime parameters are included in the csv file
    elif (
        annuity_cost in element_df.columns
        and missing_columns
        and cost_columns != missing_columns
    ):
        scenario_group = "annuity partial cost params"
    # scenario group "annuity all cost params": annuity parameter is included and all of capex, opex fix
    # and lifetime parameters are included in the csv file
    elif annuity_cost in element_df.columns and not missing_columns:
        scenario_group = "annuity all cost params"
    # scenario group "no annuity partial/all cost params": annuity parameter is not included and at least one
    # of capex, opex fix and lifetime parameters are included in the csv file
    elif annuity_cost not in element_df.columns and cost_columns != missing_columns:
        scenario_group = "no annuity partial/all cost params"
    # scenario group "no annuity no cost params": the annuity parameter is not included and neither are
    # capex, opex fix and lifetime parameters in the csv file
    elif annuity_cost not in element_df.columns and cost_columns == missing_columns:
        scenario_group = "no annuity no cost params"

    # ---------------- POSSIBLE SCENARIOS FOR EACH SCENARIO GROUP ----------------
    # loop through each entry in the csv file
    for index, row in element_df.iterrows():
        # define the row name
        row_name = row["name"]
        if scenario_group == "annuity no cost params":
            # scenario "annuity empty no cost params": the annuity parameter is left empty and the other cost
            # parameters have not been included
            if pd.isna(row[annuity_cost]):
                scenario = "annuity empty no cost params"
            # scenario "annuity defined no cost params": the annuity parameter is defined and the other cost
            # parameters have not been included
            else:
                scenario = "annuity defined no cost params"
        elif scenario_group == "annuity partial cost params":
            # scenario "annuity empty partial cost params": the annuity parameter is left empty and only
            # some other financial parameters are included
            if pd.isna(row[annuity_cost]):
                scenario = "annuity empty partial cost params"
            # scenario "annuity defined partial cost params": the annuity parameter is defined and only
            # some other financial parameters are included
            else:
                scenario = "annuity defined partial cost params"
        elif scenario_group == "annuity all cost params":
            # store the parameters
            capex = row["capex"]
            opex_fix = row["opex_fix"]
            lifetime = row["lifetime"]
            # scenario "annuity empty all cost params": the annuity parameter is left empty and all of the
            # other financial parameters are defined
            if (
                pd.isna(row[annuity_cost])
                and pd.notna(capex)
                and pd.notna(opex_fix)
                and pd.notna(lifetime)
            ):
                scenario = "annuity empty all cost params defined"
            # scenario "annuity all cost params some empty": the annuity parameter is either defined or empty,
            # but at least one of 'capex', 'opex_fix' and 'lifetime' is left empty
            elif pd.isna(capex) or pd.isna(opex_fix) or pd.isna(lifetime):
                scenario = "annuity all cost params some empty"
            # scenario "annuity defined all cost params defined": both the annuity parameter is defined and
            # all the other financial parameters are defined
            else:
                scenario = "annuity defined all cost params defined"
        elif scenario_group == "no annuity partial/all cost params":
            # store the parameters
            capex = row["capex"]
            opex_fix = row["opex_fix"]
            lifetime = row["lifetime"]
            # scenario "no annuity partial/all cost params empty": at least one of 'capex', 'opex_fix' and
            # 'lifetime' is left empty
            if pd.isna(capex) or pd.isna(opex_fix) or pd.isna(lifetime):
                scenario = "no annuity partial/all cost params empty"
            # scenario "no annuity all cost params defined": all financial parameters are defined
            else:
                scenario = "no annuity all cost params defined"
        elif scenario_group == "no annuity no cost params":
            # scenario "no annuity no cost params": neither the annuity or financial parameters are defined
            scenario = "no annuity no cost params"

        # ---------------- ACTIONS TAKEN FOR EACH SCENARIO ----------------
        if scenario == "annuity empty no cost params":
            # raise value error
            raise ValueError(
                f"'{annuity_cost}' (the annuity) has been left empty for '{row_name}' "
                f"in '{element}', and 'capex', 'opex_fix' and 'lifetime' have not "
                f" been included. \nEither the annuity ('{annuity_cost}') must be "
                f"directly stated or all of the other financial parameters must be stated "
                f"to calculate the annuity."
            )
        elif scenario == "annuity defined no cost params":
            # log info message
            logger.info(
                f"The {annuity_cost} is directly used for '{row_name}' in '{element}'."
            )
        elif scenario == "annuity empty partial cost params":
            # raise value error
            raise ValueError(
                f"'{annuity_cost}' (the annuity) has been left empty for '{row_name}' "
                f"in '{element}', and not all of 'capex', 'opex_fix' and 'lifetime' have"
                f" been included. \nEither the annuity ('{annuity_cost}') must be "
                f"directly stated or all of the other financial parameters must be stated "
                f"to calculate the annuity."
            )
        elif scenario == "annuity defined partial cost params":
            # log warning message
            logging.warning(
                f"'{annuity_cost}' (the annuity) has been defined and some but not all "
                f"of 'capex', 'opex_fix' and 'lifetime' have been defined for {row_name} "
                f"in {element}. The annuity will be directly used but be aware that some "
                f"cost results will not be calculated."
            )
        elif scenario == "annuity empty all cost params defined":
            # calculate the annuity using the calculate_annuity function
            capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
            # update the dataframe
            element_df.at[index, annuity_cost] = float(capacity_cost)
            element_df.at[index, annuity_cost_raw] = float(capacity_cost)
            # log info message
            logger.info(
                f"the annuity ('{annuity_cost}') has been calculated and updated for"
                f" '{row_name}' in '{element}'."
            )
        elif scenario == "annuity all cost params some empty":
            # log warning message
            logging.warning(
                f"One or more of 'capex', 'opex_fix' and 'lifetime' have been left "
                f"empty for {row_name} in {element}. The annuity will be directly used "
                f"but be aware that some cost results will not be calculated."
            )
        elif scenario == "annuity defined all cost params defined":
            capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
            # update the dataframe
            element_df.at[index, annuity_cost] = float(capacity_cost)
            element_df.at[index, annuity_cost_raw] = float(capacity_cost)
            # if all parameters are defined, the user is asked if they want to calculate the annuity
            # from the capex, opex_fix and lifetime or use the annuity directly
            logger.info(
                f"All parameters ('capex', 'opex_fix', 'lifetime') and '{annuity_cost}' are "
                f"provided for '{row_name}' in '{element}'. \nThe defined annuity has been replaced with "
                f"the calculated value from capex, opex_fix and lifetime."
            )
        elif scenario == "no annuity partial/all cost params empty":
            # raise value error
            raise ValueError(
                f"One or more of 'capex', 'opex_fix' and 'lifetime' have been left "
                f"empty for {row_name} in {element}. Please enter values or remove the"
                f" parameters and include "
                f"the 'capacity_cost'."
            )
        elif scenario == "no annuity all cost params defined":
            # calculate the annuity using the calculate_annuity function
            capacity_cost = calculate_annuity(capex, opex_fix, lifetime, wacc)
            # update the dataframe
            element_df.at[index, annuity_cost] = float(capacity_cost)
            element_df.at[index, annuity_cost_raw] = float(capacity_cost)
            # log info message
            logger.info(
                f"the annuity ('{annuity_cost}') has been calculated and updated for"
                f" '{row_name}' in '{element}'."
            )
        elif scenario == "no annuity no cost params":
            logger.info(
                f"Component '{row_name}' of element '{element}' does not contain '{annuity_cost}' parameter. Skipping..."
            )
    # Reset marginal_cost to resource_cost removing potential artefacts of MOO runs
    if 'marginal_cost' in element_df.columns and 'resource_cost' in element_df.columns:
        element_df['marginal_cost'] = element_df['resource_cost']
        logger.info(f"Reset marginal_cost to resource_cost for all components in '{element}'")
    elif 'marginal_cost' in element_df.columns:
        # If resource_cost doesn't exist, set marginal_cost to 0
        element_df['marginal_cost'] = 0.0
        logger.info(f"Reset marginal_cost to 0.0 for all components in '{element}'")

    # save the updated dataframe to the csv file
    element_df.to_csv(element_path, sep=";", index=False)
    return


def pre_processing_custom_attributes(element_path, element_df, custom_attributes):
    # ToDo: confirm if this function is needed, whether the attributes need to be added to
    #  'output_parameters' or if this is not necessary
    """Updates the 'output_parameters' field in the CSV file for the specified element if custom
    attributes are defined.

    :param element_path: path of the csv file
    :param element_df: dataframe containing data from the csv file
    :param custom_attributes: list of custom attributes included in the model (defined in compute.py)
    """
    # iterate over each entry in the dataframe (from csv file)
    for index, row in element_df.iterrows():
        # create empty custom attributes dict
        custom_attributes_dict = {}
        # set boolean to false
        has_custom_attributes = False
        # check if custom_attributes is not none before iterating
        if custom_attributes is not None:
            # check if any of the custom attributes list are in the dataframe columns
            for attribute in custom_attributes:
                if attribute in element_df.columns:
                    value = row[attribute]
                    # add the attribute to the custom attributes dict
                    custom_attributes_dict[attribute] = value
                    # set boolean to true
                    has_custom_attributes = True
                # if custom attributes are found for this row, add them to 'output_parameters'
                if has_custom_attributes:
                    output_parameters_str = json.dumps(
                        {"custom_attributes": custom_attributes_dict}
                    )
                    element_df.at[index, "output_parameters"] = output_parameters_str
                else:
                    # no custom attributes found, do not update 'output_parameters'
                    continue
    # write the updated dataframe back to the csv file
    element_df.to_csv(element_path, sep=";", index=False)
    return has_custom_attributes


def moo_profiles_cleanup(scenario_dir, dp, suffix=""):
    """ Remove dynamically added timeseries from all resources in /sequences/ based on suffix"""
    for res in dp.resources:
        if "/sequences/" in res.descriptor["path"]:
            df = pd.DataFrame.from_records(res.read(keyed=True))
            fields_to_remove = [f.name for f in res.schema.fields if f.name.endswith(suffix)]

            if fields_to_remove:
                # Remove from descriptor
                res.descriptor["schema"]["fields"] = [
                    f for f in res.descriptor["schema"]["fields"] if f["name"] not in fields_to_remove
                ]
                # Remove from CSV
                df = df[[col for col in df.columns if col not in fields_to_remove]]
                if "timeindex" in df.columns:
                    df["timeindex"] = pd.to_datetime(df["timeindex"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                df.to_csv(os.path.join(scenario_dir, res.descriptor["path"]), sep=";", index=False)

                # Rebuild metadata and safe datapackage.json
                dp.remove_resource(res.name)
                dp.add_resource(res.descriptor)
                dp.commit()
                dp.save(os.path.join(scenario_dir, "datapackage.json"))


def pre_processing(scenario_dir, wacc, custom_attributes=None, moo=False, moo_wf=None):
    """Performs pre-processing of input scenario data before running the model.

    :param scenario_dir: scenario directory path
    :param wacc: weighted average cost of capital (WACC) applied throughout the model (%)
    :param custom_attributes: list of custom attributes included in the model (defined in compute.py), default is None
    :param moo: whether the multi-objective optimization is activated, default is False
    :param moo_wf: dictionary of moo weight factors
    """

    logger.info("Pre-processing activated")

    dp = scenario_datapackage(scenario_dir)

    if moo_wf is None:
        moo = False
        logger.info("No weight factors for multi-objective optimization provided")

    # Provide moo_suffix for clean up and regeneration of moo profiles
    moo_suffix = "moo_profile"
    moo_profiles_cleanup(scenario_dir, dp, suffix=moo_suffix)
    if moo is False:
        logger.info(f"Optimization activated for only costs")
    else:
        logger.info(f"Multi-objective optimization activated")
        # Create a DataFrame from the resource that has cf-aware-profile or cf_aware in it
        cf_aware_name = "cf-aware-profile"
        try:
            cf_aware_res, cf_aware_df = get_moo_timeseries(dp, ts_name=cf_aware_name)  # Unit: dimensionless
        except ValueError:
            cf_aware_name = "cf_aware"
            cf_aware_res, cf_aware_df = get_moo_timeseries(dp, ts_name=cf_aware_name)  # Unit: dimensionless
        existing_cf_fields = set(cf_aware_df.columns)
        for col in cf_aware_df.columns:
            # Convert to float
            cf_aware_df[col] = cf_aware_df[col].apply(
                lambda v: float(v) if isinstance(v, Decimal) else v
            )

    # locate the elements directory
    for res in dp.resources:
        if "/elements/" in res.descriptor["path"]:
            try:
                resource_data = pd.DataFrame.from_records(res.read(keyed=True))
            except tableschema.exceptions.CastError as err:
                if err.errors:
                    logging.error(
                        f"The resource {res.name} has the following casting errors: {','.join([str(e) for e in err.errors])}")
                else:
                    logging.error(f"The resource {res.name} has the following casting error: {err}")
                resource_data = pd.DataFrame()

            element = res.name
            element_path = os.path.join(scenario_dir, res.descriptor["path"])
            element_df = resource_data.copy()

            for col in element_df.columns:
                # Convert to float
                element_df[col] = element_df[col].apply(
                    lambda v: float(v) if isinstance(v, Decimal) else v
                )

            if moo is False:
                # performs pre-processing of additional cost data (capex, opex_fix, lifetime)
                pre_processing_costs(wacc, element, element_path, element_df)
                # cast 'marginal_cost' to number
                for f in res.descriptor["schema"]["fields"]:
                    if f["name"] == "marginal_cost":
                        f["type"] = "number"
                # remove potential foreign keys for marginal_cost
                res.descriptor["schema"]["foreignKeys"] = [
                    fk for fk in res.descriptor["schema"]["foreignKeys"] if fk["fields"] != "marginal_cost"
                ]
            else:
                # performs pre-processing of cost data while taking multiple objectives (emissions, water
                # footprint, land requirement) into account, returns cf_aware_df with new profiles
                cf_aware_df = pre_processing_moo(
                    wacc, res, element, element_path, element_df, moo_wf, moo_suffix, cf_aware_df, cf_aware_name, cf_aware_res
                )
                # cast 'marginal_cost' to string because it is a foreign key (name of a profile),
                # unless it's one of the following exceptions
                element_type = element_df["type"].iloc[0]
                if element_type not in ["bus", "load", "excess", "crop"]:
                    for f in res.descriptor["schema"]["fields"]:
                        if f["name"] == "marginal_cost":
                            f["type"] = "string"

            # performs pre-processing for custom attributes (e.g. emission factor, renewable factor, land
            # requirement)
            has_custom_attrs = pre_processing_custom_attributes(
                element_path, element_df, custom_attributes
            )
            if has_custom_attrs:
                existing_res_fields = [f.name for f in res.schema.fields]
                if "output_parameters" not in existing_res_fields:
                    # Add field to descriptor
                    res.descriptor["schema"]["fields"].append({
                        "name": "output_parameters",
                        "type": "object",
                        "format": "default"
                    })

            # update element metadata
            dp.remove_resource(res.name)
            dp.add_resource(res.descriptor)
            dp.commit()

    if moo is True:
        # get the new columns of cf_aware_df and add them as new fields to the resource metadata, save the resource as csv
        for col in cf_aware_df:
            if col not in existing_cf_fields:
                cf_aware_res.descriptor["schema"]["fields"].append({
                    "name": col,
                    "type": "number",
                    "format": "default"
                })
        if "timeindex" in cf_aware_df.columns:
            cf_aware_df["timeindex"] = pd.to_datetime(cf_aware_df["timeindex"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        cf_aware_df.to_csv(cf_aware_res.source, index=False, sep=";")

        # update metadata of the resource containing cf aware and the new moo profiles
        dp.remove_resource(cf_aware_res.name)
        dp.add_resource(cf_aware_res.descriptor)
        dp.commit()

    # save metadata
    dp.save(os.path.join(scenario_dir, "datapackage.json"))

    logger.info("Pre-processing completed")
    return
