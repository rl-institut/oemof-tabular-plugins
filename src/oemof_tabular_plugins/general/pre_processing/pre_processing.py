import os
import pandas as pd
import logging
from oemof.tools import logger, economics
import json
import datapackage as dp
import tableschema
from decimal import Decimal
from copy import deepcopy
from .pre_processing_moo import pre_processing_moo, get_moo_timeseries, moo_profiles_cleanup

logger.define_logging()


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


def pre_processing_costs(wacc, element, element_df):
    """
    Pre-process cost parameters.

    Design:
    - Normalize inputs using COLUMN_RULES
    - Compute capacity_cost deterministically:
        Case 0 - capacity_cost present and zero, no other cost params: pass (no intent)
        Case 1 - capacity_cost present and non-zero, no other cost params: set to zero and info
        Case 2a - capacity_cost present, only annuity as cost params: use annuity
        Case 2b - capacity_cost present, all cost params: use annuity and info
        Case 3 - capacity_cost present, only capex, opex_fix, lifetime as cost params: compute capacity_cost
        Case 4 - capacity_cost present, none of the above: set to zero and warn (broken intent)
    - Compute marginal_cost deterministically:
        marginal_cost present, resource_cost present: set marginal_cost to resource_cost
        only marginal_cost present: set to zero and warn
    """
    # ---------------- COLUMN RULES ----------------
    COLUMN_RULES = {
        "annuity": {"strict_positive": True},
        "capex": {"strict_positive": True},
        "lifetime": {"strict_positive": True},
        "opex_fix": {"strict_positive": False},
        "resource_cost": {"strict_positive": False},
    }

    # ---------------- NORMALIZATION ----------------
    for col, rules in COLUMN_RULES.items():
        if col not in element_df.columns:
            continue  # <-- do NOT create columns

        if rules["strict_positive"]:
            element_df[col] = element_df[col].mask(
                (element_df[col] <= 0) | (pd.isna(element_df[col])),
                pd.NA
            )
        else:
            element_df[col] = element_df[col].where(
                pd.notna(element_df[col]),
                pd.NA
            )

    # ================= CAPACITY COST =================
    if "capacity_cost" in element_df.columns:
        for index, row in element_df.iterrows():
            row_name = row["name"]
            capacity_cost = row["capacity_cost"]

            capex = row.get("capex")
            opex = row.get("opex_fix")
            lifetime = row.get("lifetime")
            annuity = row.get("annuity")

            has_annuity = pd.notna(annuity)
            has_all_cost_params = all([
                pd.notna(capex),
                pd.notna(opex),
                pd.notna(lifetime)
            ])
            has_any_cost_input = any([
                has_annuity,
                pd.notna(capex),
                pd.notna(opex),
                pd.notna(lifetime)
            ])

            # -------- CASE 0: intentional zero --------
            if capacity_cost == 0 and not has_any_cost_input:
                pass

            # -------- CASE 1: no cost inputs --------
            elif capacity_cost != 0 and not has_any_cost_input:
                logging.info(
                    f"No cost inputs for '{row_name}' in '{element}', setting capacity_cost to 0.0"
                )
                capacity_cost = 0.0

            # -------- CASE 2: annuity given --------
            elif has_annuity:
                capacity_cost = annuity

                if has_all_cost_params:
                    logging.info(
                        f"Both annuity and cost parameters provided for '{row_name}' in '{element}'. "
                        f"Using annuity."
                    )

            # -------- CASE 3: compute from params --------
            elif has_all_cost_params:
                capacity_cost = calculate_annuity(capex, opex, lifetime, wacc)

            # -------- CASE 4: broken intent --------
            else:
                logging.warning(
                    f"Incomplete cost data for '{row_name}' in '{element}', setting capacity_cost to 0.0"
                )
                capacity_cost = 0.0

            element_df.at[index, "capacity_cost"] = float(capacity_cost)

    # ================= MARGINAL COST =================
    if "marginal_cost" in element_df.columns:
        if "resource_cost" in element_df.columns:
            element_df["marginal_cost"] = element_df["resource_cost"].fillna(0.0)
        else:
            logging.warning(
                f"'marginal_cost' exists but no 'resource_cost' in '{element}', setting to 0.0"
            )
            element_df["marginal_cost"] = 0.0

    return element_df


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


def pre_processing(scenario_dir, wacc, custom_attributes=None, moo=False, moo_wf=None):
    """Performs pre-processing of input scenario data before running the model.

      :param scenario_dir: scenario directory path
      :param wacc: weighted average cost of capital (WACC) applied throughout the model (%)
      :param custom_attributes: list of custom attributes included in the model (defined in compute.py), default is None
      :param moo: whether the multi-objective optimization is activated, default is False
      :param moo_wf: dictionary of moo weight factors
      """
    logging.info("Pre-processing activated")

    dp = scenario_datapackage(scenario_dir)

    if moo and moo_wf is None:
        logging.warning("MOO activated but no weight factors provided. Deactivate MOO...")
        moo = False

    moo_suffix = "moo_profile"

    # ---------------- CLEANUP ----------------
    moo_profiles_cleanup(scenario_dir, dp, suffix=moo_suffix)

    # ---------------- MOO SETUP ----------------
    if moo:
        logging.info("Multi-objective optimization activated")

        cf_aware_name = "cf-aware-profile"
        try:
            cf_aware_res, cf_aware_df = get_moo_timeseries(dp, ts_name=cf_aware_name)
        except ValueError:
            cf_aware_name = "cf_aware"
            cf_aware_res, cf_aware_df = get_moo_timeseries(dp, ts_name=cf_aware_name)

        existing_cf_fields = set(cf_aware_df.columns)

        # normalize Decimal → float
        cf_aware_df = cf_aware_df.map(
            lambda v: float(v) if isinstance(v, Decimal) else v
        )
    else:
        logging.info("Cost-only optimization activated")

    # ---------------- PROCESS ELEMENTS ----------------
    updated_resources = []

    for res in dp.resources:
        if "/elements/" not in res.descriptor["path"]:
            continue

        element = res.name
        element_path = os.path.join(scenario_dir, res.descriptor["path"])

        try:
            element_df = pd.DataFrame.from_records(res.read(keyed=True))
        except Exception as err:
            logging.error(f"Error reading {res.name}: {err}")
            element_df = pd.DataFrame()

        # normalize Decimal → float
        element_df = element_df.map(
            lambda v: float(v) if isinstance(v, Decimal) else v
        )

        # -------- ALWAYS run cost preprocessing --------
        element_df = pre_processing_costs(wacc, element, element_df)

        # default: marginal_cost numeric
        for f in res.descriptor["schema"]["fields"]:
            if f["name"] == "marginal_cost":
                f["type"] = "number"

        # remove FK by default
        res.descriptor["schema"]["foreignKeys"] = [
            fk for fk in res.descriptor["schema"]["foreignKeys"]
            if fk["fields"] != "marginal_cost"
        ]

        # -------- MOO --------
        if moo:
            cf_aware_df, profiles_created = pre_processing_moo(
                wacc, res, element, element_path, element_df,
                moo_wf, moo_suffix, cf_aware_df, cf_aware_name, cf_aware_res
            )

            if profiles_created:
                for f in res.descriptor["schema"]["fields"]:
                    if f["name"] == "marginal_cost":
                        f["type"] = "string"

        # -------- CUSTOM ATTRIBUTES --------
        has_custom_attrs = pre_processing_custom_attributes(
            element_path, element_df, custom_attributes
        )

        if has_custom_attrs:
            existing_fields = [f["name"] for f in res.descriptor["schema"]["fields"]]
            if "output_parameters" not in existing_fields:
                res.descriptor["schema"]["fields"].append({
                    "name": "output_parameters",
                    "type": "object",
                    "format": "default"
                })

        # -------- WRITE CSV --------
        element_df.to_csv(element_path, sep=";", index=False)

        updated_resources.append(res)

    # ---------------- COMMIT ELEMENT METADATA ----------------
    for res in updated_resources:
        dp.remove_resource(res.name)
        dp.add_resource(res.descriptor)

    dp.commit()

    # ---------------- WRITE MOO TIMESERIES ----------------
    if moo:
        for col in cf_aware_df.columns:
            if col not in existing_cf_fields:
                cf_aware_res.descriptor["schema"]["fields"].append({
                    "name": col,
                    "type": "number",
                    "format": "default"
                })

        if "timeindex" in cf_aware_df.columns:
            cf_aware_df["timeindex"] = pd.to_datetime(
                cf_aware_df["timeindex"]
            ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        cf_aware_df.to_csv(cf_aware_res.source, index=False, sep=";")

        dp.remove_resource(cf_aware_res.name)
        dp.add_resource(cf_aware_res.descriptor)
        dp.commit()

    # ---------------- FINAL SAVE ----------------
    dp.save(os.path.join(scenario_dir, "datapackage.json"))

    logging.info("Pre-processing completed")
