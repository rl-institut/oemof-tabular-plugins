# from .pre_processing import calculate_annuity
from oemof.tools import logger, economics
import os
import numpy as np
from datapackage import Package
import pandas as pd
import logging
import csv
from decimal import Decimal
import tableschema

# from .pre_processing import calculate_annuity

NO_MOO_VARIABLE_SCEN = "no moo variables"  # this scenario potentially can be skipped; as this would correspond with SOO
MOO_VARIABLE_SCEN = "moo variable calculation"
MOO_DISPATCHABLE_SCEN = "moo variable calculation with dispatchable"

MOO_ANNUITY = "annuity"

def moo_profiles_cleanup(scenario_dir, dp, suffix=""):
    """Remove dynamically added timeseries (by suffix) from /sequences/ resources."""

    modified_resources = []

    # --- only check resources in /sequences/ ---
    for res in dp.resources:
        if "/sequences/" not in res.descriptor["path"]:
            continue

        # --- check metadata first (avoid unnecessary reads) ---
        fields_to_remove = [
            f.name for f in res.schema.fields if f.name.endswith(suffix)
        ]
        if not fields_to_remove:
            continue

        # --- read only if needed ---
        df = pd.DataFrame.from_records(res.read(keyed=True))

        # --- drop columns ---
        df = df.drop(columns=fields_to_remove, errors="ignore")

        # --- normalize timeindex ---
        if "timeindex" in df.columns:
            df["timeindex"] = pd.to_datetime(df["timeindex"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # --- write CSV ---
        df.to_csv(os.path.join(scenario_dir, res.descriptor["path"]), sep=";", index=False)

        # --- update descriptor ---
        res.descriptor["schema"]["fields"] = [
            f for f in res.descriptor["schema"]["fields"]
            if f["name"] not in fields_to_remove
        ]

        modified_resources.append(res)

    # --- commit once ---
    if modified_resources:
        for res in modified_resources:
            dp.remove_resource(res.name)
            dp.add_resource(res.descriptor)

        dp.commit()


def get_moo_timeseries(dp, ts_name=""):
    """ """
    for res in dp.resources:
        if "/sequences/" in res.descriptor["path"]:
            field_names = [f.name for f in res.schema.fields]
            if ts_name in field_names:
                try:
                    df = pd.DataFrame.from_records(res.read(keyed=True))
                except tableschema.exceptions.CastError as err:
                    if err.errors:
                        logging.error(
                            f"The resource {res.name} has the following casting errors: "
                            f"{','.join([str(e) for e in err.errors])}"
                        )
                    else:
                        logging.error(f"The resource {res.name} has the following casting error: {err}")
                    df = pd.DataFrame()

                return res, df

    #TODO: dp.descriptor['name'] not present yet for ScenarioBuilder scenarios
    dp_name = os.path.basename(os.path.normpath(dp.base_path))
    raise ValueError(
        f"'{ts_name}' could not be found in any file under '/sequences/' of datapackage '{dp_name}'"
    )


def pre_processing_moo(
    wacc,
    res,
    element,
    element_path,
    element_df,
    moo_wf,
    moo_suffix,
    cf_aware_df,
    cf_aware_name,
    cf_aware_res
):
    """This function will run the multi-objective optimization

    The outcome is that the main costs 'capacity_cost' will be replaced by an aggregated
    indicator representing the multi-objective optimization goals.
    This function will replace the pre_processing_costs function if moo is set to True

    Inputs
    Weights: defined by model-user e.g. percentages up to 1
        (in OptiMG, the user defining this will be e.g. local prosumers)
    0.5 for costs 0.2 for emissions 0.2 for land requirement 0.1 for water dissipated
    Has to add up to 1, otherwise error

    Normalization: done with global values
    User sets costs are defined in the CSV e.g. CAPEX, OPEX fix, lifetime
        -> this cost is normalised based on global GDP
        -> value is calculated for proportion of cost to global GDP
    Same applies for total emissions
        -> the value is normalised based on total global annual GHG emissions
    Land requirements
        ->  the value is normalised based on the worlds surface area
    Water dissipated
        -> the value is normalised based on e.g. global availability
    Then these values are added together

    User includes specific cost, specific emission factor, specific land requirement, specific water footprint
    This function will take those, normalise based on normalisation data (global)
    One aggregated value will be calculated based on adding these values
    This value will be entered in the csv file under 'capacity_cost'
    The csv file will be updated

    Applies pre-processing costs to the input CSV files, where the annuity ('capacity_cost') is either
    used directly if stated, or if left empty then calculated using the calculate_annuity function,
    or if all parameters are stated a choice is given.
    :param wacc: weighted average cost of capital (WACC) applied throughout the model (%)
    :param element: csv filename
    :param element_path: path of the csv file
    :param element_df: dataframe containing data from the csv file
    :param scenario_dir: scenario directory path
    """

    # ---------------- SAFE ----------------
    def safe(val):
        if val is None:
            return 0
        try:
            return 0 if np.isnan(val) else val
        except TypeError:
            return val

    def to_float_safe(row, col):
        if col not in row or row[col] is None:
            return None
        try:
            val = float(row[col])
            return val if not np.isnan(val) else None
        except (TypeError, ValueError):
            return None

    # ---------------- WEIGHTS ----------------
    wf_cost = moo_wf["wf_cost"]
    wf_ghg = moo_wf["wf_ghg"]
    wf_lr = moo_wf["wf_lr"]
    wf_wf = moo_wf["wf_wf"]

    # TODO: is this safehandling necessary/sufficient?
    if not np.isclose(wf_cost + wf_ghg + wf_lr + wf_wf, 1.0):
        raise ValueError("MOO weights must sum to 1")

    # ---------------- GLOBALS ----------------
    global_GDP = 1.10e14
    global_GHG = 3.74e14
    global_land_surface = 1.49e14
    global_annual_deprived_water = 7.91e13
    normalization_factor = 1e15

    cf_aware = cf_aware_df[cf_aware_name]

    # ---------------- OUTPUT COLS ----------------
    moo_variable_var = "marginal_cost"
    moo_variable_fix = (
        "storage_capacity_cost"
        if "storage_capacity_cost" in element_df.columns
        else "capacity_cost"
    )

    # ---------------- TEMP STORAGE ----------------
    cf_aware_temp_df = pd.DataFrame(index=cf_aware_df.index)
    row_temp_dict = {}

    # ---------------- LOOP ----------------
    for index, row in element_df.iterrows():
        row_name = row["name"]

        # ---- inputs ----
        capacity_cost = to_float_safe(row, "capacity_cost")
        marginal_cost = to_float_safe(row, "marginal_cost")

        ghg = to_float_safe(row, "ghg_emission_factor")
        land = to_float_safe(row, "land_requirement_factor")
        water_direct = to_float_safe(row, "water_direct")
        water_indirect = to_float_safe(row, "indirect_water_consumption_factor")

        has_capacity = capacity_cost is not None
        has_flow_inputs = any(
            v is not None for v in [marginal_cost, ghg, water_direct, water_indirect]
        )

        # -------- CAPACITY MOO --------
        if has_capacity:
            moo_capacity = (
                           capacity_cost / global_GDP * wf_cost
                           + safe(land) / global_land_surface * wf_lr
                           ) * normalization_factor

            if not np.isnan(moo_capacity):
                element_df.at[index, moo_variable_fix] = float(moo_capacity)

        # -------- FLOW MOO (UNIFIED) --------
        if has_flow_inputs:
            water_total = safe(water_direct) + safe(water_indirect)

            moo_flow = (
                    safe(marginal_cost) / global_GDP * wf_cost
                    + safe(ghg) / global_GHG * wf_ghg
                    + cf_aware * water_total / global_annual_deprived_water * wf_wf
                    ) + normalization_factor

            if moo_flow is not None and not np.isnan(moo_flow).any():
                ts_header = f"{row_name}_{moo_suffix}"
                cf_aware_temp_df[ts_header] = moo_flow
                row_temp_dict[index] = ts_header

        if not has_capacity and not has_flow_inputs:
            logging.info(f"Skipping '{row_name}' in '{element}'")

    # ---------------- CHECK VARIABILITY ----------------
    any_variable = any(
        len(cf_aware_temp_df[col].unique()) > 1
        for col in cf_aware_temp_df.columns
    )

    if any_variable:
        # ---- write timeseries ----
        for col in cf_aware_temp_df.columns:
            cf_aware_df[col] = cf_aware_temp_df[col]

        for index, col in row_temp_dict.items():
            element_df.at[index, moo_variable_var] = col

        # ---- FK ----
        fk_exists = any(
            fk.get("fields") == moo_variable_var and
            fk.get("reference", {}).get("resource") == cf_aware_res.name
            for fk in res.descriptor["schema"]["foreignKeys"]
        )

        if not fk_exists:
            res.descriptor["schema"]["foreignKeys"].append({
                "fields": moo_variable_var,
                "reference": {"resource": cf_aware_res.name}
            })

        profiles_created = True

    else:
        # ---- write constants ----
        for index, col in row_temp_dict.items():
            const_val = float(cf_aware_temp_df[col].iloc[0])
            element_df.at[index, moo_variable_var] = const_val

        profiles_created = False

    # ---------------- SAVE ----------------
    element_df.to_csv(element_path, sep=";", index=False)

    return cf_aware_df, profiles_created
