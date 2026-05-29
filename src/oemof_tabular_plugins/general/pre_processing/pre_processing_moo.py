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
    """Return (resource, dataframe) for a timeseries containing ts_name column."""

    matches = []

    # --- only check resources in /sequences/ ---
    for res in dp.resources:
        if "/sequences/" not in res.descriptor["path"]:
            continue

        # --- check metadata first (avoid unnecessary reads) ---
        field_names = [f.name for f in res.schema.fields]
        if ts_name not in field_names:
            continue

        # --- read the resource if it has ts_name and write to df
        try:
            df = pd.DataFrame.from_records(res.read(keyed=True))
        except tableschema.exceptions.CastError as err:
            if err.errors:
                logging.error(
                    f"{res.name} casting errors: {','.join(map(str, err.errors))}"
                )
            else:
                logging.error(f"{res.name} casting error: {err}")
            df = pd.DataFrame()

        matches.append((res, df))

    if len(matches) == 0:
        raise ValueError(f"No timeseries with column '{ts_name}' found")

    if len(matches) > 1:
        logging.warning(f"Multiple timeseries with column '{ts_name}' found, first match will be used")

    return matches[0]


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
    """
    Apply multi-objective optimization (MOO) preprocessing to an element's cost and flow parameters.

    This function is intended to run **after standard cost preprocessing**. If MOO is active,
    it recalculates the element's cost and flow indicators based on multiple objectives, overriding
    the standard capacity_cost and marginal_cost values.

    The MOO calculation includes:
        - Capital and fixed costs (capacity_cost or storage_capacity_cost)
        - Greenhouse gas emissions
        - Land requirement
        - Water consumption (direct + indirect)

    The MOO indicators are normalized using global values and optionally scaled by a large factor
    (default 1e15) to avoid numerical underflow in optimization.

    Parameters
    ----------
    wacc : float
        Weighted average cost of capital (WACC) [%] used for annuity calculations if needed.
    res : DataPackage Resource
        The resource object corresponding to the element being preprocessed.
    element : str
        Name of the element (e.g., 'hydropower', 'pv-panel').
    element_path : str
        Path to the CSV file of the element; updated values are written here.
    element_df : pd.DataFrame
        DataFrame containing the element's data.
    moo_wf : dict
        Dictionary with weights for the multi-objective optimization goals:
            - 'wf_cost': weight for cost
            - 'wf_ghg': weight for greenhouse gas emissions
            - 'wf_lr': weight for land requirement
            - 'wf_wf': weight for water footprint
        The weights must sum to 1.0.
    moo_suffix : str
        Suffix used for naming new MOO profile columns in timeseries (e.g., '_moo_profile').
    cf_aware_df : pd.DataFrame
        DataFrame containing the CF-aware timeseries for water consumption normalization.
    cf_aware_name : str
        Column name in cf_aware_df corresponding to the CF-aware factor.
    cf_aware_res : DataPackage Resource
        Resource corresponding to cf_aware_df; used to add foreign keys if variable profiles are created.

    Returns
    -------
    cf_aware_df : pd.DataFrame
        Updated CF-aware DataFrame including any newly generated MOO profiles.
    profiles_created : bool
        True if at least one variable timeseries profile was created; False if all profiles were constant.

    Notes
    -----
    - This function **overwrites** capacity_cost and marginal_cost values if MOO is active.
    - If 'capacity_cost' is missing or zero, it is handled silently; no warnings are issued for intentional zero values.
    - Flow-related values (marginal_cost, ghg_emission_factor, water_direct, indirect_water_consumption_factor)
      missing or NaN are treated as zero in the MOO calculation.
    - All MOO indicators are multiplied by a normalization factor (default 1e15) to bring them to a comparable scale.
    - Variable timeseries are linked via foreign keys to cf_aware_res if applicable.
    - The function updates the element CSV in-place.
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

    # ---------------- GLOBAL INPUTS FOR NORMALIZATION ----------------
    global_GDP = 1.10e14    # forecasted for 2024, Unit: [USD/a],
    # Source: IMF (2024) https://www.imf.org/en/Publications/WEO/weo-database/2024/April/weo-report?c=512,914,612,171,614,311,213,911,314,193,122,912,313,419,513,316,913,124,339,638,514,218,963,616,223,516,918,748,618,624,522,622,156,626,628,228,924,233,632,636,634,238,662,960,423,935,128,611,321,243,248,469,253,642,643,939,734,644,819,172,132,646,648,915,134,652,174,328,258,656,654,336,263,268,532,944,176,534,536,429,433,178,436,136,343,158,439,916,664,826,542,967,443,917,544,941,446,666,668,672,946,137,546,674,676,548,556,678,181,867,682,684,273,868,921,948,943,686,688,518,728,836,558,138,196,278,692,694,962,142,449,564,565,283,853,288,293,566,964,182,359,453,968,922,714,862,135,716,456,722,942,718,724,576,936,961,813,726,199,733,184,524,361,362,364,732,366,144,146,463,528,923,738,578,537,742,866,369,744,186,925,869,746,926,466,112,111,298,927,846,299,582,487,474,754,698,&s=NGDPD,&sy=2022&ey=2029&ssm=0&scsm=1&scc=0&ssd=1&ssc=0&sic=0&sort=country&ds=.&br=1

    global_GHG = 3.74e14    # global CO2 emission in 2023; [kgCO2/a],
    # Source: iea (2024) https://www.iea.org/reports/co2-emissions-in-2023/executive-summary
    global_land_surface = 1.49e14   # Unit: m²
    global_annual_deprived_water = 7.91e13  # Unit: [m³/a], Source: EU JRC (2017)
    # https://data.europa.eu/doi/10.2760/88930
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
        capacity_cost = to_float_safe(row, moo_variable_fix)
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
                    ) * normalization_factor

            if moo_flow is not None and not np.isnan(moo_flow).any():
                ts_header = f"{row_name}_{moo_suffix}"
                cf_aware_temp_df[ts_header] = moo_flow
                row_temp_dict[index] = ts_header

        if not has_capacity and not has_flow_inputs:
            logging.info(f"MOO: Skipping '{row_name}' in '{element}'")

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
