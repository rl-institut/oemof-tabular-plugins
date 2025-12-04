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


def to_float(row, key):
    # Key missing
    if key not in row:
        logging.warning(f"Row {row.name}: Missing key '{key}'. Setting to '0.0'.")
        return 0.0

    value = row[key]

    # Value present but not convertible
    try:
        return float(value)
    except (TypeError, ValueError):
        logging.warning(f"Row {row.name}: Could not convert '{key}'='{value}' to float. Setting to '0.0'.")
        return 0.0


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
    annuity = round(annuity_capex + annuity_opex_fix, 2)
    return annuity


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


def pre_processing_moo(wacc, res, element, element_path, element_df, moo_wf, moo_suffix, cf_aware_df, cf_aware_name, cf_aware_res):
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
    # ---------------- MOO Normalization PARAMS ----------------
    # Global Inputs (used for normalization)
    global_GDP = 1.10 * 10**14  # forecasted for 2024, Unit: [USD/a], Source: IMF (2024)
    # https://www.imf.org/en/Publications/WEO/weo-database/2024/April/weo-report?c=512,914,612,171,614,311,213,911,314,193,122,912,313,419,513,316,913,124,339,638,514,218,963,616,223,516,918,748,618,624,522,622,156,626,628,228,924,233,632,636,634,238,662,960,423,935,128,611,321,243,248,469,253,642,643,939,734,644,819,172,132,646,648,915,134,652,174,328,258,656,654,336,263,268,532,944,176,534,536,429,433,178,436,136,343,158,439,916,664,826,542,967,443,917,544,941,446,666,668,672,946,137,546,674,676,548,556,678,181,867,682,684,273,868,921,948,943,686,688,518,728,836,558,138,196,278,692,694,962,142,449,564,565,283,853,288,293,566,964,182,359,453,968,922,714,862,135,716,456,722,942,718,724,576,936,961,813,726,199,733,184,524,361,362,364,732,366,144,146,463,528,923,738,578,537,742,866,369,744,186,925,869,746,926,466,112,111,298,927,846,299,582,487,474,754,698,&s=NGDPD,&sy=2022&ey=2029&ssm=0&scsm=1&scc=0&ssd=1&ssc=0&sic=0&sort=country&ds=.&br=1
    global_GHG = (
        3.74 * 10**14
    )  # global CO2 emission in 2023; [kgCO2/a], Source: iea (2024);
    # https://www.iea.org/reports/co2-emissions-in-2023/executive-summary
    global_land_surface = 1.49 * 10**14  # Unit: m²
    global_annual_deprived_water = 7.91 * 10**13  # Unit: [m³/a], Source: EU JRC (2017)
    # https://data.europa.eu/doi/10.2760/88930

    # Get cf_aware and the file where it was found

    # TODO cf_aware shall be collected automatically for specific location (in WEFESiteAnalyst)
    # the factors can be found here: https://wulca-waterlca.org/aware/download-aware-factors/
    cf_aware = cf_aware_df[cf_aware_name]

    # -------------- MOO Customizable Weights ------------------
    wf_cost = moo_wf["wf_cost"]
    wf_ghg = moo_wf["wf_ghg"]
    wf_lr = moo_wf["wf_lr"]
    wf_wf = moo_wf["wf_wf"]

    # TODO Create GUI interface so web app can directly provide customizable weights

    # ---------------- Assigning MOO variables in csv ----------------
    moo_variable_var = "marginal_cost"
    # Every element that has the column 'storage_capacity_cost' is assumed to only contain components of type storage
    # where 'storage_capacity_cost' represents the cost parameter to be calculated.
    # All other elements require 'capacity_cost' as the cost parameter to be calculated.
    moo_variable_fix = "storage_capacity_cost" if "storage_capacity_cost" in element_df.columns else "capacity_cost"


    # ---------------- Possible SCENARIOS ----------------
    # Different calculation of cost parameters for different component types,
    # assuming all components of an element (resource) have the same type
    # TODO: add missing types (eg of water components), eventually link to TYPEMAP directly
    element_type = element_df["type"].iloc[0]
    if element_type in ["bus", "load", "excess", "crop"]:
        scenario = NO_MOO_VARIABLE_SCEN
    elif element_type in [
        "conversion",
        "hydropower",
        "mimo",
        "pv-panel",
        "storage",
        # "toilets",
        "volatile",
        "wastewater_treatment",
        "water_filtration",
        "water-pump",
        "water_treatment",
        "water-filtration",
        "wind-turbine"
    ]:
        scenario = MOO_VARIABLE_SCEN
    elif element in [
        "dispatchable",
        "energy_sources",
        "water_sources",
    ]:
        scenario = MOO_DISPATCHABLE_SCEN
    else:
        raise ValueError(
            f"The technology defined in {element} cannot be used for multi-objective optimization at the moment"
        )

    # ---------------- ACTIONS TAKEN FOR EACH SCENARIO ----------------
    cf_aware_temp_df = pd.DataFrame(index=cf_aware_df.index)  # temporary storage of moo profiles
    row_temp_dict = {}    # temporary storage mapping the rows of element_df to the moo profile names
    for index, row in element_df.iterrows():
        # define the row name
        row_name = row["name"]
        if scenario == MOO_VARIABLE_SCEN:
            if MOO_ANNUITY not in element_df.columns:
                element_df[MOO_ANNUITY] = None
            # store the parameters
            capex = row["capex"]
            opex_fix = row["opex_fix"]
            lifetime = row["lifetime"]
            if "resource_cost" in row:
                carrier_cost = row["resource_cost"]
            else:
                raise AttributeError(
                    f"The column 'resource_cost' is missing from component {row_name} within resource '{element}'"
                    f" and is needed for multi-objective cost calculation. "
                    f"The resource_cost is the cost of one unit of flow (could be EUR/kWh or EUR/kg, EUR/m³ etc "
                )

            ghg_emission_factor = to_float(row, "ghg_emission_factor")
            land_requirement_factor = to_float(row, "land_requirement_factor")
            water_consumption_factor = to_float(row, "water_consumption_factor")
            resource_cost = to_float(row, "resource_cost")

            logging.debug(f"capex: {capex}, lifetime: {lifetime}, wacc: {wacc}")
            annuity = calculate_annuity(capex, opex_fix, lifetime, wacc)
            moo_variable_capacity = (
                annuity / global_GDP * wf_cost
                + land_requirement_factor / global_land_surface * wf_lr
            ) * 10**15
            moo_variable_flow = 10**15 * (
                resource_cost / global_GDP * wf_cost
                + ghg_emission_factor / global_GHG * wf_ghg
                + cf_aware
                * water_consumption_factor
                / global_annual_deprived_water
                * wf_wf
            )

            # moo variables are expanded by 10e15 to have numbers in range which will not be reduced while optimization
            if not np.isnan(moo_variable_capacity):
                element_df.at[index, moo_variable_fix] = float(moo_variable_capacity)
                # log info message
                logger.info(
                    f"'{moo_variable_fix}' has been calculated and updated for"
                    f" '{row_name}' in '{element}'."
                )
            else:
                logging.warning(
                    f"'{moo_variable_fix}' could not be calculated and will not be updated for"
                    f" '{row_name}' in '{element}'. Capex: {capex}, lifetime: {lifetime}, wacc: {wacc}"
                )

            if moo_variable_flow is not None and not np.isnan(moo_variable_flow).any():
                ts_header = f"{row_name}_{moo_suffix}"
                cf_aware_temp_df[ts_header] = moo_variable_flow
                row_temp_dict[index] = ts_header

                logger.info(
                    f"'{moo_variable_var}' has been calculated and updated for"
                    f" '{row_name}' in '{element}'."
                )
            else:
                logging.warning(
                    f"'{moo_variable_var}' could not be calculated and will not be updated for"
                    f" '{row_name}' in '{element}'."
                )

            if not np.isnan(annuity):
                element_df.at[index, MOO_ANNUITY] = float(annuity)
                # log info message
                logger.info(
                    f"'{MOO_ANNUITY}' has been calculated and updated for"
                    f" '{row_name}' in '{element}'."
                )
            else:
                logging.warning(
                    f"'{MOO_ANNUITY}' could not be calculated and will not be updated for"
                    f" '{row_name}' in '{element}'. Capex: {capex}, lifetime: {lifetime}, wacc: {wacc}"
                )

        elif scenario == MOO_DISPATCHABLE_SCEN:
            # store the parameters
            ghg_emission_factor = to_float(row, "ghg_emission_factor")
            water_consumption_factor = to_float(row, "water_consumption_factor")
            indirect_water_consumption_factor = to_float(row, "indirect_water_consumption_factor")
            resource_cost = to_float(row, "resource_cost")

            moo_variable_flow = 10**15 * (
                resource_cost / global_GDP * wf_cost
                + ghg_emission_factor / global_GHG * wf_ghg
                + cf_aware
                * (water_consumption_factor + indirect_water_consumption_factor)
                / global_annual_deprived_water
                * wf_wf
            )

            if moo_variable_flow is not None and not np.isnan(moo_variable_flow).any():
                ts_header = f"{row_name}_{moo_suffix}"
                cf_aware_temp_df[ts_header] = moo_variable_flow
                row_temp_dict[index] = ts_header

                logger.info(
                    f"'{row_name}' is a dispatchable source.'{moo_variable_var}' has been calculated for"
                    f" '{row_name}' in '{element}'."
                )
            else:
                logging.warning(
                    f"'{moo_variable_var}' could not be calculated and will not be updated for"
                    f" '{row_name}' in '{element}'."
                )

        elif scenario == "no moo indicator":
            logger.info(
                f"'{row_name}' of element '{element}' does not contain '{moo_variable_fix}' parameter. Skipping..."
            )

    # check if any of the moo profiles is a variable profile:
    # any_variable=False means all profiles created from this element are constant and unnecessary,
    # marginal_cost will be kept as a constant, numeric value
    any_variable = any(len(cf_aware_temp_df[col].unique()) > 1 for col in cf_aware_temp_df.columns)

    if any_variable:
        # 1) transfer all columns to cf_aware_df
        for col in cf_aware_temp_df.columns:
            cf_aware_df[col] = cf_aware_temp_df[col]

        # 2) set element_df to profile names
        for index, col in row_temp_dict.items():
            element_df.at[index, moo_variable_var] = col

        # 3) add foreign key (once)
        fk_exists = any(
            fk.get("fields") == moo_variable_var and fk.get("reference", {}).get("resource") == cf_aware_res.name
            for fk in res.descriptor["schema"]["foreignKeys"]
        )
        if not fk_exists:
            res.descriptor["schema"]["foreignKeys"].append({
                "fields": moo_variable_var,
                "reference": {"resource": cf_aware_res.name}
            })

        profiles_created = True

    else:
        # All columns are constant → write numeric values directly
        # This will also be the case for 'scenario == "no moo indicator"' as cf_aware_temp_df will be empty
        for index, col in row_temp_dict.items():
            const_val = float(cf_aware_temp_df[col].iloc[0])
            element_df.at[index, moo_variable_var] = const_val

        profiles_created = False


    logging.debug(element_path)

    # save the updated element dataframe to the csv file
    element_df.to_csv(element_path, sep=";", index=False)
    return cf_aware_df, profiles_created
