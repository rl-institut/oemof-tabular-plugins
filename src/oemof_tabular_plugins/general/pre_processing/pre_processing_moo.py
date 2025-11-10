# from .pre_processing import calculate_annuity
from oemof.tools import logger, economics
import os
import numpy as np
from datapackage import Package
import pandas as pd
import logging
import csv

# from .pre_processing import calculate_annuity

NO_MOO_VARIABLE_SCEN = "no moo variables"  # this scenario potentially can be skipped; as this would correspond with SOO
MOO_VARIABLE_SCEN = "moo variable calculation"
MOO_DISPATCHABLE_SCEN = "moo variable calculation with dispatchable"

MOO_ANNUITY = "annuity"


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


def add_moo_timeseries(ts_values, ts_header, sequences_path=None):

    df = pd.read_csv(sequences_path, sep=";")
    df[ts_header] = ts_values
    df.to_csv(sequences_path, index=False, sep=";")


def get_moo_timeseries(scenario_dir, ts_name=""):
    scenario_name = os.path.basename(scenario_dir)
    sequences_dir = os.path.join(scenario_dir, "data", "sequences")
    sequences_path = None

    for file in os.listdir(sequences_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(sequences_dir, file)
            try:
                df_header = pd.read_csv(file_path, nrows=0, sep=";")
                if ts_name in df_header.columns:
                    sequences_path = file_path
                    break
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    if sequences_path is None:
        raise ValueError(
            f"'{ts_name}' could not be found in any file under 'sequences' of datapackage '{scenario_name}'"
        )

    df = pd.read_csv(sequences_path, sep=";")
    return df[ts_name], sequences_path


def pre_processing_moo(wacc, element, element_path, element_df, scenario_dir, moo_wf):
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
    cf_aware, cf_aware_path = get_moo_timeseries(scenario_dir, ts_name="cf-aware-profile")  # Unit: dimensionless
    # TODO cf_aware shall be collected automatically for specific location (in WEFESiteAnalyst)
    # the factors can be found here: https://wulca-waterlca.org/aware/download-aware-factors/

    # -------------- MOO Customizable Weights ------------------
    wf_cost = moo_wf["wf_cost"]
    wf_ghg = moo_wf["wf_ghg"]
    wf_lr = moo_wf["wf_lr"]
    wf_wf = moo_wf["wf_wf"]

    # TODO Create GUI interface so web app can directly provide customizable weights

    # ---------------- Assigning MOO variables in csv ----------------
    moo_variable_var = "marginal_cost"
    # annuity = "annuity"
    # for every element other than storage, the fixed moo optimization variable is 'capacity_cost'
    # for storage, the fixed moo optimization variable is 'storage_capacity_cost'
    if element != "storage.csv":
        moo_variable_fix = "capacity_cost"
    else:
        moo_variable_fix = "storage_capacity_cost"


    # ---------------- Possible SCENARIOS ----------------
    if element in ["bus.csv", "load.csv", "excess.csv", "crop.csv"]:
        scenario = NO_MOO_VARIABLE_SCEN
    elif element in [
        "conversion.csv",
        "energy_conversion.csv",
        "hydropower.csv",
        "mimo.csv",
        "pv_panel.csv",
        "storage.csv",
        "toilets.csv",
        "volatile.csv",
        "wastewater_treatment.csv",
        "water_filtration.csv",
        "water_pumps.csv",
        "water_treatment.csv",
        "wind_turbine.csv"
    ]:
        scenario = MOO_VARIABLE_SCEN
    elif element in [
        "dispatchable.csv",
        "energy_sources.csv",
        "water_sources.csv",
    ]:
        scenario = MOO_DISPATCHABLE_SCEN
    else:
        raise ValueError(
            f"The technology defined in {element} cannot be used for multi-objective optimization at the moment"
        )

    # ---------------- RESET COST COLUMNS TO FORCE RECALCULATION ----------------
    # This ensures idempotent preprocessing - removes MOO artifacts from previous runs
    # and forces recalculation from base parameters (capex, opex_fix, lifetime, resource_cost)

    # Reset capacity_cost for components that will be recalculated
    if scenario in [MOO_VARIABLE_SCEN]:
        if moo_variable_fix in element_df.columns:
            for index, row in element_df.iterrows():
                # Check if this row has the required parameters for MOO calculation
                has_cost_params = all([
                    'capex' in element_df.columns and pd.notna(row.get('capex')),
                    'opex_fix' in element_df.columns and pd.notna(row.get('opex_fix')),
                    'lifetime' in element_df.columns and pd.notna(row.get('lifetime')),
                    'resource_cost' in element_df.columns and pd.notna(row.get('resource_cost'))
                ])
                if has_cost_params:
                    # Clear capacity_cost to force recalculation
                    element_df.at[index, moo_variable_fix] = None
            logger.info(f"Reset '{moo_variable_fix}' to force recalculation in '{element}'")

    # Reset marginal_cost for all MOO scenarios that will create time series profiles
    if scenario in [MOO_VARIABLE_SCEN, MOO_DISPATCHABLE_SCEN]:
        if moo_variable_var in element_df.columns:
            for index, row in element_df.iterrows():
                # Clear marginal_cost to force recalculation
                element_df.at[index, moo_variable_var] = None
            logger.info(f"Reset '{moo_variable_var}' to force recalculation in '{element}'")

    # ---------------- ACTIONS TAKEN FOR EACH SCENARIO ----------------
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

            ghg_emission_factor = row["ghg_emission_factor"]
            land_requirement_factor = row["land_requirement_factor"]
            water_consumption_factor = row["water_consumption_factor"]
            resource_cost = row["resource_cost"]

            logging.debug(f"capex: {capex}, lifetime: {lifetime}, wacc: {wacc}")
            annuity = calculate_annuity(capex, opex_fix, lifetime, wacc)
            moo_variable_capacity = (
                annuity / global_GDP * wf_cost
                + land_requirement_factor / global_land_surface * wf_lr
            ) * 10**15

            # Calculate variable flow cost
            # CRITICAL: Only include time-varying cf_aware when wf_wf > 0
            # When wf_wf=0, multiplying cf_aware (time series) by 0 creates numerical
            # artifacts that can make the result slightly non-constant, causing different
            # optimization behavior even though mathematically it should be identical.
            if wf_wf > 0:
                moo_variable_flow = 10**15 * (
                    resource_cost / global_GDP * wf_cost
                    + ghg_emission_factor / global_GHG * wf_ghg
                    + cf_aware
                    * water_consumption_factor
                    / global_annual_deprived_water
                    * wf_wf
                )
            else:
                # Exclude cf_aware term to ensure clean constant result
                moo_variable_flow = 10**15 * (
                    resource_cost / global_GDP * wf_cost
                    + ghg_emission_factor / global_GHG * wf_ghg
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

            # if isinstance(moo_variable_flow, np.ndarray):
            #     element_df.at[index, moo_variable_var] = "cf_aware"

            # TODO change this to insert it into sequences

            if moo_variable_flow is not None and not np.isnan(moo_variable_flow).any():
                # TODO should save the moo_variable_flow as a sequence and write the sequence header here instead of a float
                # save this into "moo_profile.csv" or "moo_variable_flow.csv", cf_aware should stay in volatile profile
                ts_header = f"{row_name}_mc_profile"
                add_moo_timeseries(
                    ts_values=moo_variable_flow,
                    ts_header=ts_header,
                    sequences_path=cf_aware_path,
                )
                # import pdb;
                # pdb.set_trace()
                element_df.at[index, moo_variable_var] = ts_header
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
            ghg_emission_factor = row["ghg_emission_factor"]
            water_consumption_factor = row["water_consumption_factor"]
            indirect_water_consumption_factor = row["indirect_water_consumption_factor"]
            resource_cost = row["resource_cost"]

            # Calculate variable flow cost
            # CRITICAL: Only include time-varying cf_aware when wf_wf > 0
            # When wf_wf=0, multiplying cf_aware (time series) by 0 creates numerical
            # artifacts that can make the result slightly non-constant, causing different
            # optimization behavior even though mathematically it should be identical.
            if wf_wf > 0:
                moo_variable_flow = 10**15 * (
                    resource_cost / global_GDP * wf_cost
                    + ghg_emission_factor / global_GHG * wf_ghg
                    + cf_aware
                    * (water_consumption_factor + indirect_water_consumption_factor)
                    / global_annual_deprived_water
                    * wf_wf
                )
            else:
                # Exclude cf_aware term to ensure clean constant result
                moo_variable_flow = 10**15 * (
                    resource_cost / global_GDP * wf_cost
                    + ghg_emission_factor / global_GHG * wf_ghg
                )

            # TODO change this to insert it into sequences
            if moo_variable_flow is not None and not np.isnan(moo_variable_flow).any():
                ts_header = f"{row_name}_mc_profile"
                add_moo_timeseries(
                    ts_values=moo_variable_flow,
                    ts_header=ts_header,
                    sequences_path=cf_aware_path,
                )
                element_df.at[index, moo_variable_var] = ts_header
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
    logging.debug(element_path)

    # save the updated dataframe to the csv file
    element_df.to_csv(element_path, sep=";", index=False)
    return
