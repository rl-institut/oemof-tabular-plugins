# from .pre_processing import calculate_annuity
from oemof.tools import logger, economics

# from .pre_processing import calculate_annuity

NO_MOO_VARIABLE_SCEN = "no moo variables"
MOO_VARIABLE_SCEN = "moo variable calculation"
MOO_DISPATCHABLE_SCEN = "moo variable calculation with dispatchable"


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


def pre_processing_moo(wacc, element, element_path, element_df):
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
    """
    # ---------------- MOO Normalization PARAMS ----------------
    # Global Inputs (used for normalization)
    global_GDP = (
        109.529 * 10**12
    )  # forecasted for 2024, Unit: [USD/a], Source: IMF (2024)
    # https://www.imf.org/en/Publications/WEO/weo-database/2024/April/weo-report?c=512,914,612,171,614,311,213,911,314,193,122,912,313,419,513,316,913,124,339,638,514,218,963,616,223,516,918,748,618,624,522,622,156,626,628,228,924,233,632,636,634,238,662,960,423,935,128,611,321,243,248,469,253,642,643,939,734,644,819,172,132,646,648,915,134,652,174,328,258,656,654,336,263,268,532,944,176,534,536,429,433,178,436,136,343,158,439,916,664,826,542,967,443,917,544,941,446,666,668,672,946,137,546,674,676,548,556,678,181,867,682,684,273,868,921,948,943,686,688,518,728,836,558,138,196,278,692,694,962,142,449,564,565,283,853,288,293,566,964,182,359,453,968,922,714,862,135,716,456,722,942,718,724,576,936,961,813,726,199,733,184,524,361,362,364,732,366,144,146,463,528,923,738,578,537,742,866,369,744,186,925,869,746,926,466,112,111,298,927,846,299,582,487,474,754,698,&s=NGDPD,&sy=2022&ey=2029&ssm=0&scsm=1&scc=0&ssd=1&ssc=0&sic=0&sort=country&ds=.&br=1
    global_GHG = (
        37.4 * 10**9
    )  # global CO2 emission in 2023; [tCO2/a], Source: iea (2024);
    # https://www.iea.org/reports/co2-emissions-in-2023/executive-summary
    global_land_surface = 148.94 * 10**12  # Unit: m²
    total_renewable_water_resources = 54 * 10**9  # Unit: [m³/a], Source: CIA (2011),
    # https://www.cia.gov/the-world-factbook/field/total-renewable-water-resources/

    # -------------- MOO Customizable Weights ------------------
    wf_cost = 0.2
    wf_ghg = 0.2
    wf_lr = 0.3
    wf_wf = 0.3
    # TODO Create GUI interface so web app can directly provide customizable weights

    # ---------------- Assigning MOO variables in csv ----------------
    moo_variable_var = "marginal_cost"
    # for every element other than storage, the fixed moo variable is 'capacity_cost'
    # for storage, the annuity cost parameter is 'storage_capacity_cost'
    if element != "storage.csv":
        moo_variable_fix = "capacity_cost"
    else:
        moo_variable_fix = "storage_capacity_cost"

        # loop through each entry in the csv file

    # ---------------- Possible SCENARIOS ----------------

    if element in ["bus.csv", "load.csv", "excess.csv", "crop.csv"]:
        scenario = NO_MOO_VARIABLE_SCEN
    elif element in ["conversion.csv", "mimo.csv", "storage.csv", "volatile.csv"]:
        scenario = MOO_VARIABLE_SCEN
    elif element == "dispatchable.csv":
        scenario = MOO_DISPATCHABLE_SCEN
    else:
        raise ValueError(
            f"The technology defined in {element} cannot be used for multi-objective at the moment"
        )

    # ---------------- ACTIONS TAKEN FOR EACH SCENARIO ----------------
    for index, row in element_df.iterrows():
        # define the row name
        row_name = row["name"]
        if scenario == MOO_VARIABLE_SCEN:
            # store the parameters
            capex = row["capex"]
            opex_fix = row["opex_fix"]
            lifetime = row["lifetime"]
            #carrier_cost = row["carrier_cost"]
            ghg_emission_factor = row["ghg_emission_factor"]
            land_requirement_factor = row["land_requirement_factor"]
            water_footprint_factor = row["water_footprint_factor"]

            annuity = calculate_annuity(capex, opex_fix, lifetime, wacc)
            moo_variable_capacity = (
                annuity / global_GDP * wf_cost
                + land_requirement_factor / global_land_surface * wf_lr
            )
            moo_variable_flow = (
            ghg_emission_factor / global_GHG * wf_ghg
                + water_footprint_factor / total_renewable_water_resources * wf_wf
            )

            element_df.at[index, moo_variable_fix] = float(moo_variable_capacity)
            element_df.at[index, moo_variable_var] = float(moo_variable_flow)

            # log info message
            logger.info(
                f"'{moo_variable_fix}' and {moo_variable_var} have been calculated and updated for"
                f" '{row_name}' in '{element}'."
            )

        elif scenario == MOO_DISPATCHABLE_SCEN:
            # store the parameters
            ghg_emission_factor = row["ghg_emission_factor"]
            water_footprint_factor = row["water_footprint_factor"]

            moo_variable_flow = (
                ghg_emission_factor / global_GHG * wf_ghg
                + water_footprint_factor / total_renewable_water_resources * wf_wf
            )
            element_df.at[index, moo_variable_var] = float(moo_variable_flow)
            logger.info(
                f"'{element}' is a disptachable source.'{moo_variable_var}' has been calculated for"
                f" '{row_name}' in '{element}'."
            )

        elif scenario == "no moo indicator":
            logger.info(
                f"'{element}' does not contain '{moo_variable_fix}' parameter. Skipping..."
            )
    # save the updated dataframe to the csv file
    element_df.to_csv(element_path, sep=";", index=False)
    return
