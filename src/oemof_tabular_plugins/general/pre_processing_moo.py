import os
import pandas as pd
import logging
from oemof.tools import logger, economics
from pre_processing import calculate_annuity

logger.define_logging()

#  Inputs
#  are provided by the GUI end-user; and partly fetched by component specs file;
#  until the specs structure is not defined, the inputs are provided in this src script for testing purposes

# Global Inputs (used for normalization)
global_GDP = 109.529*10**12  # forecasted for 2024, Unit: [USD/a], Source: IMF (2024)
# https://www.imf.org/en/Publications/WEO/weo-database/2024/April/weo-report?c=512,914,612,171,614,311,213,911,314,193,122,912,313,419,513,316,913,124,339,638,514,218,963,616,223,516,918,748,618,624,522,622,156,626,628,228,924,233,632,636,634,238,662,960,423,935,128,611,321,243,248,469,253,642,643,939,734,644,819,172,132,646,648,915,134,652,174,328,258,656,654,336,263,268,532,944,176,534,536,429,433,178,436,136,343,158,439,916,664,826,542,967,443,917,544,941,446,666,668,672,946,137,546,674,676,548,556,678,181,867,682,684,273,868,921,948,943,686,688,518,728,836,558,138,196,278,692,694,962,142,449,564,565,283,853,288,293,566,964,182,359,453,968,922,714,862,135,716,456,722,942,718,724,576,936,961,813,726,199,733,184,524,361,362,364,732,366,144,146,463,528,923,738,578,537,742,866,369,744,186,925,869,746,926,466,112,111,298,927,846,299,582,487,474,754,698,&s=NGDPD,&sy=2022&ey=2029&ssm=0&scsm=1&scc=0&ssd=1&ssc=0&sic=0&sort=country&ds=.&br=1
global_GHG = 37.4*10**9   # global CO2 emission in 2023; [tCO2/a], Source: iea (2024);
# https://www.iea.org/reports/co2-emissions-in-2023/executive-summary
global_land_surface = 148.94*10**12   # Unit: m²
total_renewable_water_resources = 54*10**9   # Unit: [m³/a], Source: CIA (2011),
# https://www.cia.gov/the-world-factbook/field/total-renewable-water-resources/

# National Inputs
wacc = 0.06

# Weights
wf_cost = 0.2
wf_GHG = 0.2
wf_lr = 0.3
wf_wf = 0.3

# Component Specs
# PV_panel
capex = 400  # [USD/kWp]
opex_fix = 20  # [USD/kWp/year]
lifetime = 20  # [years]
land_requirement = 1  # [m²]
GHG_emission = 0  # [tCO2eq/kWh]
water_footprint = 10  # [m³/kWp/year]


def pre_processing_moo(capex, opex_fix, lifetime, wacc, land_requirement, GHG_emission, water_footprint, global_GDP,
                       global_GHG, global_land_surface, total_renewable_water_resources, wf_cost, wf_GHG, wf_lr, wf_wf):
    """This function will run the multi-objective optimization

    The outcome is that the main costs 'capacity_cost' will be replaced by an aggregated
    indicator representing the multi-objective optimization goals.
    This function will replace the pre_processing_costs function if moo is set to True

    :param capex: CAPEX (currency/kWp) *or the unit you choose to use throughout the model e.g. MW/GW
    :param opex_fix: fixed OPEX (currency/kWp/year)
    :param lifetime: lifetime of the component (years)
    :param wacc: weighted average cost of capital (WACC) applied throughout the model (%)
    :param land_requirement: land requirement (m²)
    :param GHG_emission: GHG emission per unit of flow (e.g, electricity) (tCO2eq/KWh)
    :param water_footprint: water footprint (m³)
    :param global_GDP: global GDP of specific year (currency/year)
    :param global_GHG: global greenhouse gas emission of a specifc year [tCO2eq/year]
    :param global_land_surface: global land surface [m²]
    :param total_renewable_water_resources: global freshwater resources which are annually renewed [m³/a]

    Inputs
    Weights: defined by model-user e.g. percentages up to 1
        (in the OptiMG GUI, the user defining the weights will be e.g. local prosumers)
    0.5 for costs 0.2 for emissions 0.2 for land requirement 0.1 for water dissipated
    Has to add up to 1, otherwise error

    Normalization: done with global values
    User sets costs are defined in the CSV e.g. CAPEX, OPEX fix, lifetime
        -> this cost is normalised based on global GDP
        -> value is calculated for proportion of cost to global GDP
    Same applies for total emissions
        -> the value is normalised based on total global annual GHG emissions
    Land requirements
        ->  the value is normalised based on the world's land surface area
    Water Footprint
        -> the value is normalised based on the total renewable water resources
    Then these values are added together

    User includes specific cost, specific emission factor, specific land requirement, specific water footprint
    This function takes these as inputs, normalise them based on global reference data, multiplies them by weights
    to represent end-user preferences, and adds them up to calculate an aggregated (cost) indicator
    This value will be entered in the csv file under 'capacity_cost'
    The csv file will be updated

    :return:
    """

    annuity_total = calculate_annuity(capex, opex_fix, lifetime, wacc)
    # Unit: (currency/KWp/year)

    aggregated_indicator = (annuity_total/global_GDP * wf_cost + GHG_emission/global_GHG * wf_GHG + land_requirement /
                            global_land_surface * wf_lr + water_footprint/total_renewable_water_resources * wf_wf)

    return aggregated_indicator


aggregated_indicator = pre_processing_moo(capex, opex_fix, lifetime, wacc, land_requirement, GHG_emission,
                                          water_footprint, global_GDP, global_GHG, global_land_surface,
                                          total_renewable_water_resources, wf_cost, wf_GHG, wf_lr, wf_wf)


print(aggregated_indicator)
