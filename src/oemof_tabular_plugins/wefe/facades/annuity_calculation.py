from oemof.tools import economics   # based on ciaras pre_processing

# pv data
capex = 2970.8
opex_fix = 55.0
lifetime = 30.0
wacc = 0.06

# for belfaa xgap=0.35
biomass_sum = 782.4618001798001 # g/m^2
p_rated = 450   # W per panel
area_apv = 3.7  # m^2 per panel
kWp = 38.9
electricity_sum = 105400

# SIMPLE
HI = 0.68

# ahmed wilaya data for belfaa (0.66 avg price - 0.25 prod cost)
tomato_gain = 0.41   # EUR/kg

# ECB 24/07/24
eur_usd = 1.0848   # EUR 1 = USD 1.0848


annuity_capex = economics.annuity(capex, lifetime, wacc)
annuity_kwp = round(annuity_capex + opex_fix, 2)
print(biomass_sum)
annuity_m2 = annuity_kwp * p_rated * 1e-3 / area_apv  # update annuity from USD/kWp to USD/m^2
apv_gain = biomass_sum * HI * 1e-3 * tomato_gain * eur_usd   # USD/(m^2 a)
annuity_apv = annuity_m2 - apv_gain
print(annuity_m2, annuity_apv)
print(annuity_apv * 320)
print(apv_gain * 320)

norm_power = electricity_sum / kWp
print(norm_power)

electricity_price = 0.88
usd_mar = 9.8676

electricity_revenue = 0.88 * 1/9.8676   # USD/kWh
print(electricity_revenue * 105400)

