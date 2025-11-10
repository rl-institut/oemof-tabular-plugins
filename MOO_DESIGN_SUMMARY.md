# MOO Design Summary - Quick Reference

## Core Difference: SOO vs MOO

### Single-Objective Optimization (SOO)
```
Cost Minimization Only

capacity_cost = annuity [EUR/year]
marginal_cost = resource_cost [EUR/unit]

Solver: minimize(Σ capacity_cost*inv + Σ marginal_cost*flow)
```

### Multi-Objective Optimization (MOO)
```
Minimize Aggregated Environmental + Economic Indicator

capacity_cost = (annuity/GDP)*wf_cost + (land_factor/area)*wf_lr [dimensionless * 10^15]
marginal_cost = (res_cost/GDP)*wf_cost + (ghg/GHG)*wf_ghg + cf*water/deprived*wf_wf [dimensionless * 10^15]

Solver: minimize(Σ capacity_cost*inv + Σ marginal_cost*flow)
```

**Same solver code, different variables!**

---

## Key Formulas

### Annuity Calculation (SOO)
```python
annuity = economics.annuity(capex, lifetime, wacc) + opex_fix
        = [wacc*(1+wacc)^life/((1+wacc)^life - 1)] * capex + opex_fix
```

### MOO Capacity Cost (Fixed costs)
```python
moo_capacity_cost = (annuity / global_GDP) * wf_cost 
                  + (land_requirement_factor / global_land_surface) * wf_lr
                  * 10^15
```

### MOO Marginal Cost (Variable costs)
```python
moo_marginal_cost = (resource_cost / global_GDP) * wf_cost
                  + (ghg_emission_factor / global_GHG) * wf_ghg
                  + cf_aware * (water_consumption_factor / global_annual_deprived_water) * wf_wf
                  * 10^15
```

### Environmental Indicators (Post-processing, both SOO & MOO)
```python
ghg_emissions = aggregated_flow * ghg_emission_factor
land_requirement = (capacity_total OR investments) * land_requirement_factor
water_consumption = aggregated_flow * (water_consumption_factor + indirect_water_consumption_factor)
```

---

## Global Normalization Baselines

| Baseline | Value | Unit | Source |
|----------|-------|------|--------|
| global_GDP | 1.10 * 10^14 | USD/year | IMF 2024 forecast |
| global_GHG | 3.74 * 10^14 | kg CO2-eq/year | IEA 2023 |
| global_land_surface | 1.49 * 10^14 | m² | - |
| global_annual_deprived_water | 7.91 * 10^13 | m³/year | EU JRC 2017 |

---

## Custom Attributes Required for MOO

Must be present in CSV files for components with both capacity and marginal costs:

| Attribute | Example Value | Unit | Used In |
|-----------|---------------|------|---------|
| `ghg_emission_factor` | 0.5 | kg CO2-eq / unit | Emissions term |
| `land_requirement_factor` | 10 | m² / MW | Land term |
| `water_consumption_factor` | 100 | m³ / MWh | Water term |
| `indirect_water_consumption_factor` | 50 | m³ / MWh | Water term (dispatchable only) |
| `resource_cost` | 50 | EUR / MWh | Cost term |
| `capex` | 1000000 | EUR / MW | Annuity calculation |
| `opex_fix` | 20000 | EUR / MW / year | Annuity calculation |
| `lifetime` | 20 | years | Annuity calculation |

---

## Weight Factors (moo_wf)

Example:
```python
moo_wf = {
    "wf_cost": 0.5,   # 50% - Cost minimization
    "wf_ghg": 0.2,    # 20% - GHG minimization
    "wf_lr": 0.2,     # 20% - Land requirement minimization
    "wf_wf": 0.1,     # 10% - Water footprint minimization
}
# MUST SUM TO 1.0
```

---

## Component Classification for MOO

### NO_MOO_VARIABLE_SCEN
Elements: bus, load, excess, crop
Action: Skip MOO processing (no costs to aggregate)

### MOO_VARIABLE_SCEN (Fixed + Variable)
Elements: conversion, volatile, storage, mimo, pv_panel, wind_turbine, etc.
Calculates: capacity_cost AND marginal_cost

### MOO_DISPATCHABLE_SCEN (Variable only)
Elements: dispatchable, energy_sources, water_sources
Calculates: marginal_cost ONLY (no capacity_cost)
Includes: Both direct + indirect water consumption

---

## Pre-processing Flow

```
Input CSV (element.csv)
    ↓
Detect scenario group (has annuity? has cost params?)
    ↓
Calculate annuity if needed
    ↓
IF moo == False:
    └─ Set capacity_cost = annuity
    └─ Set marginal_cost = resource_cost
ELSE IF moo == True:
    └─ Set capacity_cost = aggregated_indicator (MOO formula)
    └─ Create marginal_cost profile (time series)
    └─ Save to sequences CSV
    └─ Preserve annuity for post-processing
    ↓
Process custom_attributes (serialize to output_parameters)
    ↓
Update CSV file
```

---

## Key Files and Line References

### Pre-processing
- **SOO**: `/general/pre_processing/pre_processing.py`
  - calculate_annuity(): lines 11-23
  - pre_processing_costs(): lines 26-266
  
- **MOO**: `/general/pre_processing/pre_processing_moo.py`
  - Global constants: lines 106-116
  - Component classification: lines 143-170
  - Capacity cost calculation: lines 199-214
  - Marginal cost calculation: lines 203-210
  - Dispatchable calculation: lines 266-299

### Post-processing
- `/datapackage/post_processing.py`
  - Custom attributes: lines 20-45
  - annuity_total(): lines 77-91
  - variable_cost_moo(): lines 151-158
  - total_annual_cost_moo(): lines 118-123
  - Environmental calculations: lines 169-217
  - CALCULATED_OUTPUTS: lines 446-550
  - CALCULATED_KPIS: lines 556-663

### Main Orchestration
- `/script/compute.py`
  - compute_scenario(): lines 32-145
  - Router logic: pre_processing() call at line 87

---

## Post-Processing: Economic Cost Reporting

**Critical Design Point**: MOO uses aggregated indicators for optimization, but reports true economic costs!

```python
# During optimization (solver sees):
capacity_cost_moo = scaled_aggregated_indicator (10^-15 magnitude)
marginal_cost_moo = scaled_aggregated_indicator (10^-15 magnitude)

# During post-processing (results reported as):
annualized_capex = annuity * investments         [True economic cost]
variable_cost_moo = resource_cost * flows        [True economic cost]
total_annual_cost_moo = sum(annualized_capex + variable_cost_moo)  [True total]
```

**This ensures transparency**: Users see actual economic costs, not MOO indicators.

---

## Why 10^15 Scaling?

Problem:
```
Normalized value = (component_value / global_baseline)
Example: (1000 EUR/yr) / (1.1e14 USD/yr) = 9.1e-12
```

Solver treats values < 1e-10 as essentially zero due to numerical precision.

Solution:
```
Scaled value = 9.1e-12 * 10^15 = 9,100
```

Now solver maintains precision within typical tolerance (1e-6 relative).

Post-processing reverses this implicitly by using original `annuity` and `resource_cost` values.

---

## AWARE Factor

**What is it**: Dimensionless regionalized scarcity characterization factor
**Source**: WULCA-WaterLCA (https://wulca-waterlca.org/aware/download-aware-factors/)
**Purpose**: Adjusts water consumption by regional water scarcity

**Formula**:
```
water_impact = cf_aware * water_consumption_factor * flows
```

**Current Implementation**: Loaded from sequences CSV (hard-coded per scenario)
**TODO**: Automate lookup by geographic coordinates

---

## Summary Table: What's Preserved vs What's Transformed

| Field | In SOO | In MOO | Post-processing Uses |
|-------|--------|--------|----------------------|
| `capacity_cost` | annuity | MOO indicator | NOT used (see below) |
| `marginal_cost` | resource_cost | MOO indicator | NOT used (see below) |
| `annuity` | = capacity_cost | raw economic | annuity * investments |
| `resource_cost` | flow cost | preserved | resource_cost * flows |
| `ghg_emission_factor` | calculated post-opt | calculated post-opt | same |
| `land_requirement_factor` | calculated post-opt | calculated post-opt | same |
| `water_consumption_factor` | calculated post-opt | calculated post-opt | same |

**Key insight**: Even in MOO mode, post-processing ignores the MOO-transformed optimization variables and recalculates true economic costs from preserved raw fields.

