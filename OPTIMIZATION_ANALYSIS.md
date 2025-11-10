# oemof-tabular-plugins: Optimization Implementation Analysis

## Executive Summary

The repository implements both **single-objective optimization (SOO)** using cost minimization and **multi-objective optimization (MOO)** combining cost, GHG emissions, land requirements, and water consumption. The key distinction is in how optimization objectives are encoded during pre-processing:

- **SOO**: Optimization variables (`capacity_cost`, `marginal_cost`) directly contain economic costs
- **MOO**: Optimization variables contain aggregated, dimensionless indicators combining normalized cost + environmental metrics

---

## 1. SINGLE-OBJECTIVE OPTIMIZATION (SOO) - Cost Minimization

### 1.1 Implementation Flow

```
Input CSV Data (CAPEX, OPEX, lifetime) 
    ↓
Pre-processing: calculate_annuity()
    ↓
CSV Updated: capacity_cost, marginal_cost, annuity
    ↓
Energy System Built
    ↓
Solver Minimizes: investment_costs + variable_costs
    ↓
Results Post-processed
```

### 1.2 Pre-processing: Cost Calculation

**File**: `src/oemof_tabular_plugins/general/pre_processing/pre_processing.py`

#### annuity Calculation Function (lines 11-23)
```python
def calculate_annuity(capex, opex_fix, lifetime, wacc):
    annuity_capex = economics.annuity(capex, lifetime, wacc)
    annuity_opex_fix = opex_fix
    annuity_total = round(annuity_capex + annuity_opex_fix, 2)
    return annuity_total
```

**Formula**:
```
annuity = CRF(wacc, lifetime) * capex + opex_fix

where CRF = wacc * (1 + wacc)^lifetime / ((1 + wacc)^lifetime - 1)
```

#### Pre-processing Workflow (lines 26-266)

1. **Reset mechanism** (lines 48-61): Clears `capacity_cost` for rows with cost parameters to force recalculation
   - Removes MOO artifacts from previous runs
   
2. **Scenario Group Detection** (lines 67-91):
   - Determines if annuity, cost params, or both are provided
   - Routes to appropriate calculation strategy

3. **Main Processing Loop** (lines 94-266):
   - For `annuity empty all cost params defined`: Calculate using `calculate_annuity()`
   - For `annuity defined all cost params defined`: Recalculate from cost params
   - For `no annuity all cost params defined`: Calculate from cost params
   - For `annuity no cost params`: Use provided annuity directly

4. **Marginal Cost Handling** (lines 239-262):
   - Resets marginal_cost to `resource_cost` (or 0) in sequences
   - Ensures variable costs are properly set for SOO

#### Calculated Fields for SOO

| Field | Type | Purpose | Used in |
|-------|------|---------|---------|
| `capacity_cost` | Scalar | Annuity per capacity unit | Investment cost calculation |
| `storage_capacity_cost` | Scalar | Annuity for storage | Storage investment cost |
| `marginal_cost` | Scalar/Profile | Cost per flow unit | Variable cost calculation |
| `annuity` | Scalar | Raw annuity (= capacity_cost) | Post-processing |

---

## 2. MULTI-OBJECTIVE OPTIMIZATION (MOO)

### 2.1 Concept Overview

MOO aggregates multiple objectives into **single dimensionless indicators** using:
- **Normalization**: Each metric normalized to global baseline
- **Weight factors**: User-defined importance weights summing to 1
- **Aggregation**: Weighted sum creates single optimization variable

**Objectives**:
1. **Cost** (wf_cost): EUR / year
2. **GHG Emissions** (wf_ghg): kg CO2-eq / year
3. **Land Requirement** (wf_lr): m²
4. **Water Footprint** (wf_wf): m³ / year (scarcity-weighted)

### 2.2 Pre-processing: MOO Calculation

**File**: `src/oemof_tabular_plugins/general/pre_processing/pre_processing_moo.py`

#### Global Normalization Constants (lines 106-116)

```python
global_GDP = 1.10 * 10^14  # USD/a (2024 forecast, IMF)
global_GHG = 3.74 * 10^14  # kg CO2/a (2023, IEA)
global_land_surface = 1.49 * 10^14  # m²
global_annual_deprived_water = 7.91 * 10^13  # m³/a (EU JRC 2017)
```

**These are used as divisors to normalize local values to global proportions.**

#### Weight Factor Dictionary (lines 124-127)

```python
moo_wf = {
    "wf_cost": 0.5,      # Cost minimization weight
    "wf_ghg": 0.2,       # GHG minimization weight
    "wf_lr": 0.2,        # Land requirement weight
    "wf_wf": 0.1,        # Water footprint weight
}
# Must sum to 1.0
```

#### MOO Calculation: Fixed Capacity Cost (lines 199-214)

**For conversion/volatile/storage components**:

```python
moo_variable_capacity = (
    annuity / global_GDP * wf_cost
    + land_requirement_factor / global_land_surface * wf_lr
) * 10^15

element_df.at[index, "capacity_cost"] = moo_variable_capacity
```

**Formula breakdown**:
- **Cost term**: `(annuity / global_GDP) * wf_cost`
  - Normalizes annual investment cost to proportion of global GDP
  - Scales by cost weight
  
- **Land term**: `(land_requirement_factor / global_land_surface) * wf_lr`
  - Normalizes land per unit capacity to global surface area
  - Scales by land weight
  
- **Scaling factor**: `* 10^15`
  - Ensures resulting values maintain precision during optimization
  - Prevents small normalized values from being rounded away

#### MOO Calculation: Variable Marginal Cost (lines 203-210)

```python
moo_variable_flow = 10^15 * (
    resource_cost / global_GDP * wf_cost
    + ghg_emission_factor / global_GHG * wf_ghg
    + cf_aware * water_consumption_factor / global_annual_deprived_water * wf_wf
)

element_df.at[index, "marginal_cost"] = ts_header  # Points to sequence profile
add_moo_timeseries(moo_variable_flow, ts_header, cf_aware_path)
```

**Formula breakdown**:
- **Cost term**: `(resource_cost / global_GDP) * wf_cost`
  - Per-unit flow cost normalized to global GDP
  
- **Emissions term**: `(ghg_emission_factor / global_GHG) * wf_ghg`
  - GHG per unit flow normalized to global annual emissions
  
- **Water term**: `cf_aware * (water_consumption_factor / global_deprived_water) * wf_wf`
  - Water per unit flow multiplied by regionalized AWARE factor
  - AWARE = scarcity-weighted characterization factor (dimensionless)
  - Normalizes to global deprived water availability
  
- **Scaling factor**: `* 10^15`
  - Applied to entire sum for precision

#### Custom Attributes Required for MOO (lines 192-195)

```python
ghg_emission_factor      # [kg CO2-eq / unit]
land_requirement_factor  # [m² / unit capacity]
water_consumption_factor # [m³ / unit flow]
resource_cost           # [EUR / unit flow]
```

**Where unit = MW, kW, m³/h, kg, etc. (defined by user)**

#### MOO Component Classification (lines 143-170)

1. **NO_MOO_VARIABLE_SCEN**: Bus, Load, Excess, Crop
   - No optimization variables → skipped

2. **MOO_VARIABLE_SCEN**: Conversion, Volatile, Storage, etc.
   - Both `capacity_cost` (fixed) and `marginal_cost` (flow) calculated

3. **MOO_DISPATCHABLE_SCEN**: Dispatchable, Water Sources
   - Only `marginal_cost` (flow) calculated
   - No `capacity_cost` (already fixed/free)

#### Dispatchable Source Calculation (lines 266-299)

```python
moo_variable_flow = 10^15 * (
    resource_cost / global_GDP * wf_cost
    + ghg_emission_factor / global_GHG * wf_ghg
    + cf_aware * (water_consumption_factor + indirect_water_consumption_factor)
      / global_annual_deprived_water * wf_wf
)
```

Includes both direct and indirect water consumption.

### 2.3 Pre-processing Workflow (lines 309-361)

**File**: `src/oemof_tabular_plugins/general/pre_processing/pre_processing.py:309-361`

```python
def pre_processing(scenario_dir, wacc, custom_attributes=None, moo=False, moo_wf=None):
    if moo is False or moo_wf is None:
        # Single-objective: standard cost calculation
        pre_processing_costs(scenario_dir, wacc, element, element_path, element_df)
    else:
        # Multi-objective: aggregated indicator calculation
        pre_processing_moo(wacc, element, element_path, element_df, scenario_dir, moo_wf)
    
    # Both paths run custom attributes processing
    pre_processing_custom_attributes(element_path, element_df, custom_attributes)
```

---

## 3. CUSTOM ATTRIBUTES AND THEIR ROLE

### 3.1 Supported Custom Attributes

**File**: `src/oemof_tabular_plugins/datapackage/post_processing.py:20-45`

**Economic**:
- `resource_cost` - Per-unit flow cost (EUR/kWh, EUR/kg, EUR/m³, etc.)
- `capacity_cost` - Annuity per capacity unit
- `storage_capacity_cost` - Annuity for storage capacity
- `annuity` - Raw calculated annuity (economic only)

**Environmental**:
- `ghg_emission_factor` - GHG per unit flow (kg CO2-eq/unit)
- `land_requirement_factor` - Land per unit capacity (m²/unit)
- `water_consumption_factor` - Direct water per unit flow (m³/unit)
- `indirect_water_consumption_factor` - Indirect water per unit flow (m³/unit)

**Other**:
- `renewable_factor` - Renewable share (0-1)
- `emission_factor` - Generic emission factor

### 3.2 Custom Attributes in Pre-processing

**File**: `src/oemof_tabular_plugins/general/pre_processing/pre_processing.py:269-306`

```python
def pre_processing_custom_attributes(element_path, element_df, custom_attributes):
    for index, row in element_df.iterrows():
        custom_attributes_dict = {}
        has_custom_attributes = False
        
        if custom_attributes is not None:
            for attribute in custom_attributes:
                if attribute in element_df.columns:
                    value = row[attribute]
                    custom_attributes_dict[attribute] = value
                    has_custom_attributes = True
        
        if has_custom_attributes:
            output_parameters_str = json.dumps(
                {"custom_attributes": custom_attributes_dict}
            )
            element_df.at[index, "output_parameters"] = output_parameters_str
```

**Result**: Custom attributes serialized to `output_parameters` column for retrieval during post-processing.

---

## 4. POST-PROCESSING AND RESULTS

### 4.1 SOO Post-processing Calculations

**File**: `src/oemof_tabular_plugins/datapackage/post_processing.py:77-159`

#### Annuity Calculation (lines 77-91)
```python
def compute_annuity_total(results_df):
    investments = results_df.investments
    if "storage" in results_df.name:
        return results_df.storage_capacity_cost * investments
    else:
        return results_df.capacity_cost * investments
```

**Result**: Annual investment cost for each component

#### Variable Costs - SOO (lines 137-149)
```python
def compute_variable_costs(results_df):
    if results_df.name[1] == "out":
        if "marginal_cost" not in results_df.index:
            return None
        return results_df.marginal_cost * results_df.aggregated_flow
    elif results_df.name[1] == "in":
        if "carrier_cost" not in results_df.index:
            return None
        return results_df.carrier_cost * results_df.aggregated_flow
```

**Result**: Annual variable cost (Economic cost)

### 4.2 MOO Post-processing Calculations

**File**: `src/oemof_tabular_plugins/datapackage/post_processing.py:106-159`

#### Annualized CAPEX - MOO (lines 106-115)
```python
def compute_annualized_capex_moo(results_df):
    investments = results_df.investments
    if investments is None:
        investments = 0
    return results_df.annuity * investments
```

**Uses**: Raw `annuity` (economic) for cost reporting
**NOT**: `capacity_cost` (which contains MOO aggregated value)

#### Variable Costs - MOO (lines 151-158)
```python
def compute_variable_cost_moo(results_df):
    """Uses resource_cost (economic) not marginal_cost (MOO indicator)"""
    return results_df.resource_cost * results_df.aggregated_flow

def compute_variable_cost_total_moo(results_df):
    return results_df["variable_cost_moo"].sum()
```

**Result**: Total annual variable cost (in economic units, not MOO indicators)

#### Total Annual Cost - MOO (lines 118-123)
```python
def compute_total_annual_cost_moo(results_df):
    """Aggregated system cost from annuity + resource_cost"""
    total_system_annuity = results_df["annualized_capex"].sum()
    total_system_variable_cost = results_df["variable_cost_moo"].sum()
    total_annual_cost = total_system_annuity + total_system_variable_cost
    return total_annual_cost
```

**Key**: Uses economic metrics (`annuity`, `resource_cost`) NOT MOO aggregated values

### 4.3 Environmental Indicators Calculated (Both SOO and MOO)

**File**: `src/oemof_tabular_plugins/datapackage/post_processing.py:177-217`

| Indicator | Calculation | Custom Attribute | Unit |
|-----------|-----------|-------------------|------|
| GHG Emissions | flow * ghg_emission_factor | ghg_emission_factor | kg CO2-eq |
| Land Requirement (added) | investments * land_requirement_factor | land_requirement_factor | m² |
| Land Requirement (total) | capacity_total * land_requirement_factor | land_requirement_factor | m² |
| Water Consumption | flow * water_consumption_factor | water_consumption_factor | m³ |
| Indirect Water | flow * indirect_water_consumption_factor | indirect_water_consumption_factor | m³ |

### 4.4 System-Level KPIs - MOO (lines 556-663)

**File**: `src/oemof_tabular_plugins/datapackage/post_processing.py`

```python
CALCULATED_KPIS = [
    # Total annual cost from MOO optimization
    {
        "column_name": "total_annual_cost_moo",
        "operation": compute_total_annual_cost_moo,
        "argument_names": ["annualized_capex", "variable_cost_moo"],
    },
    # Environmental totals (calculated post-optimization)
    {
        "column_name": "ghg_emissions_total",
        "operation": compute_ghg_emissions_total,
        "argument_names": ["ghg_emissions"],
    },
    {
        "column_name": "land_requirement_total",
        "operation": compute_system_land_requirement_total,
        "argument_names": ["land_requirement_total"],
    },
    {
        "column_name": "water_scarcity_footprint",
        "operation": compute_water_scarcity_footprint,
        "argument_names": ["indirect_water_consumption", "water_consumption"],
    },
]
```

---

## 5. SOLVER BEHAVIOR: SOO vs MOO

### 5.1 Single-Objective Optimization

**File**: `src/oemof_tabular_plugins/script/compute.py:113-129`

```python
# Build model
m = Model(es)

# Solve
m.solve("cbc")  # Minimize: sum(capacity_cost * investments + marginal_cost * flows)
```

**Objective**: Minimize total economic cost only

### 5.2 Multi-Objective Optimization

Same solver call, but with transformed objective:

```python
# capacity_cost = (annuity/GDP)*wf_cost + (land_req_factor/surface)*wf_lr
# marginal_cost = (resource_cost/GDP)*wf_cost + (ghg_factor/GHG)*wf_ghg + cf_aware*(water_factor/deprived_water)*wf_wf

# Minimize: sum(capacity_cost * investments + marginal_cost * flows)
# Where capacity_cost and marginal_cost are aggregated indicators
```

**Objective**: Minimize weighted combination of normalized metrics

**Key Insight**: Solver code is identical; transformation happens in pre-processing variables.

---

## 6. FLOW DIAGRAM: Complete MOO Workflow

```
CSV Input
├─ CAPEX, OPEX, lifetime, resource_cost
├─ ghg_emission_factor, land_requirement_factor
└─ water_consumption_factor

    ↓ [pre_processing_moo.py]

Pre-Processing Calculations
├─ annuity = CRF(wacc,lifetime)*capex + opex_fix
├─ moo_capacity_cost = (annuity/GDP)*wf_cost + (land_factor/area)*wf_lr
└─ moo_marginal_cost_profile = (resource_cost/GDP)*wf_cost 
                             + (ghg_factor/GHG)*wf_ghg
                             + cf_aware*(water_factor/deprived)*wf_wf
                             [SCALED by 10^15]

    ↓ [Update CSV]

Updated CSV
├─ capacity_cost ← moo_capacity_cost
├─ marginal_cost ← sequence_header
├─ annuity ← raw_annuity (preserved)
└─ sequences file updated with moo_marginal_cost_profile

    ↓ [compute_scenario]

Energy System Creation & Model Building
├─ Load from datapackage.json
├─ Create components with MOO-transformed costs
└─ Build oemof.solph.Model

    ↓ [Solver: CBC]

Optimization
└─ Minimize: Σ(moo_capacity_cost * inv) + Σ(moo_marginal_cost * flow)
   Where both terms are dimensionless aggregated indicators

    ↓ [Post-Processing]

Results Extraction
├─ investments, flows (from solver)
├─ capacity_total = capacity + investments
└─ annualized_capex = annuity * investments [Economic value]
└─ variable_cost_moo = resource_cost * flows [Economic value]

    ↓

KPIs Calculation
├─ total_annual_cost_moo = sum(annualized_capex + variable_cost_moo)
├─ ghg_emissions_total = sum(flows * ghg_emission_factor)
├─ land_requirement_total = sum(capacity_total * land_requirement_factor)
└─ water_scarcity_footprint = sum(flows * cf_aware * water_factor)

    ↓

Visualization (Dash App)
├─ Economic costs [USD/year]
├─ Environmental impacts [kg CO2-eq, m², m³]
└─ System composition [capacities, flows]
```

---

## 7. KEY DESIGN PATTERNS AND CONSIDERATIONS

### 7.1 Normalization Strategy

**Why Global Constants?**
- Makes optimization objectives dimensionless and comparable
- Enables weighting between incompatible units (EUR vs kg vs m²)
- Prevents unit-dependent solution bias

**Formula Pattern**:
```
Normalized_Value = (Component_Value / Global_Baseline) * Weight
```

Examples:
- Cost: `(annuity[EUR/yr] / GDP[EUR/yr]) * wf_cost` → dimensionless
- Emissions: `(ghg[kg CO2] / global_emissions[kg CO2]) * wf_ghg` → dimensionless

### 7.2 Scaling by 10^15

**Purpose**: Numerical stability during optimization
- Normalized values often range 10^-15 to 10^-10
- Solver precision → rounded away at such scales
- Multiply by 10^15 → rescales to range 10^0 to 10^5
- Solver maintains precision
- Economic post-processing reverses this implicitly (uses `annuity`, `resource_cost` directly)

### 7.3 Resource Cost vs Marginal Cost

**In SOO**:
```
marginal_cost [CSV] = economic cost per unit flow
```

**In MOO**:
```
marginal_cost [sequence] = dimensionless MOO indicator (scaled by 10^15)
resource_cost [CSV] = preserved for post-processing economic calculations
```

**Post-processing**: Always uses `resource_cost`, not `marginal_cost`, to calculate actual costs

### 7.4 Annuity Preservation

**In SOO**:
```
annuity = capacity_cost (same value)
```

**In MOO**:
```
capacity_cost = (annuity/GDP)*wf_cost + land_term  [MOO aggregated]
annuity = preserved raw economic value
```

**Purpose**: Post-processing can report true economic costs even when optimization used MOO indicators

### 7.5 AWARE Factor Integration

```python
cf_aware_profile  # Load from sequences/volatile_profile.csv
                  # Dimensionless regionalized scarcity characterization
                  
water_footprint = cf_aware * water_consumption * wf_wf / global_deprived_water
```

Enables location-based water impact assessment.

---

## 8. CONTROL FLOW IN MAIN COMPUTE FUNCTION

**File**: `src/oemof_tabular_plugins/script/compute.py:32-145`

```python
def compute_scenario(
    scenario_dir,
    results_path,
    wacc,
    scenario_name=None,
    custom_attributes=None,
    typemap=None,
    moo=False,              # ← MOO activation flag
    moo_wf=None,            # ← Weight factors dict
    dash_app=False,
    parameters_units=None,
    infer_bus_carrier=True,
    skip_preprocessing=False,
    skip_infer_datapackage_metadata=False,
    save_raw_results=True,
):
    # 1. PRE-PROCESSING
    if skip_preprocessing is False:
        pre_processing(
            scenario_dir, 
            wacc, 
            custom_attributes, 
            moo,        # Passed to routing logic
            moo_wf      # Passed with weight factors
        )
    
    # 2. METADATA INFERENCE
    if skip_infer_datapackage_metadata is False:
        otp_building.infer_metadata_from_data(...)
    
    # 3. ENERGY SYSTEM CREATION
    es = EnergySystem.from_datapackage(
        os.path.join(scenario_dir, "datapackage.json"),
        typemap=typemap,
    )
    
    # 4. MODEL BUILDING & SOLVING
    m = Model(es)
    m.add_constraints_from_datapackage(...)
    m.solve("cbc")  # ← Same solver for both SOO and MOO
    
    # 5. POST-PROCESSING
    return post_processing(
        params,
        es,
        results_path,
        dp_path=os.path.join(scenario_dir, "datapackage.json"),
        dash_app=dash_app,
        parameters_units=parameters_units,
    )
```

**Router Logic in pre_processing()** (lines 309-361):
```python
if moo is False or moo_wf is None:
    pre_processing_costs(scenario_dir, wacc, ...)  # SOO path
else:
    pre_processing_moo(wacc, ...)  # MOO path
```

---

## 9. TESTING AND EXAMPLE SCENARIOS

### 9.1 Test Files

**File**: `tests/test_pre_processing.py`
- Tests annuity calculation with various parameter combinations
- Validates cost scenarios (annuity + params, no annuity + params, etc.)
- Checks error handling for missing/incomplete parameters

**File**: `tests/test_post_processing.py`
- Tests post-processing calculations
- Validates KPI aggregation

### 9.2 Example Configuration

**File**: `examples/scripts/compute.py:65-116`

```python
# MOO Configuration
moo = False  # Set to True to activate MOO

moo_wf = {
    "wf_cost": 0.5,   # 50% weight on cost
    "wf_ghg": 0.2,    # 20% weight on GHG emissions
    "wf_lr": 0.2,     # 20% weight on land requirements
    "wf_wf": 0.1,     # 10% weight on water footprint
}
```

### 9.3 Custom Attributes Example

```python
custom_attributes = [
    "ghg_emission_factor",
    "renewable_factor",
    "land_requirement_factor",
    "water_consumption_factor",
    "indirect_water_consumption_factor",
    "resource_cost",
    "annuity"
]
```

---

## 10. COMPARISON TABLE: SOO vs MOO

| Aspect | Single-Objective (SOO) | Multi-Objective (MOO) |
|--------|------------------------|----------------------|
| **Optimization Variable** | Economic cost | Dimensionless aggregated indicator |
| **capacity_cost** | annuity [EUR/yr] | (annuity/GDP)*wf_c + (lr_factor/area)*wf_lr [10^-15] |
| **marginal_cost** | resource_cost [EUR/unit] | (rc/GDP)*wf_c + (ghg/GHG)*wf_g + cf*wf*wf_w [10^-15] |
| **Solver Objective** | minimize(sum(costs)) | minimize(sum(aggregated_indicators)) |
| **Post-processing Costs** | Directly from optimization | Calculated from resource_cost + annuity (not from optimization) |
| **Environmental Metrics** | Calculated post-optimization | Calculated post-optimization (same as SOO) |
| **Weight Factors** | N/A | User-defined, sum=1 |
| **Normalization** | None | Global baselines (GDP, GHG, area, water) |
| **Scaling** | None | 10^15 for numerical stability |

---

## 11. POTENTIAL ISSUES AND DESIGN NOTES

### 11.1 MOO Post-processing Discrepancy

**Current Design**:
```python
total_annual_cost_moo = sum(annuity * investments) + sum(resource_cost * flows)
```

**Note**: Calculation uses raw `annuity` and `resource_cost`, **not** the MOO-transformed `capacity_cost` and `marginal_cost`.

**Implication**: Economic cost reported in post-processing is the true economic cost, not the MOO-optimized cost. This is intentional for transparency.

### 11.2 AWARE Factor Handling

**Current**: Hard-coded lookup (`cf_aware_profile`)

```python
cf_aware, cf_aware_path = get_moo_timeseries(
    scenario_dir, ts_name="cf-aware-profile"
)
```

**TODO** (line 120): "cf_aware shall be collected automatically for specific location (in WEFESiteAnalyst)"

### 11.3 Marginal Cost Profile Strategy

**Current**:
```python
ts_header = f"{row_name}_mc_profile"
add_moo_timeseries(
    ts_values=moo_variable_flow,
    ts_header=ts_header,
    sequences_path=cf_aware_path,
)
element_df.at[index, "marginal_cost"] = ts_header
```

**Note**: References a column name (ts_header) that points to a profile in a CSV file, not the value itself.

---

## 12. FILE STRUCTURE SUMMARY

```
src/oemof_tabular_plugins/
├── general/
│   ├── pre_processing/
│   │   ├── pre_processing.py (lines 11-266)
│   │   │   ├── calculate_annuity() [11-23]
│   │   │   └── pre_processing_costs() [26-266]
│   │   │       ├── Cost parameter detection [67-91]
│   │   │       ├── Scenario routing [94-266]
│   │   │       └── Marginal cost reset [239-262]
│   │   └── pre_processing_moo.py (lines 19-309)
│   │       ├── Global normalization constants [106-116]
│   │       ├── pre_processing_moo() [66-309]
│   │       │   ├── Component classification [143-170]
│   │       │   ├── MOO_VARIABLE_SCEN processing [176-264]
│   │       │   │   ├── Capacity cost calculation [199-214]
│   │       │   │   └── Marginal cost calculation [203-210]
│   │       │   └── MOO_DISPATCHABLE_SCEN processing [266-299]
│   └── post_processing/
│       ├── post_processing.py (general KPI calculations)
│       └── post_processing_moo.py (stub, not implemented)
├── datapackage/
│   └── post_processing.py (lines 1-672)
│       ├── Custom attributes [20-45]
│       ├── Calculation functions [54-217]
│       ├── CALCULATED_OUTPUTS [446-550]
│       └── CALCULATED_KPIS [556-663]
└── script/
    └── compute.py (lines 32-145)
        └── compute_scenario() - main orchestration
```

---

## 13. CONCLUSIONS

### Design Philosophy

1. **Pre-processing transforms**: Cost/environmental data → optimization variables
2. **Solver-agnostic**: Same optimization code works for SOO and MOO (variables are different)
3. **Transparent economics**: Post-processing always uses raw metrics, not MOO aggregated values
4. **Global normalization**: Enables comparison of incompatible units (EUR, kg, m², m³)
5. **Numerical stability**: 10^15 scaling prevents solver precision loss

### Key Takeaways

- **SOO**: Direct economic cost minimization
- **MOO**: Minimizes aggregated, normalized, dimensionless indicators combining cost + environmental objectives
- **Custom attributes**: Enable flexible scenario definition without code changes
- **Post-processing**: Decoupled from optimization; calculates true economic and environmental metrics regardless of optimization objective

