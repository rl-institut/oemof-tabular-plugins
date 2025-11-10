# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

oemof-tabular-plugins is a Python package that extends [oemof-tabular](https://github.com/oemof/oemof-tabular) with custom constraint facades for energy system optimization. The package is designed to be modular, allowing different domain-specific implementations (e.g., general, WEFE - Water-Energy-Food-Ecosystem) to coexist.

## Development Setup

```bash
# Install development dependencies
pip install -r requirements/dev_requirements.txt

# Install pre-commit hooks (required before first commit)
pre-commit install
```

The pre-commit hooks enforce Black formatting automatically. If code is not formatted, Black will format it for you - simply stage the changes and commit again.

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_pre_processing.py

# Run specific test
pytest tests/test_pre_processing.py::test_calculate_annuity
```

Test files are located in `tests/` and include:
- `test_pre_processing.py` - Tests for cost calculations and preprocessing
- `test_post_processing.py` - Tests for results processing
- `test_water_components.py` - Tests for WEFE-specific components

Test input data is in `tests/test_inputs*/` directories.

## Running Scenarios

```bash
# Run example scenario computation
python examples/scripts/compute.py

# Run sensitivity analysis
python examples/scripts/compute_sensitivity.py

# Display results
python examples/scripts/results.py
```

Example scenarios are in `examples/scenarios/` (e.g., `aiwa_24`, `arusi_8760`, `general_basic`).

## Code Formatting

```bash
# Format all code with Black
black .

# The pre-commit hook will automatically run Black on commit
```

## Documentation

```bash
# Build documentation locally with MkDocs
mkdocs serve

# Documentation will be available at http://127.0.0.1:8000
```

Documentation source files are in `docs/` and use MkDocs with the Material theme.

## Command-Line Tools

The package provides two CLI tools (defined in `pyproject.toml`):

```bash
# Convert datapackage to JSON
dp_to_json <args>

# Convert JSON to datapackage
json_to_dp <args>
```

## Architecture

### Module Structure

The package is organized into domain-specific modules under `src/oemof_tabular_plugins/`:

- **`general/`** - Core functionality for general energy system modeling
  - `constraints/` - Generic constraint implementations
  - `pre_processing/` - Input data preprocessing (cost calculations, MOO setup)
  - `post_processing/` - Results processing and visualization (GUI/Dash app)

- **`wefe/`** - Water-Energy-Food-Ecosystem specific extensions
  - `constraints/` - WEFE-specific constraints
  - `facades/` - Component facades (PVPanel, WindTurbine, Crops, Water systems, etc.)
  - `global_specs/` - Crop specs, PV modules, wind turbines, soil specs
  - `post_processing/` - WEFE-specific results processing

- **`datapackage/`** - Datapackage handling
  - `building.py` - Metadata inference
  - `config.py` - Configuration management
  - `dp_to_json.py` / `json_to_dp.py` - Format conversion
  - `post_processing.py` - Results serialization

- **`script/`** - Main computation orchestration
  - `compute.py` - Core scenario computation workflow

### Constraint Type Maps

The package uses a merging pattern for constraint type maps:

```python
# In __init__.py
from .general import CONSTRAINT_TYPE_MAP as ctm_general
from .wefe import CONSTRAINT_TYPE_MAP as ctm_wefe

CONSTRAINT_TYPE_MAP = ctm_general | ctm_wefe
```

Each module defines its own `CONSTRAINT_TYPE_MAP` which gets merged at the package level.

### WEFE Type Map

WEFE extends the base oemof.tabular TYPEMAP with custom facades:

```python
WEFE_TYPEMAP = {
    "water-pump": WaterPump,
    "water-filtration": WaterFiltration,
    "crop": SimpleCrop,
    "mimo-crop": MimoCrop,
    "pv-panel": PVPanel,
    "mimo": MIMO,
    "apv": APV,
    "inverter": Inverter,
    "hydropower": RRHydropower,
    "river-flow": Volatile,
    "wind-turbine": WindTurbine
}
WEFE_TYPEMAP.update(TYPEMAP)  # Merge with base oemof.tabular TYPEMAP
```

### Scenario Computation Workflow

The main computation flow (in `src/oemof_tabular_plugins/script/compute.py::compute_scenario()`):

1. **Pre-processing** - Calculate annuities from CAPEX/OPEX/lifetime, handle MOO weight factors
2. **Metadata inference** - Update `datapackage.json` from CSV files
3. **Energy system creation** - Build oemof.solph EnergySystem from datapackage
4. **Model building** - Create optimization model, add constraints
5. **Solving** - Run solver (CBC by default)
6. **Post-processing** - Process results, optionally launch Dash visualization app

### Multi-Objective Optimization (MOO)

The package supports MOO with customizable weight factors for:
- Cost minimization
- GHG emissions minimization
- Land requirement minimization
- Water scarcity footprint minimization

Weight factors are specified in `moo_wf` dict and processed during pre-processing.

### Custom Attributes

The framework allows custom attributes to be added to components via the `custom_attributes` parameter. Common attributes include:
- `ghg_emission_factor`, `renewable_factor`
- `land_requirement_factor`, `water_consumption_factor`
- `resource_cost`, `annuity`

## Git Workflow

This project follows a feature-branch workflow:

- **Main branch**: `production` (not `main` or `master`)
- **Branch naming**: `feature/<issue-nr>-<short-description>` (e.g., `feature/42-add-new-constraint`)
- **Commit messages**: Imperative mood, <50 chars, end with issue number (e.g., "Add MOO weight factor validation #42")
- **Pull requests**: Target `production` branch, include `Close #<issue-nr>` in description

Before committing:
1. Add your changes to CHANGELOG.md
2. On first commit: Add your name to CITATION.cff
3. Ensure Black formatting passes (pre-commit will handle this)

## Release Process

1. Create release candidate with version `vX.Y.Zrc1`
2. Build package: `python prepare_release.py`
3. Upload to PyPI: `twine upload dist/*`
4. Test installation in fresh environment
5. Iterate on release candidates (rc2, rc3, etc.) if issues found
6. Make final release from `production` branch

## Important Notes

- All cost calculations use WACC (Weighted Average Cost of Capital)
- Scenarios use datapackage format with CSV files in `data/elements/` and `data/sequences/`
- Results include raw oemof.solph outputs and processed/aggregated outputs
- The package requires `oemof.tabular`, `numpy==1.26.0`, `dash`, and `oemof-industry`
