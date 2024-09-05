# Changelog

All notable changes to this project will be documented in this file. <br>
For each version, important additions, changes and removals are listed here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.2] dev - 202Y-MM-DD
### Added
- pre-commit configuration (lint files with black)
- general_custom_attributes example
- tests for pre-rpocessing of custom attributes
- layout for multi-objective optimization (MOO)
- a new workflow for automated tests
- new function for adding additional foreign keys to datapackage.json in src/oemof_tabular_plugins/datapackage/building.py
- function `infer_busses_carrier` to get the carriers assigned to each bus either explicitly (ie directly defined in `bus.csv` resource) or implicitly (ie via the carriers assigned to components connected to busses). The explicit method can lead to logical errors such as two carriers assigned to a single bus which is then raised as an error.
- Facade SimpleCrop in wefe subpackage to simulate a crop growth without accounting for water in (following this reference https://doi.org/10.1016/j.eja.2019.01.009), a test model is provided in `examples/scenarios/test_crop`.

### Changed
- structure of the general package and the wefe package
- pre-processing.py to include the option for adding custom attributes e.g. renewable factor, emission factor, land requirement
- scripts/infer.py files and included foreign keys function where necessary
- Now `validate_datapackage` of facades must return the resource (modified or not)

### Removed
- the hydrogen package

## [0.0.1] Initial Release - Hello Super-Repo - 2024-04-09

### Added
- pyproject.toml
- setup.py to .gitignore
- new requirements files inside a requirements folder
- created `oemof_tabular_plugins/` inside `src/` folder
- created examples folder with scenarios for user to play with
- created subpackages for general, hydrogen and wefe
- added pre and post processing in the general package

### Changed
- Changelog
- USERS.cff
- CITATIONS.cff
- moved requirements from a file to a folder structure similar to the one of Numpy
- LICENSE (MIT License)
