# Getting Started

## Installing oemof-tabular-plugins (as a package)
1. Install a solver (e.g. cbc) for oemof-solph:
   - Read [this guide](https://oemof-solph.readthedocs.io/en/latest/readme.html#installing-a-solver)
   - Once installed, the path where the cbc.exe file is stored should be added to your system path environment variables
   - Then your computer needs to be restarted
2. Create a new repository or clone/open an existing repository
3. Create a virtual environment (with python version >= 3.9, 3.10 or above is recommended) e.g. with conda or virtualenv
4. Activate your virtual environment
5. Install the oemof-tabular-plugins package using:
    ``pip install oemof-tabular-plugins``
   - Potential errors you might come across:
       - WARNING: for windows users, you might get an error when installing oemof-tabular: ‘ERROR: Could not build wheels
      for cchardet, which is required to install pyproject.toml-based projects’
       - Visit here for guidelines on how to fix the error:
     https://github.com/twintproject/twint/issues/1407#issuecomment-1141734344
6. Check that the solver (e.g. cbc) has been installed correctly by typing the following in your terminal:
   ``oemof_installation_test``
    - If successful, the following message will appear:
    ```*****************************
    Solver installed with oemof:

    cbc: working
    glpk: not working
    gurobi: not working
    cplex: not working

    *****************************
    oemof successfully installed.
    *****************************
    ```
7. Once all of these steps have been completed successfully, you are ready to run scenarios!

## Installing oemof-tabular-plugins (for development)
1. Install a solver (e.g. cbc) for oemof-solph:
   - Read [this guide](https://oemof-solph.readthedocs.io/en/latest/readme.html#installing-a-solver)
   - Once installed, the path where the cbc.exe file is stored should be added to your system path environment variables
   - Then your computer needs to be restarted
2. Clone oemof-tabular-plugins:
   ``git clone https://github.com/rl-institut/oemof-tabular-plugins.git``
2. Create a virtual environment (with python version >= 3.9, 3.10 or above is recommended) e.g. with conda or virtualenv
3. Activate your virtual environment
4. Install the repository as a package using:
``pip install -e .``
    - this installs the requirements and treats oemof-tabular-plugins as a package but from the current branch version
    - Potential errors you might come across:
       - WARNING: for windows users, you might get an error when installing oemof-tabular: ‘ERROR: Could not build wheels
      for cchardet, which is required to install pyproject.toml-based projects’
       - Visit here for guidelines on how to fix the error:
     https://github.com/twintproject/twint/issues/1407#issuecomment-1141734344
5. Check that the solver (e.g. cbc) has been installed correctly by typing the following in your terminal:
   ``oemof_installation_test``
    - If successful, the following message will appear:
    ```*****************************
    Solver installed with oemof:

    cbc: working
    glpk: not working
    gurobi: not working
    cplex: not working

    *****************************
    oemof successfully installed.
    *****************************
6. Now you can create your own branch or pull an existing branch and start developing!

## Using oemof-tabular-plugins (as a package)
1.	Set up scenarios in your project repository
2.	Run compute.py
3.	Evaluate results stored in ‘results’ directory

Set up scenarios in your project repository
Once oemof-tabular-plugins has been installed as a package in your repository, you can start to use it for your own case studies. The structure of your repository should be set up similarly to below:

|-- project_name
    |-- scenarios
	 |-- scenario_name_1
	     |-- data
  |-- constraints
      |-- constraint.csv
		  |-- elements
             |-- bus.csv
             |-- conversion.csv
             |-- dispatchable.csv
             |-- excess.csv
             |-- load.csv
             |-- storage.csv
             |-- volatile.csv
		  |-- sequences
             |-- load_profile.csv
             |-- volatile_profile.csv
	     |-- scripts
  |-- .gitkeep
	 |-- scenario_name_2
	 |-- scenario_name_3
    |-- scripts
	 |-- compute.py

### Creating scenarios
Inside your scenario, there will be the ‘data’ and ‘scripts’ directories. The ‘data’ directory contains all of your scenario input data and the ‘scripts’ directory can contain any scripts that are specific to the scenario.

### Data/constraints (optional)
The ‘constraints’ directory is optional and only needs to be included if you are setting some kind of constraint for your energy system optimization e.g. setting the maximum annual emissions limit or a minimum renewable share.  Here will contain the CSV file/s defining the constraints in your system. The parameters that should be included are:
THIS SHOULD BE FORMATTED AS A TABLE
Name	unique name identifier of your constraint
	required
Type	constraint type (as defined in the CONSTRAINT_TYPE_MAP)	required
Limit	constraint value (minimum or maximum)
	required
Keyword	label that is defines which components are involved in the constraint, must be consistent between the constraint definition and the components it refers to	required

### Data/elements (required)
The ‘elements’ directory is required for running your scenario and contains the input CSV files where the components and parameters of the energy system are defined. The CSV files are typically named according to the corresponding facade. Details of the types of existing facades and their parameters are outlined here: https://oemof-tabular.readthedocs.io/en/latest/facades.html

### Data/sequences (required)
The ‘sequences’ directory is required for running your scenario and contains the input CSV files which are all of the time series involved in your scenario. This will be profiles for e.g. volatile renewable energy sources and demand load profiles, or anything related to a component that is variable. All time series in your model must be of the same length, and each entry represents the value at a given time-step.

### Running compute.py
The ‘compute.py’ script is located in the generic (not scenario-specific) ‘scripts’ directory, and this is where the scenarios are ran. Here you define the scenario/s you want to run and then after running compute.py, the results for the scenario/s are automatically saved in the results folder. The main things that happen in compute.py are:
- Adding any custom facades to the TYPEMAP. See [here](https://github.com/oemof/oemof-tabular/blob/dev/src/oemof/tabular/facades/__init__.py) for the already existing facades in oemof-tabular
- Defining the type of optimization you want to perform (cost-optimization or multi-objective optimization)
- Pre-processing of input data in preparation for energy system creation
- Building of metadata (datapackage.json) from CSV input files
- Create oemof-solph energy system object from the datapackage.json file
- Create model from energy system object
- Add constraints to the model
- Solve the oemof-solph energy system model using a solver e.g. cbc
- Post-processing of the oemof-solph results for easier analysis

## Introduction
Oemof-tabular-plugins is an extension package for oemof-tabular, offering additional functionalities to ease the set-up of diverse energy systems. Before using oemof-tabular-plugins, it is advised to familiarise yourself with [oemof-tabular](https://oemof-tabular.readthedocs.io/en/latest/index.html). The package contains various sub-packages: a general package which contains any additional features that have the potential to be useful across energy system types, and project type
specific packages (e.g. one for water-energy-food systems, hydrogen systems or multi-nodal energy systems) which contain additional features specific to that project type. At present, there are very few sub-packages but the aim is to create an
extensive package library for various project types for easy implementation in future projects. Each sub-package contains any methods required to pre-process the data, any components that are specific to the energy system type (e.g. an agrivoltaic plant),
any constraints specific to the energy system (e.g. maximum land use) and any additional post-processing of data.

## Examples
The provided examples are designed solely to demonstrate the setup process for an energy system using CSV files for different cases. The input data used in these examples is not accurate and should not be used for any analysis. These examples are based on a limited three-hour period and are meant only to illustrate the format and structure of the input files. For actual analysis, you should use your own input data and longer timeseries.

### general_basic
This example demonstrates how an energy system is set up in oemof-tabular with no additional oemof-tabular-plugins functionalities. The annuity (‘capacity_cost’) is calculated prior to the CSV files and inputted directly

### general_add_cost_inputs
This example shows how you can implement CAPEX, OPEX fix and lifetime as additional cost parameters. If you choose to implement your costs like this, the annuity (‘capacity_cost’) gets automatically calculated in pre-processing and added to the input CSV files before the energy system is created.

### general_custom_attributes/wefe_custom_attributes
These examples shows how you can include additional parameters for your components that aren’t already default parameters. As it stands, the new custom attributes that can be included are ‘renewable_factor’, ‘emission_factor’, ‘land_requirement_factor’ and ‘water_footprint_factor’. They have to be named exactly like this to work and have the results for them considered properly.

### general_constraints
This example shows how you can apply constraints to your CSV files. Only constraints that have been defined in oemof-solph/oemof-tabular and/or oemof-tabular-plugins can be applied – a pyomo constraint and a constraint façade are needed to apply the constraints in your CSV files.

### wefe_pv_panel
This example shows how you can use the detailed PV panel façade that has been defined in the  oemof-tabular-plugins wefe directory. It outlines the parameters that can be included in the CSV files.

### wefe_reverse_osmosis
This example demonstrates how you can use a simple multiple-input multiple-output converter within a system including energy and water.

## General
This sub-package contains additional functionalities to oemof-tabular, applicable to a wide range of energy systems.

## Pre-processing
The pre-processing module contains additional functionalities for pre-processing of input data.

Calculate_annuity: This function is there to automatically calculate the annuity based on the CAPEX, OPEX fix and lifetime of a component.

Pre_processing: The idea of the pre-processing module is to offer users flexibility in how they input data. The module provides an option for users to either directly input the annuity (as commonly done in oemof-tabular) or input the capital expenditure (CAPEX), fixed operational expenditure (OPEX fix), and lifetime parameters. In the latter case, the annuity is automatically calculated using the calculate_annuity function. This approach eliminates the need for manual data pre-processing before inputting it into CSV files.

## Constraints

The constraints module offers the constraint functionalities to be later used in the constraint facades.

Minimum renewable share: The renewable share of the optimized energy system must reach at least the minimal renewable share set in the constraint. This value is calculated from the total renewable energy generation of the system in relation to the total energy generation.

In order for this constraint to work, you must set the ‘renewable_factor’ parameter to 1 (if renewable source) or 0 (if non-renewable source). This is defined under ‘custom_attributes’ in ‘output_parameters’. See example X

## Constraint facades

These are required for oemof-tabular. They are classes that encapsulate the logic for building and applying a specific type of constraint in an energy system optimization model.

MinimumRenewableShare: a specific type of constraint façade that enforces a minimum renewable share constraint in the energy system optimization model. It checks for flows with a specified keyword attribute (‘renewable_factor’) and applies the constraint to those flows.

A constraint type map is then defined containing all of the generic constraints, which will be referenced when adding constraints to the model. See X

## Post-processing

The post-processing module stores and saves the relevant techno-economic and environmental results in an easy to read format.
