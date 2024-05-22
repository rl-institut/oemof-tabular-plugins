
.. figure:: https://user-images.githubusercontent.com/14353512/185425447-85dbcde9-f3a2-4f06-a2db-0dee43af2f5f.png
    :align: left
    :target: https://github.com/rl-institut/super-repo/
    :alt: Repo logo

=======
General
=======
The general subpackage contains a range of methods and functionalities that can be applied for any project. As a
foundational component of the library, the tools are not project-specific, and this subpackage can be used
in combination with project specific subpackages.

Constraints
===========
The general constraints subpackage contains all of the constraints that are not seen to be project specific.
In :ref:`constraints.py <constraints/constraints.py>`, the constraints are defined and added to a Pyomo model.
In :ref:`constraint_facades.py <constraints/constraint_facades.py>`, a mapping is
provided between constraint types and their corresponding facade classes.

Available constraints
---------------------
- **Minimum renewable share** - this constraint sets a minimum renewable share on the total energy supply

Post-processing
===============
The general post-processing subpackage contains the functionalities involved in post-processing and visualising
of the (oemof) results for any results that are seen to be used in a wide range of projects. There is the
option to apply post-processing methods for a cost-only optimization (in :ref:`post_processing.py <post_processing/post_processing.py>` or for a
multi-objective optimization (in :ref:`post_processing_moo.py <post_processing/post_processing_moo.py>`)
(to be implemented).

General results (cost-optimization)
-----------------------------------
- **Excess generation**: the excess energy generation for each energy vector
- **Specific system costs**: total costs from optimization relative to total demand
- **Renewable share**: total renewable share of supply
- **Total emissions**: total annual emissions of energy system
- **Capacities**
- **Flows**
- **Costs**

General results (multi-objective-optimization)
----------------------------------------------
- **TO BE COMPLETED**

Pre-processing
===============
The pre-processing subpackage contains the functionalities involved in pre-processing of input data before
creating an oemof model from the datapackage.json file. There is the option to apply post-processing methods
for a cost-only optimization (in :ref:`pre_processing.py <pre_processing/pre_processing.py>` or for a
multi-objective optimization (in :ref:`pre_processing_moo.py <pre_processing/pre_processing_moo.py>`)
(to be implemented).

General pre-processing options (cost-optimization)
--------------------------------------------------
- Pre-processing costs:
  this function allows the user to input either the annuity directly or the CAPEX, OPEX fix, and lifetime, or both.

  - If only the annuity is given,
    this is directly used in the model, but some post-processing results will not be calculated (e.g. upfront investment costs)
  - If only the CAPEX, OPEX fix and lifetime are given,
    the annuity is calculated directly from these values using the calculate_annuity function
  - If both are given,
    the user is asked whether to use the defined annuity (but warned that there might be discrepencies between the
    annuity and CAPEX, OPEX fix and lifetime values) or to replace the annuity with the calculated value using
    the calculate_annuity function

- Pre-processing custom attributes:
  this function allows the user to define custom attributes in their components e.g. renewable factor, emission
  factor etc. These attributes are not already parameters in the component facade.

General pre-processing options (multi-objective-optimization)
------------------------------------------------------------
- **TO BE COMPLETED**
