
.. figure:: https://user-images.githubusercontent.com/14353512/185425447-85dbcde9-f3a2-4f06-a2db-0dee43af2f5f.png
    :align: left
    :target: https://github.com/rl-institut/super-repo/
    :alt: Repo logo

====
WEFE
====
The wefe subpackage contains methods and functionalities that are specific to projects involving the
water-energy-food-environment nexus. This includes additional components that are mainly found in WEFE
systems that contain more detail than the general oemof-tabular components, global specs that can be
utilised across multiple WEFE projects, and post-processing methods that are specific to WEFE projects.
This should be used in addition to the general subpackage.

Facades
=======
The facades subpackage contains all of the facades that have been created specifically for WEFE systems,
where more detail is required than possible in the generic oemof-tabular facade. Each facade can
represent a component type, where inputs, methods and outputs are defined.

Available facades
-----------------
- **PV panel** - this more detailed PV panel takes in global irradiance and outputs electricity, taking
  the temperature factor into consideration.

- **Agrivoltaics** - TO BE COMPLETED

Global specs
============
The global specs subpackage contains various specs that can be used across multiple WEFE projects.
TO BE COMPLETED

Post-processing
===============
The post-processing subpackage contains the methods that are specific to WEFE system results.
TO BE COMPLETED
