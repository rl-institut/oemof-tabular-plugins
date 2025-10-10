
.. figure:: https://user-images.githubusercontent.com/14353512/185425447-85dbcde9-f3a2-4f06-a2db-0dee43af2f5f.png
    :align: left
    :target: https://github.com/rl-institut/super-repo/
    :alt: Repo logo

=====================
oemof-tabular-plugins
=====================

**oemof-tabular-plugins (otp) adds on specific characterisitcs to oemof-tabular. Among others, it is used for modeling and optimizing water, energy, food, and environment (WEFE) components serving as base for the repository WEFEConfigurator.**

.. list-table::
   :widths: auto

   * - License
     - |badge_license|
   * - Documentation
     - |badge_documentation|
   * - Publication
     -
   * - Development
     - |badge_issue_open| |badge_issue_closes| |badge_pr_open| |badge_pr_closes| |badge_black|
   * - Community
     - |badge_contributing| |badge_contributors|

.. contents::
    :depth: 2
    :local:
    :backlinks: top

Installation
============
For using oemof-tabular-plugins, install it using pip. For allowing the use of the most recent features, we recommend to directly install otp from the production branch. Therefore, use anaconda prompt and move to the local repository of oemof-tabular-plugins. Then install otp uisng

.. code::

    pip install -e .


OTP requires specific versions oemof-tabular (e.g. commit 09346649f75389d9fdafa62c24ae5e95cc0cf291 on dev) and oemof-industry (pip install oemof-industry==0.1.1rc2)
For installing a suitable version of oemof-tabular, clone oemof-tabular to your local machine. Open the repository using e.g. "Git Bash". On the dev branch checkout out to the specific commit needed

.. code::

    git checkout 09346649f75389d9fdafa62c24ae5e95cc0cf291

Then use an anaconda prompt and move to the local repository of oemof-tabular. Install the oemof-tabular version you are on locally by 

.. code::

    pip install .

Currently (as of Oct10 2025) commit 09346649f75389d9fdafa62c24ae5e95cc0cf291 is the latest oemof-tabular commit on dev. Therefore you do not have to checkout on it to install oemof-tabular.

The suitable oemof.industry version can be installed by specifiying the required version.

.. code::

    pip install oemof-industry==0.1.1rc2

In case you would like to visualize the topology of the WEFE system you are modeling and optimizing, oemof.visio and graphviz are required.

.. code::

    pip install oemof.visio[network]
    pip install graphviz

On Windows machines, you additionally have to download Graphviz (https://graphviz.org/download/) and install it on your system. During installation, make sure to activate "add PATH" variables to ensure that the executable can found. Afterwards, restart your environment.


Introduction
============
Among others, in otp you can model optimize integrated WEFE systems. To start you can specify the scenario which you would like to model. Therefore open in examples/scripts/compute.py" and type or uncomment a scenario which you would like to run e.g. "Arusi8760". Moreover you can define whether you would like to run multi-objective-optimization or not (MOO=True or MOO=False). Run the scenario by executing compute.py

.. code::

    python compute.py

This builds a the scenario described in csv files in the scenario folder and optimizes it using time-series based optimization. The results are presented in a dash app hosted on a local server.

In general showcases different features to complement `oemof-tabular <https://github.com/oemof/oemof-tabular>`_ to add constraints specific to certain uses of it

.. code::

    import oemof_tabular_plugin as otp
    # one can import the full constraint map
    from otp import CONSTRAINT_TYPE_MAP
    # or just the one relevant for a specific usecase
    from otp.hydrogen import CONSTRAINT_TYPE_MAP

Documentation
=============
| The documentation is created with Markdown using `MkDocs <https://www.mkdocs.org/>`_.
| All files are stored in the ``docs`` folder of the repository.
| A **GitHub Actions** deploys the ``production`` branch on a **GitHub Page**.
| The documentation page is `rl-institut.github.io/super-repo/ <https://rl-institut.github.io/super-repo/>`_

Collaboration
=============
| Everyone is invited to develop this repository with good intentions.
| Please follow the workflow described in the `CONTRIBUTING.md <CONTRIBUTING.md>`_.

License and Citation
====================
| The code of this repository is licensed under the **MIT License** (MIT).
| See `LICENSE.txt <LICENSE.txt>`_ for rights and obligations.
| See the *Cite this repository* function or `CITATION.cff <CITATION.cff>`_ for citation of this repository.
| Copyright: `super-repo <https://github.com/rl-institut/super-repo/>`_ © `Reiner Lemoine Institut <https://reiner-lemoine-institut.de/>`_ | `MIT <LICENSE.txt>`_


.. |badge_license| image:: https://img.shields.io/github/license/rl-institut/super-repo
    :target: LICENSE.txt
    :alt: License

.. |badge_documentation| image:: https://img.shields.io/github/actions/workflow/status/rl-institut/super-repo/gh-pages.yml?branch=production
    :target: https://rl-institut.github.io/super-repo/
    :alt: Documentation

.. |badge_contributing| image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :alt: contributions

.. |badge_contributors| image:: https://img.shields.io/badge/all_contributors-1-orange.svg?style=flat-square
    :alt: contributors

.. |badge_issue_open| image:: https://img.shields.io/github/issues-raw/rl-institut/super-repo
    :alt: open issues

.. |badge_issue_closes| image:: https://img.shields.io/github/issues-closed-raw/rl-institut/super-repo
    :alt: closes issues

.. |badge_pr_open| image:: https://img.shields.io/github/issues-pr-raw/rl-institut/super-repo
    :alt: open pull requests

.. |badge_pr_closes| image:: https://img.shields.io/github/issues-pr-closed-raw/rl-institut/super-repo
    :alt: closes pull requests

.. |badge_black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black linting badge
