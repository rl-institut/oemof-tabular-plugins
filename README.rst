
.. figure:: https://user-images.githubusercontent.com/14353512/185425447-85dbcde9-f3a2-4f06-a2db-0dee43af2f5f.png
    :align: left
    :target: https://github.com/rl-institut/super-repo/
    :alt: Repo logo

=====================
oemof-tabular-plugins
=====================

**A dummy repo to pitch an idea to oemof community**

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

Introduction
============
This is to showcase how this repository could be used to complement `oemof-tabular <https://github.com/oemof/oemof-tabular>`_ to add constraints specific to certain uses of it

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
