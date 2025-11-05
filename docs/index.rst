.. _main_sections:

=======================================
HydroMT-Wflow: Wflow plugin for HydroMT
=======================================

|pypi| |conda_forge| |docs_latest| |docs_stable| |license| |doi| |binder| |sonarqube_coverage| |sonarqube|

HydroMT_ (Hydro Model Tools) is an open-source Python package that facilitates the process of
building and analyzing spatial geoscientific models with a focus on water system models.
It does so by automating the workflow to go from raw data to a complete model instance which
is ready to run and to analyze model results once the simulation has finished.

This plugin provides an implementation of the model API for the Wflow_ model.

.. grid:: 2
    :gutter: 2

    .. grid-item-card::
        :text-align: center
        :link: getting_started/intro
        :link-type: doc

        :octicon:`rocket;5em;sd-text-icon blue-icon`
        +++
        **Getting Started**

        First time user? Learn how to install, configure, and start using HydroMT-Wflow
        effectively.

    .. grid-item-card::
        :text-align: center
        :link: user_guide/index
        :link-type: doc

        :octicon:`book;5em;sd-text-icon blue-icon`
        +++
        **User Guide**

        Explore detailed guides on model setup, configuration, and workflows.

    .. grid-item-card::
        :text-align: center
        :link: user_guide/2_sbm_model/1_methods_components
        :link-type: doc

        :octicon:`list-unordered;5em;sd-text-icon blue-icon`
        +++
        **Wflow SBM methods**

        Regular user? Here is a quick access to the available methods to prepare a Wflow
        SBM model.

    .. grid-item-card::
        :text-align: center
        :link: user_guide/3_sediment_model/1_methods_components
        :link-type: doc

        :octicon:`list-unordered;5em;sd-text-icon blue-icon`
        +++
        **Wflow Sediment methods**

        Regular user? Here is a quick access to the available methods to prepare a Wflow
        Sediment model.

    .. grid-item-card::
        :text-align: center
        :link: https://deltares.github.io/hydromt/stable/guides/user_guide/model_build.html

        :octicon:`device-desktop;5em;sd-text-icon blue-icon`
        +++
        **Building models with HydroMT**

        Learn more about the basics of model building with HydroMT in the core documentation.

    .. grid-item-card::
        :text-align: center
        :link: https://deltares.github.io/hydromt/stable/guides/advanced_user/data_prepare_cat.html

        :octicon:`stack;5em;sd-text-icon blue-icon`
        +++
        **Preparing a Data Catalog**

        Learn more about the HydroMT Data Catalog and how to prepare your own in the
        core documentation.


    .. grid-item-card::
        :text-align: center
        :link: api/index
        :link-type: doc

        :octicon:`code-square;5em;sd-text-icon blue-icon`
        +++
        **API Reference**

        Access the full API documentation for HydroMT-Wflow's modules and functions.

    .. grid-item-card::
        :text-align: center
        :link: changelog
        :link-type: doc

        :octicon:`graph;5em;sd-text-icon blue-icon`
        +++
        **What's new?**

        Want to learn what are the new developments and features in the new release?
        Check the changelog.

.. toctree::
   :titlesonly:
   :includehidden:
   :hidden:

   getting_started/intro.rst
   user_guide/index.rst
   API <api/index.rst>
   dev_guide/intro.rst


.. _Hydromt: https://deltares.github.io/hydromt/latest/
.. _Wflow: https://github.com/Deltares/Wflow.jl

.. |sonarqube| image:: https://sonarcloud.io/api/project_badges/measure?project=Deltares_hydromt_wflow&metric=alert_status
    :target: https://sonarcloud.io/summary/new_code?id=Deltares_hydromt_wflow
    :alt: SonarQube status

.. |sonarqube_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=Deltares_hydromt_wflow&metric=coverage
    :alt: Coverage
    :target: https://sonarcloud.io/summary/new_code?id=Deltares_hydromt_wflow

.. |docs_latest| image:: https://img.shields.io/badge/docs-latest-brightgreen.svg
    :target: https://deltares.github.io/hydromt_wflow/latest
    :alt: Latest developers docs

.. |docs_stable| image:: https://img.shields.io/badge/docs-stable-brightgreen.svg
    :target: https://deltares.github.io/hydromt_wflow/stable
    :alt: Stable docs last release

.. |pypi| image:: https://img.shields.io/pypi/v/hydromt_wflow.svg?style=flat
    :target: https://pypi.org/project/hydromt_wflow/
    :alt: PyPI

.. |conda_forge| image:: https://anaconda.org/conda-forge/hydromt_wflow/badges/version.svg
    :target: https://anaconda.org/conda-forge/hydromt_wflow
    :alt: Conda-Forge

.. |binder| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Deltares/hydromt_wflow/main?urlpath=lab/tree/examples

.. |doi| image:: https://zenodo.org/badge/356210291.svg
    :alt: Zenodo
    :target: https://zenodo.org/badge/latestdoi/356210291

.. |license| image:: https://img.shields.io/github/license/Deltares/hydromt_wflow
    :alt: License
    :target: https://github.com/Deltares/hydromt_wflow/blob/main/LICENSE
