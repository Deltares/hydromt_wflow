.. _readme:

=======================================
HydroMT-Wflow: Wflow plugin for HydroMT
=======================================

|pypi| |conda_forge| |docs_latest| |docs_stable| |codecov| |license| |doi| |binder|

What is the HydroMT-Wflow plugin
--------------------------------
HydroMT_ (Hydro Model Tools) is an open-source Python package that facilitates the process of
building and analyzing spatial geoscientific models with a focus on water system models.
It does so by automating the workflow to go from raw data to a complete model instance which
is ready to run and to analyze model results once the simulation has finished.
This plugin provides an implementation of the model API for the Wflow_ model.

Why HydroMT-Wflow?
------------------
Setting up distributed hydrological models typically requires many (manual) steps
to process input data and might therefore be time consuming and hard to reproduce.
Especially improving models based on global-local geospatial datasets, which are
rapidly becoming available at increasingly high resolutions, might be challenging.
HydroMT-Wflow aims to make the Wflow model building and updating processes **fast**, **modular** and **reproducible**
and to facilitate the analysis of the model results.

How to use HydroMT-Wflow?
-------------------------
The HydroMT-Wflow plugin can be used as a **command line** application, which provides commands to *build*,
*update* and *clip* a Wflow model with a single line, or **from python** to exploit its rich interface.
You can learn more about how to use HydroMT-Wflow in its `online documentation. <https://deltares.github.io/hydromt_wflow/latest/getting_started/intro>`_
For a smooth installing experience we recommend installing HydroMT-Wflow and its dependencies
from conda-forge in a clean environment, see `installation guide. <https://deltares.github.io/hydromt_wflow/latest/getting_started/installation>`_

How to cite?
------------
For publications, please cite our work using the DOI provided in the Zenodo badge |doi| that points to the latest release.

How to contribute?
-------------------
If you find any issues in the code or documentation feel free to leave an issue on the `github issue tracker. <https://github.com/Deltares/hydromt_wflow/issues>`_
You can find information about how to contribute to the HydroMT project at our `contributing page. <https://deltares.github.io/hydromt/latest/dev/contributing>`_

HydroMT seeks active contribution from the (hydro) geoscientific community.
So far, it has been developed and tested with a range of `Deltares <https://www.deltares.nl/en/>`_ models, but
we believe it is applicable to a much wider set of geoscientific models and are
happy to discuss how it can be implemented for your model.

.. _Hydromt: https://deltares.github.io/hydromt/latest/
.. _Wflow: https://github.com/Deltares/Wflow.jl

.. |codecov| image:: https://codecov.io/gh/Deltares/hydromt_wflow/branch/main/graph/badge.svg?token=ss3EgmwHhH
    :target: https://codecov.io/gh/Deltares/hydromt_wflow

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
