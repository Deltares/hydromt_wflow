# -*- coding: utf-8 -*-
import os
from pathlib import Path
import shutil
import sys
import hydromt_wflow

# -- Path setup --------------------------------------------------------------
DOCS_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, DOCS_ROOT.as_posix())

STATIC_DIR = DOCS_ROOT / "_static"
BUILD_DIR = DOCS_ROOT / "_build"
TEMPLATES_DIR = DOCS_ROOT / "_templates"
EXAMPLES_SRC = DOCS_ROOT.parent / "examples"
EXAMPLES_DOCS = DOCS_ROOT / "_examples"

def _rel_path(target: Path, relative_to: Path = DOCS_ROOT) -> str:
    return Path(os.path.relpath(target, start=relative_to)).as_posix()

# Copy notebooks to include in docs
shutil.rmtree(EXAMPLES_DOCS, ignore_errors=True)
shutil.copytree(EXAMPLES_SRC, EXAMPLES_DOCS)

for path in [DOCS_ROOT, STATIC_DIR, TEMPLATES_DIR, EXAMPLES_DOCS]:
    if not path.exists():
        raise FileNotFoundError(f"Expected documentation folder not found: {path.as_posix()}. Check your setup.")

# -- Project information -----------------------------------------------------
project = "HydroMT Wflow"
copyright = "Deltares"
author = "Dirk Eilander"
version = hydromt_wflow.__version__
bare_version = hydromt_wflow.__version__
doc_version = bare_version[: bare_version.find("dev") - 1]

# -- General configuration ------------------------------------------------
extensions = [
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
]
templates_path = [_rel_path(TEMPLATES_DIR)]
source_suffix = [".rst"] # nbsphinx self-registers for .ipynb files
master_doc = "index"
language = "en"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]
pygments_style = "sphinx"
todo_include_todos = False

autosummary_generate = True
add_module_names = False
autodoc_default_options = {
    "member-order": "bysource",
    "autoclass-content": "init",
}

# -- HTML output options ----------------------------------------------
html_theme = "pydata_sphinx_theme"
html_logo = _rel_path(STATIC_DIR / "hydromt-icon.svg")
html_favicon = _rel_path(STATIC_DIR / "hydromt-icon.svg")
html_static_path = [_rel_path(STATIC_DIR)]
html_css_files = ["theme-deltares.css"]
html_theme_options = {
    "show_nav_level": 1,
    "navbar_align": "left",
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Deltares/hydromt_wflow",
            "icon": "fab fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Wflow",
            "url": "https://deltares.github.io/Wflow.jl/dev/",
            "icon": _rel_path(STATIC_DIR / "wflow_logo.png"),
            "type": "local",
        },
        {
            "name": "Deltares",
            "url": "https://www.deltares.nl/en/",
            "icon": _rel_path(STATIC_DIR / "deltares-blue.svg"),
            "type": "local",
        },
    ],
    "external_links": [
        {
            "name": "HydroMT core",
            "url": "https://deltares.github.io/hydromt/latest/index.html",
        },
    ],
    "logo": {
        "text": "HydroMT Wflow",
    },
    "navbar_center": ["version-switcher", "navbar-nav"],
    "navbar_end": ["navbar-icon-links"],
    "switcher": {
        "json_url": "https://raw.githubusercontent.com/Deltares/hydromt_wflow/gh-pages/switcher.json",
        "version_match": doc_version,
    },
}
html_context = {
    "github_url": "https://github.com",
    "github_user": "Deltares",
    "github_repo": "hydromt_wflow",
    "github_version": "main",
    "doc_path": "docs",
    "default_mode": "light",
}

# -- Output Options ------------------------------------------
htmlhelp_basename = "hydromt_wflow_doc"
latex_documents = [
    (
        master_doc,
        "hydromt_wflow.tex",
        "HydroMT wflow plugin Documentation",
        [author],
        "manual",
    ),
]
man_pages = [
    (master_doc, "hydromt_wflow", "HydroMT wflow plugin Documentation", [author], 1)
]
texinfo_documents = [
    (
        master_doc,
        "hydromt_wflow",
        "HydroMT wflow plugin Documentation",
        author,
        "HydroMT wflow plugin",
        "Build and analyze wflow models like a data-wizard.",
        "Miscellaneous",
    ),
]

# -- Intersphinx -----------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    "rasterio": ("https://rasterio.readthedocs.io/en/latest", None),
    "geopandas": ("https://geopandas.org/en/stable", None),
    "xarray": ("https://xarray.pydata.org/en/stable", None),
    "hydromt": ("https://deltares.github.io/hydromt/latest/", None),
}

# -- nbsphinx --------------------------------------------------------------
# create a label for each notebook: ``example-<notebook-filename>`` (no path, no suffix)
nbsphinx_prolog = r"""
{% set filename = env.docname.split('/')[-1] %}

.. _example-{{ filename }}:

.. TIP::

    .. raw:: html

        <div>
            For an interactive online version click here:
            <a href="https://mybinder.org/v2/gh/Deltares/hydromt_wflow/main?urlpath=lab/tree/examples/{{ filename }}.ipynb" target="_blank" rel="noopener noreferrer"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg"></a>
        </div>
"""
nbsphinx_execute = "always"
nbsphinx_timeout = 300

# -- Linkcheck -------------------------------------------------------------
# nitpicky = True # warn about all references where the target cannot be found
linkcheck_ignore = [
    r'https://localhost:\d+/',
    'https://doi.org/10.1029/2018JG004881',
    r'https://deltares.github.io/hydromt_wflow/.*'
]
