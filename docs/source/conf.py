# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Ensure the package is importable for autodoc
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'frappe'
copyright = '2026, Tomohiro C. Yoshida'
author = 'Tomohiro C. Yoshida'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
]

autosummary_generate = True

# Mock heavy/optional dependencies so autodoc works on Read the Docs
autodoc_mock_imports = [
    "astroquery",
    "astroquery.linelists",
    "astroquery.linelists.cdms",
    "astropy",
    "astropy.constants",
    "jax",
    "jax.numpy",
    "jax.random",
    "jax.scipy",
    "jax.scipy.interpolate",
    "numpyro",
    "numpyro.distributions",
    "numpyro.infer",
    "numpyro.infer.autoguide",
    "numpyro.optim",
    "matplotlib",
    "matplotlib.pyplot",
    "scipy",
    "scipy.special",
]

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']



