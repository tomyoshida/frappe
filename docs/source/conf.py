# Configuration file for the Sphinx documentation builder.
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Ensure the package is importable for autodoc
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'FRAPPE'
copyright = '2026, Tomohiro C. Yoshida'
author = 'Tomohiro C. Yoshida'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'autoapi.extension',
    #'sphinx.ext.autodoc',
    #'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
    'nbsphinx',
    'IPython.sphinxext.ipython_console_highlighting' ,
    'sphinxcontrib.video',
]

autoapi_type = 'python'
autoapi_dirs = ['../../frappe'] 

nbsphinx_execute = 'never'

autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
    #'imported-members',
]

autosummary_generate = True

# Mock heavy/optional dependencies so autodoc works on Read the Docs

autodoc_mock_imports = [
    "astroquery",
    "astroquery.linelists",
    "astroquery.linelists.cdms",
    "astropy",
    "astropy.constants",
    "casatools",
    "casatasks", 
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
exclude_patterns = ['_build', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

def setup(app):
    app.add_css_file('custom.css')









######## notebook settings ###########

import nbformat

def truncate_notebook_outputs(app, docname, source):
    # docname を確認し、ipynbファイルでない場合はスキップ
    # Read the Docsのビルド環境でも安全なように拡張子チェックを簡略化
    if not app.env.doc2path(docname).endswith('.ipynb'):
        return

    try:
        # source[0] はリストの第1要素にドキュメントの内容が文字列で入っています
        nb = nbformat.reads(source[0], as_version=4)
        changed = False

        for cell in nb.cells:
            if cell.cell_type == 'code' and 'outputs' in cell:
                for output in cell.outputs:
                    # 1. 標準出力 (stream: print文など) を3行に制限
                    if output.output_type == 'stream' and 'text' in output:
                        lines = output.text.splitlines()
                        if len(lines) > 5:
                            output.text = '\n'.join(lines[:5]) + '\n... [Output Truncated]'
                            changed = True
                    
                    # 2. 実行結果 (execute_result: セル末尾の評価値) を3行に制限
                    elif output.output_type == 'execute_result' and 'data' in output:
                        if 'text/plain' in output.data:
                            text_data = output.data['text/plain']
                            lines = text_data.splitlines()
                            if len(lines) > 5:
                                output.data['text/plain'] = '\n'.join(lines[:5]) + '\n... [Output Truncated]'
                                changed = True
        
        if changed:
            # 変更があった場合のみ書き戻す
            source[0] = nbformat.writes(nb)
            
    except Exception:
        # 万が一解析に失敗してもビルドを止めない
        pass

def setup(app):
    # source-readイベントに接続
    app.connect('source-read', truncate_notebook_outputs)