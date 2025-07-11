# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os,sys, types
# sys.path.insert(0, os.path.abspath('..'))
# sys.path.insert(0,os.path.abspath('../pywiggle'))
sys.modules['pywiggle._wiggle'] = types.ModuleType('pywiggle._wiggle')

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.viewcode','sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Present auto-documented members in source order (rather than alphabetical).
autodoc_member_order = 'bysource'

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'wiggle'
copyright = u"Copyright (c) 2025-2035, Mathew Syriac Madhavacheril"
author = u"Mathew Syriac Madhavacheril"

def get_version():
    try:
        from importlib.metadata import version
        return version("pywiggle")
    except Exception:
        return "unknown"

release = get_version()
version = ".".join(release.split(".")[:2])


# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False



# -- Options for HTML output -------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a
# theme further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Options for HTMLHelp output ---------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'pywiggledoc'
