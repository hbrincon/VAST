# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.  If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
#sys.path.insert(0, os.path.abspath('../../VoidFinder/'))
#sys.path.insert(1, os.path.abspath('../../Vsquared/'))
#sys.path.insert(0, os.path.abspath('../../VoidFinder/vast/voidfinder/'))
print('sys.path:', sys.path)
for entry in sys.path:
    if 'site-packages' in entry:
        print(entry)

#from vast.voidfinder import filter_galaxies
import vispy
print('vispy:', vispy)
#print('vf.__file__:', vf.__file__)
#parent_directory = os.path.dirname(vast.__file__)
#print(os.listdir(parent_directory + '/voidfinder/'))

# -- Project information -----------------------------------------------------

project = 'VAST'
copyright = "2021, Kelly A. Douglass, Dylan Veyrat, Stephen W. O'Neill Jr., Segev BenZvi, Fatima Zaidouni, Michaela Guzzetti"
author = "Kelly A. Douglass, Dylan Veyrat, Stephen W. O'Neill Jr., Segev BenZvi, Fatima Zaidouni, Michaela Guzzetti"

# The full version, including alpha/beta/rc tags
release = 'v1.0.0-alpha'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'numpydoc',
    'sphinxarg.ext'
    #'pyquickhelper.sphinxext.sphinx_autosignature'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['.templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


#html_logo = 'filename.png'


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['.static']
