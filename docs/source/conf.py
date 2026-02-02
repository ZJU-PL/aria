# sys.path.insert(0, os.path.abspath('..'))

import os
import re

# -- Project information -----------------------------------------------------

project = "ARIA"
copyright = "2024-2025, ZJU Programming Languages and Automated Reasoning Group"
author = "ZJU Programming Languages and Automated Reasoning Group"

# Read version from pyproject.toml
pyproject_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'pyproject.toml')
with open(pyproject_path, 'r') as f:
    content = f.read()
    version_match = re.search(r'^version = ["\']([^"\']+)["\']', content, re.MULTILINE)
    if version_match:
        release = version_match.group(1)
    else:
        release = "0.1.0"

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx_rtd_theme'
]

# Alternative themes you could use instead:
# For alabaster theme (built-in)
# html_theme = 'alabaster'

# For classic theme (built-in)
# html_theme = 'classic'

# For nature theme (built-in)
# html_theme = 'nature'

templates_path = ['_templates']
exclude_patterns = []

# HTML output options
html_theme = 'sphinx_rtd_theme'  # ReadTheDocs theme
# html_static_path = ['_static']
html_title = 'ARIA Documentation'

# LaTeX output options
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
}
