"""NeuroMesh documentation"""

from __future__ import annotations

from datetime import datetime

project = "NeuroMesh"
author = "ARPL"
project_copyright = f"{datetime.now():%Y}, {author}"
release = "humble"

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

html_theme = "sphinx_book_theme"
html_title = project
html_theme_options = {
    "repository_url": "https://github.com/arplaboratory/neuromesh",
    "use_repository_button": True,
}

html_static_path = ["_static"]
html_js_files = [
    "mermaid.min.js",
    "mermaid-init.js",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

suppress_warnings = [
    "docutils",
    "toc.not_included",
    "ref.ref",
    "misc.highlighting_failure",
]
