"""NeuroMesh Documentation."""

from __future__ import annotations

from datetime import datetime

project = "NeuroMesh"
author = "Long Quang"
project_copyright = f"{datetime.now():%Y}, {author}"
copyright = project_copyright
release = "1.0.0"

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

html_js_files = [
    "mermaid.min.js",
    "mermaid-init.js",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "tasklist",
]

suppress_warnings = [
    "docutils",
    "toc.not_included",
    "ref.ref",
    "misc.highlighting_failure",
]

html_theme = "sphinx_book_theme"
html_title = project
html_theme_options = {
    "repository_url": "https://github.com/arplaboratory/neuromesh",
    "use_repository_button": True,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
html_static_path = ["_static"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]