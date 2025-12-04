#!/usr/bin/env python3
"""
Simple script to convert markdown files to HTML with consistent styling.
Usage: python md_to_html.py [markdown_file]
Or run without arguments to convert all markdown files in docs/
"""

import sys
import os
from pathlib import Path
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension

# HTML template with navigation
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CCAI9012 - {title}</title>
    <link rel="stylesheet" href="docs-style.css">
</head>
<body>
    <div class="container">
        <nav id="sidebar">
            <div class="sidebar-header">
                <h2>CCAI9012</h2>
            </div>
            <ul class="nav-menu">
                <li><a href="index.html"{home_active}>Home</a></li>
                <li><a href="timetable.html"{timetable_active}>Timetable</a></li>
                <li><a href="installation.html"{installation_active}>Installation Guide</a></li>
                <li><a href="starter_kits.html"{starter_active}>Starter Kits</a></li>
                <li><a href="reading_material.html"{reading_active}>Reading Materials</a></li>
                <li><a href="datasets.html"{datasets_active}>Datasets Reference</a></li>
                <li><a href="api/index.html">API Documentation</a></li>
            </ul>
        </nav>

        <main id="content">
{content}
        </main>
    </div>
</body>
</html>
"""

# Page titles and active states
PAGE_CONFIG = {
    'installation.md': {
        'title': 'Installation Guide',
        'html_file': 'installation.html',
        'active': 'installation_active'
    },
    'starter_kits.md': {
        'title': 'Starter Kits',
        'html_file': 'starter_kits.html',
        'active': 'starter_active'
    },
    'reading_material.md': {
        'title': 'Reading Materials',
        'html_file': 'reading_material.html',
        'active': 'reading_active'
    },
    'datasets.md': {
        'title': 'Datasets Reference',
        'html_file': 'datasets.html',
        'active': 'datasets_active'
    },
    'timetable.md': {
        'title': 'Course Timetable',
        'html_file': 'timetable.html',
        'active': 'timetable_active'
    }
}


def convert_md_to_html(md_file_path, output_dir=None):
    """Convert a markdown file to HTML with styling"""

    md_file = Path(md_file_path)
    if not md_file.exists():
        print(f"Error: File {md_file} not found")
        return False

    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Get page configuration
    config = PAGE_CONFIG.get(md_file.name)
    if not config:
        print(f"Warning: No configuration found for {md_file.name}, using defaults")
        title = md_file.stem.replace('_', ' ').title()
        html_filename = md_file.stem + '.html'
        active_key = None
    else:
        title = config['title']
        html_filename = config['html_file']
        active_key = config['active']

    # Convert markdown to HTML
    md = markdown.Markdown(extensions=[
        TableExtension(),
        FencedCodeExtension(),
        'nl2br',  # Convert newlines to <br>
        'sane_lists'  # Better list handling
    ])
    html_content = md.convert(md_content)

    # Set active navigation state
    nav_states = {
        'home_active': '',
        'installation_active': '',
        'starter_active': '',
        'reading_active': '',
        'datasets_active': '',
        'timetable_active': ''
    }
    if active_key:
        nav_states[active_key] = ' class="active"'

    # Fill template
    full_html = HTML_TEMPLATE.format(
        title=title,
        content=html_content,
        **nav_states
    )

    # Determine output path
    if output_dir:
        output_path = Path(output_dir) / html_filename
    else:
        # Output directly to docs/ directory (same as markdown files)
        output_path = md_file.parent / html_filename

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_html)

    print(f"✓ Converted: {md_file.name} → {output_path}")
    return True


def convert_all_docs():
    """Convert all markdown files in docs/ directory"""
    docs_dir = Path(__file__).parent
    md_files = list(docs_dir.glob('*.md'))

    if not md_files:
        print("No markdown files found in docs/ directory")
        return

    print(f"Found {len(md_files)} markdown file(s)")
    print("-" * 50)

    success_count = 0
    for md_file in md_files:
        if convert_md_to_html(md_file):
            success_count += 1

    print("-" * 50)
    print(f"Converted {success_count}/{len(md_files)} files successfully")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Convert specific file
        md_file = sys.argv[1]
        convert_md_to_html(md_file)
    else:
        # Convert all markdown files
        convert_all_docs()
