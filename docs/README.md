# Markdown to HTML Converter

This is a simple, lightweight tool to automatically convert Markdown files into HTML files with a consistent style and navigation bar.

## Usage

### Method 1: Convert all Markdown files (recommended)

```bash
cd docs
python md_to_html.py
```

This will automatically convert all `.md` files under the `docs/` directory into HTML and save them to the `docs/ccai9012/` directory.

### Method 2: Convert a single file

```bash
python md_to_html.py datasets.md
```

## Features

- ✅ Automatically maintains a consistent navigation bar and style
- ✅ Supports Markdown tables
- ✅ Supports code block syntax highlighting
- ✅ Automatically sets the correct page title
- ✅ Automatically highlights the current page in the navigation

## Workflow

1. Edit your Markdown files (e.g., `datasets.md`)
2. Run `python md_to_html.py`
3. The HTML files are automatically updated in the `ccai9012/` directory
4. Open the HTML file in your browser to view the result

## Dependencies

You need the `markdown` library:

```bash
pip install markdown
```

## Notes

- Markdown files should be placed in the `docs/` directory
- HTML output will be saved to the `docs/ccai9012/` directory
- Make sure the `docs-style.css` file exists in the `ccai9012/` directory
