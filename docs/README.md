# Documentation Site Builder

This directory provides a lightweight static documentation site generation workflow:

- Put Markdown source files in `docs/md/`
- Output generated HTML pages to `docs/` (repository `docs/` is the site root)
- Homepage is generated to `docs/index.html` (GitHub Pages expects this)
- Sidebar navigation / page mapping is centrally configured in `docs/pages.json` (supports multi-level `children`, suitable for multi-level directories like `starter_kits/`)
- The API Reference is an independent Sphinx + PyData Theme site. `docs/generate_api_doc.py` generates module sources and builds them directly into `docs/api/`; `md_to_html.py` does not rewrite API HTML or CSS.

## File structure and execution logic

- `docs/md/`: Markdown source files for site content (e.g., `docs/md/datasets.md`, `docs/md/starter_kits/index.md`)
- `docs/`: Final HTML output (site root)
  - Example: `docs/timetable.html`, `docs/starter_kits/index.html`
- `docs/index.html`: Homepage (generated from `docs/md/index.md`)
- `docs/api_source/`: Sphinx configuration, authored landing page, generated module sources, templates, and API-specific theme overrides
- `docs/api/`: Sphinx API documentation output directory (not produced by Markdown conversion)
  - Example: `docs/api/ccai9012/index.html`
- `docs/pages.json`:
  - Defines the navigation hierarchy (`pages` + `children`)
  - Defines the Markdown → HTML mapping (`source` / `output`)
  - Can configure homepage cards (`home_card`), etc.

## Usage (recommended order)

### Build everything with one command

Activate the course environment, then run the build script from the repository root:

```bash
conda activate ccai9012
bash docs/make_docs.sh
```

The script can also be called from inside `docs/`. It first rebuilds the complete API Reference and then converts every non-empty Markdown page under `docs/md/`. Only pages registered in `docs/pages.json` appear in the global sidebar.

### Step 1: Generate the API Reference for `ccai9012`

You can run this from either the repository root or the `docs/` directory (recommended to run under `docs/` since paths are more intuitive):

```bash
cd docs
python generate_api_doc.py
```

This step regenerates the public-object module pages and runs a strict Sphinx build (`-W --keep-going`) into:

- `docs/api/ccai9012/...`

### Step 2: Convert the Markdown under `docs/md/` into site HTML

```bash
cd docs
python md_to_html.py
```

This step will:

- Convert every non-empty Markdown file under `docs/md/` into HTML
- Read `docs/pages.json` to determine which pages appear in the global sidebar and where registered pages are written
- Write all Markdown pages under `docs/` (the site root), including nested paths such as `docs/starter_kits/...`
- Generate a unified sidebar, stable `h2`/`h3` anchors, and responsive “On this page” navigation
- Leave the independently generated API subsite unchanged

## Adding new pages (supports multi-level directories, e.g., starter-kits)

There are two types of course page:

- **Linked-only page:** generated and reachable from links in another page, but absent from the global sidebar. Add its Markdown file under `docs/md/`; no `pages.json` entry is needed. By default, `docs/md/example.md` becomes `docs/example.html`, preserving subdirectories.
- **Navigation page:** generated and displayed in the global sidebar. Add its Markdown file and register it in `docs/pages.json`.

To add a navigation page, complete these steps:

1. Add the Markdown source under `docs/md/`.

   Examples: `docs/md/weekly_syllabus.md` or `docs/md/starter_kits/extra/topic_a.md`.

2. Add a node to the appropriate position in the `pages` tree in `docs/pages.json`. To display the page below an existing page in the sidebar, place the new node in that parent's `children` list.

```json
{
  "key": "weekly_syllabus",
  "label": "Weekly Syllabus",
  "title": "Weekly Syllabus",
  "source": "weekly_syllabus.md",
  "output": "weekly_syllabus.html"
}
```

The fields mean:

- `key`: unique identifier used to highlight the active navigation item.
- `label`: text displayed in the sidebar.
- `title`: text used in the browser's `<title>`.
- `source`: Markdown path relative to `docs/md/`.
- `output`: generated HTML path relative to `docs/`.
- `children`: optional list of nested pages; it can be nested to additional levels.
- `home_card`: optional homepage-card settings. Omit it when the page should appear only in navigation.

Do not add `href` when a page already has `source` and `output`; the generator derives its link from `output`.

3. Run the complete build and check that every page is converted:

```bash
conda activate ccai9012
bash docs/make_docs.sh
```

The final line reports the course-site entry point. The course-page stage reports the numbers of navigation and linked-only pages, followed by `Converted N/N files successfully`; both final numbers must match.

## Adding API Reference pages

Do not edit `docs/api/` or `docs/api_source/generated/` manually because both contain generated files.

- Add or update public classes and functions in a module under `ccai9012/` and provide Google-style docstrings.
- When adding a new public module, add its pedagogical overview, use cases, workflow, and related Starter Kits to `MODULE_METADATA` in `docs/generate_api_doc.py`.
- Run `bash docs/make_docs.sh`. Sphinx automatically creates the module page and individual class, function, and method pages.
- The API build uses warnings as errors, so missing metadata or invalid documentation references stop the build.
