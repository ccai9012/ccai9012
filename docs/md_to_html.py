#!/usr/bin/env python3
"""Build the Markdown-based CCAI9012 course website.

Page mapping and navigation come from ``docs/pages.json``. API documentation is
built independently by ``generate_api_doc.py`` and is never rewritten here.
"""

from __future__ import annotations

import html as html_lib
import json
import os
import sys
from pathlib import Path

import markdown
from markdown.extensions.fenced_code import FencedCodeExtension
from markdown.extensions.tables import TableExtension
from markdown.extensions.toc import TocExtension


DOCS_DIR = Path(__file__).resolve().parent
MD_ROOT = DOCS_DIR / "md"
SITE_ROOT = DOCS_DIR


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title_full}</title>
    <link rel="stylesheet" href="{css_href}">
    <script src="{script_href}" defer></script>
</head>
<body>
    <a class="skip-link" href="#content">Skip to content</a>
    <div class="container">
        <nav id="sidebar" aria-label="Course navigation">
            <div class="sidebar-header"><h2>{site_name}</h2></div>
            <ul class="nav-menu">
{nav_items}
            </ul>
        </nav>
        <div class="page-shell{page_shell_modifier}">
            <main id="content">
{mobile_toc}
{content}
            </main>
{desktop_toc}
        </div>
    </div>
</body>
</html>
"""


def _load_site_config() -> dict:
    """Load and validate the site navigation configuration."""
    config_path = DOCS_DIR / "pages.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc

    if not isinstance(config, dict) or config.get("version") != 1:
        raise ValueError("docs/pages.json must be a version 1 JSON object")
    if not isinstance(config.get("site"), dict) or not isinstance(config.get("pages"), list):
        raise ValueError("docs/pages.json requires object 'site' and list 'pages' fields")

    seen_keys: set[str] = set()

    def validate(nodes: list[dict]) -> None:
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError("Every pages entry must be an object")
            key = node.get("key")
            if not isinstance(key, str) or not key or key in seen_keys:
                raise ValueError(f"Invalid or duplicated page key: {key!r}")
            seen_keys.add(key)
            for field in ("href", "source", "output"):
                value = node.get(field)
                if value is not None and not isinstance(value, str):
                    raise ValueError(f"pages[{key}].{field} must be a string")
            href = node.get("href")
            if isinstance(href, str) and (href.startswith("/") or href.startswith("..")):
                raise ValueError(f"pages[{key}].href must stay under docs/: {href!r}")
            children = node.get("children", [])
            if not isinstance(children, list):
                raise ValueError(f"pages[{key}].children must be a list")
            validate(children)

    validate(config["pages"])
    return config


def _flatten_pages(config: dict) -> list[dict]:
    """Return navigation nodes in display order with hierarchy metadata."""
    flattened: list[dict] = []

    def walk(nodes: list[dict], level: int, parents: tuple[str, ...], child_class: str | None) -> None:
        for node in nodes:
            inherited_class = node.get("nav_children_class") or child_class
            flattened.append({
                "node": node,
                "level": level,
                "parents": parents,
                "children_class": inherited_class,
            })
            children = node.get("children", [])
            if children:
                walk(children, level + 1, parents + (node["key"],), inherited_class)

    walk(config["pages"], 0, (), None)
    return flattened


def _page_href(node: dict) -> str | None:
    """Return a configured page's site-root-relative output URL."""
    if node.get("href"):
        return node["href"]
    if node.get("output"):
        return node["output"]
    if node.get("source"):
        return Path(node["source"]).with_suffix(".html").as_posix()
    return None


def _pages_by_source(config: dict) -> dict[str, dict]:
    pages: dict[str, dict] = {}
    for entry in _flatten_pages(config):
        node = entry["node"]
        source = node.get("source")
        if not source:
            continue
        if source in pages:
            raise ValueError(f"Duplicated Markdown source in pages.json: {source}")
        pages[source] = {
            "title": node.get("title") or Path(source).stem.replace("_", " ").title(),
            "output": node.get("output") or Path(source).with_suffix(".html").as_posix(),
            "nav_key": node["key"],
        }
    return pages


def _base_href(output_path: Path) -> str:
    depth = len(output_path.parent.resolve().relative_to(SITE_ROOT.resolve()).parts)
    return "../" * depth


def _css_href(config: dict, output_path: Path) -> str:
    css = config["site"].get("css", {})
    css_path = css.get("path") if isinstance(css, dict) else None
    if isinstance(css_path, str) and css_path.strip():
        return _base_href(output_path) + css_path.strip()
    legacy = css.get("subdir_href") if output_path.parent != SITE_ROOT else css.get("top_level_href")
    return legacy if isinstance(legacy, str) and legacy else "docs-style.css"


def _new_markdown() -> markdown.Markdown:
    """Create a converter with stable, unique h2/h3 anchors."""
    return markdown.Markdown(extensions=[
        TableExtension(),
        FencedCodeExtension(),
        TocExtension(permalink=False, toc_depth="2-3"),
        "sane_lists",
        "smarty",
    ])


def _render_nav(config: dict, *, base_href: str, active_key: str | None) -> str:
    lines: list[str] = []
    for entry in _flatten_pages(config):
        node = entry["node"]
        href = _page_href(node)
        if not href:
            continue
        label = node.get("label") or node.get("title") or node["key"]
        active = node["key"] == active_key or node["key"] in next(
            (item["parents"] for item in _flatten_pages(config) if item["node"]["key"] == active_key),
            (),
        )
        anchor_class = ' class="active"' if active else ""
        item_class = entry["children_class"] if entry["level"] > 0 else None
        item_attr = f' class="{html_lib.escape(item_class)}"' if item_class else ""
        lines.append(
            f'                <li{item_attr}><a href="{base_href}{html_lib.escape(href)}"'
            f'{anchor_class}>{html_lib.escape(str(label))}</a></li>'
        )
    return "\n".join(lines)


def _render_toc_items(tokens: list[dict]) -> str:
    items: list[str] = []

    def walk(children: list[dict]) -> None:
        for token in children:
            level, anchor, name = token.get("level"), token.get("id"), token.get("name")
            if level in (2, 3) and isinstance(anchor, str) and isinstance(name, str):
                items.append(
                    f'<li class="toc-level-{level}"><a href="#{html_lib.escape(anchor)}">'
                    f'{html_lib.escape(name)}</a></li>'
                )
            nested = token.get("children")
            if isinstance(nested, list):
                walk(nested)

    walk(tokens)
    return "\n".join(items)


def _render_on_this_page(tokens: list[dict], *, mobile: bool) -> str:
    items = _render_toc_items(tokens)
    if not items:
        return ""
    if mobile:
        return "\n".join([
            '<details class="page-toc-mobile">',
            '  <summary>On this page</summary>',
            '  <nav aria-label="On this page"><ul>',
            items,
            '  </ul></nav>',
            '</details>',
        ])
    return "\n".join([
        '<aside class="page-toc">',
        '  <nav aria-label="On this page">',
        '    <p class="page-toc-title">On this page</p>',
        '    <ul>',
        items,
        '    </ul>',
        '  </nav>',
        '</aside>',
    ])


def _render_home_cards(config: dict) -> str:
    cards: list[tuple[int, str]] = []
    home = next((entry["node"] for entry in _flatten_pages(config) if entry["node"]["key"] == "home"), {})
    heading = html_lib.escape(home.get("home_cards_title", "Documentation"))
    for entry in _flatten_pages(config):
        node = entry["node"]
        card = node.get("home_card")
        href = _page_href(node)
        if not isinstance(card, dict) or not card.get("show") or not href:
            continue
        try:
            order = int(card.get("order", 10_000))
        except (TypeError, ValueError):
            order = 10_000
        title = card.get("title") or node.get("title") or node.get("label") or node["key"]
        markup = "\n".join([
            '    <div class="card">',
            f'      <h3>{html_lib.escape(card.get("icon", ""))} {html_lib.escape(str(title))}</h3>',
            f'      <p>{html_lib.escape(card.get("description", ""))}</p>',
            f'      <a href="{html_lib.escape(href)}" class="btn">{html_lib.escape(card.get("button", "Open →"))}</a>',
            '    </div>',
        ])
        cards.append((order, markup))
    cards.sort(key=lambda item: item[0])
    return "\n".join([
        '<section class="quick-links">',
        f'  <h2>{heading}</h2>',
        '  <div class="card-grid">',
        *(markup for _, markup in cards),
        '  </div>',
        '</section>',
    ])


def convert_md_to_html(
    md_file_path: str | os.PathLike,
    output_dir: str | os.PathLike | None = None,
) -> bool:
    """Convert one Markdown source into its configured course-site page."""
    config = _load_site_config()
    md_file = Path(md_file_path)
    if not md_file.exists():
        candidate = MD_ROOT / md_file
        if candidate.exists():
            md_file = candidate
        else:
            print(f"Error: File {md_file} not found", flush=True)
            return False

    try:
        source_key = md_file.resolve().relative_to(MD_ROOT.resolve()).as_posix()
    except ValueError:
        source_key = md_file.name
    page = _pages_by_source(config).get(source_key, {})
    title = page.get("title") or md_file.stem.replace("_", " ").title()
    html_rel = page.get("output") or Path(source_key).with_suffix(".html").as_posix()
    output_path = (Path(output_dir) if output_dir else SITE_ROOT) / html_rel
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converter = _new_markdown()
    content = converter.convert(md_file.read_text(encoding="utf-8"))
    if source_key == "index.md":
        content += "\n" + _render_home_cards(config)

    base_href = _base_href(output_path)
    site = config["site"]
    full_html = HTML_TEMPLATE.format(
        title_full=html_lib.escape(f"{site.get('title_prefix', '')}{title}"),
        css_href=_css_href(config, output_path),
        script_href=base_href + "on-this-page.js",
        page_shell_modifier="" if converter.toc_tokens else " page-shell--no-toc",
        site_name=html_lib.escape(site.get("name", "CCAI9012")),
        nav_items=_render_nav(config, base_href=base_href, active_key=page.get("nav_key")),
        mobile_toc=_render_on_this_page(converter.toc_tokens, mobile=True),
        content=content,
        desktop_toc=_render_on_this_page(converter.toc_tokens, mobile=False),
    )
    output_path.write_text(full_html, encoding="utf-8")
    print(f"✓ Converted: {source_key} → {output_path}", flush=True)
    return True


def convert_all_docs() -> None:
    """Convert every non-empty Markdown page; use ``pages.json`` for navigation."""
    pages = _pages_by_source(_load_site_config())
    declared_sources = set(pages)
    existing_sources = {
        path.relative_to(MD_ROOT).as_posix()
        for path in MD_ROOT.rglob("*.md")
        if path.read_text(encoding="utf-8").strip()
    }
    missing_sources = sorted(declared_sources - existing_sources)
    if missing_sources:
        raise ValueError(
            "pages.json sources that are missing or empty: " + ", ".join(missing_sources)
        )

    linked_only_sources = sorted(existing_sources - declared_sources)
    sources = [MD_ROOT / source for source in pages]
    sources.extend(MD_ROOT / source for source in linked_only_sources)
    print(
        f"Found {len(sources)} non-empty markdown file(s): "
        f"{len(pages)} navigation page(s), {len(linked_only_sources)} linked-only page(s)",
        flush=True,
    )
    print("-" * 50, flush=True)
    successes = sum(convert_md_to_html(source) for source in sources)
    print("-" * 50, flush=True)
    print(f"Converted {successes}/{len(sources)} files successfully", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        convert_md_to_html(sys.argv[1])
    else:
        convert_all_docs()
