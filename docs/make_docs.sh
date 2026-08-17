#!/usr/bin/env bash

set -euo pipefail

DOCS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$DOCS_DIR")"

cd "$REPO_ROOT"

echo "Building API Reference..."
python docs/generate_api_doc.py

echo "Building course pages..."
python docs/md_to_html.py

echo "Documentation build complete: docs/index.html"
