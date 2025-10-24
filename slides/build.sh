#!/bin/bash
# Quick build script for KTH thesis presentation

set -e

echo "Building KTH Thesis Presentation..."

# HTML (for presentation with interactive Sankey diagrams)
echo "→ Building HTML..."
marp presentation.md --html --theme kth-theme.css -o presentation.html

# PDF (backup, iframes won't render)
echo "→ Building PDF..."
marp presentation.md --theme kth-theme.css --allow-local-files -o presentation.pdf

echo "✓ Build complete!"
echo "  - presentation.html (use this for the defense)"
echo "  - presentation.pdf (backup/printing)"
echo ""
echo "To present: open presentation.html in a browser and press F11 for fullscreen"

