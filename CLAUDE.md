# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

## Build Commands

**Compile thesis:**
```bash
# Compile with LaTeX Workshop extension in VS Code, or manually:
latexmk -pdf -xelatex main.tex
```

**Dependencies (Ubuntu/WSL):**
```bash
sudo apt update && sudo apt install texlive-full latexmk biber ttf-mscorefonts-installer
fc-cache -fv  # refresh fontconfig for XeLaTeX fonts
```

## Repository Architecture

### Core Structure
- `main.tex` - Main thesis document with proper KTH template setup
- `sections/` - Individual thesis chapters as separate .tex files
- `notes/` - Detailed research notes and project documentation
- `setup/` - LaTeX configuration, title page, and KTH branding
- `references.bib` - Bibliography (currently minimal, needs expansion)
- `plotting/` - Python scripts to create the plots in this thesis by using the WandB API
  - Uses `uv` for dependency management (run `uv sync` to install dependencies)
  - Execute scripts with `uv run -m scripts.script_name`
  - **Important**: For creating or modifying thesis plots, use the `thesis-plot-manager` agent which contains detailed plotting guidelines and directory structure


Use `\todoinline{}` commands to mark sections needing development with instructive ideas for what should come next

**For thesis writing and LaTeX composition, use the `thesis-writer` agent which specializes in academic prose and research narrative development.**

### Academic Style (for later refinement)
- Target venue: KTH thesis, potentially adapted for academic conference
- Emphasize novel "hill-climbing the coding agent gradient" experiment

## Plotting Overview

The `plotting/` directory contains Python scripts for generating thesis figures using the WandB API. The directory follows a structured organization:

- **scripts/**: Core plotting scripts organized by purpose
  - Main directory: Primary thesis figure generation scripts
  - `analysis/`: Research analysis scripts (patterns, behaviors)
  - `exploration/`: Data exploration and discovery
  - `debugging/`: Temporary debugging scripts
- **utils/**: Shared utilities (table parsing, data processing)
- **figures/plots/**: Auto-organized output by plot type (analysis, temporal, sankey, comparison)
- **cache/**: Cached WandB table downloads for faster re-runs

### Key principles:
- All plots include run IDs in filenames for reproducibility
- Use existing utilities (especially TableExtractor for HTML parsing)
- Scripts are executed with `uv run -m scripts.script_name`
- Output automatically organized by plot type

**For detailed plotting guidelines and creating new plots, use the `thesis-plot-manager` agent.**