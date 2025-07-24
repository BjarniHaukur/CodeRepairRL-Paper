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


Use `\todoinline{}` commands to mark sections needing development with instructive ideas for what should come next

### Academic Style (for later refinement)
- Target venue: KTH thesis, potentially adapted for academic conference
- Emphasize novel "hill-climbing the coding agent gradient" experiment