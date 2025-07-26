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
  - See **Plotting Guidelines** below for detailed organization rules


Use `\todoinline{}` commands to mark sections needing development with instructive ideas for what should come next

### Academic Style (for later refinement)
- Target venue: KTH thesis, potentially adapted for academic conference
- Emphasize novel "hill-climbing the coding agent gradient" experiment

## Plotting Guidelines

### Directory Structure
```
plotting/
├── utils/               # Common utilities - NEVER modify existing functions
│   ├── table_parser.py  # HTML table parsing (TableExtractor class)
│   ├── data_processor.py # Future: common data processing
│   └── plotting_utils.py # Future: common plotting helpers
├── figures/             # Output directory - AUTO-CREATED
│   └── plots/           # Organized by plot type
│       ├── analysis/    # Distributions, clustering, performance analysis
│       ├── temporal/    # Time-series, evolution, training progress
│       ├── sankey/      # Flow diagrams, command evolution
│       └── comparison/  # Side-by-side comparisons, baselines
├── scripts/             # Core plotting scripts ONLY
│   ├── template.py      # COPY THIS for new plots
│   ├── command_evolution_sankey.py
│   ├── individual_problem_analysis.py
│   ├── temporal_analysis.py
│   ├── analysis/        # Research analysis (rm patterns, behavior)
│   ├── exploration/     # Data exploration and discovery
│   └── debugging/       # Temporary debugging (DELETE when done)
├── plot_config.py       # Global config, colors, styling
├── wandb_utils.py       # WandB API utilities
└── pyproject.toml       # Dependencies
```

### File Naming Convention
**ALWAYS use this format**: `{descriptive_name}_{run_id}.{extension}`

**Examples**:
- `figures/plots/analysis/problem_distribution_tfk08zx2.png`
- `figures/plots/temporal/reward_evolution_tfk08zx2.png` 
- `figures/plots/sankey/command_flow_early_tfk08zx2.html`
- `figures/plots/comparison/baseline_vs_improved_tfk08zx2.png`

### Code Organization Rules

1. **ALWAYS start from template.py**
   ```bash
   cp scripts/template.py scripts/your_new_plot.py
   ```

2. **Use clean imports** (no sys.path manipulation needed):
   ```python
   from plot_config import create_figure, save_figure, get_output_filename
   from wandb_utils import get_run, get_history
   from utils.table_parser import TableExtractor  # For rollout data
   ```

3. **Use TableExtractor for HTML table parsing** (NO code duplication):
   ```python
   extractor = TableExtractor()
   tables = extractor.extract_all_training_tables(run)
   rollout_data = extractor.extract_rollout_data(tables)
   ```

4. **Use auto-categorized saving**:
   ```python
   # Auto-detects plot type from name
   save_figure(fig, "reward_distribution_analysis")
   
   # Or specify explicitly
   save_figure(fig, "baseline_comparison", plot_type="comparison")
   ```

5. **Include WandB URLs in comments**:
   ```python
   RUN_ID = "tfk08zx2"  # https://wandb.ai/entity/project/runs/tfk08zx2
   ```

### Plot Type Guidelines

- **analysis/**: Static analysis, distributions, clustering
- **temporal/**: Time-series data, training curves, evolution
- **sankey/**: Flow diagrams, command sequences, transitions  
- **comparison/**: Side-by-side comparisons, A/B testing

### Common Utilities Usage

- **TableExtractor**: Use for ALL HTML table parsing
- **plot_config.py**: Get colors, styling, output paths
- **wandb_utils.py**: WandB API interactions
- **NEVER duplicate code** - if you need new common functionality, add to utils/

### Quality Standards

1. **Always include run IDs** in filenames for reproducibility
2. **Auto-organize by plot type** using the naming convention
3. **Reuse existing utilities** - never reimplement table parsing
4. **Follow the template structure** for consistency
5. **Include descriptive comments** with WandB URLs

### Script Organization Rules

**DO NOT pollute the main scripts/ directory!**

- **scripts/** - ONLY core plotting scripts that generate thesis figures
- **scripts/analysis/** - Research analysis scripts (e.g., rm behavior patterns)
- **scripts/exploration/** - Data exploration and run discovery
- **scripts/debugging/** - Temporary debugging scripts (DELETE after use)

**Important**:
- Use `replace=True` in WandB downloads to avoid disk caching
- Delete debugging scripts when done investigating
- If unsure, put new scripts in appropriate subdirectory, not root