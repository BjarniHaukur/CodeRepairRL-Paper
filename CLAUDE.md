# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a LaTeX thesis repository for "CodeRepairRL" - research on agent-in-the-loop reinforcement learning for automated code repair. The thesis explores training LLMs to fix bugs by embedding coding agents into the RL training loop, comparing minimalist vs heavyweight scaffolding approaches.

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

### Key Documentation Files
- `notes/PROJECT.md` - Comprehensive project overview, research questions, technical approach
- `notes/PAPER.md` - Detailed thesis outline with section breakdowns
- `notes/AGENT_RL_INTEGRATION.md` - Technical details on agent-in-the-loop implementation
- `notes/DATASETS.md` - Information about SWE-Gym, SWE-Bench-Verified datasets
- `notes/RESOURCES.md` - Literature review and related work compilation

### Thesis Content Status
- **Sections 1-6**: Currently contain only template content - all substantive research content needs to be written
- **Abstract/Introduction**: Template placeholders only
- **Method/Results**: Empty sections waiting for actual research content
- **Bibliography**: Only contains one entry, needs comprehensive academic references

## Research Context

### Core Innovation
The thesis pioneered "agent-in-the-loop reinforcement learning" where coding agents are embedded directly into RL training loops, enabling:
- Multi-step interactive debugging rather than single-pass generation
- Direct reinforcement in realistic development environments  
- Comparison of minimalist (nano-agent) vs heavyweight scaffolding approaches

### Technical Approach
- **Training**: Two-stage pipeline (SFT → GRPO reinforcement learning)
- **Models**: Qwen3 continued RL training (not traditional fine-tuning)
- **Scaffolds**: Nano-agent (minimalist) vs Aider-style (heavyweight)
- **Evaluation**: SWE-Bench-Verified, potentially Defects4J for Java generalization

## Writing Guidelines

### Writing Focus
- **Current stage**: Focus on SUBSTANCE over academic style/polish
- Early drafts should prioritize getting ideas down and building complete arguments
- Style refinement comes later in the writing process
- Use `\todoinline{}` commands to mark sections needing development with instructive ideas for what should come next

### Academic Style (for later refinement)
- Target venue: KTH thesis, potentially adapted for academic conference
- Emphasize novel "hill-climbing the coding agent gradient" experiment
- Highlight significance: first open-source replication of agent-in-loop RL for code

### Content Priorities
1. **Method section**: Core technical contribution of agent-RL integration
2. **Introduction**: Problem statement and novel approach positioning  
3. **Background**: Literature review on RL for code, agent scaffolding
4. **Results**: Experimental outcomes demonstrating monotonic improvement

### Key Technical Terms
- Agent-in-the-loop RL
- GRPO (Group Relative Policy Optimization)
- Scaffold complexity (minimalist vs heavyweight)
- SWE-Gym/SWE-Bench evaluation environments
- Nano-agent architecture

## Development Notes

The thesis content draws heavily from detailed notes in `notes/` directory. When writing sections, reference these files for comprehensive technical details and research context. The goal is demonstrating that LLMs can learn terminal navigation and codebase interaction through RL - a significant open-source achievement.