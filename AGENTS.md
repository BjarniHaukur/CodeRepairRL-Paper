# AGENTS.md

This document provides guidance for coding agents (e.g., Cursor, Claude Code) contributing to this repository. The project is a KTH thesis that investigates reinforcement learning for automated program repair using a single minimalist agent agent (the “Nano” agent). All edits should respect the thesis’ academic tone, emphasize methodological clarity, and maintain reproducibility.

The repository is organized around a conventional LaTeX thesis. The entry point is `main.tex`, which includes chapter files under `sections/`. LaTeX configuration and KTH-specific assets live in `setup/`. Bibliographic entries are collected in `references.bib`.

Current research questions emphasize Nano-only reinforcement learning with three focal points: harness adaptation and success scaling across Qwen3 model sizes on SWE-Bench-Verified, multilingual transfer on a 50-task SWE-Bench-Multilingual holdout, and scaffold transfer to Mini-SWE-Agent-/Aider-/OpenHands-style interfaces.	odoinline{Refresh once scaffold evaluations solidify.} Insert concise `\todoinline{}` placeholders where quantitative updates, figures, and tables will later be integrated, and avoid speculative claims beyond this scope.\todoinline{Update this summary if additional evaluations are reinstated.}

All authoring should assume a single agent paradigm. Remove or avoid references to multiple scaffolds unless explicitly restoring prior context for comparison. Where the thesis discusses “scaffolding” in a generic sense, ensure the text clearly distinguishes the implemented Nano agent from broader agentic tooling ecosystems.

Finally, maintain an academic narrative style. Favor cohesive exposition over enumerated lists, except where short fenced code blocks clarify concrete commands (compilation, dependency setup, or plot execution). When adding content that is tentative or data-dependent, annotate with `\todoinline{}` and include a short directive indicating what evidence or analysis will be added later.
