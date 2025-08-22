Thesis revision checklist (Intro, Background, Method)

Style and consistency
- [ ] Reduce subsubsections; prefer sections/subsections; keep paragraphs over long bullet lists
- [x] Use cref

Introduction
- [ ] Align central narrative in `sections/1-introduction.tex`
  - [ ] Make the opening motivation (open, reproducible methods) coherent with the main problem framing
  - [ ] Reframe "Problem" to emphasize learning agency via RL (not proposing agency itself), tying to the open, reproducible agenda
- [x] Consider retitle in `main.tex` to: "Learning Agency in the Terminal: Lessons from Repository-Level Reinforcement Learning"
- [ ] Justify Nano, CodeRepairRL, etc., as vehicles for the above scope; tie explicitly to initial motivation
- [ ] DeepSWE comparison: double-check compute normalization and references; clearly state our novelty (test-free reward, much lower GPU cost)

Background
- [~] Move MDP to background and make it shorter
- [ ] Change APR with LLMs to Agentless / Agentic
  - What about the basic ones? 
- [ ] Cite and briefly describe relevant systems (e.g., SWE-Agent, OpenHands/OpenDevin, Aider; use citable sources)
- [ ] Discuss and cite Terminal- and TauBench, cite SOTA numbers perhaps, cite
- [ ] Dr. GRPO write and cite
- [ ] Dapo write and cite
- [ ] Remember more of these

Method
- [ ] Split current Chapter 3 into three chapters and update `main.tex` includes
  - [ ] Chapter: "Nano: Terminal-Based Coding Agent" (current 3.2 + add a short example run)
  - [ ] Chapter: "Efficient Agentic Learning Framework" (current 3.3, 3.5, 3.6, 3.8, 3.9)
  - [ ] Chapter: "Experimental Methodology" (current 3.4; optionally include rewards from 3.7)
- [ ] Reduce bullet points across these chapters; convert to flowing prose (keep only essential lists)
- [ ] Replace the method diagram (clearer, less busy); keep label `fig:method-diagram`
- [ ] Standardize GRPO/GSPO notation (single symbol set) and finish the GSPO stability rationale

Carryover notes (integrated as tasks)
- [ ] Weave the "environments are the new datasets" motivation into Intro framing and transitions
- [ ] Highlight novelty vs. DeepSWE: test-free reward signal and lower GPU cost; emphasize low engineering friction enabling broader environments/tasks
- [ ] Consider brief discussion of in-context learning vs. RL for agentic behavior (scope to a short contrast in Intro/Background)