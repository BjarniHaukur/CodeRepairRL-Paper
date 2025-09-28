Thesis revision checklist (Intro, Background, Method)

"Our method is parameter efficient and has lower developmental friction making it easier to integrate in training with a large collage of diverse RL environments"

- [ ] Bit awkward around whether we should use "scaffold" vs "harness". Might require both but currently paragraphs just use them at random almost.

Background
- [~] Move MDP to background and make it shorter
- [x] Change APR with LLMs to Agentless / Agentic
  - [x] What about the basic ones? 
- [x] Merge LLM APR and related work
- [ ] Add a section to the RL background of how agentic coding has a well posited place
- [x] Cite and briefly describe relevant systems (e.g., SWE-Agent, OpenHands/OpenDevin, Aider; use citable sources)

- [ ] Discuss TauBench, cite SOTA numbers perhaps, cite
- [x] Dr. GRPO write and cite
- [x] Dapo write and cite
- [x] Remember more of these
 - GSPO suffices I think
- [ ] "The ecosystem has evolved by leaps and bounds since I started"

- [ ] Vibe write the methods IS EASY JUST DO IT
- [ ] Vibe write the results, we will have my current results and just add the more polished ones


Misc
- [ ] Move specific lora efficiency gains discussion to method

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

Method
- [ ] Split current Chapter 3 into three chapters and update `main.tex` includes
  - [ ] Chapter: "Nano: Terminal-Based Coding Agent" (current 3.2 + add a short example run)
  - [ ] Chapter: "Efficient Agentic Learning Framework" (current 3.3, 3.5, 3.6, 3.8, 3.9)
  - [ ] Chapter: "Experimental Methodology" (current 3.4; optionally include rewards from 3.7)
- [ ] Reduce bullet points across these chapters; convert to flowing prose (keep only essential lists)
- [ ] Replace the method diagram (clearer, less busy); keep label `fig:method-diagram`
- [ ] Standardize GRPO/GSPO notation (single symbol set) and finish the GSPO stability rationale

Chapter 4: The Work
- [ ] Move practical engineering details (TRL/vLLM modifications, NCCL sync, ZeRO+LoRA+checkpointing scheduling) from Methods into Chapter 4 where appropriate
- [ ] Fill in quantitative metrics (compute hours, sync latency, VRAM during gathers, throughput), cross-reference Appendix
- [ ] Ensure \ac{} usage for all acronyms and align tone with single-agent scope

Carryover notes (integrated as tasks)
- [ ] Weave the "environments are the new datasets" motivation into Intro framing and transitions
- [~] Highlight novelty vs. DeepSWE: test-free reward signal and lower GPU cost; emphasize low engineering friction enabling broader environments/tasks
- [~] We use SWE-RL's patch based reward scheme, talk about it in method but credit in background? It ties nicely into the "low friction" angle. It is a nascent trend in the field to train models on many environments concurrently. Having an environment that requires extensive dockerized setups for test-driven / execution driven rewards is hard to get working along with other environments. That is where our execution-free patch similarity comes in nicely. And we can easily support more programming languages.
- [ ] Consider brief discussion of in-context learning vs. RL for agentic behavior (scope to a short contrast in Intro/Background)