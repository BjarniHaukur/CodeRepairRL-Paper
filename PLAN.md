# Thesis Restructuring and Quality Plan

This plan tracks the restructuring of the thesis to follow the KTH template with clear separation between methodology (Ch 3), implementation (Ch 4), and results (Ch 5).

---

## MONPERRUS FEEDBACK

- [x] Change name to Learning Agency in the Terminal with Repository-Level Reinforcement Learning
- [x] abstract too long
- [x] problem statement is very important  "The problem I am addressing IS..."
- [x] RQs are a bit weak (merged RQ0+RQ3 → RQ1, renumbered to 3 RQs)
- [x] 1.3 should be Contribution
- [x] harness <-> scaffold
- [-] list of scaffolds along with list of models
- [x] in background, move RL + Lora up, now it is: "apr, apr, apr, theory, theory, apr"
- [x] current 2.5 can split it up a bit more clear
- [x] rename "Methodology" to something more impressive
- [x] 3.3 should start with "why we made Nano", not "what it is"
- [x] "Sidestepping" sounds like we are avoiding the diff problem, we should describe it as what it IS, not WHAT IT IS NOT"
- [x] could add more in policy optimization
- [x] skip SLURM
- [x] rename 4 to something more impressive: "Contributions to …"
- [x] in 5, add RQ methodology before showing the results
- [x] figure 5.2.1, add apply_patch + shell
- [-] like figure 5.2.3, do per tool success rate
- [-] add episode example to results
- [x] swe bench table remove first two (500(
- [x] make everything in the table into higher->better, flip the values
- [x] rq4, maybe remove, maybe move to RQ0, definitely make it shorter
- [x] a short 6.3.1 to an extremely long 6.3.2. is bad smell
- [x] Maybe call it: "Broader Reflections on Reinforcement Learning"
- [x] 6.2 can be moved into 5 as the final / summary section
- [x] Chapter 2: Go from most theoretical to most practical
- [x] Chapter 6: From most factual, to most theoretical
- [x] Adding stuff detracts from my cool stuff

---

## ✅ Supervisor Feedback Session (Acknowledgements, Abstract, Minor Fixes)

### Completed Items
- [x] **Acknowledgements section** - Enhanced from draft to polished 5-paragraph structure with NAISS infrastructure, ASSERT-KTH book club, open-source ecosystem, and personal thanks
- [x] **Abstract rewrite** - Complete publication-quality abstract with clear structure (problem → solution → method → results → impact), data placeholders ready for fill-in
- [x] **Keywords finalized** - 8-term list including GSPO, online learning, tool-augmented LMs, multilingual generalization
- [x] **Chapter 4 title change** - "The Work" → "Implementation" for better clarity
- [x] **Section 3.10 title cleanup** - Removed "(brief)" from "Evaluation Protocol"
- [x] **Chapter reference fix** - Fixed "Chapter Chapter 3" to "\cref{ch:method}" in Chapter 4 opening
- [x] **Code overflow fix (page 29)** - Broke long `subprocess.check_output()` line into multiple lines in verbatim block
- [x] **RQ3 rephrasing** - Updated RQ3 throughout all chapters from "Does the execution-free multilingual curriculum generalize beyond Python?" to "Does execution-free RL enable effective multilingual training without language-specific engineering?" to better reflect the language-agnostic design principle
- [x] **Removed scaffold transfer references** - Cleaned up stray reference to cross-scaffold experiments in Chapter 1 that no longer exists

### Confirmed Exclusions
- [x] **No multi-scaffold experiments in thesis** - Aider/OpenHands/Mini-SWE-Agent transfer experiments moved to subsequent paper (confirmed with supervisor)

---

## ✅ Recent Session (RQ Restructuring & Prose Polish)

### Research Question Restructuring
- [x] **Removed old RQ3 (Scaffold Transfer)** - moved to concurrent conference paper
- [x] **Redefined RQ1**: "How does GSPO training improve Nano harness adaptation?" (harness-level efficiency metrics, command usage evolution)
- [x] **Redefined RQ2**: "Does execution-free patch-similarity RL training improve SWE-Bench-Verified performance?" (test-verified success rates)
- [x] **Redefined RQ3**: "Does execution-free RL enable effective multilingual training without language-specific engineering?" (language-agnostic training validation, 50-task holdout, per-language rewards)
- [x] Updated RQ formulations throughout: Ch 1 (Introduction), Ch 3 (Methodology), Ch 5 (Results), Ch 6 (Conclusions)
- [x] Adjusted analysis claims in RQ1 from "statistical significance testing" to "qualitative examination of command usage patterns" (honest about data limitations)

### Model Scope Simplification
- [x] **Primary focus: Qwen3-14B only** in all main chapters
- [x] Moved Qwen3-8B and Llama3.1-8B comparisons to Appendix B (`app:model-comparison`)
- [x] Added appendix section documenting: Qwen3-8B (capacity scaling) and Llama3.1-8B (~3x lower rewards, illustrating importance of tool-calling capability)
- [x] Updated all "model scaling" references to focus on 14B
- [x] Removed 30B references from Ch 3, Ch 4, Ch 5
- [x] Updated ZeRO discussion: ZeRO-2 only (removed ZeRO-3 content)
- [x] Updated resource requirements to reflect 14B focus

### Prose Polishing (Chapters 1, 2, 3)
- [x] **Chapter 1 (Introduction)**: Improved technical precision, removed redundancy, clarified execution-free approach
- [x] **Chapter 2 (Background)**: Enhanced technical language, credited GitBug-Java, improved APR taxonomy clarity, updated dataset descriptions (750/250 split motivation)
- [x] **Chapter 3 (Methodology)**: Tightened technical descriptions, improved observation/action space clarity, refined reward design explanation, clarified GSPO adoption rationale

### Holistic Review
- [x] Reviewed Chapters 1 & 2 holistically for quality (assessed as "quite good" for master's thesis standard)
- [x] Removed awkward "constant training conditions across experiments" sentence (no longer relevant with single-model focus)

## ✅ Completed Major Items

### Chapter 3: Methods (NEW)
- [x] Create new chapter skeleton `sections/3-method-new.tex` with agreed structure
- [x] Migrate Nano agent details (observations, actions, termination, sandbox, truncation=2000 chars)
- [x] **Add formal RL integration notation** (h_t, y_t, c_t, o_t, R(τ) with interaction cycle)
- [x] Specify datasets: Python-only (≈2400 SWE-Gym) and mixed 1k curriculum (750 Python + 250 multilingual; 50 held out)
- [x] Define reward: canonical git diff; per-file SequenceMatcher; dual-mask loss with **extensive mathematical formalization**
- [x] **Expand masked loss computation** with motivation, dual-mask strategy, perplexity analysis, algorithmic benefits
- [x] Replace RL algorithm section with GSPO + modifications; **add GRPO advantages explanation**
- [x] **Expand Qwen model selection rationale** with tool-calling superiority and code understanding capabilities
- [x] Clarify model choice and adaptation (Qwen3 8B/14B/30B-A3B; LoRA r=32 α=64; 12k context)
- [x] Summarize infrastructure (vLLM serving, NCCL live sync, SLURM isolation)
- [x] **Add comprehensive decoding and exploration policy** with temperature parameter analysis
- [x] **Expand multilingual training protocol** with language distribution and evaluation strategy
- [x] Add brief Evaluation Protocol (primary: SWE-Bench-Verified; secondary: patch-similarity, per-language rewards)
- [x] Add Reproducibility essentials section
- [x] Add dedicated "Training Environment" section with training-inference duality
- [x] Remove references to separate multilingual protocol and multi-component reward

### Chapter 4: The Work (NEW)
- [x] Create new chapter skeleton `sections/4-work-new.tex`
- [x] Outline subsections narrating implementation
- [x] **Add detailed NCCL weight synchronization implementation** (150-300ms, differential updates, memory-bounded)
- [x] **Expand practical challenges section** (tool-call formatting, trajectory serialization, reward determinism, coordination, monitoring)
- [x] **Create compute-efficient infrastructure section** (LoRA, DeepSpeed ZeRO-2/3, gradient checkpointing, BF16, PagedAttention)
- [x] **Move computational performance analysis from Ch 5** and expand with throughput, latency, scalability, cost-effectiveness
- [x] Add training runs and protocol description
- [x] Document ablations and diagnostics

### Chapter 5: Results Restructuring
- [x] **Remove 4 methodology subsections** that duplicated Chapter 3
- [x] **Move Computational Performance Analysis to Chapter 4**
- [x] **Align RQ sections with Chapter 1** (titles, wording, metrics)
- [x] Update chapter opening to reference evaluation protocol from Ch 3
- [x] Remove discussion/implications (moved to Ch 6)

### Chapter 6: Conclusions Enhancement
- [x] **Add comprehensive "Lessons Learned" section** (§6.7) with 4 subsections on system complexity, design choices, methodology, and broader insights
- [x] **Move "Broader Implications" from Ch 5** with Design Philosophy and Improvement Potential subsections

### Appendices
- [x] **Create formal masked loss mathematical derivation** with problem formulation, dual-mask equations, perplexity decomposition, algorithmic properties
- [x] Add MLP and VRAM acronyms

### Structure and Consistency
- [x] **RQ consistency audit and alignment** - Verified and aligned all Research Questions across Ch 1, 3, and 5
  - Fixed terminology ("limited-shot" → "few-shot")
  - Added explicit RQ signposting in Ch 3 evaluation section
  - Verified metrics alignment across all chapters
- [x] **Thesis structure harmonization** - Established consistent sectioning hierarchy
  - Ch 3: Converted 7 unnumbered subsections to numbered (Nano Agent components now in ToC)
  - Ch 4: Removed redundant "Implementation Overview", promoted 13 paragraphs/subsections to proper hierarchy
  - Ch 5: Promoted "Qualitative Improvement Analysis" to subsection
  - Ch 6: Promoted 4 key conceptual insights to subsections
  - Result: +25 meaningful ToC entries, consistent 3-level hierarchy
- [x] **Dual-masking strategy repositioning** - Minimized narrative bloat
  - Compressed Ch 3 explanation from ~30 lines to 5 sentences
  - Kept full mathematical formulation in Appendix
  - Positioned as implementation detail, not contribution claim
- [x] **Reproducibility section migration** - Moved from Ch 3 (Methodology) to Ch 4 (Implementation)
  - Better template compliance (methodology vs. deliverables)
  - Merged with existing Ch 4 content for comprehensive documentation
- [x] **Section transitions and flow** - Added bridging text between major sections
  - Ch 3: Nano Agent section introduction
  - Ch 4: Training Runs protocol bridge
  - Fixed Ch 6 empty subsection headers
  - Fixed typos ("on of" → "one of")
- [x] **Todoinline audit and cleanup** - Comprehensive audit of 82 markers
  - ✅ Phase 1 (Category A): Removed 8 obsolete/completed markers
  - ✅ Phase 2 (Category C): Completed 15 editorial items (citations, appendix refs, prose additions)
  - ✅ Phase 3 (Category D): Resolved 2 structural decisions (deleted both)
  - **Remaining: 56 todoinlines** (26 removed/completed from original 82)

---

## 📝 Todoinline Refinement: ACT NOW vs DO IN THE END

**56 todoinlines remaining** - refined into actionable categories:

### ✏️ ACT NOW (18 items) - Can address without experiments
These require writing/documentation but no experimental data:
- **Ch 1 (2)**: Verify SOTA numbers note, add APR citation placeholder
- **Ch 2 (1)**: Review math notation clarity
- **Ch 3 (2)**: Document known dataset versions, evaluation seeds/subsets structure
- **Appendices (4)**: Document config structures (vLLM, TRL, SLURM, NCCL) with value placeholders
- **Ch 5 (1)**: Write GSPO stability discussion (qualitative, no numbers)
- **Pre-content (2)**: Delete/integrate specialist comment, keywords already done

### ⏳ DO IN THE END (38 items) - Need experimental completion
Cannot proceed without training results:
- **Ch 1 (5)**: Scope tightening, tool metrics, language counts, harness configs, conference alignment
- **Ch 3 (2)**: Exact batch sizes, update counts, final harness budgets
- **Ch 4 (5)**: Training counts (updates/epochs/hours), performance tables, sync latencies, GPU scaling, cost analysis
- **Ch 5 (24)**: All results, tables, comparisons, analyses
- **Ch 6 (8)**: All conclusion items depend on Ch 5 results
- **Pre-content (3)**: Abstract numbers, transfer results, efficiency metrics

**Note:** "ACT NOW" items deferred until after current flow-state writing session to avoid context-switching.

---

## 🔄 In Progress / Partial

### Content Migration
- [~] Migrate all valuable content from OLD chapters 3 & 4 (MOSTLY DONE - core content migrated)
  - [x] Formal RL notation
  - [x] Masked loss detail
  - [x] GRPO advantages
  - [x] Model selection rationale
  - [x] Decoding policy
  - [x] Multilingual protocol
  - [x] NCCL synchronization
  - [x] Practical challenges
  - [x] Compute infrastructure
  - [x] Lessons learned
  - [ ] Review if anything else valuable remains in OLD files

### Acronyms
- [~] Acronym audit: ensure all occurrences use `\ac{}` consistently
  - [x] Chapter 3-new
  - [x] Chapter 4-new
  - [x] Chapter 6
  - [x] Appendices
  - [ ] Chapter 1 (Introduction)
  - [ ] Chapter 2 (Background)
  - [ ] Chapter 5 (Results)

---

## 📋 High-Priority Next Steps (30min-1hr each)

### Content Quality

- [ ] **Review all `\todoinline{}` markers** - Create prioritized list of which need data first vs. which can wait
  - Scan all chapters and list critical vs. nice-to-have todos
  - Mark which are data-dependent (need experiments) vs. editorial (need writing)

- [ ] **Cross-reference audit** - Ensure all `\ref{}` and `\cref{}` point to correct labels
  - Check Chapter 5 references to methodology sections
  - Verify figure and table references exist
  - Fix any broken cross-references from restructuring

- [x] **RQ consistency check** - Verify exact alignment between Ch 1, 3, and 5
  - Compare RQ wording in Introduction vs. Methods vs. Results
  - Ensure metrics listed match across all three chapters
  - Check hypothesis statements are consistent
  - ✅ COMPLETED: All RQs aligned, "limited-shot" → "few-shot", added RQ signposting in Ch 3

### Chapter-Specific Polish

- [ ] **Chapter 2 (Background) alignment check** - Ensure Background doesn't contradict new Methods
  - Check if Background discusses methods that changed in Ch 3
  - Verify terminology consistency (e.g., "GSPO" not "GRPO")
  - Update any forward-references to Chapter 3 structure

- [ ] **Chapter 1 section §1.4 Goals alignment** - Update if necessary to match new methodology
  - Check if Goals section needs updating based on new RQ formulations
  - Verify deliverables listed match what's in Ch 4

- [ ] **Abstract update** - Rewrite to reflect restructured thesis and key contributions
  - Emphasize single-agent Nano approach
  - Highlight compute-efficiency narrative
  - Match structure: Ch 3 methods → Ch 4 work → Ch 5 results → Ch 6 conclusions

### Technical Correctness

- [ ] **Terminology consistency audit** - Create glossary of key terms and check usage
  - "scaffold" vs. "harness" vs. "agent" - consistent usage?
  - "execution-free" vs. "test-free" vs. "patch-similarity-based"
  - "online RL" vs. "experiential learning" - when to use which?

- [x] **Bibliography completeness check** - Verify all cited works exist in references.bib
  - ✅ All 39 active citation keys present in references.bib
  - ✅ 0 undefined citations
  - ✅ GSPO (7×), GRPO (3×), Dr.GRPO (2×), DAPO (1×) all cited correctly
  - ✅ Biber runs cleanly with no warnings

- [ ] **Figure and table caption review** - Ensure captions are descriptive and self-contained
  - Each caption should explain what the figure shows without reading main text
  - Add missing figure placeholders for referenced-but-missing figures
  - Check figure numbering is sequential

### Structural Polish

- [x] **Transition sentences between sections** - Add connecting prose between major sections
  - Add transition at end of Ch 3 pointing to Ch 4
  - Add transition at end of Ch 4 pointing to Ch 5
  - Check section-to-section flow within chapters
  - ✅ COMPLETED: Added bridging sentences in Ch 3 (Nano Agent intro), Ch 4 (Training Runs intro), fixed Ch 6 hierarchy

- [ ] **Chapter opening paragraphs** - Ensure each chapter has clear roadmap paragraph
  - Ch 2: "This chapter reviews..."
  - Ch 3: "This chapter presents..."
  - Ch 4: "This chapter describes..."
  - Ch 5: "This chapter reports..."

- [ ] **Final section summaries** - Add brief summary paragraphs to major sections
  - Ch 3 end: summarize methodology before implementation
  - Ch 4 end: summarize what was built before results
  - Ch 5 end: already has summary, review for completeness

### Pre-Content & Metadata

- [ ] **Acknowledgments** - Write acknowledgments section in 0-pre-content.tex
  - Thank advisors, infrastructure providers (NAISS?), funding sources
  - Acknowledge open-source tools used

- [ ] **Swedish abstract** - Translate abstract to Swedish per KTH requirement
  - Wait until English abstract is finalized
  - Place in 0-pre-content.tex

- [ ] **Keywords** - Define 5-7 keywords in 0-pre-content.tex
  - Should include: Reinforcement Learning, Automated Program Repair, Coding Agents
  - Add domain-specific terms

---

## 🎯 Medium-Priority Items (after data collection)

### Results Chapter Population

- [ ] **Populate main results table** (§5.2) with actual performance numbers
  - SWE-Bench-Verified success rates
  - Baseline comparisons
  - Statistical significance tests

- [ ] **Fill RQ1 harness adaptation results** (§5.3)
  - Tool success rate progressions
  - Invalid call rate reductions
  - Model size comparison table

- [ ] **Fill RQ2 multilingual results** (§5.4)
  - Per-language reward deltas
  - Bootstrap confidence intervals
  - Holdout performance table

- [ ] **Fill RQ3 scaffold transfer results** (§5.5)
  - Zero-shot transfer success rates
  - Few-shot adaptation curves
  - Cross-harness comparison table

### Performance Metrics Documentation

- [ ] **Add performance metrics to Chapter 4** (§4.7)
  - Episodes per hour by model size
  - Peak VRAM measurements
  - Sync latency percentiles
  - GPU utilization stats
  - Total compute hours table

---

## 🔮 Lower-Priority / Polish Items

### Visual Content

- [ ] **Generate missing figures** referenced in text
  - Agent loop architecture diagram (Ch 3)
  - System architecture diagram (Ch 4)
  - Training sequence diagram (exists, verify placement)
  - Nano rollout example (exists, verify caption)

- [ ] **Create supplementary figures** for better understanding
  - Reward computation flowchart
  - Training-inference duality diagram
  - Dual-mask visualization

### Appendix Expansion

- [ ] **Add SLURM templates** to Appendix per todos in Ch 4
  - Job submission scripts
  - Environment setup

- [ ] **Add configuration fragments** to Appendix
  - DeepSpeed config YAML
  - vLLM server flags
  - LoRA configuration

- [ ] **Document NCCL environment** in Appendix
  - Communication group setup
  - Gathering pseudo-code

### Future Work Section

- [ ] **Expand Future Work** in Chapter 6 based on limitations
  - Test-based rewards section
  - Multi-objective optimization
  - Longer training investigations

---

## 📝 Notes on Methodology

- **Single-agent Nano paradigm**: All content should emphasize the minimalist single-agent approach
- **Compute-efficiency narrative**: Throughout thesis, emphasize academic-accessible compute requirements (not industrial-scale)
- **Execution-free rewards**: This is a key distinguishing feature enabling multilingual training
- **Template compliance**: Strict separation - Ch 3 = methods, Ch 4 = implementation, Ch 5 = results only, Ch 6 = discussion

---

## Estimated Status

**Overall thesis completeness**: ~75%
- **Structure**: 95% (template-compliant, well-organized)
- **Content migration**: 90% (core content in place)
- **Polish**: 60% (needs cross-references, transitions, consistency checks)
- **Data**: 20% (many todos waiting for experimental results)
- **Pre-content**: 40% (abstract, acknowledgments, keywords need work)

**Ready for**: Intensive polish phase while experiments run in parallel