# Thesis TODOs

## Immediate Data & Analysis Capture
- [ ] Launch the final SWE-Bench-Verified evaluation for the Qwen3-14B GSPO checkpoint (record job id and ETA in the lab log).
- [ ] Start the corresponding SWE-Bench-Verified run for the Qwen3-8B GSPO checkpoint to validate the observed uplift.
- [ ] Queue the SWE-Bench-Verified pass for Qwen3-30B-A3B so all reported model sizes share the same evaluation protocol.
- [ ] Run the reference Llama3.1-8B evaluation to document the non-convergent baseline in Appendix materials.
- [ ] Compute pre/post average reward on the 50-sample multilingual holdout split and export the plot to `plotting/figures/multilingual_holdout.png`.
- [ ] Extract tool-success, invalid-call, and action-count metrics from `tfk08zx2` into `plotting/derived/harness_metrics.csv` for RQ1 analysis.
- [ ] Aggregate GPU hours, peak VRAM during NCCL gathers, and sync latency stats into `notes/compute.md` for citation in Methods and Results.

## Introduction (`sections/1-introduction.tex`)
- [ ] Refresh the SWE-Bench-Verified SOTA paragraph with current numbers and cite Claude~4.1, Kimi-K2, and Qwen3-Coder system cards.
- [ ] Replace the “which traditional paper to cite here” marker with canonical pre-LLM APR references (GenProg, SemFix, Prophet).
- [ ] Rewrite the RL feasibility discussion to explain how patch-similarity rewards make GSPO viable without heuristics, citing SWE-RL or related work.
- [x] Add TauBench as the sole forward-looking terminal benchmark and note that broader cross-harness studies are deferred.
- [x] Clarify that the thesis reports pure GSPO+KL training without SFT or distillation baselines.
- [x] Align the listed research questions with the final Results chapter scope (SWE-Bench-Verified focus plus multilingual holdout analysis).
- [x] Update the methodology teaser to name the 750 SWE-Gym + 250 SWE-Bench-Multilingual curriculum explicitly.
- [ ] Add a short preview of the cross-scaffold evaluation (Mini-SWE-Agent/Aider/OpenHands) so readers expect RQ3 later.
- [ ] Confirm the Benefits/Ethics section notes whether enhanced Apptainer isolation was enabled in the final experiments.
- [ ] Fix the Outline subsection by renaming “Chapter 4” properly and replacing the placeholder sentence for Chapter~\ref{ch:work-new}.

## Background (`sections/2-background.tex`)
- [ ] Insert the sentence explaining that many scaffolds treat the LLM as a callable function, complicating end-to-end RL, addressing the pending TODO.
- [ ] Expand the PPO→GRPO→Dr.GRPO progression so GSPO appears as the culminating method with KL regularization for small batch regimes.
- [ ] Replace the “bla” placeholder beneath the clipping equation with an explanation of how the group baseline interacts with asymmetric clipping.
- [ ] Rewrite the GSPO subsection to describe its motivation (sequence-level ratios, inference/training mismatch) and cite the primary source.

## Methodology & System Design (`sections/3-method-new.tex`)
- [ ] Document dataset versions, commit hashes, and filtering rules for the 750 SWE-Gym + 250 SWE-Bench-Multilingual curriculum.
- [ ] Add the promised reward aggregation table showing per-file patch similarity roll-ups to episode scores.
- [ ] Record the effective batch size (post masking) and total update count for the main GSPO+KL run.
- [ ] Cite the TRL modules modified for GSPO support and summarize the Nano-specific API extensions.
- [ ] Insert measured VRAM reduction factors and NCCL sync latencies with forward references to the appendix.
- [ ] Specify seeds and held-out splits for both the main curriculum and the 50-sample multilingual evaluation.
- [ ] Once figures are exported, add cross-references for the agent loop, system architecture, and rollout illustrations.

## The Work (`sections/4-work-new.tex`)
- [ ] Add a concise repository layout figure or bullet map highlighting the trainer, serving stack, and logging pipelines.
- [ ] List the key vLLM server flags, batching parameters, and scheduler adjustments that enforced deterministic tool calls.
- [ ] Report before/after peak VRAM and sync latencies for the live adapter synchronization upgrade.
- [ ] Name the TRL files touched for GSPO integration and summarize the API changes (dual-masked loss, KL term).
- [ ] Document the SLURM templates and environment modules used to launch GSPO runs.
- [ ] Provide concrete counts for updates, curriculum epochs, compute hours, and cluster specs for each reported model size.
- [ ] Summarize ablation attempts (e.g., KL weight sweeps, curriculum variants) in a compact table with qualitative outcomes.
- [ ] Populate the performance characterization table with VRAM, p50/p95 latency, and tokens/sec for the main model sizes.
- [ ] Cross-reference the appendix sections that contain configuration fragments and sync measurement details.

## Experimental Results (`sections/5-results.tex`)
- [ ] Draft the chapter opener summarizing overall SWE-Bench-Verified improvements from GSPO training.
- [ ] Describe the three learning phases using the reward/component curves from Figures~\ref{fig:training-loss} and \ref{fig:reward-components}.
- [ ] Compare GSPO stability versus PPO using gradient variance or loss volatility statistics from training logs.
- [ ] Explain computational efficiency gains (wall-clock, GPU util, sample efficiency) achieved by the training–inference duality.
- [ ] Build the main SWE-Bench-Verified table covering base vs GSPO checkpoints for Qwen3-8B/14B/30B and the Llama3.1-8B reference.
- [ ] Add the McNemar-style significance analysis with confidence intervals and effect sizes for each model comparison.
- [ ] Summarize RQ1 harness metrics (tool success, invalid calls, action efficiency) using the exported traces.
- [ ] Present the multilingual holdout pre/post analysis (table + plot) and discuss observed reward shifts.
- [ ] Detail the non-convergent Llama3.1-8B behaviour as a contrastive case in the analysis subsection.
- [ ] Fill in the ablation table (KL weight, curriculum composition) and interpret the critical components.
- [ ] Build the cross-scaffold transfer table summarizing zero-shot results across Mini-SWE-Agent/Aider/OpenHands.
- [ ] Draft the few-shot adaptation paragraph and learning-curve description for cross-scaffold transfer.
- [ ] Add qualitative analysis bullets for scaffold-specific behaviours referencing transcript excerpts.
- [ ] Write the learning-rate sensitivity paragraph with tested values and stability observations.
- [ ] Discuss trajectory-length impacts using the plotted distributions.
- [ ] Create the failure-mode table focused on SWE-Bench-Verified error categories and add commentary on file localization challenges.
- [ ] Complete the training performance metrics table and follow with scalability and cost-effectiveness narratives grounded in the recorded compute stats.
- [ ] Conclude the chapter with completed RQ1–RQ3 summary bullets referencing the SWE-Bench and multilingual findings.
- [ ] Add the “Design Philosophy” reflection (minimalist tools + GSPO) and the “Future Potential” note about scaling trends within the constraint of single-agent training.

## Conclusions (`sections/6-conclusions.tex`)
- [ ] Update the Cross-Model Portability paragraph to cite the finalized Qwen3/Llama3.1 results.
- [ ] Insert empirical evidence for RQ1 once harness metrics are finalized.
- [ ] Add the RQ2 summary referencing the multi-model SWE-Bench-Verified table.
- [ ] Summarize the multilingual holdout improvements for RQ3 with confidence intervals.
- [ ] Document in the conclusions that broader cross-harness studies are deferred while referencing the multilingual holdout evidence.
- [ ] Add RQ3 scaffold-transfer takeaways once zero-/few-shot experiments conclude.
- [ ] Replace the “pending validation” note with citations to the completed training curves and performance comparisons in Chapter~\ref{ch:results}.

## Appendices & Supplementary Material (`sections/appendices.tex`)
- [ ] Fill in the sampling-parameter table (temperature/top-p/tool budgets) for training and SWE-Bench evaluation.
- [ ] Add supplementary SWE-Bench-Verified breakdowns (per-category success, tool metrics) once the main tables are locked.
- [ ] Include the multilingual holdout plot and extended statistics backing the RQ3 discussion.

## Cross-Chapter Consistency
- [x] Remove any remaining references to SFT baselines, HumanEval, or cross-harness benchmarks that are now out of scope.
- [x] Verify that RQ numbering and chapter references stay aligned once all edits land.
- [ ] Run a final spellcheck/capitalization pass for recurring terms (Nano, GSPO, SWE-Bench-Verified) after textual updates.
