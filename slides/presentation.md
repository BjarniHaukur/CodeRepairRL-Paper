---
marp: true
theme: kth-theme
paginate: true
math: katex
html: true
---

<!-- _class: lead -->

# Learning Agency in the Terminal
## Lessons from Repository-Level Reinforcement Learning

**Bjarni Haukur Bjarnason**

**Supervisor**: André Afonso Nunes Silva
**Examiner**: Martin Monperrus
**Opponent**: Han Tu

<div style="height: 80px;"></div>

**October 2025**

---

# Motivation

Frontier coding agents achieve impressive results, but **training methods remain proprietary**.

Even "open-source" releases don't include training recipes.

<div style="display: flex; align-items: center; justify-content: center; gap: 80px; margin-top: 40px;">
<div style="min-width: 300px;">

### Reasons why?

- Prohibitively expensive
- Prohibitively complicated

</div>

<div style="border-left: 4px solid #555; height: 120px;"></div>

<div style="min-width: 300px;">

### Solution?

- Make it cheaper
- Make simplifying assumptions

</div>
</div>

<div style="height: 80px;"></div>

<div style="display: flex; justify-content: center; align-items: center; gap: 100px;">
<img src="images/anthropic.png" style="height: 70px; width: auto;">
<img src="images/openai.png" style="height: 70px; width: auto;">
<img src="images/qwen.png" style="height: 70px; width: auto;">
<img src="images/kimi.png" style="height: 70px; width: auto;">
</div>

<!--
I'd wager that all of us in this room of heard of and used LLM coding agents. Frontier systems like Claude Code and Codex are incredibly useful, but their training methods are completely proprietary.
Even teams like Alibaba Qwen and MoonshotAI Kimi who do "open" research FALL SHORT when it comes publishing full recipes. Just weights and algorithms.
Thus the open resource community suffers

- Two barriers: cost (tens of thousands of GPU-hours), complexity (distributed systems)
- Our solution: execution-free rewards + language-agnostic design = cheap & simple
- Result: one of the first fully open training recipes for agentic debugging
-->

---

# Core Contributions

**1. Training infrastructure for online RL on LLMss**
- Implemented live weight synchronization
- Enables multi-turn, tool-using, asynchronous episodes

**2. Academic-scale feasibility**
- 6× A100 GPUs for 32B training, **12× less than DeepSWE**
- 3x A100 GPUs for 14B training

**3. Execution-free training paradigm**
- Single reward function across 10 languages
- No test infrastructure required

<!-- Speaker notes:
In this work, I contribute in my own small way on three fronts
1.) Created training infrastmructure to train ulti-turn online rl
2.) Cheap: 120× less compute than comparable systems (academic feasibility)
3.) Simple: execution-free rewards eliminate infrastructure complexity, enable multilingual training
-->

---

# Research Questions

**RQ0: Training Convergence**
Does GSPO training converge effectively with execution-free rewards?

**RQ1: Harness Adaptation**
How does RL training improve operational behavior within the Nano harness?

**RQ2: SWE-Bench Performance**
Does execution-free RL improve test-verified success on SWE-Bench-Verified?

**RQ3: Multilingual Generalization**
Does execution-free RL improve performance across languages without language-specific engineering?

<!-- Speaker notes:
And As measured by these four research questions:
Does training converge? Does the agent learn to use tools better? Does it improve on real benchmarks? Does it work across languages?
-->

---

<!-- _class: section-title -->

# Background & Related Work

<!-- Speaker notes:
First I will quickly go through some of the foundational background
-->


---

# APR with LLMs

**Scaffold-Free**: Direct patch generation
- RepairLLaMA: Fine-tuned on bug-fix pairs
- CodeRL: RL with unit test feedback at function level

**Agentless**: Script-driven interaction
- External harness curates context, applies edits
- Model doesn't control exploration strategy

**Agentic**: Model-driven interaction
- SWE-Agent, OpenHands, mini-swe-agent
- Model autonomously navigates, edits, iterates

<!-- Large language models have enabled an entirely new class of Automatic Program Repair methods
To highlight our defintion of agentic, we contrast it with other types of LLM approaches 

1. Mostly uses LLM as functions that can learn patterns
2. Uses LLMs as scripting, with some amount of autonomy
3. Full autonomy in the given task
-->

---

# Why Agentic LLM Systems Enable RL

**LLMs have sufficient priors to attempt real world tasks coherently**
- Stumbling attempts are VASTLY better than starting from nothing

**Agentic systems enable End-to-End Learning**:
- Model controls all actions → clean credit assignment
- Actions → observations → rewards (full causal chain)
- No scaffold interference breaking the gradient signal
- Result: RL can optimize exploration strategies, not just patterns

<!-- Speaker notes:
And what I find really exciting is that

LLMs provide the foundation to attempt real tasks, and agentic systems let the model control everything—what to explore, when to submit. This creates a clean gradient signal. Contrast with agentless: scaffold decides navigation, breaks causal chain, prevents learning exploration strategies. Now we can finally apply decades of RL research to optimize policies for real-world debugging.
-->

---


# Policy Optimization PPO Foundation

**Proximal Policy Optimization** (Schulman et al., 2017)

Clipped surrogate objective with value baseline:

$$
J_{\text{PPO}}(\theta) = \mathbb{E}\left[\frac{1}{|y|}\sum_{t=1}^{|y|} \min(w_t \hat{A}_t, \text{clip}(w_t) \hat{A}_t)\right] - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

Where $w_t = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t \mid x, y_{<t})}$ (per-token importance ratio)

**Advantages**: Stable updates via clipping, well-established
**Drawback**: Separate value network $V_\phi$ doubles memory footprint

<!-- 
Policy optimization methods are ways to directly improve an agent’s decision-making policy by adjusting its parameters to maximize expected rewards through experience..

First and foremost of those methods is PPO, I don´t have time to go into too much detail but basically:



We optimize 
-->

---

# Policy Optimization: GRPO Innovation

**Group Relative Policy Optimization** (Shao et al., 2024)

**Key Insight**: Replace learned value baseline with group-relative baseline

Sample $G$ responses per query, compute group advantage:

$$
A_i = \frac{r(x, y_i) - \mu}{\sigma + \varepsilon}, \quad \mu = \frac{1}{G}\sum_{j=1}^G r(x, y_j)
$$

$$
J_{\text{GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \min(w_{i,t} A_i, \text{clip}(w_{i,t}) A_i)\right]
$$

**Advantages**: No value network, lower memory, simpler training
**Issue**: Per-token importance ratios unstable for long sequences


<!-- Speaker notes:
GRPO's innovation was eliminating the value network by using group-relative rewards. Instead of training a value network, we directly let determine 
-->

---

# Policy Optimization: GSPO Advancement

**Group Sequence Policy Optimization** (Yuan et al., 2025)

**Core Innovation**: Sequence-level importance weighting


$$
s_i(\theta) = \left(\frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)}\right)^{1/|y_i|} = \exp\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log\frac{\pi_\theta(y_{i,t})}{\pi_{\theta_{\text{old}}}(y_{i,t})}\right)
$$

$$
J_{\text{GSPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^G \min(s_i A_i, \text{clip}(s_i) A_i)\right]
$$

**GRPO → GSPO Benefits**:
- Superior stability: single weight per sequence vs. noisy per-token ratios
- Robustness: tolerates training/inference engine precision differences


<!-- Speaker notes:
We use the more stable variant but I won't go into too much detail
-->

---

# Positioning: Related Concurrent Work

**SWE-RL** from Meta (Wei et al., 2025):
- Agentless GRPO with patch-similarity rewards
- **Static single-turn**: full context provided, no interactive tool use

**DeepSWE** from Agentica (Zhang et al., 2025):
- Agentic GRPO with test-based rewards
- 72 H100s

**CodeRepairRL (ours)**:
- Agentic GSPO with patch-similarity
- 3 A100s, **24x** less

---

# Model Selection: Qwen3-14B

**Why Qwen3?**
- Hybrid reasoning model with strong tool-calling capabilities
- Community consensus: strongest open-weight models at the time

**Why 14B?**
- **Practical constraint**: 8B too large for 2 GPUs, too small for 3
- 32B requires 6 GPUs minimum
- 14B optimal fit for 3× A100 allocation

<!-- Speaker notes:
Need power of 2 for both training and inference
-->

---

# Datasets

**Training (1,000 multilingual tasks)**:
- SWE-Gym: 750 Python debugging tasks from real repositories
- SWE-Bench-Multilingual: 250 tasks across 9 languages (Rust, Java, PHP, Ruby, JS, TS, Go, C, C++)

**Evaluation (500 python tasks)**:
- SWE-Bench-Verified: 500 Python tasks


<!-- Speaker notes:
All are datasets buggy repositories from Github with instructions to fix them
-->

---

<!-- _class: section-title -->

# Methodology & Implementation


<!-- Speaker notes:
Now let's see how we built the training system—starting with design decisions and implementation details.
-->

---

# Nano Agent

<div class="columns">
<div>

**Fully agentic**
- No repository summaries
- No pre-computed context
- Starts with a blank slate + a mission

**Two tools only**
- `shell` to navigate and read
- `apply_patch` to affect files

</div>
<div>

![width:500px](../plotting/figures/nano_blank.png)

</div>
</div>

<!-- First I'll tell you about Nano. It is very similar to coding agents you know, but simpler and tuned specifically to work well in training.  -->


---

# Illustrative Nano episode

```
shell(cmd="ls src/")
drwxr-xr-x  utils/
-rw-r--r--  main.py
-rw-r--r--  config.py

shell(cmd="grep -n 'def process' src/utils.py")
42:def process_data(data):
43:    return data.strip().lower()

apply_patch(
  file_path="src/utils.py",
  old_content="return data.strip().lower()",
  new_content="return data.strip().lower().replace(' ', '_')"
)
Patch applied successfully.
```

Agent explores → identifies bug → applies targeted fix

<!-- Speaker notes:
To illustrate more clearly...
-->

---

# Sidestepping Diff Generation Complexity

**Nano uses semantic search-replace → git computes canonical diff**

<div style="padding: 12px; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 24px; line-height: 1.6; margin: 20px 0; white-space: pre;">$ git diff
diff --git a/src/utils.py b/src/utils.py
index abc123..def456 100644
<span style="color: #0066cc;">--- a/src/utils.py</span>
<span style="color: #0066cc;">+++ b/src/utils.py</span>
<span style="color: #008080;">@@ -42,1 +42,1 @@ def process_data(data):</span>
<span style="color: #cc0000;">-    return data.strip().lower()</span>
<span style="color: #009900;">+    return data.strip().lower().replace(' ', '_')</span></div>

**Eliminates brittle diff formatting errors**

--- 


# Patch-Similarity Examples

**No Similarity (R < 0.1)** — Same file, different functions → minimal overlap

<div style="display: flex; justify-content: center; margin: 20px 0;">
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">

<div>
<strong>Agent Patch:</strong>

<div style="padding: 12px; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 24px; line-height: 1.5; margin-top: 8px; white-space: pre;">
<span style="color: #0066cc;">--- a/src/utils.py</span>
<span style="color: #0066cc;">+++ b/src/utils.py</span>
<span style="color: #008080;">@@ -42,1 +42,1 @@</span>
<span style="color: #cc0000;">-    return data.strip().lower()</span>
<span style="color: #009900;">+    return data.strip().lower().replace(' ', '_')</span>
</div>

</div>

<div>
<strong>Ground Truth Patch:</strong>

<div style="padding: 12px; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 24px; line-height: 1.5; margin-top: 8px; white-space: pre;">
<span style="color: #0066cc;">--- a/src/utils.py</span>
<span style="color: #0066cc;">+++ b/src/utils.py</span>
<span style="color: #008080;">@@ -8,1 +8,1 @@</span>
<span style="color: #cc0000;">-MAX_SIZE = 1024</span>
<span style="color: #009900;">+MAX_SIZE = 2048</span>
</div>

</div>

</div>
</div>

<!-- 
With the agent generated diff
Our reward function averages sequence similarities across the diffs of files affected.

Correlates with functional similarity but does not fully overlap.-->

---

# Patch-Similarity Examples

**Partial Match (R ≈ 0.75)** — Same file, same idea, different structure

<div style="display: flex; justify-content: center; margin: 20px 0;">
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">

<div>
<strong>Agent Patch:</strong>

<div style="padding: 12px; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 24px; line-height: 1.5; margin-top: 8px; white-space: pre;">
<span style="color: #0066cc;">--- a/parser.py</span>
<span style="color: #0066cc;">+++ b/parser.py</span>
<span style="color: #008080;">@@ -18,3 +18,4 @@</span>
 def parse(text):
<span style="color: #cc0000;">-    return text</span>
<span style="color: #009900;">+    tokens = text.split()</span>
<span style="color: #009900;">+    return tokens</span>
</div>

</div>

<div>
<strong>Ground Truth Patch:</strong>

<div style="padding: 12px; border-radius: 4px; font-family: 'Courier New', monospace; font-size: 24px; line-height: 1.5; margin-top: 8px; white-space: pre;">
<span style="color: #0066cc;">--- a/parser.py</span>
<span style="color: #0066cc;">+++ b/parser.py</span>
<span style="color: #008080;">@@ -18,1 +18,1 @@</span>
<span style="color: #cc0000;">-    return text</span>
<span style="color: #009900;">+    return text.split()</span>
</div>

</div>

</div>
</div>


---

# Training Infrastructure

![width:1100px](../plotting/figures/training_sequence_diagram.png)


<!-- I'll linger on this for a bit

LLMs are still only weights and biases, we need sophisticated endoint logic to make them useful (e.g. tool calling, conversation turns etc.)

-->

---

# Compute Optimizations Overview

| Component | Optimization | Benefit |
|-----------|--------------|---------|
| **Parameters** | LoRA (rank 32) | ~98% fewer trainable params |
| **Optimizer** | DeepSpeed ZeRO-2 | Shard states across GPUs |
| **Gradients** | Accumulation (4 steps) | 4× effective batch size |
| **Activations** | Gradient checkpointing | ~40-60% activation memory |
| **Operations** | Fused Triton kernels | ~40% peak memory (TRL) |
| **Precision** | BF16 mixed precision | 2× memory reduction |
| **Episodes** | Explicit limits + warnings | Better reward signals |

**Compositionality**: Each optimization addresses distinct bottleneck

**Result**: 14B training on 2× A100, 1x A100 for inference

<!-- Speaker notes:
Making 14B training fit on 3 A100s required stacking every memory optimization available—this is how we achieved 24× GPU reduction.

Significantly lower bar to entry.
-->

---

<!-- _class: section-title -->

# Experimental Results


<!-- Speaker notes:
Now the payoff—did it work? Four research questions about convergence, adaptation, multilingual generalization, and system performance.
-->


---

# RQ0: Training Convergence

![width:1150px](../plotting/figures/plots/temporal/reward_over_time_8dc73bp4.png)

**Yes—training converges. Rewards doubled, variance increased.**



<!-- Speaker notes:
Yes—training converges! 
- No variance collapse: strong gradient signal throughout
- 3 A100s, 2 days wall-clock, 144 GPU-hours total
-->

---

# Why Variance Matters

In group-relative methods like GSPO, advantages compare each response against within-group statistics:

$$
A_i = \frac{r(x, y_i) - \mu}{\sigma + \varepsilon}
$$

**When all rewards are similar**:

- Advantages approach zero → no gradient signal → learning stops

**Our variance increased throughout training**:
- Policy generates diverse outcomes with different rewards
- Implies there is more to learn


<!-- Speaker notes:
Why does increasing variance matter? In GSPO, variance collapse kills learning—advantages approach zero and gradients disappear.
- GSPO advantages normalize by group statistics: A = (r - μ) / σ
- If σ collapses: all advantages → 0, no gradient signal
- Our variance increased: policy generates diverse outcomes with different rewards
- This proves the policy gradient signal stayed strong throughout training
-->

---

# RQ1: Harness Adaptation

![width:920px](../plotting/figures/plots/temporal/tool_success_rates_ema0.05_8dc73bp4.png)

<!-- 
Harness adaptation through tool calling accuracies and 
Shell success rates nearly doubled—and this happened during the budget-exhaustion phase, so it's learning better tool use, not just selectivity.
- Shell: 45% → 80% (+78% improvement)
- apply_patch: volatile but trending up
- Key: improvements during high-call-count phase = learning better generation
- Not just: learning to be selective
-->

---

# RQ1: Harness Adaptation

![width:920px](../plotting/figures/plots/temporal/command_trend_direct_ema0.05_8dc73bp4.png)

<!-- Speaker notes:
Strategic shift from basic navigation to focused debugging—grep emerges, while ls/cd decline.
- Early: heavy ls, cd (basic navigation)
- Late: grep, focused searches (targeted debugging)
- apply_patch spike at step 200: over-exploration during reward dip
- Evidence: policy learns strategic exploration patterns
-->

---

<!-- _class: iframe-slide -->

# Early Training Command Flow

<div class="iframe-wrapper">
<div style="transform: scale(0.4); transform-origin: center center;">
<iframe src="../plotting/figures/plots/sankey/early_training_sankey_T25_8dc73bp4.html" style="width: 2800px; height: 1400px; border: none;"></iframe>
</div>
</div>

<div class="smaller">
Command distribution and transition patterns during early training (first 20% episodes)
</div>



<!-- Speaker notes:
Early training: chaotic command flow with lots of basic navigation—look at the thickness of cd and ls flows.
-->

---

<!-- _class: iframe-slide -->

# Late Training Command Flow

<div class="iframe-wrapper">
<div style="transform: scale(0.4); transform-origin: center center;">
<iframe src="../plotting/figures/plots/sankey/late_training_sankey_T25_8dc73bp4.html" style="width: 2800px; height: 1400px; border: none;"></iframe>
</div>
</div>

<div class="smaller">
Command distribution and transition patterns after training convergence (last 20% episodes)
</div>

<!-- Speaker notes:
Late training: much more focused—grep dominates, less random navigation, cleaner transitions to apply_patch.
-->

---

# RQ2: SWE-Bench Performance

| Metric | Baseline | Ours (step 460) | Change |
|--------|----------|-----------------|--------|
| **Completed instances** | 186 (37.2%) | 388 (77.6%) | **+108%** |
| **Resolved instances** | 36 (7.2%) | 31 (6.2%) | -13.9% |
| **Empty patches** | 313 (62.6%) | 111 (22.2%) | **-64.5%** |

**Mixed results**: Operational competence improved dramatically, functional correctness flat
- Same harness, no scaffold engineering, identical evaluation settings
- Completion rate more than doubles, test success unchanged
- Interpretation: Acquired harness operation (Tier 1), not yet functional correctness (Tier 2)



<!-- Speaker notes:
Mixed results: completion rates doubled, but test-verified success stayed flat—we learned operational competence but not yet functional correctness.
- Completion: 37% → 78% (+108%) — huge improvement
- Test-verified resolve: 7.2% → 6.2% (flat, within error margins)
- Empty patches: down 64% (313 → 111)
- Interpretation: acquired Tier 1 capability (reliable operation) but not Tier 2 (functional correctness)
- Rewards increased 54%, so learning is happening—just not translating to tests yet
-->

---

# RQ2: Reward Progression Context

| Metric | Baseline | Ours (step 460) |
|--------|----------|-----------------|
| **Mean patch-similarity reward** | 0.122 | **0.189** |
| **Improvement** | — | **+54%** |



**Optimistic interpretation:**
1. First: Escape zero-reward regime ✓
2. Then: Climb toward functional correctness ← (not yet)



<!-- Speaker notes:
Patch-similarity rewards did increase substantially—54% improvement—validating that learning occurred.
- Baseline mean: 0.122, Ours: 0.189 (+54%)
- Baseline heavily skewed: 313 empty patches (zero reward by definition)
- Natural hierarchy: first escape zero-reward regime, then climb toward quality
- Step 460: successfully acquired Tier 1, beginning Tier 2 climb
-->

---

# RQ3: Multilingual Generalization

![width:920px](../plotting/figures/plots/analysis/language_reward_epochs_n1000_8dc73bp4.png)

**Yes—consistent improvements across all 9 languages (epoch 1 → epoch 2)**
- Python-heavy curriculum (750/1,000 tasks)
- Language-agnostic reward enables unified training


<!-- Speaker notes:
All nine languages show measurable improvements from epoch 1 to epoch 2—validation that patch-similarity training works across diverse ecosystems.
- All languages: consistent gains
- Java, Ruby, PHP: highest absolute (0.12-0.18)
- Rust, TypeScript: lower absolute but clear progression (0.06-0.08)
- Evidence: language-agnostic reward enables unified training
-->

---

<!-- _class: section-title -->

# Discussion & Conclusions


<!-- Speaker notes:
Let's synthesize what we learned, acknowledge limitations, and chart future directions.
-->

---

# Key Findings Summary

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 40px;">

<div>

**RQ0: Training Convergence (✓)**
- Rewards doubled, variance increased
- Execution-free rewards sufficient

</div>

<div>

**RQ1: Harness Adaptation (✓)**
- Shell success: 45% → 80%
- Strategic command emergence
- Policy-harness co-adaptation

</div>

<div>

**RQ2: SWE-Bench Benchmarks (!)**
- Completion: 37% → 78%
- Test success: flat (7.2% → 6.2%)
- Operational, not yet functional

</div>

<div>

**RQ3: Multilingual (✓)**
- Improvements across all 9 languages
- Language-agnostic validated

</div>

</div>


<!-- Speaker notes:
- RQ0: rewards doubled, variance increased = convergence ✓
- RQ1: operational improvements across all metrics ✓
- RQ2: completion doubled, test success flat (operational but not functional) ⚠️
- RQ3: all 9 languages improved, language-agnostic validated ✓
-->

---

# Future Directions

<div style="display: grid; grid-template-columns: 1fr auto 1fr; gap: 60px; margin-top: 40px;">

<div style="display: flex; flex-direction: column; gap: 40px;">

<h3 style="margin-top: 0; text-align: center;">Further Optimizations</h3>

<div>

**Cross-group walltime optimization**
- Slightly off-policy episodes
- Eliminate idle time during training

</div>

<div>

**Unified training-inference architecture**
- Co-locate on single node
- Lower GPU requirements (3 → 1-2)

</div>

</div>

<div style="border-left: 3px solid #004791; height: 100%;"></div>

<div style="display: flex; flex-direction: column; gap: 40px;">

<h3 style="margin-top: 0; text-align: center;">Further Abilities</h3>

<div>

**Extended training**
- Train for a longer time
- Train 32B for longer duration

</div>

<div>

**Multi-task curriculum training**
- Concurrent training on diverse tasks

</div>

</div>

</div>


<!-- Speaker notes:
Several promising directions. Infrastructure: eliminate idle time, reduce GPU requirements. Most important: extended training with larger models—our curves haven't plateaued, DeepSWE showed this works. Multi-task: does debugging skill transfer? And the big open question: where do execution-free methods plateau, or do they scale all the way?
-->

---

<!-- _class: lead -->

# Questions?

**Thank you for your attention**

Bjarni Haukur Bjarnason
bhbj@kth.se

KTH Royal Institute of Technology
Division of Theoretical Computer Science