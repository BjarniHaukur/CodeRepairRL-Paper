# Project Details for CodeRepairRL

This thesis explores transforming large language models from passive learners—trained purely on observational data—into active agents capable of experiential learning through direct interaction with their environment. Concretely, I embed a coding agent (Aider) into an asynchronous, OpenAI-compatible vLLM server to create real-time reinforcement learning loops where the model actively navigates and edits code repositories to fix bugs. The training dataset, SWE-Gym, consists of real-world GitHub repositories with known issues and corresponding resolved patches, which serve as oracle ground truths. By rewarding the model based on the similarity of its generated patches to these oracle patches, the agent learns to perform context-aware navigation and issue resolution, rather than merely replicating static patterns. Evaluating generalization capabilities on the more challenging SWE-Bench-Verified dataset, this research investigates whether experiential, agent-in-the-loop reinforcement learning produces more robust and transferable code-repair skills compared to traditional synchronous RL or conventional fine-tuning methods.

In this thesis, I differentiate between two distinct types of coding agent scaffolding: a lightweight approach, exemplified by OpenAI's Codex agent, which interacts solely through basic terminal commands (e.g., ls, cat, grep) to navigate and understand repositories; and a heavyweight approach, represented by tools like Aider, which incorporate extensive contextual understanding features such as heuristic-based repository summarization (RepoMap) and internal reasoning dialogues to guide file operations. By training identical language models within these two contrasting scaffolds, I aim to evaluate the performance differences resulting from varying degrees of environmental support and contextual assistance, thereby exploring the trade-offs between scaffold complexity and agent efficiency.


## Research Questions

**RQ1**
Does integrating a scaffold into the reinforcement learning training loop significantly improve automated code repair performance relative to the original, non-RL-tuned model?
- Base assumption


**RQ2**
Does a minimalist coding scaffold (with simple, terminal-style interactions and minimal assumptions) yield better performance in RL-based code repair than a heavily engineered coding scaffold (with extensive support such as repository context mapping and simulated dialogues)?
- Bitter lesson angle, are fewer assumptions better in the long run?
- Motivated by my belief that companies like Anthropic and OpenAI are already doing this kind of training on their base models.
- A lot of work (in "heavyweight" scaffolds) goes into corraling models to behave a certain way. Is it perhaps better to keep it simple, then train the models to behave properly.


**RQ3**
To what extent do the performance gains from scaffold-in-the-loop RL training for code repair generalize beyond the training environment, such as to program repair tasks in other programming languages and to general code-generation benchmarks like HumanEval?
- Does it generalize? 
- Is this approach simply fine-tuning the models to behaver better inside their relative scaffolds or does it make them qualitatively better?

### Secondary version
**RQ 1**  
Does reinforcement learning conducted within a coding scaffold raise automated program-repair accuracy beyond what is achieved by the same model without reinforcement learning?

**RQ 2**  
When the identical reinforcement-learning protocol is applied, does a minimalist scaffold confer at least as much improvement in program repair as a heavyweight scaffold that provides extensive engineered context?

**RQ 3**  
Do the performance gains obtained through scaffold-based reinforcement learning (i) transfer to bug-fixing in another programming language when the scaffold is reused and (ii) persist on standard code-generation tasks run without any scaffold at all?


## Implementation Note

This project uses a [custom fork of TRL](https://github.com/BjarniHaukur/trl) (Transformer Reinforcement Learning), originally enhanced for better Weights & Biases logging. The fork has evolved to support integrating coding agents directly into the reinforcement learning loop, with contributions being prepared for potential integration into the main TRL repository.

## Technical Innovation: Agent-in-the-Loop Reinforcement Learning

Our project pioneers a novel approach that integrates agent frameworks directly into the reinforcement learning training loop. By implementing an OpenAI-compatible API server with vLLM that supports asynchronous token streaming, we enable:

1. **Multi-step, interactive agent training** - Rather than single-pass generation, agents can perform complex sequences of actions including repository exploration, code editing, and command execution.

2. **Parallelized inference with independent control** - Multiple agents operate concurrently on the same tasks, with independent control over their generation process (e.g., one might execute bash commands while another writes code).

3. **Direct reinforcement in realistic environments** - Training occurs in the same environment where models will be deployed, rewarding effective tool use and complex reasoning paths that lead to correct solutions.

This approach significantly lowers the barrier to entry for research on tool-using, environment-interactive agent training, bringing open-source capabilities closer to what we believe large AI labs are implementing internally.

## Implementation Approach

### Current Implementation: Enhancing TRL/GRPOTrainer with Agentic Capabilities

The core of our implementation involves integrating agentic capabilities directly into the reinforcement learning pipeline:

- **Integration with Aider**: We replace the standard `model.generate` functionality with `aider.Coder.chat`, leveraging Aider's built-in agentic coding tools for repository exploration, understanding, and multi-file change management.

- **Outcome-Supervised Reward Modeling**: Rather than attempting to reward intermediate steps of the process (which would be highly complex), we focus on rewarding the final patch generated. This allows us to assess and reinforce behaviors that lead to superior outcomes.

- **Parallel Agent Execution**: Our architecture enables running multiple coding agents in parallel, each addressing the same task but potentially taking different approaches. This parallelism not only improves training efficiency but also allows us to discover diverse solution strategies.

### Future Directions

While our current implementation focuses on patch similarity for reward modeling, several promising extensions are under consideration:

#### Test-Driven Reinforcement Learning

A natural evolution of our approach would implement a reward system based on passing test cases rather than similarity to reference solutions:

- Deploy scalable Kubernetes infrastructure to parallelize test case execution across multiple model outputs
- Reward functional correctness while allowing creative problem-solving approaches
- Provide greater flexibility compared to "golden patch" approaches
- Encourage models to develop diverse solution strategies that still satisfy requirements

This approach would require significant infrastructure development but would offer a more generalizable and robust measure of code correctness than patch similarity alone.

## Objectives

### Code Repair

Our primary objective remains generating accurate code patches to fix specific software issues. Initially, we reward the model based on the accuracy of the generated diffs using a search/replace approach that compares model-generated diffs against oracle solutions. With our agent-in-the-loop approach, we can now explore more complex multi-file edits while maintaining evaluation simplicity. Ultimately, we aim to use software tests as the oracle for correctness, though this presents challenges due to the extensive compute requirements for running comprehensive test suites. **While test execution and support for additional languages such as Java are promising extensions to this work, the current focus is on Python and diff-based evaluation.**

### Code Implementation from Specification

We also explore the task of implementing code from specifications using our stack.py dataset. This represents a related but distinct challenge from code repair, with fewer existing datasets in this category. The model must generate a complete implementation based on specifications, testing its understanding of requirements, adherence to contextual style, and ability to produce functional code. Our agent integration approach is particularly valuable here, as it allows for incremental implementation and self-testing.

### Vulnerability Classification

As a secondary objective, we address classification of software vulnerabilities. This task evaluates our model's ability to reason about complex software vulnerabilities, effectively creating a test-time compute-enabled, process-verifiable classifier. Unlike traditional black-box models, our approach provides explicit reasoning paths, enhancing transparency and trustworthiness.

### Benchmark Performance Evaluation

A key goal of this project is to observe and measure performance improvements on established benchmarks including SWE-Bench, Aider-polyglot, and similar evaluation frameworks. These benchmarks provide standardized measurement of code understanding, editing, and implementation capabilities, allowing us to quantify the effectiveness of our agent-in-the-loop reinforcement learning approach compared to existing methods.

For detailed run results and progress on our objectives, see our [WandB project page](https://wandb.ai/assert-kth/TTC).

## Post-Training Perspective

Post-training of LLMs is a complex, multivariate process consisting of many specialized steps across various domains, ultimately resulting in powerful general-purpose models. While achieving a comprehensive post-training pipeline is beyond the scope of a master's thesis, our project aims to demonstrate a significant performance improvement in the specific domain of code repair through agent-in-the-loop training. We believe such targeted improvements represent one valuable step in the broader post-training stage of frontier models.

## Experimental Plan & Benchmark Rationale

Our agent‑in‑the‑loop code‑repair thesis hinges on **showing real generalisation, not memorisation**.  
To that end we split data into **(i) a Python-focused training set** that lets the agent learn tooling skills, and **(ii) evaluation suites across both Python and Java** that test generalization capabilities.

| Phase | Corpus | Language(s) | Why it's in‑scope |
|-------|--------|-------------|-------------------|
| **Train / fine‑tune** | **SWE‑Gym** (≈ 2.4 k tasks) | Python | RL‑ready, containerised tasks—perfect for teaching interactive tool use. |
| **Evaluate (primary)** | **SWE‑Bench‑Verified** (≈ 2.3 k bugs) | Python | Harder, multi‑file patches in modern repos; measures generalization beyond SWE‑Gym. |
| **Evaluate (generalization)** | **Defects4J v2.0** (835 bugs) | Java | Tests cross-language generalization capabilities, addressing RQ3 about transfer learning. |
| **Evaluate (generalization)** | **GitBug‑Java** (199 recent bugs) | Java | 2020‑2023 GitHub issues with modern Java ecosystems; further tests language transfer. |
| **Evaluate (sanity / optional)** | **BugsInPy** + **QuixBugs** | Python & Java | Quick cross‑language smoke tests; catch trivial regressions before full runs. |

**Why this split works**

* **Focus on Python training** – Concentrates learning efforts on a single language ecosystem, avoiding dilution of the training signal.
* **Cross-language evaluation** – Java evaluation directly addresses RQ3, measuring how well Python-learned repair skills transfer to a syntactically different language.
* **Temporal gap** – Training data comes from curated SWE-Gym snapshots; evaluation bugs are more recent (2020 – 2024), reducing leakage risk.
* **Task diversity** – From simpler Python tasks to Java repairs, giving a complete picture of the agent's generalization capabilities.
* **Comparability** – SWE‑Bench is the emerging LLM standard for Python; Defects4J provides a traditional baseline for Java evaluations.

Success criterion: **≥ 5 pp absolute improvement** over strong zero‑shot LLM baselines on SWE‑Bench‑Verified for in-domain performance, plus *any* measurable improvement on Java datasets would demonstrate cross-language transfer. This directly validates RQ3 by showing whether agent skills learned in Python environments can generalize beyond the training domain.


## Compute Efficiency

Since LLMs are inherently large, training them can be challenging for individual researchers. Our project has moved to a streamlined approach using DeepSpeed and Accelerate for distributed training. We use "ZeRO-Stage 3" DeepSpeed configuration, where optimizer state, gradients, and model parameters are all sharded between GPUs.

## Definitions and Intuitions

In our project, we use specific terms that are crucial for understanding our approach:

- **KL-divergence (Kullback-Leibler divergence)**: A fundamental metric in information theory that measures how one probability distribution differs from another. In our reinforcement learning setup, we use it to quantify how much our updated policy ($\pi_{t+1}$) diverges from our reference policy ($\pi_t$) as the training progresses. This helps ensure our model learns gradually and stably.

- **Loss**: In the context of our model, 'loss' refers to the discrepancy between the reference policy and the update policy. It is a measure of how much the model's predictions deviate from the expected outcomes based on the evolving reference policy.

### Conceptual Insights

Our approach to optimizing models in this project is guided by several foundational ideas:

- In SFT, we optimize the model to output the exact sequence of tokens, whereas in GRPO (even if the text is identical), we optimize the parameters to produce any sequence that maximizes reward.

- This approach effectively bootstraps the model onto itself, as it uses its own policy as a baseline for improvement.

- The more capable the base model, the more significant the returns from our RL training, provided the RL environment is sufficiently challenging (and not overly so).

- By training agents in realistic environments with real tool use, we reinforce not just token generation but effective problem-solving strategies and environmental interactions. 