## Core Concepts
| Term | Meaning / Intent |
|------|------------------|
| **Automated Code Repair** | The task of automatically generating patches that change buggy source code into a version that passes its reference test suite or matches an oracle patch. |
| **Coding Agent** | An LLM that can read, navigate and edit a code-base through tool calls (e.g. reading files, running commands, applying patches). |
| **Scaffold** | The external framework that mediates those tool calls, feeds observations back to the model and enforces a step-by-step dialogue. Think: the “shell” the coding agent lives in. |
| **Agent-in-the-Loop RL** | Reinforcement learning where the LLM is trained *while* acting through the scaffold, receiving a reward only after each full bug-fix attempt. |

## Scaffolding Taxonomy
| Term | Meaning / Entails |
|------|-------------------|
| **Lightweight Scaffold** | Minimal tool surface: typically shell commands (`ls`, `cat`, `grep`, patch apply) plus plain text prompts. No repository summary or artificial coaching. |
| **Heavyweight Scaffold** | Adds engineered assistance such as RepoMap summaries, heuristic file ranking, or simulated “internal” dialogue turns that guide the model. Example: Aider. |
| **Tool Call / Action Step** | A single atomic operation executed by the scaffold on behalf of the model (open file, write patch, run tests). |
| **Episode** | One complete attempt to fix a bug: sequence of tool calls → final patch → reward. |


## Training Pipeline Components
| Term | Meaning / Entails |
|------|-------------------|
| **OpenAI-compatible vLLM Server** | Async inference backend exposing the OpenAI Chat API; streams tokens and accepts weight updates during training. |
| **Parallel Agents** | Running several agent instances concurrently (distinct repos / GPUs) to accelerate data collection for RL. |


## Evaluation Metrics
| Metric | Definition |
|--------|------------|
| **Success Rate / Solve Rate** | Percentage of issues where the generated patch is identical to the oracle or passes all tests. |
| **Pass@k** | Probability that *any* of the top-k generated candidate patches is correct. |
| **Diff Similarity Score** | Normalised measure (0-1) of how close a generated patch matches the oracle diff. |
| **Actions-per-Solve** | Median number of tool calls taken for a successful fix (proxy for efficiency). |