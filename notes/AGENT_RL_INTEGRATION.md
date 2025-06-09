# Agent Integration into RL Training Loops

## Overview

This document explores our approach to integrating agent frameworks into reinforcement learning (RL) training for large language models (LLMs). By implementing an OpenAI-compatible API server with vLLM, we enable existing agent scaffolding to be directly incorporated into the RL training loop, allowing for more complex and realistic training scenarios.

## Key Insights

### From Observation to Interactive Experience

Our approach pioneers a method that fuses asynchronous token streaming with distributed weight synchronization, enabling real-time reinforcement in coding applications—where the necessary "sensory feedback" is already conveyed in language.

1. **Simultaneous Training and Serving**  
   - In an RL scenario, we're not just feeding prompts and waiting for outputs (i.e., "observing") but actively using those outputs for training—rewarding or penalizing the model in real time.
   - The inference server isn't static. Instead, it receives immediate weight updates (or partial updates like LoRA) as soon as a response is evaluated.

2. **Direct Reinforcement via OpenAI-Compatible API**  
   - By implementing an OpenAI-compatible API endpoint, we can plug nearly any LLM application directly into an RL training loop.
   - A heuristic that measures output quality lets us reinforce desirable behavior immediately.

3. **Real-Time Weight Synchronization**  
   - Every time an agent (or a coding agent like Aider) produces an output, the updated weights are sent back and immediately synchronized across all inference processes.
   - The system must handle fast and reliable weight updates via distributed communication so that all serving processes reflect the latest model improvements.

4. **Parallelized Inference with Multiple Agents**  
   - We run multiple agents in parallel on the same task so that many concurrent API calls are made.
   - Every batch of these responses results in an update that is synchronized across instances.

5. **Coding Domain as a Natural Fit**  
   - The coding domain is particularly well-suited for this integrated training/serving paradigm because coding is inherently language-based.
   - The scaffolding mainly involves extracting code blocks and interpreting them as diffs or commands—making the feedback loop straightforward.
   - Tasks like program repair benefit greatly from multi-step approaches that current synchronous `.generate()` calls cannot support effectively.

## Motivation and Challenges

### Current Limitations

Currently, TRL only supports synchronous, batched `.generate()` calls for inference. This restricts the types of rollouts that can be created, especially in domains that benefit from having multi-step approaches, tool use, or environment interaction.

For example, in tasks from SWE-Gym, the model needs to generate code edits for real repositories. To do this in one `.generate()` call, the user must manually construct the relevant repo context and later parse outputs like diffs to extract useful reward signals. This makes experimentation slow and feels like "reinventing the wheel."

Rather than building ad-hoc scaffolding from scratch, integrating existing coding agents like Aider directly into the training loop is more efficient. These agents already support rich workflows such as repo mapping, diff parsing, and iterative interaction—and they use the OpenAI API interface.


## Proposed Solution: OpenAI-Compatible vLLM API Server

### Integration of AsyncLLMEngine

- **Native Streaming Capability:**  
  The `AsyncLLMEngine` is designed to yield token deltas via an async generator, providing genuine, real-time streaming outputs.
  
- **Concurrency and Scalability:**  
  An asynchronous architecture inherently supports many concurrent API calls using an event loop, reducing blocking and providing lower latency—key for interfacing directly with an RL training loop.
  
- **Integration with Weight Sharing:**  
  This approach requires re-engineering parts of our system to adapt weight sharing via collective_rpc into an asynchronous context, laying the foundation for a robust, scalable solution.

### Implementation Details

The implementation mirrors the weight syncing logic from `trl/scripts/vllm_serve.py`, but offloads most complexity to the existing `vllm.entrypoints.openai.api_server` infrastructure.

This enables:
- Training on significantly more complex rollouts than the standard synchronous `.generate()` endpoint can support
- Seamless integration with existing agent frameworks via the OpenAI API interface
- Reproduction of pipelines similar to those used in state-of-the-art models like OpenHands LM 32B