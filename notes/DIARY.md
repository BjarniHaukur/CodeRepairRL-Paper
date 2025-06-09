# Project Diary


## Backlog

<details>
<summary>General Research Tasks</summary>
- [x] Look at how SWE-Bench works exactly, should we train for that behavior end2end?
- [ ] Read S∗: Test Time Scaling for Code Generation: https://arxiv.org/pdf/2502.14382
- [ ] Read MUFIN **paper**
</details>

<details>
<summary>Technical, non critical tasks</summary>
- [ ] Make median/std log to train_median/train_std (WandB)
  - for convenience
  - annoying to have to scroll throught those statistics when all I want is the actual value
</details>

## May 19 - May 26

### Writing
- [ ] Write a lot while berzileus is down (BREAK INTO TASKS)

### TRL
- [x] Create a PR and @ maintainer
- [ ] Update email and issue when good training run is going
- [ ] Bug?
  - /home/ubuntu/CodeRepairRL/.venv/lib/python3.11/site-packages/torch/utils/checkpoint.py:87: UserWarning: None of the None of the inputs have inputs have requires_grad=True. Gradients will be None
- [x] Bug: multi_turn is_eos

### CodeRepairRL
- [x] Add unsloth FastLanguageModel?
  - can't, unsloth doesn't support latest trl
  - and a large portion of their performance gains are through quantization which I'm not sure I can support
- [x] Enable gradient checkpointing
- [x] liger loss
- [x] Completion length bug 
- [x] Understand and configure grad_accum_steps / similar
  - Do some debug prints to figure out where and how they impact VRAM utilization
- [x] Setup two grpo.yaml files, one for short context / single turn, another for long context
- [x] Setup deepspeed
  - zero 2 is our best option
- [x] Tune the scripts
  - 8B, 14B, 32B
  - 8k context minimum
  - 1 inference server gets us really far
  - we need way more training compute
- [ ] See if aggressively clipping the grad helps
- [ ] Why is Nano not producing a diff? Skill issue or context_window issue?
- [ ] SFT stage?
  - I believe my use falls under the academic section of their clause
  - Generate N completions per SWE-GYM[:32]
  - Train on best-of-N if diff similarity > 0.8?

### Nano
- [x] Set no_logs flag or similar
- [x] Ensure context utilization
- [x] Make a strucured tool call log
  - Could be useful to penalize commands downstream during training
  - like cat (token expensive and "narrow) and reward rg (token efficient)
  - make each tool return an output dict instead
- [x] Return final token count
  - if log mode active, print in end
- [x] Make __version__ accessible without complicating the build process and without having to write it in two places
- [x] Make a minimal yet nice looking print in the beginning of a session if verbose
- [x] Modularization
  - simple and a bit easier to read
- [x] Create tool
  - disabled by default
- [x] Deepwiki mcp tool
  - disabled by default, would be cool to see if Nano "reward" hacks with this.
- [ ] Test on SWE-Gym, see how often Nano manages to produce a diff
  - With small models
  - With varying context


## May 12 - May 19
Entries got deleted, did A LOT more

### trl
- [x] Update to latest
- [x] Support DP in vllm_serve_async
- [x] Write test cases for vllm_serve_async
  - Post: They pass
- [x] Setup cli
  - Post: workaround, still use vllm's
- [x] Generalize handling of prompt, completion, tools in GenerationResult
- [x] Patch GRPOTrainer to handle vllm_mode="async_server" without requiring prompts
- [x] Bugfix - Kernel version
  - [2025-05-17 16:20:37,654][accelerate.utils.other][WARNING] - Detected kernel version 4.18.0, which is below the recommended minimum of 5.5.0; this can cause the process to hang. It is recommended to upgrade the kernel to the minimum version or higher.
  - Post: Nothing to do, this is the kernel version of Berzileus, could hang, could work
- [x] Bugfix - Jumbled generations
  - Weight Syncing does not work with quantization
  - Removed load_in_8bit
  - Should probably raise a warning...
- [x] Bugfix - prompt_ids device
- [x] Bugfix - error processing chat template
  - Post: Qwen3's jinja template is bugged
- [ ] Bugfix - robustly parse .generate, if something goes wrong we should not crash in the trainer
  - We should always have a prompt at least, but perhaps not generations
- [x] Change it from vllm_client_cls to rollout_func?
  - rollout_fn is not as clear that it should do the entire batch
  - but it removes the unneeded complexity of having to extend VLLMClient for no reason except inheriting the vLLM url
  - it has the same "signiture" as reward_func which is good
- [ ] Update GitHub issue, @ the maintainer
- [ ]  

### nano
- [x] Fix issue with logging and multiprocess (same timestamp)
- [x] Added failure handling for subprocess commands to show raw terminal output
- [x] Update the feedback from the `shell` tool isn't "verbose" enough.
- [ ] Add max tokens
  - after _tool_reply we call litellm.token_counter, if < context_window - TOKEN_THRESHOLD then append system warning to finish now, then do 1 more _chat with completion tokens configured correctly (-epsilon), then break 

### CodeRepairRL
- [x] Update config files
- [x] Update NanoAgent
- [x] Low hanging fruit debugging
- [x] NanoAgents fail
  - Isolate. Serve the model and initialize the client myself, see where it fails
  - Seems like GRPOTrainer is not happy with empty GenerationResults
  - Post: it was Qwen3 jinja template issue
- [x] Update slurm scripts
  - Will make medium / large match when I'm ready for those
- [x] Add FlashInfer
- [ ] Bugfix - tokenizer parallelism
  - huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks... To disable this warning, you can either: Avoid using `tokenizers` before the fork if possible. Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
- [x] Bugfix - No label_names provided for model class `PeftModelForCausalLM`
- [x] **Smol training run**
- [x] nano_rollout_func throughput testing
  - ThreadPool seems to be faster
- [ ] **8B training run**

## May 5 - May 12

### nano-"codex"
- [x] Rename, "lame to have openai"
- [x] Enable calling "apply_patch" many times
- [x] At the end of .run() always return the unified diff
- [x] Debug reasoning models
  - {enabled: false} doesnt seem to work
- [x] Add verbosity flag
- [x] CLI 
- [x] Clarify that "cat-ing" an entire file is not a good strategy?
- [x] "We expect the agent to fail in certain situations, but we will directly reinforce favorable behaviors with outcome supervision..."
- [x] Transfer and email
- [x] Ready for deployment

### CodeRepairRL tasks
- [x] Implement NanoAgent client
- [x] Stress test paralellizability
  - 2 Nanos in paralell are faster than 16 but not bottlenecking
- [x] Should I reintroduce unsloth? It is much faster...
  - No, atleast not all of it, its too much hassle
  - I could have FastLanguageModel from unsloth only, but that requires carefully disabling trainer monkey patch
- [x] Unified diff rewards
- [x] Get vllm_container.sif working
  - Needs to have bleeding edge vllm, flash-attn, etc.
  - Post: Maybe revert back to uv / pip and try to fix that way.
- [x] Run vllm serve and hook up Nano to it.
  - Should work, just being paranoid
- [x] Integrate vllm_container into train_container.sif
- [x] Run vllm-serve-async and observe the LLM endpoint working
  - Post: It was simpler to create a new entrypoint instead of working with TRL's argparse
    - "vllm-serve-async --model ..."
- [X] Hook up one Nano instance with the trl vllm-serve
- [x] Run vllm-serve-async and observe the weight sync endpoints working
  - "AsyncLLM engine has no argument `llm_engine`"
- [ ] Smol training run, 2 thin GPUs, Qwen3-0.6B Lora
  - Find problems with my trl changes
  - Probably something wrong in ._generate_and_score
- [ ] **Have a full training run going before the monday meeting**

### trl tasks
- [ ] Make system prompt + task specification into "prompt"
  - Double check where AutoTokenizer places the tools. I'd assume into the system prompt.



## April 28 - May 5

### Writing Tasks
- [ ] Write something in relevant research
- [x] Write down some research questions
- [x] Setup paper skeleton (PAPER.md) 
- [x] Update .md files
  - This includes a lot of the content of my final thesis
  - Useful as model context
- [x] Setup local latex
  - It's been really annoying to use Overleaf lately
  - Frequently exceed compilation time
  - Slow in general
  - Bad writing experience IMO
  - Could link my repo to Overleaf for easier sharing, or just push the latest pdf always

### Scaffold Tasks
- [x] SimpleAgent for debugging
- [x] Double check that apptainer codex works on berzileus (test with o4-mini)
- [x] Implement "nano-codex"
- [ ] Fix local SWE-Agent implementation
  - Option 1: Configure it to run inside an apptainer (though this means docker inside apptainer, which is not ideal)
  - Option 2: Debug the call stack to identify why it fails when running locally
- [ ] Resolve Codex issues
  - Currently not working for any team member
  - Only o4-mini seems viable; other models are essentially unusable
  - Primary issue: models fail to properly utilize tools, especially the apply_patch function
- [x] Is parallel apptainer execution feasible?
  - Determine if this creates a performance bottleneck
  - Check execution speed (acceptable if <2s)
  - Test nested apptainer functionality: Can we launch an apptainer from within another apptainer with different configurations?
  - Post:
    - Yes, shouldn't be a problem
    - Hopefully don't run into problems in practice

### TRL Tasks
- [x] Setup a developmental fix
- [ ] Bugcheck with SimpleAgent
- [ ] Bugcheck with NanoCodexAgent
<details><summary>If I want to merge</summary>


- [ ] Think about the VLLMClient abstraction I am proposing
  - I like the extra customizability
  - I don't like delegating such a complex class to the user when they only need to implement one method

- [ ] Deal with EOS masking in GRPOTrainer

- [ ] Ensure the input/output is correct
  - Should mirror vllm_client.generate() exactly

- [ ] Implement and support AsyncVLLMClient
  - The "correct" thing to do would be to replace all generating behavior with a Client
  - Let's just focus on adding this one thing with minimal flags and checks
  - "Explore then contract", get this working, implement tests, then refine the idea
</details>

### CodeRepairRL Tasks
- [ ] SWE-Bench-verified setup
- [ ] Aider-polyglot setup


## April 21 - April 28
Mostly spent this week prepping for a hard exam

### Tasks:
- [x] Verify Aider eligibility
  - Can work, needs a relatively extensive fork to completely remove context pruning and everything that violates my inference-training context parity requirements.
- [x] Verify SWE-Agent eligibility
  - A much more "mature" code base
  - However, it seems to have some internal issues with their local deployment, it is a recent feature I believe
- [x] Verify Codex eligibility
  - Probably, but need to wait a bit longer
  - Needs npm, can install it from an apptainer
  - It doesn't seem to work that well, their approach is to offload everything onto the model, e.g. repowide understanding by navigating in the terminal
  - Fails 100% of the time when it is not o4-mini


## April 14 - April 21

### Tasks:
- [ ] Start writing
  - Doesn't need to be a lot
  - I'm not too freaked out about this, since I have a LOT of written material spread out over these many documents over what I have been doing, what papers I've been reading, etc.
- [x] Scope out the project
  - I am setting very ambitious goals, especially since there is so little time left
  - But I want to train my Aider-Qwen2.5-Coder on both SWE-Gym and Defects4j (mine correct patches from repairbench)
  - (Because most of the program repair literature is on Python and Java)
  - Then I want to evaluate my model on SWE-Bench-Verified and on Gitbug-Java.
  - The latter hasn't been done I believe, nice to do since we progress and combine "in-house" work  
- [x] Update PROJECT.md
- [x] Read r2e_gym
  - https://www.monperrus.net/martin/fat-reading-notes, use this style of "Review"
  - https://arxiv.org/pdf/2504.07164
- [x] Read Multi-SWE-bench: A Multilingual Benchmark for Issue Resolving
- [x] Read Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation
  - https://zhuohaoyu.github.io/assets/files/orps_icml_preprint.pdf
- [x] Read SEAlign: Alignment Training for Software Engineering Agent
  - https://arxiv.org/pdf/2503.18455
- [x] Refactor AgentManager
  - Now we import the abstract from our trl fork and implement our own custom agent
- [x] Test MultiProcessAiderAgent
 - Works well, think the abstraction of data dicts in, updated data dicts out is nice
- [x] Update the issue to show this more mature use case
- [ ] Implement and support AsyncVLLMClient
  - The "correct" thing to do would be do replace all generating behavior with a Client
  - Let's just focus on adding this one thing with minimal flags and checks
  - "Explore then contract", get this working, implement tests, then refine the idea
- [x] Find out how to deactivate the "context" pruning in aider
  - Or reconstruct it, shouldn't be that hard
  - Perhaps just add my own list of messages, only append new messages (Aider seems to change previous messages), remove the last message if it looks weird
  - Post:
    - Aider "violates" my 1-to-1 requirement
    - It modifies the conversation history, removing context, presumably to help less capable models "follow" the conversation better
    - I don't think Qwen32B or even Qwen7B would struggle too much with having that extended context
    - Also makes inference less efficient, invalidating a lot of KV-Cache
- [x] Rename issue, send email
  - Mayebe ping willcobb or create an issue on the Verifiers side pointing to this
  - Post: Maybe ping maintainer / willccbb. Verifiers can 

### Berzelius contingent tasks:
- [ ] Fix container, problem with trl vllm-serve
  - When Berzelius is back up
  - Now I get "backend not found" error
  - I think its due to "Linux/image" version, not flash-attn as I thought before
  - Can run vLLM myself on the container
- [ ] Train Aider-Qwen-2.5-Coder-32B
  - Ensure that we are turning off bash commands for the agent (VIP)

## April 8 - April 14, 2025

#### Keywords
Multi-turn, end2end, reinforcement training of coding agents.

Training, not fine-tuning? (maybe a bit controversial)

We would certainly have superior domain specific performance within e.g. Aider but the goal is to improve "agentic" coding behavior in general. What I want to illustrate in my thesis is that this isn't a fine-tuning technique, but perhaps one of the key steps in the post-training stage of the next iteration of frontier models (circa Claude-3.7-sonnet).

"in-context training", "direct agent reinforcement", "agent-in-the-loop training"

I don't think trl will be mergable after these changes and do we like this name? Agent-In-The-Loop Reinforcement Trainer (AITLRT)

#### Tasks:
- [ ] Start writing
  - Doesn't need to be a lot,
  - Did some DeepResearch, but need to focus on this more.
- [x] Unslop README.md
  - Move some of it to docs/PROJECT.md
  - Perfect for AI context
  - Also just nice to have if someone wants a deeper understanding of everything
  - This way it doesn't clog up the README.md
  - Add a section at the bottom of the README.md detailing what exactly all the .md files are
  - Also just useful for later writin
- [x] Test APIProxy against a real endpoint
  - Launch a vllm server, make a call to the proxy which calls to vllm
  - Verify that chats are stored and the format of the requests
  - Post:
    - Figured out a nicer abstraction, APIProxy no longer a thing
- [x] Implement an OpenAI compatible endpoint
  - https://github.com/huggingface/trl/issues/3284
- [x] Test Aider against our vllm_serve_openai_endpoint.py
  - Is there and identifying header we can use for our purposes:
    - [ ] yes, [x] no
  - Or do we need to map requests/responses to the prompts which caused them:
- [x] Test MultiProcessAider against our endpoint
  - Observe N Aiders working concurrently
  - Find possible errors in OpenAI endpoint emulation
  - Ensure that vllm dynamic batching works properly with this approach
- [x] Fix NotImplementedError for vllm attn_cls
  - Probably some version thing
  - Try going to vanilla trl, and launch trl vllm-serve
  - If that doesn't work, roll back
  - Find which vllm/trl/vllm_flash_attn/triton/etc. versions work

## April 1 - April 7, 2025 (SLURM issues were a priority, but couldn't work on them bc. Berzileus was down)

#### Tasks:
- [x] Jobscripts vastly simplified
- [x] Make hf and uv cache point to my project directory instead of my home directory
- [x] Ran a sanity check test that two apptainer containers can talk to each other via HTTP
- [x] Training runs with grpo.use_vllm=false work
- [x] Paralellized Aider test
  - Under extras/
  - Multiprocess: clone a SWE-Bench-Lite repo, change directory, run aider, it maps the repo, suppress the terminal outputs ...
  - It is possible to access the entire conversation history in textual format if we fork the repo
  - Perhaps best to do this on the vLLM side. Make it "agent agnostic". Gather all the requests in a buffer, then in our training loop, when Aider exits, we call a new api_endpoint which gives us the entire history and resets the buffer.
- [x] Transfer ownership of my fork to ASSERT-KTH
- [x] Change vllm-serve
  - Make an OpenAI compatible endpoint
  - Gather chat histories in a buffer
  - New endpoint to get histories and resets the buffer
- [x] Change GRPOTrainer ".generate()" stage
  - Replace "vllm_client.generate" with paralellized Aider instances (echoes extras/paralellized_aider.py)
  - Modify API_ENDPOINT to our vllm server
  - Post:
    - A hacky, probably non-functional impl in place (monkeypatch middle-ware)
    - This is a bit tricky to get right
    - Would be way simpler to customize Aider, but if this is done correcly, it could be quite profound (as an example at least)
- [x] Create a "repo repair dataset"
  - We can just prefetch all the repos (and initialize Aider cache for speed perhaps)
  - Store the path to the tempfolder under "repo_folder" key
  - Create "get_repo_repair_dataset" in swe_bench.py (has no system prompt, that is delegated to the agent)
  - Post:
    - We can't prefetch the repos because of the repeating nature of GPRO
    - We need N copies of the repo so that .aider-cache doesn't get reused or have any race condition scenarios
    - This copying behaviour happens inside GRPOTrainer so we would need to implement it there.

## March 24 - March 30, 2025

#### Tasks
- [x] Remove unsloth for now, that library is a nightmare to work with
  - Simplifies the config by a lot
- [x] Reward callbacks only receive prompt and completion as args, rest is kwargs
- [x] Simplify diff
  - Just search/replace, refactor for possible future multi-file edits
- [x] Simplify rewards  
  - All scaled between 0-1, reward weighting done in the script
- [x] Forked trl
  - Ideas such as integrating Aider will require it
  - For now, the only change is better wandb logging
- [x] Support newest trl version
  - Something changed in the last few weeks in trl
  - Need to run vllm serve myself, makes sense
  - Simply need to add trl vllm-serve to the .sh scripts
  - Post:
    - We kind of need multi gpu for this to work well, I get cryptic NCCL errors when trying to force it to occupy only 1 gpu
    - For simple testing we can simply do uv run -m src.train_grpo grpo.use_vllm=false
    - But for larger scale training we run the .sh scripts
- [x] Deepspeed / accelerate support
  - Post:
    - Needs testing, should be pretty good
    - Opted for "ZeRO-Stage 3" deepspeed, where optimizer state/gradients/model parameters are all sharded between GPUs
    - This isn't really necessary for training LoRAs, but I'd like to move past those ultimately  
- [x] Stack "repair" on KTH DGX-100
- [x] Added .from_unified_diff to SearchReplace
- [x] Added git utils for datasets which require us to clone repositories
  - Tested a few different methods, the fastest was to create an empty repo, add the origin, then fetch only the relevant commit 

#### Ideas
We could simply:
1. host vLLM
2. run Aider Coder or similar on a repo, tasked with e.g. fixing a bug
3. it returns a diff which we score, and we weigh the entire conversation history with the rewards
I.e. directly reinforce performance when using aider. Requires slight refactoring of trl.GRPOTrainer, but not too bad.
This vastly simplifies performing multi file edits and stuff. Perhaps issues with batching



## March 17 - March 23, 2025
Was a bit sick during the weekened which is when I am normally the most productive

#### Tasks
- [x] Fix apptainer and test_job.sh
  - I know it works to cp ttc.sif to my project directory then do apptainer exec ttc.sif uv run ...
  - Maybe some discrepancy between that and what test_job.sh is doing
  - Or some complexity I'm not aware about in srun
- [x] Fix some refactor mismatches
- [x] Stack implementation dataset
  - It's not exactly the type of dataset we are looking for but its pretty good
- [x] Read LLM Post-Training: A Deep Dive into Reasoning Large Language Models: https://arxiv.org/pdf/2502.21321

#### Ideas
- What if we use AST to extract docstrings of python code on the Stack.
  - Sufficiently detailed docstrings can be mutated (via LLM) into a detailed description of the desired functionality of a function or class
  - We then remove said function/class but provide the rest of the code as context


## March 10 - March 16, 2025

#### Supervisor Updates
- Using test cases as rewards may require alot of SWE to run each test case on Kubernetes
  - Issues with scalability

#### Tasks
- [x] Fix diff.py, it is VERY sloppy
- [x] Fix reward.py, it sloppy
  - Get rid of get_diff entirely
  - Make correctness and quality rewards as separate as possible
  - Merge strict quality with partial quality, since the latter smoothly transitions from 0-1
  - Also add a method to diff.py to compare two diffs of the same type 
- [x] Side quest: Fast QwQ-32B inference on 1 of KTH's H100
- [x] Check out verifiers, seems like exactly what I need, could even contribute to the repo
  - paralell execution env where we can call dockerized tests?
- [x] Check out lighteval, would be cool to check e.g. humaneval before and after
  - looks good, much better than deepeval which I was using before


#### Thoughts
- Should we do have an agent scaffolding like SWE-Agent / OpenHands / Agentless
- Would be really cool to reinforce tool usage but that requires a lot of engineering
- Bug fix diff matching is probably almost equivalent to running the tests
  - At least for simple bugs which only require changing a couple lines
- Perhaps a more interesting task is implementing something new with well defined behaviour
  - Then oracle patch diff matching is a vastly different objective from passing the tests
  - Can we create a synthetic dataset for this?


## March 3 - March 9, 2025

#### Supervisor Updates
- Notes from meeting:
  - I need to scope out my project more, decide what task I am optimizing for exactly
  - Is that code execution RL? 
  - Will I be doing a ML paper? Data curation and cleaning paper / agentic coder paper?
  - Am I targeting SWE-Bench? RepairBench?

#### Tasks
- [x] Scope out my project
- [ ] Look at how SWE-Bench works exactly, should we train for that behavior end2end?
- [ ] Fix SLURM scripts (get a basic training run going)

#### Thoughts
- Should I keep on using GRPOTrainer?
- Should I fork verifiable envs or create my own (execution consideration)
