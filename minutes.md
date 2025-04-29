Minutes of December 9th
Bjarni, André

News:
= Talked about possible (yet unlikely) methods of automatically labelling features.
- Talked about manually labelling features being difficult and time consuming, most likely use data to identify whether a feature we are looking for exists (i.e. not charting the entire feature space).
- Talked about code vulnerabilities being an interesting avenue.

-----------------------------------------------------------------------

Minutes of December 19th
Bjarni, André

News:
- Bjarni has started to play/implement/train SAEs (cross-coders) for the QWEN models.
    - https://github.com/BjarniHaukur/minimal-ml/blob/master/crosscoders.ipynb
- Bjarni mentioned that SAEs aren't necessarily the best way of steering model behaviour. More so a a way of "proving" that the features we find real. Perhapas deemphasize steering in the title and use steering as an evaluation. 

Surveys/Lists:
- https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite


Papers: ([] is an arbitrary measure of how interesting / relevant I found the reading material)
    - Transformers:
        - attention paper (bengio) [**---]
        - Attention is all you need [**---]
    - Superposition:
        - https://transformer-circuits.pub/2022/toy_model/index.html  [***--]
    - SAEs:
        - https://transformer-circuits.pub/2023/monosemantic-features/index.html [*****]
        - (Sparse Autoencoders Find Highly Interpretable Features in Language Models) https://arxiv.org/pdf/2309.08600 [*****]
        - (Scaling and evaluating sparse autoencoders) https://arxiv.org/pdf/2406.04093 [****-]
        - (Improving Dictionary Learning with Gated Sparse Autoencoders) https://arxiv.org/pdf/2404.16014 [****-]
        - (Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2) https://arxiv.org/pdf/2408.05147 [***--]
        - (Steering Language Model Refusal with Sparse Autoencoders) https://arxiv.org/pdf/2411.11296 [****-]
        - (Automatically Interpreting Millions of Features in Large Language Models) https://arxiv.org/pdf/2410.13928 [***--]

    - Other mech interp:
        - (Thinking Like Transformers) https://arxiv.org/pdf/2106.06981 [xxxxx]
        - (Learning Transformer Programs) https://arxiv.org/pdf/2306.01128 [xxxxx]
        - (Tracr: Compiled Transformers as a Laboratory for Interpretability) https://arxiv.org/pdf/2301.05062 [xxxxx]
        - Look for activation patching

Datasets:
    - Vulnerabilities:
        - https://github.com/wagner-group/diversevul
        - https://github.com/secureIT-project/CVEfixes
        - https://huggingface.co/datasets/aisecurity/known_exploited_vulnerabilities
        - https://huggingface.co/datasets/Coriolan/smart-contract-vulnerabilities
        - https://huggingface.co/datasets/DanCip/filtered-vulnerabilities
        - https://huggingface.co/datasets/leoreycar/sc_vulnerabilities
    - Code:
        - https://huggingface.co/datasets/bigcode/the-stack-v2
        - https://huggingface.co/datasets/bigcode/commitpackft

Ideas:
- Generate synthetic data to train SAEs (insert vulnerabilities, flaws, translate, etc)

Actions:
- Search for surveys on SAEs
- Search for datasets (code, vulnerabilities, multiple languages)
- Hone in on the "scope" of this thesis as we read these papers

------------------------------------------------------------------

Minutes of January 9th
Bjarni, André

Ideas:
- To tackle the problem of activation reconstruction after clamping features:
    - Pretrain an LLM with SAEs built-in from scratch
    - Fine-tune the LLM after training an SAEs with high-quality data, to "merge" the layers
    - Clamp the SAE during the SAE training to improve activation reconstruction after clamping
- For MoE models, can we find semantic features unique to each expert?
    - Maybe for Concept Language Models (i.e., not token language model) it gets closer to this idea

Questions:
- When clamping, which layers do we clamp?

Datasets:
- https://huggingface.co/datasets/bigcode/the-stack-v2-dedup
- See datasets.md for vulnerabilities

Actions:
- Add Bjarni to OpenAI (ask Martin, email: bhbj@kth.se), Add Bjarni to Google (ask Martin, email: bjarnihaukur11@gmail.com)
- Added to Anthropic, OpenRouter

https://github.com/ASSERT-KTH/CodeSAE

------------------------------------------------------------------

Minutes of January 20th

Ideas:
- Add instruct samples (e.g. repair tasks, translation tasks, etc)
- Add execution (e.g. scratchpad traces) samples

Done:
- PrimeVuln data gathered and processed

Actions:
- Train an SAE on Qwen-1.5B with PrimeVuln
- Setup and run AutoInterpret pipeline
- Find interesting features (e.g. specific code structures, etc)
- Setup DeepEval and run on reconstructed activations (no clamping, nX sampling)

Future:
- Integrate TheStack + tasks-specific datasets
- Start experimenting with integrating the SAE into the model
    - Fine-tune with SAE inplace
    - Pre-train a model with SAE in place
        - Baseline would be speedrun on NanoGPT
     
------------------------------------------------------------------

Minutes of January 27th (Pseudo, for book keeping)

Problems:
- SAE training requires quite alot of data

Ideas:
- Maybe we can pre train on TheStack then post train on PrimeVul if we do not find vulnerability features
    - Sounds kinda weird since by definition none of the features found are represented but perhaps features which do not appear as often in PrimeVul can be overwritten
- Training the SAE + LLM together:
    - a.) We optimize for SAE reconstruction task and LLM next token prediction jointly
        - If we do not optimize the reconstruction task, the SAE might just become a poorly performing MLP layer instead of being an autoencoder
    - b.) We freeze the SAE and train only the layers of the transformer which come after the SAE intervention
    - ...

Done:
- SAE training
- TheStack and Primevul datasets
- A few SAE architectures
- ...

WIP:
- AutoInterp, xml formatting on tokens where feature activate (<active:1.3>some_token</active>), inplace
- Benchmarking, multiple choice benchmarks supported, harder to make small LLMs adhere to syntax especially when we ask them to first deliberate on their answers. (Maybe deliberate, then do another round asking it to output its answer directly with a SchemaValidator)

Actions:
- Longer training run
- Log circuitsvis HTML straigt to wandb to give us a feel for how the model develops with time
    -  e.g. feature activations on a code snippet or top-k tokens corresponding to features
- Log recovered loss?
- Log MMLU every 2-300 steps?

Future:
- Analyze the features
    - AutoInterp
    - Probing the features with PrimeVul test set (e.g. analyzing which features fire for a particular CWE class both before and after the vulnerability has been fixed and compare)
- If this work is meaningful, then we can register our model directly on HuggingFace quite easily
    - users would need to install a small custom library we can put on pypi
    - AutoConfig.register("new-model", NewModelConfig); AutoModel.register(NewModelConfig, NewModel)
        - where the NewModels is an extension we write of the Auto classes
- We could also publish to Neuronpedia, need to double check the requirements of that on the programming side but the last time I checked it did not look too complicated 



-------------------------------------

Minutes of Feb 4th

One/two-week sprint, to surf the wave of RL:
- Bjarni will take a week to explore RL (GRPO) on coding tasks.
    1) PrimeVul, signal comes from identifying the correct CVE
    2) Use Defects4J as training data (compiler + tests signal) -> GitBug-Java as eval
        - https://github.com/ASSERT-KTH/repairbench-framework/
- At the end, we decide whether to continue this direction or go back to SAEs (or some flavour of both, e.g. SAE on the reasoning traces; SAEs trained while doing RL)

- Low-Rank Adapting Models for Sparse Autoencoders (https://arxiv.org/pdf/2501.19406)
    - They train a LoRA on top of a frozen LLM and trained SAE, similar to Bjarni's idea of post-training with SAE reconstructed activations

SAE project:
- Bjarni is setting up Berzelius account
- Will likely use SAELens

----------------------------------

Minutes of Feb 18th

- Thesis is now on reasoning models

Plan:
    - Step 1: Explore GRPO on vulnerability detection (done)
        - trained instruct models on vulnerability detection
        - managed to improve the results, but lack of sparsity in the reward might be a problem
    - Step 2: Use pre-trained reasoning model and see if we can improve vulnerability classification performance via rl
        - Utilize in-context learning, put the CWE label definitions in the context window
            - Perhaps group them together somehow
        - Or just filter out specific CWEs (top 10-20)
    - Step 3:
        - do step 2 on a larger model on Berzelius
    - Step X:
        - Generate DeepSeek-R1 reasoning traces for Defects4J
        - Setup verification service
            - cached results: https://github.com/ASSERT-KTH/repairbench-cache
        - Eval on GitBug-Java, SWE-Bench, SWE-Bench-Java
    - Step X+1: train a reasoning agent (i.e, add tool usage)
        - https://github.com/SWE-Gym/SWE-Gym
        - https://github.com/r2e-project/r2e2

-----------------------------------------------------------

Minutes of Feb 26th

- SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution (https://arxiv.org/pdf/2502.18449)
- Claude 3.7 is trained on real-world SWE tasks, going away from math/code competitions

Example jobscripts:
- https://github.com/ASSERT-KTH/RASPing/blob/paper-revamps/experiments/loss_functions/loss_functions_jobscript
- https://github.com/ASSERT-KTH/RASPing/blob/paper-revamps/experiments/train_mutations/launch_train_mutations.py

Plan:
    - Step 1: Setup Apptainer
    - Step 2: Replicate SWE-RL
        - Can still use PrimeVul
        - Create diff from vuln to fix
        - SequenceMatch for continuous reward
        - 
    - Step 3: Start on Related Research

-----------------------------------------------------------

Minutes of April 4th

News:
- Basic training (GRPO on non-agentic generations) is working

Plan:
- Finish vLLM integration (OpenAI SDK compatible endpoint, telemetry)
- Finish agentic training scripts (vLLM server + TRL trainer in same job)