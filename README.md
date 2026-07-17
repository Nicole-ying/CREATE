# CREATE: Closed-Loop Reward Evolution via Training-Evaluation Feedback

CREATE is a feedback-driven iterative reward refinement framework for LLM-generated reinforcement learning reward functions. It organizes component-level training evidence into structured diagnosis, then applies severity-gated editing along a single reward lineage.

## Installation

```bash
pip install -r requirements.txt
```

Requires: `gymnasium[box2d]>=0.29.0`, `stable-baselines3>=2.0.0`, `torch`, `openai`, `pyyaml`, `numpy`. For Ant-v4 experiments, additionally: `gymnasium[mujoco]`.

Set your DeepSeek API key:
```bash
export DEEPSEEK_API_KEY=your_key_here
```

## Quick Start

Run a full CREATE experiment on LunarLander-v3 with 5 seeds, 10 iterations per seed:

```bash
python -m create.pipeline.run_multi_seed_experiment --config configs/lunarlander.yaml
```

Or a single seed:

```bash
python -m create.pipeline.run_iterative_experiment --config configs/lunarlander.yaml --prefix test_run --seed 0
```

## Project Structure

```
create/
├── pipeline/
│   ├── run_iterative_experiment.py      # Main orchestrator (seeds, iterations, stopping)
│   ├── run_multi_seed_experiment.py     # Multi-seed wrapper
│   ├── run_reflection_agent.py          # Single-agent reflection (iteration 2+)
│   ├── run_direct_generation_pipeline.py # Initial reward generation (iteration 1)
│   ├── run_01_environment_analyzer_md.py # Environment analysis via LLM
│   ├── run_02_build_expert_context.py    # Expert schema context builder
│   ├── run_03_direct_reward_generator.py # Reward code generation + validation
│   ├── run_06_update_reward_memory.py    # Reward memory table updater
│   ├── common.py                         # Shared I/O and config loading
│   └── reflection_tools.py              # Reflection agent tool implementations
├── training/
│   ├── train_sb3_wrapper.py             # PPO training + evaluation
│   └── reward_wrapper.py                # Reward override gym wrapper
├── llm/
│   └── deepseek_client.py               # OpenAI-compatible LLM API client
└── rag/
    └── direct_context_builder.py         # Fixed expert schema context builder

configs/           # Experiment configurations
prompts/           # LLM system prompts
envs/              # Anonymized environment specifications
baselines/         # Comparison baselines
analysis/          # Figure generation + experiment narrative
knowledge_base/    # Expert knowledge for reward design
runs/              # Experiment output directory
```

## Experiments

### Main Experiments

| Experiment | Config | Command |
|---|---|---|
| CREATE (LunarLander-v3) | `configs/lunarlander.yaml` | `python -m create.pipeline.run_multi_seed_experiment --config configs/lunarlander.yaml` |
| CREATE (BipedalWalker-v3) | `configs/bipedalwalker.yaml` | `python -m create.pipeline.run_multi_seed_experiment --config configs/bipedalwalker.yaml` |
| CREATE (Ant-v4) | `configs/ant.yaml` | `python -m create.pipeline.run_iterative_experiment --config configs/ant.yaml --prefix ant_run --seed 0` |

### Ablations

| Ablation | Config | What It Tests |
|---|---|---|
| Coarse Feedback | `configs/env001_ablation_eureka_feedback_v4.yaml` | Structured evidence replaced with scalar summary |
| Unconstrained Refinement | `configs/env001_ablation_unconstrained_v4.yaml` | Editing constraints removed |

### Budget-Matched Independent Generation

```bash
python baselines/run_independent_baseline.py --seeds 5 --samples 10
```

### Held-Out Evaluation

```bash
python baselines/run_held_out_eval.py --experiments CREATE CoarseFeedback Unconstrained IndependentGen
python baselines/run_held_out_eval_bipedal.py
```

## Key Results

| Experiment | Dev Score | Held-out | Success |
|---|---|---|---|
| CREATE (LunarLander) | 228.98 ± 16.54 | 231.63 ± 12.23 | 5/5 |
| Coarse Feedback | 134.97 ± 148.43 | 126.71 ± 143.85 | 2/5 |
| Unconstrained Refinement | 114.21 ± 46.31 | 102.87 ± 40.77 | 0/5 |
| Independent Gen (budget-matched) | −0.74 ± 114.40 | 1.86 ± 114.98 | 0/5 |
| CREATE (BipedalWalker) | 311.54 ± 5.17 | 310.82 ± 5.03 | 5/5 |

## Methodology

CREATE operates in a closed loop (Algorithm 1 in the paper):

1. **Initialize**: Generate initial reward program from task interface
2. **Train**: PPO policy optimization using generated reward only
3. **Evaluate**: Development evaluation on native environment reward (20 episodes, fixed seeds)
4. **Diagnose**: Assemble component-level evidence (activation rates, magnitude shares, termination patterns) and diagnose primary failure
5. **Edit**: Apply severity-gated edit --- parameter tuning, component refactoring, or structural redesign --- targeting one component per iteration
6. **Preserve**: Best Archive stores the historically best reward-policy pair

The active reward lineage can explore edits that temporarily reduce performance, since the archive protects the best known solution.

## Citation

If you use CREATE in your research, please cite:
```
[TBD - paper under review]
```

## License

This project is released for research purposes. See LICENSE for details.
