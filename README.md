<!-- Introduces the META_HRL research codebase and its reproducible workflows. -->
# META_HRL

META_HRL is a research codebase for resource allocation in mixed near-field and
far-field massive MIMO systems using rate-splitting multiple access (RSMA),
hierarchical reinforcement learning (HRL), and context-conditioned Meta-PPO.

The project models a 128-antenna ULA downlink serving six single-antenna users.
It separates slow grouping decisions from fast power/rate allocation decisions,
then studies whether task context helps policies generalize to unseen channel
distributions.

## What is implemented

- Mixed near-/far-field Rician channels with ULA geometry and validation tools.
- One-layer RSMA and group-common-stream RSMA signal, precoding, rate, and
  feasibility-constraint models.
- Physical baselines, candidate partitions, and heuristic grouping.
- A one-step Gymnasium environment for Flat PPO and a multi-step hierarchical
  environment for staged HRL.
- PPO Worker/Manager training, context-conditioned Meta-PPO training, evaluation,
  paired benchmarks, significance tests, and sensitivity analysis.
- Configuration files, unit tests, checkpoints, logs, and analysis artifacts.

## Research question

Near-field channels depend strongly on both user distance and angle, whereas
far-field channels are primarily angle dependent. Consequently, a fixed user
grouping or resource-allocation policy can be fragile when geometry, Rician
factor, noise, path loss, or QoS requirements change.

This project investigates the following decision hierarchy:

```text
Slow timescale: Manager
  observation -> choose one of 203 canonical user partitions

Fast timescale: Worker
  channel + geometry + manager partition (+ task context)
  -> private power + group-common power + common-rate allocation
```

The reward combines sum rate, Jain fairness, QoS shortfall, and (for HRL)
group-switching cost.

## System at a glance

| Item | Default setting |
| --- | --- |
| Antennas / users | 128-element ULA / 6 users |
| Carrier frequency | 28 GHz |
| Channel | Mixed near-/far-field Rician, ideal CSI |
| User region | Angles -60 to 60 degrees; near 10-80 m, far 100-250 m |
| Base RSMA action | 13 logits: 6 private powers, 1 common power, 6 common rates |
| Flat observation | 1,572 float32 CSI and geometry features |
| HRL Worker action | 18 logits: private power, group-common power, group-common rates |
| Manager action | 203 canonical user partitions |

The default parameters are in `configs/channel/rician_near_far.yaml`,
`configs/rsma/default.yaml`, and `configs/environment/default.yaml`.

## Installation

Python 3.10 or later is required. The runtime dependencies are Gymnasium,
NumPy, and Stable-Baselines3.

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

## Quick start

### One-step RSMA environment

`OneStepRSMAEnv` samples one static channel block per episode. It returns a
1,572-dimensional observation; an action is converted from logits into valid
power and common-rate allocations via softmax projections.

```python
import numpy as np
from meta_hrl.envs import OneStepRSMAEnv

env = OneStepRSMAEnv()
observation, info = env.reset(seed=42)
next_observation, reward, terminated, truncated, info = env.step(
    np.zeros(13, dtype=np.float32)
)
```

An episode always terminates after this one step. `info` includes allocated
resources, SINR, common/private/user rates, fairness, QoS gap, and geometry.

### Flat PPO baseline

```python
from meta_hrl.training import (
    FlatPPOConfig,
    evaluate_flat_policy,
    evaluate_physical_baselines,
    train_flat_ppo,
)

result = train_flat_ppo(FlatPPOConfig(total_timesteps=100_000))
ppo_metrics = evaluate_flat_policy(result.model, episodes=100, seed=10_000)
baseline_metrics = evaluate_physical_baselines(episodes=100, seed=10_000)
```

This is a contextual-bandit baseline: each action allocates resources for an
independently sampled channel scenario.

## Hierarchical RL

`HierarchicalRSMAEnv` is the multi-step group-RSMA environment.

1. Call `reset()`.
2. The Manager selects a partition with `set_manager_action(partition_index)`.
3. The Worker repeatedly calls `step(worker_action)`.
4. After `high_level_interval` steps, the Manager can select another partition.

Each non-singleton group receives a group-common stream. Stream slots are tied
to the smallest user index in the group, so Worker action size remains fixed
even when the selected partition changes. `WorkerTrainingEnv` trains a PPO
Worker under fixed or heuristic grouping; `ManagerTrainingEnv` freezes that
Worker and trains a discrete PPO Manager from interval-level returns.

## Meta-PPO and task context

Meta tasks vary slow channel/environment properties, including near/far mix,
Rician K factor, path-loss exponent, noise, and QoS. Training, validation, and
OOD tasks are reproducibly separated by task ID.

`TaskContextBuffer` isolates transition histories per task. The baseline
`MetaContextEncoder` summarizes recent `(state, action, reward, next_state)`
transitions as a seven-dimensional deterministic context. A learned feature
branch can encode the base observation and this context with separate MLPs
before PPO actor/critic inference.

For evaluation, each requested support-episode budget rebuilds its context
buffer independently, which prevents information leakage between adaptation
curve points.

## Command-line experiments

The installed `meta-hrl` command exposes reproducible Meta-PPO workflows.

```bash
# Train one context-conditioned Meta-PPO Worker.
meta-hrl train-meta --seed 42 --timesteps 100000 --output-directory outputs/meta_train

# Paired learned-context versus raw-context benchmark.
meta-hrl benchmark --seeds 11,29,47 --timesteps 100000 --output-directory outputs/benchmark

# Export CSV, SVG adaptation curves, and paired differences.
meta-hrl analyze-benchmark outputs/benchmark/meta_ppo_benchmark.json

# Paired bootstrap confidence intervals and sign-flip tests.
meta-hrl significance outputs/benchmark/meta_ppo_benchmark.json

# Context capacity, feature width, and OOD-shift sensitivity grid.
meta-hrl sensitivity --context-capacities 16,64,128 --context-feature-dims 16,32,64 --ood-shift-scales 0.5,1.0,1.5
meta-hrl analyze-sensitivity outputs/sensitivity/meta_ppo_sensitivity.json
```

Use `meta-hrl --help` or `meta-hrl <command> --help` to inspect all arguments.
Comma-separated lists are accepted for `--seeds`, `--task-ids`, and sweep
parameters.

## Current experimental evidence

The checked-in eight-seed, 100k-timestep paired benchmark finds that the
learned-context variant underperforms the raw deterministic-context ablation on
both validation and OOD tasks. At zero support episodes, learned minus raw OOD
reward is -7.34 and sum rate is -0.405 (both p = 0.0078125); the result remains
negative through ten support episodes. Increasing support from 0 to 10 episodes
does not show a clear few-shot adaptation gain in this setup.

The three-factor sensitivity experiment identifies OOD-shift strength as the
dominant factor: terminal reward is about 0.13 for shift 0.5, versus -66.83 for
1.0 and -121.70 for 1.5. Among the examined settings, context capacity 16 and
feature dimension 32 are the strongest marginal candidates. These results make
the raw deterministic context the current reference approach; the learned branch
is retained as a documented ablation rather than claimed as an improvement.

Artifacts are available under `outputs/benchmark_8seeds/` and
`outputs/sensitivity/`.

## Repository layout

```text
configs/             Channel, RSMA, environment, training, experiment defaults
docs/                Research task framework in Markdown and DOCX
outputs/             Checkpoints, logs, benchmark results, CSV/SVG analyses
scripts/             Documentation-generation helper
src/meta_hrl/
  channel/           Geometry, near/far-field steering, Rician channel, checks
  rsma/              Signal model, precoding, rates, constraints, baselines
  grouping/          Candidate partitions, heuristics, grouping metrics
  envs/              Gymnasium, hierarchical, task, and context environments
  agents/            Policies, replay/context buffers, learned context modules
  training/          Train, evaluate, benchmark, sensitivity, analysis workflows
  utils/             Seeding, checkpointing, logging, normalization
tests/               Unit and integration tests
```

## Documentation

- Task framework: `docs/meta_hrl_task_framework.md`
- Word version: `docs/meta_hrl_task_framework.docx`
- Regenerate the Word document: `python scripts/generate_task_framework_docx.py`

The project is organized around a strict implementation order: validate the
channel model first, then RSMA calculations, then environment behavior, and
only then RL training.
