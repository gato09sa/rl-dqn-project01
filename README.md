# Project 01 — Reinforcement Learning and DQN on Gymnasium Environments

José Reinoso · Aldrin Chavez

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install swig setuptools wheel
pip install --no-build-isolation box2d-py   # LunarLander (needs swig on Python 3.14)
pip install -r requirements.txt
wandb login --host https://api.forge.coreweave.com   # optional: live metrics in W&B
```

## Structure

```
agents/
  dqn.py           # DQN: Q-network, replay buffer, target network, epsilon-greedy
                   #   flags: use_target, use_replay (ablation), double, dueling (Tier 3B)
  tabular.py       # Q-learning and SARSA on a Q-table
envs/
  discretizer.py   # uniform state discretization (CartPole / MountainCar ranges)
experiments/
  common.py        # seeding, epsilon schedule, greedy evaluation, CSV + W&B logging,
                   #   parallel runs
  plot.py          # learning curves (mean ± 1 std across seeds, x = env steps) + tables
  tier1/ tier2/ tier3/   # configs and run scripts per tier
results/
  tier1/ tier2/ tier3/   # logs, figures, trained weights, GIFs
report/            # paper
```

## Agent interface

Both agents expose the same methods, so the experiment scripts treat them alike:

| Method | Tabular | DQN |
|---|---|---|
| `act(state, epsilon)` | ε-greedy on the Q-table | ε-greedy on the Q-network |
| `greedy(state)` | ε = 0, used in evaluation | ε = 0, used in evaluation |
| `update(...)` | `update(s, a, r, s_next, terminated, a_next)` | `update()` on a replay mini-batch, returns `(loss, mean_q)` |
| `save(path)` / `load(path)` | `.npz` | `.pt` |

DQN also has `buffer.add(obs, a, r, next_obs, terminated)` and `maybe_sync_target(step)`.
Only `terminated` (not `truncated`) stops bootstrapping.

## Logging

Each run writes `results/<tier>/<variant>/seed<k>/` with `train.csv`, `eval.csv` and
`summary.json` (see `experiments/common.py`). The CSV files are the source of every
figure; W&B (entity `intelligent_agents`, project `rl-dqn-project01`) mirrors them live.
To log without network: `WANDB_MODE=offline`, then `wandb sync wandb/`.

## Parallel runs

`experiments.common.run_parallel` runs one process per (variant, seed). Our laptops
(i5-10210U) have 4 physical cores / 8 threads; use `workers=4`. Scripts that call it must
guard the call with `if __name__ == "__main__":`.

## Reproducing the figures

```bash
python -m experiments.plot --tier <tier> --variants <v1> <v2> ... --out results/<tier>/figures/<name>.png [--wandb]
```

One command per experiment will be listed here as the tiers are completed.
