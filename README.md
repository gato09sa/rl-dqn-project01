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
The Forge host (`https://api.forge.coreweave.com`) is set in `experiments/common.py`, so
each member only needs `wandb login --host https://api.forge.coreweave.com` once.
To log without network: `WANDB_MODE=offline`, then `wandb sync wandb/`.
To send test runs elsewhere: `WANDB_PROJECT=rl-dqn-smoke-test python -m experiments.sweep ... --quick`.

## Parallel runs

`experiments.common.run_parallel` runs one process per (variant, seed). Our laptops
(i5-10210U) have 4 physical cores / 8 threads; use `workers=4`. Scripts that call it must
guard the call with `if __name__ == "__main__":`.

## Reproducing the experiments

Each tier is one config file. One command trains every variant and seed (4 runs in
parallel, already-finished runs are skipped), one builds every figure and table, one
records the GIF. Add `--no-wandb` to the sweep to log only to CSV.

| Tier | Train | Figures (`results/<tier>/figures/`) | Approx. time (4 workers, i5-10210U) |
|---|---|---|---|
| 1 — CartPole-v1, tabular vs DQN, 5 seeds | `python -m experiments.sweep experiments/tier1/cartpole.json` | `python -m experiments.make_figures experiments/tier1/cartpole.json` | ~30 min |
| 2 — LunarLander-v3, DQN study + ablation, 3 seeds | `python -m experiments.sweep experiments/tier2/lunarlander.json` | `python -m experiments.make_figures experiments/tier2/lunarlander.json` | ~2–2.5 h |
| 3B — Double / Dueling DQN, 3 seeds (needs Tier 2 `default`) | `python -m experiments.sweep experiments/tier3/lunarlander_variants.json` | `python -m experiments.make_figures experiments/tier3/lunarlander_variants.json` | ~1 h |

```bash
# smoke test of a whole tier in ~1 min (results in results/<tier>_quick/, safe to delete)
python -m experiments.sweep experiments/tier2/lunarlander.json --quick --no-wandb

# only some variants / seeds
python -m experiments.sweep experiments/tier2/lunarlander.json --variants default no_target --seeds 0 1

# upload the figures and tables to W&B too
python -m experiments.make_figures experiments/tier2/lunarlander.json --wandb

# GIF of a trained agent -> results/<tier>/gifs/
python -m experiments.record_gif experiments/tier1/cartpole.json --variant dqn --seed 0
python -m experiments.record_gif experiments/tier2/lunarlander.json --variant default --seed 0
```

Every variant, its hyperparameters and the default values are documented inside the
config files (`_doc` fields). The final config of each run is also saved in its
`summary.json`.
