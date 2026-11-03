# Project 01 — Reinforcement Learning and DQN on Gymnasium Environments

José Reinoso · Aldrin Chavez

## Structure

```
agents/
  dqn.py           # shared: DQN agent (Q-network, replay buffer, target network)
  tabular.py       # Tier 1: Q-learning / SARSA
envs/
  discretizer.py   # Tier 1: state discretization
experiments/
  common.py        # shared: seeding, greedy evaluation, CSV logging
  plot.py          # shared: learning curves (mean ± band, x-axis = env steps)
  tier1/           # Tier 1 configs and run scripts
  tier2/           # Tier 2 configs and run scripts
  tier3/           # Tier 3 configs and run scripts
results/
  tier1/ tier2/ tier3/   # logs, figures, trained weights, GIFs
report/            # paper
requirements.txt
```
