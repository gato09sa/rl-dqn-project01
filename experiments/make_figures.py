"""Build every figure listed under "plots" in an experiment config.

Usage:
    python -m experiments.make_figures experiments/tier1/cartpole.json [--wandb] [--quick]

Figures go to results/<tier>/figures/<plot name>.png, each with a .md/.csv summary table.
A variant written as "tier2/default" is read from another tier (Tier 3 reuses the
Tier 2 vanilla DQN runs instead of training them again).
"""
import argparse
import os

from experiments.common import RESULTS
from experiments.plot import plot, upload_to_wandb
from experiments.train import load_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--wandb", action="store_true", help="also upload figures and tables to W&B")
    p.add_argument("--quick", action="store_true", help="plot the --quick smoke-test results")
    a = p.parse_args()

    exp = load_experiment(a.config)
    tier = exp["tier"] + ("_quick" if a.quick else "")
    for spec in exp["plots"]:
        out = os.path.join(RESULTS, tier, "figures", spec["name"] + ".png")
        variants = spec["variants"]
        if a.quick:   # cross-tier references point at the quick results too
            variants = [v.replace("/", "_quick/", 1) if "/" in v else v for v in variants]
        _, table = plot(tier, variants, out, spec.get("title", ""), spec.get("labels"))
        if a.wandb:
            upload_to_wandb(tier, out, table)


if __name__ == "__main__":
    main()
