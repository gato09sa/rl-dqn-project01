"""Run every (variant, seed) of an experiment config in parallel.

Usage:
    python -m experiments.sweep experiments/tier1/cartpole.json                 # everything
    python -m experiments.sweep experiments/tier2/lunarlander.json --variants default lr_1e-4
    python -m experiments.sweep experiments/tier1/cartpole.json --quick          # smoke test

--quick: tiny budget, seed 0 only, results in results/<tier>_quick/ (safe to delete).
Finished runs (summary.json present) are skipped, so an interrupted sweep can be resumed
by running the same command again. --force reruns them.
"""
import argparse
import os

from experiments.common import run_dir, run_parallel
from experiments.train import load_experiment, train

QUICK = {"total_steps": 3_000, "eval_every": 1_000, "eval_episodes": 2,
         "final_eval_episodes": 3, "learning_starts": 500, "eps_decay_steps": 2_000}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--variants", nargs="+", help="subset of variants (default: all)")
    p.add_argument("--seeds", nargs="+", type=int, help="subset of seeds (default: config)")
    p.add_argument("--workers", type=int, default=4,
                   help="parallel processes (4 = physical cores of an i5-10210U)")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--quick", action="store_true", help="smoke test with a tiny budget")
    p.add_argument("--force", action="store_true", help="rerun finished runs")
    a = p.parse_args()

    exp = load_experiment(a.config)
    variants = a.variants or list(exp["variants"])
    seeds = a.seeds or ([0] if a.quick else exp["seeds"])
    tier = exp["tier"] + ("_quick" if a.quick else "")
    unknown = set(variants) - set(exp["variants"])
    if unknown:
        raise SystemExit(f"unknown variants: {sorted(unknown)}")

    jobs = []
    for v in variants:
        for s in seeds:
            done = os.path.exists(os.path.join(run_dir(tier, v, s), "summary.json"))
            if a.force or not done:
                jobs.append({"config_path": a.config, "variant": v, "seed": s, "tier": tier,
                             "use_wandb": not a.no_wandb,
                             "overrides": QUICK if a.quick else None})
    print(f"{tier}: {len(jobs)} runs to do on {a.workers} workers")
    run_parallel(train, jobs, workers=a.workers)


if __name__ == "__main__":
    main()
