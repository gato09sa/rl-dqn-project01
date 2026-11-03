"""Shared experiment utilities: seeding, epsilon schedule, greedy evaluation,
logging (CSV + optional Weights & Biases), and parallel execution.

Every run writes to results/<tier>/<variant>/seed<k>/:
    train.csv     step, episode_return, episode_length            (one row per episode)
    eval.csv      step, eval_mean, eval_std[, extra columns...]    (one row per evaluation)
    summary.json  config, seed, final 100-episode evaluation, steps-to-threshold,
                  wall-clock time, hardware
The CSV files are the source of truth for the report figures; W&B is a live mirror.

W&B: runs go to entity "intelligent_agents" on the CoreWeave Forge server (the host is
stored by `wandb login --host https://api.forge.coreweave.com`). Disable with
use_wandb=False, or set WANDB_MODE=offline and upload later with `wandb sync wandb/`.

Parallel: run_parallel() starts one process per (variant, seed). The laptop CPUs
(i5-10210U) have 4 physical cores / 8 threads; 4 workers is the measured sweet spot.
Scripts that call run_parallel() MUST put that call under `if __name__ == "__main__":`
(Python 3.14 starts workers with forkserver and re-imports the main module).
"""
import csv
import json
import os
import platform
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import gymnasium as gym
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

WANDB_ENTITY = "intelligent_agents"
WANDB_PROJECT = "rl-dqn-project01"

EVAL_SEED_BASE = 10_000    # periodic evaluations always use env seeds 10000..10009
FINAL_SEED_BASE = 20_000   # final evaluation uses env seeds 20000..20099 (never seen before)


# ----------------------------------------------------------------- seeding
def set_seeds(seed):
    """Seed Python, NumPy and PyTorch. The env is seeded separately via env.reset(seed=...)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(env_id, seed):
    env = gym.make(env_id)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


# ------------------------------------------------------------- exploration
def linear_epsilon(step, eps_start, eps_end, decay_steps):
    """Linear decay from eps_start to eps_end over decay_steps env steps, then constant."""
    frac = min(1.0, step / decay_steps)
    return eps_start + frac * (eps_end - eps_start)


# -------------------------------------------------------------- evaluation
def evaluate(policy, env_id, n_episodes=10, seed_base=EVAL_SEED_BASE):
    """Greedy evaluation on a fresh env. `policy` is any function obs -> action, so the
    same code evaluates tabular agents (lambda o: agent.greedy(disc(o))) and DQN
    (agent.greedy). Returns the list of episode returns."""
    env = gym.make(env_id)
    returns = []
    for i in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + i)
        done, ret = False, 0.0
        while not done:
            obs, r, terminated, truncated, _ = env.step(policy(obs))
            ret += r
            done = terminated or truncated
        returns.append(ret)
    env.close()
    return returns


def reward_threshold(env_id):
    """Success threshold from the Gymnasium spec (CartPole 475, LunarLander 200, ...)."""
    return gym.spec(env_id).reward_threshold


# ----------------------------------------------------------------- logging
def run_dir(tier, variant, seed):
    return os.path.join(RESULTS, tier, variant, f"seed{seed}")


class RunLogger:
    """Writes train.csv / eval.csv / summary.json and mirrors everything to W&B.

    In W&B every metric uses `env_step` as its x-axis, and runs are grouped by
    "<tier>/<variant>" so the dashboard can average the seeds of one configuration.
    """

    def __init__(self, tier, variant, seed, config, use_wandb=True, eval_extra=()):
        self.dir = run_dir(tier, variant, seed)
        os.makedirs(self.dir, exist_ok=True)
        self.tier, self.variant, self.seed, self.config = tier, variant, seed, config
        self.t0 = time.time()
        self.eval_extra = list(eval_extra)
        self.threshold = config.get("threshold") or reward_threshold(config["env_id"])
        self.steps_to_threshold = None

        self._train_f = open(os.path.join(self.dir, "train.csv"), "w", newline="")
        self._eval_f = open(os.path.join(self.dir, "eval.csv"), "w", newline="")
        self._train = csv.writer(self._train_f)
        self._eval = csv.writer(self._eval_f)
        self._train.writerow(["step", "episode_return", "episode_length"])
        self._eval.writerow(["step", "eval_mean", "eval_std", "wall_time_s"] + self.eval_extra)

        self.wandb = None
        if use_wandb:
            import wandb
            self.wandb = wandb.init(
                entity=WANDB_ENTITY, project=WANDB_PROJECT,
                group=f"{tier}/{variant}", name=f"{tier}/{variant}/seed{seed}",
                job_type=tier, config={**config, "seed": seed, "tier": tier, "variant": variant},
                reinit="finish_previous",
            )
            self.wandb.define_metric("env_step")
            self.wandb.define_metric("*", step_metric="env_step")

    def log_episode(self, step, episode_return, episode_length):
        self._train.writerow([step, episode_return, episode_length])
        if self.wandb:
            self.wandb.log({"env_step": step, "train/episode_return": episode_return,
                            "train/episode_length": episode_length})

    def log_eval(self, step, returns, **extra):
        """returns: list of greedy episode returns. extra: values for the eval_extra
        columns (e.g. epsilon, avg_loss, avg_q); missing ones are left blank."""
        m, s = float(np.mean(returns)), float(np.std(returns))
        if self.steps_to_threshold is None and m >= self.threshold:
            self.steps_to_threshold = step
        self._eval.writerow([step, m, s, round(time.time() - self.t0, 2)]
                            + [extra.get(k, "") for k in self.eval_extra])
        self._eval_f.flush()
        self._train_f.flush()
        if self.wandb:
            self.wandb.log({"env_step": step, "eval/return_mean": m, "eval/return_std": s,
                            **{f"eval/{k}": v for k, v in extra.items() if v is not None}})
        return m

    def finish(self, final_returns, **extra_summary):
        """Close files, write summary.json, finish the W&B run. Returns the summary dict."""
        self._train_f.close()
        self._eval_f.close()
        summary = {
            "tier": self.tier, "variant": self.variant, "seed": self.seed,
            "config": self.config, "threshold": self.threshold,
            "final_eval_mean": float(np.mean(final_returns)),
            "final_eval_std": float(np.std(final_returns)),
            "final_eval_episodes": len(final_returns),
            "steps_to_threshold": self.steps_to_threshold,
            "wall_time_s": round(time.time() - self.t0, 2),
            "hardware": hardware_info(),
            **extra_summary,
        }
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if self.wandb:
            for k in ("final_eval_mean", "final_eval_std", "steps_to_threshold", "wall_time_s"):
                self.wandb.summary[k] = summary[k]
            self.wandb.finish()
        return summary


def hardware_info():
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo") as f:
            cpu = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    return {"cpu": cpu, "logical_cpus": os.cpu_count(), "os": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__,
            "gymnasium": gym.__version__, "numpy": np.__version__}


# ---------------------------------------------------------------- parallel
def run_parallel(fn, jobs, workers=4):
    """Run fn(**job) for every dict in `jobs`, `workers` processes at a time.
    fn must be a top-level (picklable) function; it should call
    agents.dqn.configure_torch_for_parallel(seed) first so each process uses one core.
    Returns the results in completion order."""
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, **job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            res = fut.result()
            print(f"[done] {job}", flush=True)
            results.append(res)
    return results
