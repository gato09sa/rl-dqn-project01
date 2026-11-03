"""Train one agent (tabular or DQN) for one seed, from an experiment config file.

Usage:
    python -m experiments.train experiments/tier1/cartpole.json --variant dqn --seed 0
    python -m experiments.train experiments/tier2/lunarlander.json --variant default --seed 0 --no-wandb

Config file layout (see experiments/tier*/ *.json):
    tier, env_id, seeds          where results go, which env, which seeds the sweep uses
    run                          budget and evaluation protocol, shared by every variant
    agent_defaults.tabular/.dqn  documented default hyperparameters per agent family
    variants.<name>              {"algo": "qlearning" | "sarsa" | "dqn", ...overrides}
    plots                        figures built by experiments.make_figures

Final config of a run = run + agent_defaults[family] + variants[name] (later wins).
"""
import argparse
import json

import numpy as np

from agents.dqn import DQNAgent, configure_torch_for_parallel
from agents.tabular import TabularAgent
from envs.discretizer import UniformDiscretizer
from experiments.common import (FINAL_SEED_BASE, RunLogger, evaluate, linear_epsilon,
                                make_env, run_dir, set_seeds)

TABULAR = ("qlearning", "sarsa")


def load_experiment(path):
    with open(path) as f:
        return json.load(f)


def build_config(exp, variant):
    v = exp["variants"][variant]
    family = "tabular" if v["algo"] in TABULAR else "dqn"
    cfg = {"env_id": exp["env_id"], **exp["run"], **exp["agent_defaults"][family], **v}
    cfg["variant"] = variant
    return cfg


def train(config_path, variant, seed, use_wandb=True, tier=None, overrides=None):
    """Train one (variant, seed). `tier` and `overrides` exist for quick tests
    (e.g. tier="tier1_quick", overrides={"total_steps": 5000})."""
    exp = load_experiment(config_path)
    cfg = {**build_config(exp, variant), **(overrides or {})}
    tier = tier or exp["tier"]

    configure_torch_for_parallel(seed)   # one CPU thread per process + torch seed
    set_seeds(seed)
    env = make_env(cfg["env_id"], seed)
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    tabular = cfg["algo"] in TABULAR

    if tabular:
        disc = UniformDiscretizer(cfg["bins"], cfg["ranges"])
        agent = TabularAgent(disc.n_states, n_actions, cfg["alpha"], cfg["gamma"], cfg["algo"], seed=seed)
        policy = lambda o: agent.greedy(disc(o))
        extra_cols = ["epsilon", "visited_fraction"]
    else:
        disc = None
        agent = DQNAgent(obs_dim, n_actions, cfg, seed=seed)
        policy = agent.greedy
        extra_cols = ["epsilon", "avg_loss", "avg_q"]

    log = RunLogger(tier, variant, seed, cfg, use_wandb=use_wandb, eval_extra=extra_cols)
    log.log_eval(0, evaluate(policy, cfg["env_id"], cfg["eval_episodes"]), epsilon=cfg["eps_start"])

    eps = cfg["eps_start"]
    obs, _ = env.reset(seed=seed)
    if tabular:
        s = disc(obs)
        a = agent.act(s, eps)        # SARSA needs the first action before the loop
    ep_ret, ep_len = 0.0, 0
    losses, qs = [], []

    for step in range(1, cfg["total_steps"] + 1):
        eps = linear_epsilon(step, cfg["eps_start"], cfg["eps_end"], cfg["eps_decay_steps"])

        if tabular:
            next_obs, r, terminated, truncated, _ = env.step(a)
            s_next = disc(next_obs)
            a_next = agent.act(s_next, eps)              # chosen now, used by SARSA's target
            agent.update(s, a, r, s_next, terminated, a_next)
            s, a = s_next, a_next
        else:
            a = agent.act(obs, eps)
            next_obs, r, terminated, truncated, _ = env.step(a)
            agent.buffer.add(obs, a, r, next_obs, terminated)
            if step > cfg["learning_starts"] and step % cfg["train_freq"] == 0:
                loss, q = agent.update()
                losses.append(loss)
                qs.append(q)
            agent.maybe_sync_target(step)
        obs = next_obs

        ep_ret += r
        ep_len += 1
        if terminated or truncated:
            log.log_episode(step, ep_ret, ep_len)
            ep_ret, ep_len = 0.0, 0
            obs, _ = env.reset()
            if tabular:
                s = disc(obs)
                a = agent.act(s, eps)

        if step % cfg["eval_every"] == 0:
            returns = evaluate(policy, cfg["env_id"], cfg["eval_episodes"])
            if tabular:
                log.log_eval(step, returns, epsilon=eps, visited_fraction=agent.visited_fraction())
            else:
                log.log_eval(step, returns, epsilon=eps,
                             avg_loss=float(np.mean(losses)) if losses else None,
                             avg_q=float(np.mean(qs)) if qs else None)
                losses, qs = [], []

    env.close()
    out = run_dir(tier, variant, seed)
    if tabular:
        agent.save(f"{out}/qtable.npz")
        extra = {"n_states": disc.n_states, "visited_fraction": agent.visited_fraction()}
    else:
        agent.save(f"{out}/model.pt")
        extra = {"n_updates": agent.n_updates}
    final = evaluate(policy, cfg["env_id"], cfg["final_eval_episodes"], FINAL_SEED_BASE)
    summary = log.finish(final, **extra)
    print(f"{tier}/{variant}/seed{seed}: final {summary['final_eval_mean']:.1f} "
          f"± {summary['final_eval_std']:.1f}, threshold@{summary['steps_to_threshold']}, "
          f"{summary['wall_time_s']:.0f}s", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-wandb", action="store_true")
    a = p.parse_args()
    train(a.config, a.variant, a.seed, use_wandb=not a.no_wandb)


if __name__ == "__main__":
    main()
