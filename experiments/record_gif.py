"""Record a GIF of a trained agent playing one greedy episode.

Usage:
    python -m experiments.record_gif experiments/tier2/lunarlander.json --variant default --seed 0
Output: results/<tier>/gifs/<variant>_seed<k>.gif
"""
import argparse
import os

import gymnasium as gym
import imageio.v2 as imageio

from agents.dqn import DQNAgent
from agents.tabular import TabularAgent
from envs.discretizer import UniformDiscretizer
from experiments.common import FINAL_SEED_BASE, RESULTS, run_dir
from experiments.train import TABULAR, build_config, load_experiment


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", type=int, default=0, help="training seed of the run to load")
    p.add_argument("--episode-seed", type=int, default=FINAL_SEED_BASE)
    p.add_argument("--frame-skip", type=int, default=2, help="keep 1 of every k frames")
    p.add_argument("--quick", action="store_true", help="use the --quick smoke-test results")
    a = p.parse_args()

    exp = load_experiment(a.config)
    cfg = build_config(exp, a.variant)
    tier = exp["tier"] + ("_quick" if a.quick else "")
    src = run_dir(tier, a.variant, a.seed)
    env = gym.make(cfg["env_id"], render_mode="rgb_array")
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n

    if cfg["algo"] in TABULAR:
        disc = UniformDiscretizer(cfg["bins"], cfg["ranges"])
        agent = TabularAgent(disc.n_states, n_actions, seed=0)
        agent.load(os.path.join(src, "qtable.npz"))
        policy = lambda o: agent.greedy(disc(o))
    else:
        agent = DQNAgent(obs_dim, n_actions, cfg)
        agent.load(os.path.join(src, "model.pt"))
        policy = agent.greedy

    obs, _ = env.reset(seed=a.episode_seed)
    frames, ret, done, t = [env.render()], 0.0, False, 0
    while not done:
        obs, r, terminated, truncated, _ = env.step(policy(obs))
        ret += r
        done = terminated or truncated
        t += 1
        if t % a.frame_skip == 0 or done:
            frames.append(env.render())
    env.close()

    out = os.path.join(RESULTS, tier, "gifs", f"{a.variant}_seed{a.seed}.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fps = env.metadata.get("render_fps", 50) // a.frame_skip
    imageio.mimsave(out, frames, duration=1000 / fps, loop=0)
    print(f"saved {out}: {t} steps, return {ret:.1f}")


if __name__ == "__main__":
    main()
