"""Learning curves and summary tables from results/<tier>/<variant>/seed*/ logs.

Usage (one command per figure):
    python -m experiments.plot --tier tier1 --variants tab_bins6 tab_bins10 dqn \
        --out results/tier1/figures/tabular_vs_dqn.png --title "CartPole-v1" [--wandb]

Each figure has two panels sharing the x-axis (environment steps):
    left : greedy evaluation return (eval.csv)
    right: training return, mean of the last 10 episodes (train.csv)
Lines are the mean over seeds, the shaded band is +-1 standard deviation across seeds,
and the dashed line is the environment's success threshold.
Next to the figure it writes <out>.md / <out>.csv with the summary table.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.common import RESULTS, WANDB_ENTITY, WANDB_PROJECT  # noqa: E402

# Fixed categorical order (colorblind-checked palette). Color follows the variant's
# position in --variants, so keep the same order across figures for the same variants.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(tier, variant):
    """List of dicts {seed, eval: DataFrame, train: DataFrame, summary: dict}.
    A variant written as "<other_tier>/<variant>" is read from that tier."""
    if "/" in variant:
        tier, variant = variant.split("/", 1)
    runs = []
    for d in sorted(glob.glob(os.path.join(RESULTS, tier, variant, "seed*"))):
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue   # unfinished run
        with open(os.path.join(d, "summary.json")) as f:
            summary = json.load(f)
        runs.append({"seed": summary["seed"], "summary": summary,
                     "eval": pd.read_csv(os.path.join(d, "eval.csv")),
                     "train": pd.read_csv(os.path.join(d, "train.csv"))})
    if not runs:
        raise FileNotFoundError(f"no finished runs in {os.path.join(RESULTS, tier, variant)}")
    return runs


def eval_curves(runs):
    """Evaluation points share the same steps across seeds -> stack directly."""
    n = min(len(r["eval"]) for r in runs)
    steps = runs[0]["eval"]["step"].to_numpy()[:n]
    ys = np.stack([r["eval"]["eval_mean"].to_numpy()[:n] for r in runs])
    return steps, ys


def train_curves(runs, n_points=200, window=10):
    """Episodes end at different steps in each seed: smooth each seed with a rolling
    mean over `window` episodes, then interpolate onto a common step grid."""
    last = min(r["train"]["step"].iloc[-1] for r in runs)
    grid = np.linspace(0, last, n_points)
    ys = []
    for r in runs:
        t = r["train"]
        smooth = t["episode_return"].rolling(window, min_periods=1).mean().to_numpy()
        ys.append(np.interp(grid, t["step"].to_numpy(), smooth))
    return grid, np.stack(ys)


def summary_table(tier, variants):
    rows = []
    for v in variants:
        runs = load_runs(tier, v)
        s = [r["summary"] for r in runs]
        final = np.array([x["final_eval_mean"] for x in s])
        reached = [x["steps_to_threshold"] for x in s if x["steps_to_threshold"] is not None]
        rows.append({
            "variant": v, "seeds": len(s),
            "final_eval_mean": final.mean(), "final_eval_std_across_seeds": final.std(),
            "reached_threshold": f"{len(reached)}/{len(s)}",
            "median_steps_to_threshold": int(np.median(reached)) if reached else "not reached",
            "wall_time_s_mean": np.mean([x["wall_time_s"] for x in s]),
        })
    return pd.DataFrame(rows)


def plot(tier, variants, out, title="", labels=None):
    labels = labels or variants
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    threshold = None
    for i, (v, lab) in enumerate(zip(variants, labels)):
        runs = load_runs(tier, v)
        threshold = runs[0]["summary"]["threshold"]
        color = COLORS[i % len(COLORS)]
        for ax, (x, ys) in zip(axes, (eval_curves(runs), train_curves(runs))):
            m, s = ys.mean(axis=0), ys.std(axis=0)
            ax.plot(x, m, color=color, lw=1.6, label=f"{lab} (n={len(runs)})")
            ax.fill_between(x, m - s, m + s, color=color, alpha=0.18, lw=0)

    for ax, sub in zip(axes, ("Evaluation return (greedy, 10 episodes)",
                              "Training return (last 10 episodes)")):
        if threshold is not None:
            ax.axhline(threshold, color=MUTED, lw=1, ls="--")
        ax.set_title(sub, fontsize=10, color=INK)
        ax.set_xlabel("Environment steps", color=MUTED)
        ax.grid(True, color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    axes[0].set_ylabel("Return", color=MUTED)
    handles, lbls = axes[0].get_legend_handles_labels()
    if threshold is not None:
        handles.append(plt.Line2D([], [], color=MUTED, lw=1, ls="--"))
        lbls.append(f"threshold ({threshold:g})")
    fig.legend(handles, lbls, loc="lower center", ncol=min(len(lbls), 5), frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{title}  (mean ± 1 std across seeds)" if title else "mean ± 1 std across seeds",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    table = summary_table(tier, variants)
    base = os.path.splitext(out)[0]
    table.to_csv(base + ".csv", index=False)
    with open(base + ".md", "w") as f:
        f.write(table.to_markdown(index=False, floatfmt=".1f"))
    print(f"saved {out}\n{table.to_string(index=False)}")
    return out, table


def upload_to_wandb(tier, out, table):
    import wandb
    name = os.path.splitext(os.path.basename(out))[0]
    run = wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, job_type="figures",
                     group=f"{tier}/figures", name=f"{tier}/figure/{name}")
    wandb.log({f"{tier}/{name}": wandb.Image(out),
               f"{tier}/{name}_table": wandb.Table(dataframe=table.astype(str))})
    run.finish()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tier", required=True)
    p.add_argument("--variants", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", help="legend names, same order as --variants")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="")
    p.add_argument("--wandb", action="store_true", help="also upload the figure and table to W&B")
    a = p.parse_args()
    out, table = plot(a.tier, a.variants, a.out, a.title, a.labels)
    if a.wandb:
        upload_to_wandb(a.tier, out, table)


if __name__ == "__main__":
    main()
