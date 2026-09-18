"""Summarize and plot paired post-hoc controller timing measurements."""

import argparse
import json
from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STYLES = {
    "nearest": ("Nearest", "#0072B2", "o", "-"),
    "ppo_sampled": ("Trained PPO", "#D55E00", "s", "--"),
    "random": ("Random feasible", "#009E73", "^", ":"),
}
METRICS = [
    "wall_seconds",
    "agent_seconds",
    "environment_seconds",
    "overhead_seconds",
    "steps",
    "ms_per_decision_epoch",
    "agent_ms_per_epoch",
    "environment_ms_per_epoch",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "paper/figures")
    args = parser.parse_args()
    metadata = json.loads((args.input_dir / "metadata.json").read_text())
    assert "finished" in metadata, "The benchmark is not complete"
    assert set(metadata["tasks"]) == set(range(1, 17)), (
        "Publication plots require all 16 configurations"
    )
    data = pd.read_csv(args.input_dir / "episodes.csv")
    expected = len(metadata["tasks"]) * 3 * metadata["episodes_per_configuration"]
    assert len(data) == expected
    assert not data.duplicated(["task", "policy", "seed"]).any()
    assert np.isfinite(data[METRICS]).all().all()
    assert (data["steps"] > 0).all()
    assert (data.overhead_seconds >= -1e-8).all()
    assert np.allclose(
        data.ms_per_decision_epoch, 1000 * data.wall_seconds / data.steps
    )
    for _, group in data.groupby(["task", "seed"]):
        assert set(group.policy) == set(STYLES)
        assert (
            group.day.nunique()
            == group.requests.nunique()
            == group.policy_seed.nunique()
            == 1
        )
    groups = data.groupby(["task", "fleet_size", "locations", "policy"])
    assert bool(
        np.all(groups.size().to_numpy() == metadata["episodes_per_configuration"])
    )
    rows = []
    for key, group in groups:
        task, fleet, locations, policy = cast(tuple[int, int, int, str], key)
        row: dict[str, Any] = dict(
            task=task,
            fleet_size=fleet,
            locations=locations,
            policy=policy,
            episodes=len(group),
        )
        for metric in METRICS:
            row.update(
                {
                    f"{metric}_{name}": group[metric].quantile(q)
                    for name, q in [("median", 0.5), ("q1", 0.25), ("q3", 0.75)]
                }
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.input_dir / "summary.csv", index=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10})
    for metric, ylabel, filename in [
        ("wall_seconds", "Episode time (s)", "posthoc_episode_times"),
        ("ms_per_decision_epoch", "Time per epoch (ms)", "posthoc_epoch_times"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), sharey=True)
        for panel, (ax, locations) in enumerate(
            zip(axes.flat, [5, 10, 20, 40], strict=True)
        ):
            for policy, (label, color, marker, linestyle) in STYLES.items():
                g = summary.loc[
                    (summary.locations == locations) & (summary.policy == policy)
                ].sort_values("fleet_size")
                med = g[f"{metric}_median"].to_numpy()
                ax.errorbar(
                    g.fleet_size,
                    med,
                    yerr=[med - g[f"{metric}_q1"], g[f"{metric}_q3"] - med],
                    color=color,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=1.2,
                    markersize=4,
                    capsize=2,
                    elinewidth=0.7,
                    label=label,
                )
            ax.set(
                title=f"({chr(97 + panel)}) $D={locations}$",
                xlabel="Number of vehicles",
                yscale="log" if metric == "wall_seconds" else "linear",
            )
            ax.set_xticks([100, 500, 1000, 2000], ["100", "500", "1,000", "2,000"])
            if panel % 2 == 0:
                ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25, linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
        if metric == "wall_seconds":
            axes.flat[0].set_ylim(
                summary[f"{metric}_q1"].min() * 0.85,
                summary[f"{metric}_q3"].max() * 1.2,
            )
        else:
            axes.flat[0].set_ylim(0, 1.2 * summary[f"{metric}_q3"].max())
        fig.legend(
            *axes.flat[0].get_legend_handles_labels(),
            loc="upper center",
            ncol=3,
            frameon=False,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.5, h_pad=1.4)
        for suffix in ("pdf", "png"):
            fig.savefig(
                args.output_dir / f"{filename}.{suffix}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)
    print(
        summary.loc[
            (summary.fleet_size == 2000) & (summary.locations == 40),
            ["policy"] + [f"{m}_median" for m in METRICS],
        ].to_string(index=False)
    )
    print("Verified", len(data), "paired timing records.")


if __name__ == "__main__":
    main()
