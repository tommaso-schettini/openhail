#!/usr/bin/env python
"""Plot and summarize a completed periodic PPO Slurm batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument(
        "--training-root", default="out/periodic_ppo_training", type=Path
    )
    parser.add_argument(
        "--baseline-root", default="out/simulator_analysis_v3", type=Path
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_runs(root: Path, job_id: str) -> dict[str, pd.DataFrame]:
    run_parts: dict[str, list[pd.DataFrame]] = {}
    for directory in sorted(root.glob(f"*_{job_id}_*")):
        config_path = directory / "config.json"
        metrics_path = directory / "metrics.csv"
        if not config_path.is_file() or not metrics_path.is_file():
            continue
        with config_path.open(encoding="utf-8") as file:
            metadata = json.load(file)
        infrastructure = metadata["arguments"]["infrastructure"]
        metrics = pd.read_csv(metrics_path)
        metrics["run_directory"] = directory.name
        metrics["training_seed"] = metadata["arguments"].get("seed")
        run_parts.setdefault(infrastructure, []).append(metrics)
    if not run_parts:
        raise FileNotFoundError(f"No training runs found for Slurm job {job_id}")
    runs = {
        infrastructure: pd.concat(parts, ignore_index=True)
        for infrastructure, parts in run_parts.items()
    }
    return dict(sorted(runs.items()))


def load_baselines(root: Path, infrastructures: list[str]) -> dict[str, pd.DataFrame]:
    baselines: dict[str, pd.DataFrame] = {}
    for infrastructure in infrastructures:
        matches = list(root.glob(f"*.{infrastructure}/episodes.csv"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected one baseline file for {infrastructure}, found {len(matches)}"
            )
        baselines[infrastructure] = pd.read_csv(matches[0])
    return baselines


def evaluation_curve(metrics: pd.DataFrame) -> pd.DataFrame:
    evaluation = metrics.loc[metrics["phase"] == "evaluation_sampled"]
    return (
        evaluation.groupby("episode", as_index=False)
        .agg(
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            acceptance_mean=("request_acceptance_rate", "mean"),
            acceptance_std=("request_acceptance_rate", "std"),
            samples=("reward", "size"),
        )
        .sort_values("episode")
    )


def confidence_interval(std: pd.Series, samples: pd.Series) -> pd.Series:
    return 1.96 * std.fillna(0.0) / np.sqrt(samples)


def baseline_stats(
    baseline: pd.DataFrame, metric: str
) -> dict[str, tuple[float, float]]:
    result: dict[str, tuple[float, float]] = {}
    for policy in ("nearest", "random"):
        values = baseline.loc[baseline["policy"] == policy, metric]
        mean = float(values.mean())
        ci = float(1.96 * values.std(ddof=1) / np.sqrt(len(values)))
        result[policy] = (mean, ci)
    return result


def plot_evaluation_small_multiples(
    runs: dict[str, pd.DataFrame],
    baselines: dict[str, pd.DataFrame],
    output_path: Path,
    metric: str,
    ylabel: str,
) -> None:
    figure, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    for axis, (infrastructure, metrics) in zip(axes.flat, runs.items()):
        curve = evaluation_curve(metrics)
        mean_column = f"{metric}_mean"
        std_column = f"{metric}_std"
        ci = confidence_interval(curve[std_column], curve["samples"])
        x = curve["episode"].to_numpy()
        y = curve[mean_column].to_numpy()
        interval = ci.to_numpy()
        axis.plot(x, y, color="#1769aa", linewidth=1.8, label="Periodic PPO")
        axis.fill_between(
            x,
            y - interval,
            y + interval,
            color="#1769aa",
            alpha=0.18,
            linewidth=0,
        )

        baseline_metric = "reward" if metric == "reward" else "request_acceptance_rate"
        stats = baseline_stats(baselines[infrastructure], baseline_metric)
        for policy, color, linestyle in (
            ("nearest", "#d95f02", "--"),
            ("random", "#666666", ":"),
        ):
            mean, baseline_ci = stats[policy]
            axis.axhline(
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=1.3,
                label=policy.title(),
            )
            axis.axhspan(
                mean - baseline_ci,
                mean + baseline_ci,
                color=color,
                alpha=0.07,
                linewidth=0,
            )

        best_index = int(np.argmax(y))
        axis.scatter(
            x[best_index],
            y[best_index],
            marker="*",
            s=75,
            color="#1769aa",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        axis.set_title(infrastructure)
        axis.grid(alpha=0.22)
        if metric == "acceptance":
            axis.set_ylim(0.0, max(0.55, float((y + interval).max()) + 0.02))

    for axis in axes[-1]:
        axis.set_xlabel("Training episode")
    for axis in axes[:, 0]:
        axis.set_ylabel(ylabel)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(f"Periodic PPO {ylabel.lower()} by infrastructure", y=0.997)
    figure.tight_layout(rect=(0, 0, 1, 0.925))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_final_comparison(
    runs: dict[str, pd.DataFrame],
    baselines: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    labels = list(runs)
    x = np.arange(len(labels))
    width = 0.25
    reward_values = {policy: [] for policy in ("nearest", "random", "ppo")}
    acceptance_values = {policy: [] for policy in ("nearest", "random", "ppo")}

    for infrastructure, metrics in runs.items():
        final_episode = int(metrics.loc[metrics["phase"] == "train", "episode"].max())
        final_eval = metrics.loc[
            (metrics["phase"] == "evaluation_sampled")
            & (metrics["episode"] == final_episode)
        ]
        for policy in ("nearest", "random"):
            policy_rows = baselines[infrastructure].loc[
                baselines[infrastructure]["policy"] == policy
            ]
            reward_values[policy].append(float(policy_rows["reward"].mean()))
            acceptance_values[policy].append(
                float(policy_rows["request_acceptance_rate"].mean())
            )
        reward_values["ppo"].append(float(final_eval["reward"].mean()))
        acceptance_values["ppo"].append(
            float(final_eval["request_acceptance_rate"].mean())
        )

    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    colors = {"nearest": "#d95f02", "random": "#888888", "ppo": "#1769aa"}
    for offset, policy in zip((-width, 0.0, width), ("nearest", "random", "ppo")):
        axes[0].bar(
            x + offset,
            reward_values[policy],
            width,
            label="Periodic PPO" if policy == "ppo" else policy.title(),
            color=colors[policy],
        )
        axes[1].bar(
            x + offset,
            acceptance_values[policy],
            width,
            color=colors[policy],
        )
    axes[0].set_ylabel("Mean evaluation reward")
    axes[1].set_ylabel("Request acceptance rate")
    axes[1].set_xlabel("Infrastructure configuration")
    axes[1].set_xticks(x, labels)
    axes[0].legend(ncol=3, frameon=False, loc="upper center")
    axes[0].set_title("Final policy performance")
    for axis in axes:
        axis.grid(axis="y", alpha=0.22)
        axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_training_diagnostics(runs: dict[str, pd.DataFrame], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    fields = (
        ("reward", "Training reward", False),
        ("request_acceptance_rate", "Training acceptance rate", False),
        ("explained_variance", "Critic explained variance", False),
        ("value_loss", "Value loss", True),
        ("policy_entropy", "Policy entropy", False),
        ("approximate_kl", "Joint approximate KL", False),
    )
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, len(runs)))
    for color, (infrastructure, metrics) in zip(colors, runs.items()):
        training = (
            metrics.loc[metrics["phase"] == "train"]
            .groupby("episode", as_index=False)
            .mean(numeric_only=True)
            .sort_values("episode")
        )
        for axis, (field, title, logarithmic) in zip(axes.flat, fields):
            rolling = training[field].rolling(20, min_periods=5).mean()
            axis.plot(
                training["episode"],
                rolling,
                label=infrastructure,
                color=color,
                linewidth=1.35,
            )
            axis.set_title(title)
            axis.grid(alpha=0.2)
            if logarithmic:
                axis.set_yscale("log")
    for axis in axes[-1]:
        axis.set_xlabel("Training episode")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.96),
        ncol=9,
        frameon=False,
    )
    figure.suptitle("Training diagnostics (20-episode rolling means)", y=0.998)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_reward_components(
    runs: dict[str, pd.DataFrame],
    baselines: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    labels = list(runs)
    x = np.arange(len(labels))
    width = 0.36
    ppo_service: list[float] = []
    ppo_reposition: list[float] = []
    nearest_service: list[float] = []
    nearest_reposition: list[float] = []

    for infrastructure, metrics in runs.items():
        final_episode = int(metrics.loc[metrics["phase"] == "train", "episode"].max())
        final_eval = metrics.loc[
            (metrics["phase"] == "evaluation_sampled")
            & (metrics["episode"] == final_episode)
        ]
        nearest = baselines[infrastructure].loc[
            baselines[infrastructure]["policy"] == "nearest"
        ]
        ppo_service.append(float(final_eval["service_reward"].mean()))
        ppo_reposition.append(float(final_eval["reposition_reward"].mean()))
        nearest_service.append(float(nearest["service_reward"].mean()))
        nearest_reposition.append(float(nearest["reposition_reward"].mean()))

    figure, axis = plt.subplots(figsize=(14, 6.5))
    axis.bar(
        x - width / 2,
        nearest_service,
        width,
        color="#f4a261",
        label="Nearest: service reward",
    )
    axis.bar(
        x - width / 2,
        nearest_reposition,
        width,
        color="#d95f02",
        label="Nearest: reposition reward",
    )
    axis.bar(
        x + width / 2,
        ppo_service,
        width,
        color="#79addc",
        label="Periodic PPO: service reward",
    )
    axis.bar(
        x + width / 2,
        ppo_reposition,
        width,
        color="#1769aa",
        label="Periodic PPO: reposition reward",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_xlabel("Infrastructure configuration")
    axis.set_ylabel("Mean evaluation reward component")
    axis.set_title("Final PPO and nearest-baseline reward decomposition")
    axis.legend(ncol=2, frameon=False, loc="upper left")
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def build_summary(
    runs: dict[str, pd.DataFrame], baselines: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    for infrastructure, metrics in runs.items():
        training = (
            metrics.loc[metrics["phase"] == "train"]
            .groupby("episode", as_index=False)
            .mean(numeric_only=True)
            .sort_values("episode")
        )
        curve = evaluation_curve(metrics)
        best = curve.loc[curve["reward_mean"].idxmax()]
        final = curve.iloc[-1]
        initial = curve.iloc[0]
        last_five = curve.tail(5)
        nearest = baselines[infrastructure].loc[
            baselines[infrastructure]["policy"] == "nearest"
        ]
        random = baselines[infrastructure].loc[
            baselines[infrastructure]["policy"] == "random"
        ]
        nearest_reward = float(nearest["reward"].mean())
        random_reward = float(random["reward"].mean())
        final_episode = int(final["episode"])
        final_samples = metrics.loc[
            (metrics["phase"] == "evaluation_sampled")
            & (metrics["episode"] == final_episode)
        ]
        paired = final_samples.merge(
            nearest[
                [
                    "seed",
                    "reward",
                    "service_reward",
                    "reposition_reward",
                    "request_acceptance_rate",
                ]
            ],
            on="seed",
            suffixes=("_ppo", "_nearest"),
            validate="many_to_one",
        )
        reward_difference = paired["reward_ppo"] - paired["reward_nearest"]
        reward_difference_ci = float(
            1.96 * reward_difference.std(ddof=1) / np.sqrt(len(reward_difference))
        )
        recent_training = training.tail(50)
        records.append(
            {
                "infrastructure": infrastructure,
                "training_episodes": int(training["episode"].max()),
                "evaluation_checkpoints": len(curve),
                "initial_eval_reward": initial["reward_mean"],
                "best_eval_episode": int(best["episode"]),
                "best_eval_reward": best["reward_mean"],
                "final_eval_reward": final["reward_mean"],
                "final_drop_from_best_pct": (best["reward_mean"] - final["reward_mean"])
                / best["reward_mean"],
                "last_5_eval_reward": last_five["reward_mean"].mean(),
                "nearest_reward": nearest_reward,
                "random_reward": random_reward,
                "final_gain_vs_nearest": final["reward_mean"] - nearest_reward,
                "final_gain_vs_nearest_pct": (
                    final["reward_mean"] / nearest_reward - 1.0
                ),
                "paired_gain_ci_low": reward_difference.mean() - reward_difference_ci,
                "paired_gain_ci_high": reward_difference.mean() + reward_difference_ci,
                "paired_seed_wins": int((reward_difference > 0).sum()),
                "paired_seed_count": len(reward_difference),
                "service_reward_gain_vs_nearest": (
                    paired["service_reward_ppo"] - paired["service_reward_nearest"]
                ).mean(),
                "reposition_reward_gain_vs_nearest": (
                    paired["reposition_reward_ppo"]
                    - paired["reposition_reward_nearest"]
                ).mean(),
                "final_gain_vs_random": final["reward_mean"] - random_reward,
                "final_acceptance_rate": final["acceptance_mean"],
                "nearest_acceptance_rate": nearest["request_acceptance_rate"].mean(),
                "random_acceptance_rate": random["request_acceptance_rate"].mean(),
                "critic_ev_last_50": recent_training["explained_variance"].mean(),
                "value_loss_last_50": recent_training["value_loss"].mean(),
                "entropy_last_50": recent_training["policy_entropy"].mean(),
                "approximate_kl_last_50": recent_training["approximate_kl"].mean(),
                "clip_fraction_last_50": recent_training["clip_fraction"].mean(),
                "invalid_actions": int(training["invalid_action_count"].sum()),
                "empty_masks": int(training["empty_mask_count"].sum()),
                "zero_capacity_charges": int(
                    training["zero_capacity_charge_count"].sum()
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("infrastructure")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.training_root / f"server_batch_{args.job_id}_plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.training_root, args.job_id)
    baselines = load_baselines(args.baseline_root, list(runs))

    plot_evaluation_small_multiples(
        runs,
        baselines,
        output_dir / "evaluation_reward_by_infrastructure.png",
        metric="reward",
        ylabel="Mean evaluation reward",
    )
    plot_evaluation_small_multiples(
        runs,
        baselines,
        output_dir / "evaluation_acceptance_by_infrastructure.png",
        metric="acceptance",
        ylabel="Request acceptance rate",
    )
    plot_final_comparison(runs, baselines, output_dir / "final_policy_comparison.png")
    plot_training_diagnostics(runs, output_dir / "training_diagnostics.png")
    plot_reward_components(
        runs, baselines, output_dir / "reward_component_comparison.png"
    )
    summary = build_summary(runs, baselines)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Wrote plots and summary to {output_dir}")


if __name__ == "__main__":
    main()
