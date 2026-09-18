#!/usr/bin/env python
"""Train the periodic PPO repositioning agent with closest request assignment."""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openhail.agents.periodic_ppo import PeriodicPPOAgent
from openhail.core.openhail_env import OpenhailEnv
from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils.arg_utilities import get_parser, load_config
from openhail.utils.logging_utilities import (
    TRAINING_METRIC_FIELDS,
    CsvMetricsLogger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--critic-learning-rate",
        type=float,
        default=None,
        help="Critic Adam learning rate (defaults to --learning-rate).",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--reward-scale", type=float, default=0.01)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-decisions", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-first-seed", type=int, default=100000)
    parser.add_argument("--eval-first-policy-seed", type=int, default=200000)
    parser.add_argument(
        "--greedy-eval",
        action="store_true",
        help="Also log a separately labelled deterministic evaluation.",
    )
    parser.add_argument("--simulation", default="debug_config_fast")
    parser.add_argument("--city", default="nyc")
    parser.add_argument("--horizon", default="horizon_1_2k")
    parser.add_argument("--fleet", default="fleet_20")
    parser.add_argument("--infrastructure", default="S05_1_4")
    parser.add_argument("--decision-clock", type=float, default=900.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="out/periodic_ppo_training")
    parser.add_argument("--run-name")
    parser.add_argument("--save-interval", type=int, default=0)
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


def git_dirty() -> bool | None:
    """Return whether tracked or untracked files differ from HEAD."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def create_run_directory(base: str, name: str) -> Path:
    run_directory = Path(base) / name
    run_directory.mkdir(parents=True, exist_ok=True)
    if any(run_directory.iterdir()):
        raise FileExistsError(
            f"Use a fresh training directory to preserve prior results: {run_directory}"
        )
    lock = run_directory / ".training.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Training directory is already locked: {run_directory}"
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as file:
        file.write(f"pid={os.getpid()}\n")
    atexit.register(lambda: lock.exists() and lock.unlink())
    return run_directory


def load_environment_config(args: argparse.Namespace) -> dict[str, Any]:
    parser = get_parser(
        simulation=args.simulation,
        evaluation="S1",
        city=args.city,
        horizon=args.horizon,
        fleet=args.fleet,
        infrastructure=args.infrastructure,
        agent="periodic_ppo",
    )
    config, _ = load_config(parser.parse_args([]))
    config["simulation"]["decision_clock"] = args.decision_clock
    config["simulation"]["strict_validation"] = True
    config["simulation"]["track_vehicles"] = True
    config["simulation"]["track_requests"] = True
    config["simulation"]["track_epochs"] = True
    return config


def run_episode(
    env: OpenhailEnv,
    agent: PeriodicPPOAgent,
    seed: int,
    training: bool,
    sample_actions: bool = True,
    policy_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    if policy_seed is None:
        policy_seed = seed
    agent.set_mode(training=training, sample_actions=sample_actions)
    agent.begin_episode(policy_seed)
    state, _ = env.reset(seed=seed)

    total_reward = 0.0
    service_reward = 0.0
    reposition_reward = 0.0
    steps = 0
    invalid_action_count = 0
    empty_mask_count = 0
    action_noop_count = 0
    action_reposition_count = 0
    action_charge_count = 0
    zero_capacity_charge_count = 0
    reposition_destinations = np.zeros(env.instance.D_repo, dtype=np.int64)
    charge_destinations = np.zeros(env.instance.D_repo, dtype=np.int64)
    soc_values: list[float] = []
    occupancy_values: list[float] = []
    queue_values: list[float] = []
    delta_times: list[float] = []
    policy_times: list[float] = []
    policy_entropies: list[float] = []

    done = False
    while not done:
        action = agent.choose_action(state)
        if agent.last_policy_decision:
            policy_times.append(float(state["time"]))
            policy_entropies.append(agent.last_policy_entropy)
        empty_mask_count += agent.last_empty_mask_count
        invalid_action_count += int(not env.action_space.contains(action))

        rnr_action = np.asarray(action[1], dtype=np.int32)
        action_noop_count += int((rnr_action == 0).sum())
        repositioning = (rnr_action > 0) & (rnr_action % 2 == 1)
        charging = (rnr_action > 0) & (rnr_action % 2 == 0)
        action_reposition_count += int(repositioning.sum())
        action_charge_count += int(charging.sum())
        if np.any(repositioning):
            destinations = (rnr_action[repositioning] - 1) // 2
            reposition_destinations += np.bincount(
                destinations, minlength=env.instance.D_repo
            )
        if np.any(charging):
            destinations = rnr_action[charging] // 2 - 1
            charge_destinations += np.bincount(
                destinations, minlength=env.instance.D_repo
            )
            zero_capacity_charge_count += int(
                np.count_nonzero(env.instance.charger_count[destinations] == 0)
            )

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.observe(reward, next_state, done)
        state = next_state
        total_reward += float(reward)
        service_reward += float(info["reward_components"]["service"])
        reposition_reward += float(info["reward_components"]["reposition"])
        soc_values.append(float(info["mean_soc"]))
        occupancy_values.append(float(np.mean(info["charger_occupancy"])))
        queue_values.append(float(np.mean(info["charger_queue"])))
        delta_times.append(float(info["delta_time"]))
        steps += 1

    request_statistics = env.get_episode_summary()["request_statistics"]
    requests_total = int(request_statistics["requests_total"])
    requests_served = int(request_statistics["requests_served"])
    policy_deltas = np.diff(policy_times)
    metrics: dict[str, Any] = {
        "seed": seed,
        "policy_seed": policy_seed,
        "action_selection": "sampled" if sample_actions else "greedy",
        "reward": total_reward,
        "service_reward": service_reward,
        "reposition_reward": reposition_reward,
        "requests_total": requests_total,
        "requests_served": requests_served,
        "request_acceptance_rate": (
            requests_served / requests_total if requests_total else 0.0
        ),
        "action_noop_count": action_noop_count,
        "action_reposition_count": action_reposition_count,
        "action_charge_count": action_charge_count,
        "zero_capacity_charge_count": zero_capacity_charge_count,
        "reposition_destination_counts": json.dumps(
            reposition_destinations.tolist(), separators=(",", ":")
        ),
        "charge_destination_counts": json.dumps(
            charge_destinations.tolist(), separators=(",", ":")
        ),
        "policy_decision_count": len(policy_times),
        "mean_policy_delta_time": (
            float(np.mean(policy_deltas)) if policy_deltas.size else 0.0
        ),
        "policy_entropy": (
            float(np.mean(policy_entropies)) if policy_entropies else 0.0
        ),
        "invalid_action_count": invalid_action_count,
        "empty_mask_count": empty_mask_count,
        "mean_soc": float(np.mean(soc_values)),
        "mean_charger_occupancy": float(np.mean(occupancy_values)),
        "mean_charger_queue": float(np.mean(queue_values)),
        "mean_delta_time": float(np.mean(delta_times)),
        "steps": steps,
    }
    training_stats = (
        agent.pop_training_stats()
        if training
        else PeriodicPPOAgent._empty_training_stats()
    )
    return metrics, training_stats


def complete_metric_row(
    episode: int,
    phase: str,
    metrics: dict[str, Any],
    training_stats: dict[str, float],
    learning_rate: float,
    critic_learning_rate: float,
) -> dict[str, Any]:
    return {
        "episode": episode,
        "phase": phase,
        **metrics,
        "policy_loss": training_stats["policy_loss"],
        "approximate_kl": training_stats["approximate_kl"],
        "clip_fraction": training_stats["clip_fraction"],
        "value_loss": training_stats["value_loss"],
        "explained_variance": training_stats["explained_variance"],
        "gradient_norm": training_stats["gradient_norm"],
        "policy_gradient_norm": training_stats["policy_gradient_norm"],
        "value_gradient_norm": training_stats["value_gradient_norm"],
        "optimizer_updates": int(training_stats["updates"]),
        "learning_rate": learning_rate,
        "critic_learning_rate": critic_learning_rate,
    }


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    if args.run_name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.run_name = (
            f"periodic_ppo_{args.infrastructure}_{args.num_episodes}ep_"
            f"s{args.seed}_{timestamp}"
        )
    run_directory = create_run_directory(args.output_dir, args.run_name)
    config = load_environment_config(args)
    instance = OpenhailInstance(config)
    env = OpenhailEnv(instance, config["simulation"], {"render_mode": None})
    agent = PeriodicPPOAgent(
        instance,
        training=True,
        device=args.device,
        seed=args.seed,
        learning_rate=args.learning_rate,
        critic_learning_rate=(
            args.learning_rate
            if args.critic_learning_rate is None
            else args.critic_learning_rate
        ),
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio,
        entropy_coefficient=args.entropy_coefficient,
        value_coefficient=args.value_coefficient,
        reward_scale=args.reward_scale,
        update_epochs=args.update_epochs,
        minibatch_decisions=args.minibatch_decisions,
    )

    metadata = {
        "run_name": args.run_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_hash": git_hash(),
        "git_dirty": git_dirty(),
        "arguments": vars(args),
        "agent": asdict(agent.config),
        "environment": config,
        "architecture": {
            "assignment": "closest_feasible",
            "policy_epoch": "CLOCK_EPOCH",
            "primary_evaluation": "sampled_fixed_seed_pairs",
            "best_checkpoint_metric": "mean_sampled_evaluation_reward",
            "charge_actions_require_positive_capacity": True,
            "ppo_likelihood_ratio": "joint_fleet_action",
            "critic_aggregation": "independent_self_attention_set_encoder",
            "target_location_features": "normalized_location_state",
            "location_count_independent_checkpoint": True,
            "native_actions_per_vehicle": agent.num_actions,
            "vehicle_feature_dim": agent.feature_extractor.vehicle_feature_dim,
            "location_feature_dim": agent.feature_extractor.location_feature_dim,
            "time_feature_dim": agent.feature_extractor.time_feature_dim,
        },
    }
    with (run_directory / "config.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    best_evaluation_reward = -np.inf
    metrics_path = run_directory / "metrics.csv"
    try:
        with CsvMetricsLogger(metrics_path, TRAINING_METRIC_FIELDS) as logger:
            for episode in range(1, args.num_episodes + 1):
                episode_start = time.time()
                training_seed = args.seed + episode - 1
                metrics, training_stats = run_episode(
                    env, agent, training_seed, training=True
                )
                learning_rate = agent.actor_optimizer.param_groups[0]["lr"]
                critic_learning_rate = agent.critic_optimizer.param_groups[0]["lr"]
                logger.log(
                    complete_metric_row(
                        episode,
                        "train",
                        metrics,
                        training_stats,
                        learning_rate,
                        critic_learning_rate,
                    )
                )

                evaluation_text = ""
                if args.eval_interval > 0 and episode % args.eval_interval == 0:
                    sampled_evaluation_rewards = []
                    for offset in range(args.eval_episodes):
                        evaluation_seed = args.eval_first_seed + offset
                        policy_seed = args.eval_first_policy_seed + offset
                        eval_metrics, eval_stats = run_episode(
                            env,
                            agent,
                            evaluation_seed,
                            training=False,
                            sample_actions=True,
                            policy_seed=policy_seed,
                        )
                        sampled_evaluation_rewards.append(float(eval_metrics["reward"]))
                        logger.log(
                            complete_metric_row(
                                episode,
                                "evaluation_sampled",
                                eval_metrics,
                                eval_stats,
                                learning_rate,
                                critic_learning_rate,
                            )
                        )
                    mean_sampled_evaluation = float(np.mean(sampled_evaluation_rewards))
                    evaluation_text = f" eval_sampled={mean_sampled_evaluation:.1f}"
                    if mean_sampled_evaluation > best_evaluation_reward:
                        best_evaluation_reward = mean_sampled_evaluation
                        agent.save(run_directory / "checkpoint_best.pt")

                    if args.greedy_eval:
                        greedy_evaluation_rewards = []
                        for offset in range(args.eval_episodes):
                            evaluation_seed = args.eval_first_seed + offset
                            policy_seed = args.eval_first_policy_seed + offset
                            eval_metrics, eval_stats = run_episode(
                                env,
                                agent,
                                evaluation_seed,
                                training=False,
                                sample_actions=False,
                                policy_seed=policy_seed,
                            )
                            greedy_evaluation_rewards.append(
                                float(eval_metrics["reward"])
                            )
                            logger.log(
                                complete_metric_row(
                                    episode,
                                    "evaluation_greedy",
                                    eval_metrics,
                                    eval_stats,
                                    learning_rate,
                                    critic_learning_rate,
                                )
                            )
                        evaluation_text += (
                            " eval_greedy="
                            f"{float(np.mean(greedy_evaluation_rewards)):.1f}"
                        )

                if args.save_interval > 0 and episode % args.save_interval == 0:
                    agent.save(run_directory / f"checkpoint_ep{episode}.pt")
                duration = time.time() - episode_start
                print(
                    f"episode={episode} reward={float(metrics['reward']):.1f} "
                    f"decisions={metrics['policy_decision_count']} "
                    f"loss={training_stats['policy_loss']:.4f} "
                    f"kl={training_stats['approximate_kl']:.4f} "
                    f"clip={training_stats['clip_fraction']:.3f} "
                    f"time={duration:.1f}s{evaluation_text}",
                    flush=True,
                )
        agent.save(run_directory / "checkpoint_final.pt")
    finally:
        env.close()
        lock = run_directory / ".training.lock"
        if lock.exists():
            lock.unlink()

    print(f"Training complete: {run_directory}")


if __name__ == "__main__":
    main()
