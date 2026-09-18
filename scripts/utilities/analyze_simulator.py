#!/usr/bin/env python
"""Characterize simulator states, masks, rewards, and baseline behavior.

The script runs closest/nearest and random-feasible baselines over every row in
``scripts/instances/simulator_analysis_instances.tsv``. Each configuration receives its
own output directory so rows can also be executed independently on a cluster.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Any

import numpy as np

from openhail.agents import NearestAgent, RandomAgent
from openhail.core.constants import (
    CLOCK_EPOCH,
    END_OF_CHARGE,
    END_OF_REPO,
    END_OF_SERVE,
    LOC,
    NEW_REQUEST,
    TIME,
    Q,
)
from openhail.core.openhail_env import OpenhailEnv
from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils import arg_utilities, state_utilities
from openhail.utils.logging_utilities import CsvMetricsLogger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCES = (
    PROJECT_ROOT / "scripts" / "instances" / "simulator_analysis_instances.tsv"
)

ANALYSIS_FIELDS = (
    "configuration",
    "policy",
    "seed",
    "reward",
    "service_reward",
    "reposition_reward",
    "requests_total",
    "requests_served",
    "request_acceptance_rate",
    "steps",
    "mean_delta_time",
    "min_delta_time",
    "max_delta_time",
    "mean_soc",
    "min_soc",
    "max_soc",
    "mean_charger_occupancy",
    "max_charger_occupancy",
    "mean_charger_queue",
    "max_charger_queue",
    "mean_reposition_eligible_vehicles",
    "min_feasible_actions",
    "max_feasible_actions",
    "empty_mask_count",
    "invalid_action_count",
    "action_noop_count",
    "action_reposition_count",
    "action_charge_count",
    "vehicle_x_min",
    "vehicle_x_max",
    "vehicle_y_min",
    "vehicle_y_max",
    "availability_delay_min",
    "availability_delay_max",
    "clock_epoch_count",
    "new_request_epoch_count",
    "end_service_epoch_count",
    "end_charge_epoch_count",
    "end_repo_epoch_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument(
        "--config-index",
        type=int,
        default=None,
        help="One-based TSV row to run; default runs every row.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=("nearest", "random"),
        default=("nearest", "random"),
    )
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--first-seed", type=int, default=None)
    parser.add_argument("--decision-clock", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "out" / "simulator_analysis",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    required = {
        "city",
        "simulation",
        "horizon",
        "fleet",
        "infrastructure",
        "num_episodes",
        "first_seed",
        "decision_clock",
    }
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Missing TSV columns: {sorted(missing)}")
    return rows


def load_instance_config(row: dict[str, str], decision_clock: float) -> dict:
    parser = arg_utilities.get_parser(
        simulation=row["simulation"],
        evaluation="S1",
        city=row["city"],
        horizon=row["horizon"],
        fleet=row["fleet"],
        infrastructure=row["infrastructure"],
        agent="nearest",
    )
    parsed = parser.parse_args([])
    config, _ = arg_utilities.load_config(
        parsed, source_dir=str(PROJECT_ROOT / "data" / "config")
    )
    simulation = config["simulation"]
    simulation["decision_clock"] = decision_clock
    simulation["strict_validation"] = True
    simulation["track_vehicles"] = True
    simulation["track_requests"] = True
    simulation["track_epochs"] = True
    return config


def make_agent(policy: str, instance: OpenhailInstance, seed: int):
    if policy == "nearest":
        return NearestAgent(instance)
    return RandomAgent(instance, seed=seed)


def native_action_counts(rnr_mask: np.ndarray) -> np.ndarray:
    """Count NOOP plus both native modes for every feasible destination."""
    return rnr_mask[:, 0].astype(np.int32) + 2 * rnr_mask[:, 1:].sum(axis=1)


def summarize_episode(
    configuration: str,
    policy: str,
    seed: int,
    env: OpenhailEnv,
    agent,
) -> dict[str, Any]:
    state, _ = env.reset(seed=seed)
    agent.set_doy(env.request_manager.doy)

    totals = {
        "reward": 0.0,
        "service_reward": 0.0,
        "reposition_reward": 0.0,
        "steps": 0,
        "empty_mask_count": 0,
        "invalid_action_count": 0,
        "action_noop_count": 0,
        "action_reposition_count": 0,
        "action_charge_count": 0,
    }
    delta_times: list[float] = []
    mean_socs: list[float] = []
    min_socs: list[float] = []
    max_socs: list[float] = []
    mean_occupancies: list[float] = []
    max_occupancies: list[float] = []
    mean_queues: list[float] = []
    max_queues: list[float] = []
    eligible_counts: list[int] = []
    feasible_action_counts: list[int] = []
    x_values: list[float] = []
    y_values: list[float] = []
    availability_delays: list[float] = []
    epoch_counts = {
        CLOCK_EPOCH: 0,
        NEW_REQUEST: 0,
        END_OF_SERVE: 0,
        END_OF_CHARGE: 0,
        END_OF_REPO: 0,
    }
    destination_counts = np.zeros(env.instance.D_repo, dtype=np.int64)

    done = False
    while not done:
        action = agent.choose_action(state)
        serve_action, rnr_action, _ = action

        post_assignment_V = state_utilities.update_state(
            env.instance, state["V"], state["request"], serve_action
        )
        rnr_mask, _ = state_utilities.compute_rnr_mask(env.instance, post_assignment_V)
        per_vehicle_actions = native_action_counts(rnr_mask)
        eligible_counts.append(int(np.any(rnr_mask[:, 1:], axis=1).sum()))
        feasible_action_counts.extend(per_vehicle_actions.tolist())
        totals["empty_mask_count"] += int((per_vehicle_actions == 0).sum())
        totals["invalid_action_count"] += int(not env.action_space.contains(action))

        rnr_action = np.asarray(rnr_action)
        totals["action_noop_count"] += int((rnr_action == 0).sum())
        totals["action_reposition_count"] += int(
            ((rnr_action > 0) & (rnr_action % 2 == 1)).sum()
        )
        totals["action_charge_count"] += int(
            ((rnr_action > 0) & (rnr_action % 2 == 0)).sum()
        )
        positive_actions = rnr_action[rnr_action > 0]
        if positive_actions.size:
            destinations = np.ceil(positive_actions / 2).astype(np.int64) - 1
            destination_counts += np.bincount(
                destinations, minlength=env.instance.D_repo
            )

        vehicles = state["V"]
        x_values.extend(np.asarray(vehicles[LOC])[:, 0].tolist())
        y_values.extend(np.asarray(vehicles[LOC])[:, 1].tolist())
        availability_delays.extend(
            (np.asarray(vehicles[TIME]) - float(state["time"])).tolist()
        )

        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        totals["reward"] += float(reward)
        totals["service_reward"] += info["reward_components"]["service"]
        totals["reposition_reward"] += info["reward_components"]["reposition"]
        totals["steps"] += 1
        delta_times.append(info["delta_time"])
        epoch_counts[info["epoch_type"]] = epoch_counts.get(info["epoch_type"], 0) + 1

        soc = np.asarray(state["V"][Q], dtype=float) / env.instance.ev_max_Q
        mean_socs.append(float(soc.mean()))
        min_socs.append(float(soc.min()))
        max_socs.append(float(soc.max()))
        occupancy = np.asarray(info["charger_occupancy"], dtype=float)
        queue = np.asarray(info["charger_queue"], dtype=float)
        mean_occupancies.append(float(occupancy.mean()))
        max_occupancies.append(float(occupancy.max(initial=0)))
        mean_queues.append(float(queue.mean()))
        max_queues.append(float(queue.max(initial=0)))

    summary = env.get_episode_summary()
    requests = summary.get("request_statistics", {})
    requests_total = int(requests.get("requests_total", 0))
    requests_served = int(requests.get("requests_served", 0))
    acceptance = requests_served / requests_total if requests_total else 0.0

    destination_metrics = {
        f"action_destination_{idx}_count": int(count)
        for idx, count in enumerate(destination_counts)
    }
    return {
        "configuration": configuration,
        "policy": policy,
        "seed": seed,
        **totals,
        "requests_total": requests_total,
        "requests_served": requests_served,
        "request_acceptance_rate": acceptance,
        "mean_delta_time": statistics.fmean(delta_times),
        "min_delta_time": min(delta_times),
        "max_delta_time": max(delta_times),
        "mean_soc": statistics.fmean(mean_socs),
        "min_soc": min(min_socs),
        "max_soc": max(max_socs),
        "mean_charger_occupancy": statistics.fmean(mean_occupancies),
        "max_charger_occupancy": max(max_occupancies),
        "mean_charger_queue": statistics.fmean(mean_queues),
        "max_charger_queue": max(max_queues),
        "mean_reposition_eligible_vehicles": statistics.fmean(eligible_counts),
        "min_feasible_actions": min(feasible_action_counts),
        "max_feasible_actions": max(feasible_action_counts),
        "vehicle_x_min": min(x_values),
        "vehicle_x_max": max(x_values),
        "vehicle_y_min": min(y_values),
        "vehicle_y_max": max(y_values),
        "availability_delay_min": min(availability_delays),
        "availability_delay_max": max(availability_delays),
        "clock_epoch_count": epoch_counts[CLOCK_EPOCH],
        "new_request_epoch_count": epoch_counts[NEW_REQUEST],
        "end_service_epoch_count": epoch_counts[END_OF_SERVE],
        "end_charge_epoch_count": epoch_counts[END_OF_CHARGE],
        "end_repo_epoch_count": epoch_counts[END_OF_REPO],
        **destination_metrics,
    }


def aggregate(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy in sorted({row["policy"] for row in rows}):
        policy_rows = [row for row in rows if row["policy"] == policy]
        metrics = {}
        for field in fields:
            values = [row[field] for row in policy_rows]
            if field in {"configuration", "policy"}:
                continue
            numeric = [float(value) for value in values]
            metrics[field] = {
                "mean": statistics.fmean(numeric),
                "min": min(numeric),
                "max": max(numeric),
            }
            if len(numeric) > 1:
                metrics[field]["stdev"] = statistics.stdev(numeric)
        result[policy] = metrics
    return result


def run_configuration(
    row: dict[str, str],
    policies: tuple[str, ...],
    output_base: Path,
    num_episodes: int,
    first_seed: int,
    decision_clock: float,
) -> None:
    configuration = ".".join(
        row[key] for key in ("city", "horizon", "fleet", "infrastructure")
    )
    output_dir = output_base / configuration
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_instance_config(row, decision_clock)
    instance = OpenhailInstance(config)
    env = OpenhailEnv(instance, config["simulation"], {"render_mode": None})
    episode_rows: list[dict[str, Any]] = []
    fields = ANALYSIS_FIELDS + tuple(
        f"action_destination_{idx}_count" for idx in range(instance.D_repo)
    )

    with CsvMetricsLogger(output_dir / "episodes.csv", fields) as logger:
        for policy in policies:
            for seed in range(first_seed, first_seed + num_episodes):
                agent = make_agent(policy, instance, seed)
                metrics = summarize_episode(configuration, policy, seed, env, agent)
                logger.log(metrics)
                episode_rows.append(metrics)
                print(
                    f"{configuration} | {policy} | seed={seed} | "
                    f"reward={metrics['reward']:.2f} | "
                    f"acceptance={metrics['request_acceptance_rate']:.3f}",
                    flush=True,
                )

    manifest = {
        "configuration": configuration,
        "source": row,
        "policies": list(policies),
        "num_episodes": num_episodes,
        "first_seed": first_seed,
        "decision_clock": decision_clock,
        "instance_config": config,
        "summary": aggregate(episode_rows, fields),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    rows = load_rows(args.instances)
    if args.config_index is not None:
        if not 1 <= args.config_index <= len(rows):
            raise SystemExit(
                f"--config-index must be between 1 and {len(rows)}, inclusive."
            )
        rows = [rows[args.config_index - 1]]

    for row in rows:
        num_episodes = args.num_episodes or int(row["num_episodes"])
        first_seed = (
            args.first_seed if args.first_seed is not None else int(row["first_seed"])
        )
        decision_clock = (
            args.decision_clock
            if args.decision_clock is not None
            else float(row["decision_clock"])
        )
        run_configuration(
            row,
            tuple(args.policies),
            args.output_dir,
            num_episodes,
            first_seed,
            decision_clock,
        )


if __name__ == "__main__":
    main()
