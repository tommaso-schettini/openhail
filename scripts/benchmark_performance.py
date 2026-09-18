"""Run the reproducible large-scale OpenHail performance baseline."""

import argparse
import json
import platform
import sys
import time
from collections import Counter

import numpy as np

from openhail.agents import agent_initializer
from openhail.core.openhail_env import EPOCH_TYPE_NAMES, OpenhailEnv
from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils import arg_utilities

DEFAULT_FLEET_SIZE = 2101
DEFAULT_REQUESTS_PER_DAY = 40_000
SECONDS_PER_DAY = 86_400


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fleet-size", type=int, default=DEFAULT_FLEET_SIZE)
    parser.add_argument(
        "--requests-per-day",
        type=int,
        default=DEFAULT_REQUESTS_PER_DAY,
    )
    return parser.parse_args()


def load_benchmark_config(args: argparse.Namespace) -> tuple[dict, dict]:
    config_parser = arg_utilities.get_parser(
        simulation="performance",
        evaluation="S1",
        city="nyc",
        horizon="horizon_1_40k",
        fleet="fleet_2101",
        infrastructure="S67_67_4",
        agent="nearest",
    )
    instance_config, extra_config = arg_utilities.load_config(
        config_parser.parse_args([])
    )
    instance_config["fleet"]["num_vehicles"] = args.fleet_size
    instance_config["horizon"]["num_requests"] = args.requests_per_day
    return instance_config, extra_config


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.fleet_size <= 0:
        raise ValueError("--fleet-size must be strictly positive.")
    if args.requests_per_day <= 0:
        raise ValueError("--requests-per-day must be strictly positive.")

    instance_config, extra_config = load_benchmark_config(args)

    start = time.perf_counter()
    instance = OpenhailInstance(instance_config)
    instance_seconds = time.perf_counter() - start

    agent = agent_initializer.initialize_agent(extra_config["agent"], instance)
    env = OpenhailEnv(
        instance,
        instance_config["simulation"],
        {"render_mode": None},
    )

    start = time.perf_counter()
    state, _ = env.reset(seed=args.seed)
    reset_seconds = time.perf_counter() - start
    agent.set_doy(env.request_manager.doy)

    generated_requests = int(env.request_manager.dummy_request)
    agent_seconds = 0.0
    environment_seconds = 0.0
    total_reward = 0.0
    served_requests = 0
    decisions = 0
    epoch_counts = Counter()
    peak_charger_occupancy = 0
    peak_charger_queue = 0
    terminated = False
    loop_start = time.perf_counter()

    while not terminated:
        start = time.perf_counter()
        action = agent.choose_action(state)
        after_agent = time.perf_counter()
        state, reward, terminated, truncated, info = env.step(action)
        after_environment = time.perf_counter()

        terminated = terminated or truncated
        agent_seconds += after_agent - start
        environment_seconds += after_environment - after_agent
        total_reward += reward
        served_requests += info["requests_served"]
        decisions += 1
        epoch_counts[EPOCH_TYPE_NAMES[info["epoch_type"]]] += 1
        peak_charger_occupancy = max(
            peak_charger_occupancy,
            int(np.sum(info["charger_occupancy"])),
        )
        peak_charger_queue = max(
            peak_charger_queue,
            int(np.sum(info["charger_queue"])),
        )

    loop_seconds = time.perf_counter() - loop_start
    timed_compute_seconds = agent_seconds + environment_seconds
    simulated_fleet_hours = (
        args.fleet_size * (instance.max_time - instance.start_time) / 3600.0
    )
    return {
        "system": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
        "workload": {
            "city": instance.city,
            "source_day_of_year": int(env.request_manager.doy),
            "simulation_days": (instance.max_time - instance.start_time)
            / SECONDS_PER_DAY,
            "fleet_size": args.fleet_size,
            "generated_requests": generated_requests,
            "requests_per_vehicle_day": generated_requests / args.fleet_size,
            "repository_locations": int(instance.D_repo),
            "charging_locations": int(np.count_nonzero(instance.charger_count)),
            "charging_posts": int(np.sum(instance.charger_count)),
            "seed": args.seed,
        },
        "outcomes": {
            "decisions": decisions,
            "served_requests": int(served_requests),
            "acceptance_rate": served_requests / generated_requests,
            "total_reward": float(total_reward),
            "peak_charger_occupancy": peak_charger_occupancy,
            "peak_charger_queue": peak_charger_queue,
            "decision_epochs": dict(sorted(epoch_counts.items())),
        },
        "timing": {
            "instance_seconds": instance_seconds,
            "reset_seconds": reset_seconds,
            "agent_seconds": agent_seconds,
            "environment_seconds": environment_seconds,
            "loop_seconds": loop_seconds,
            "instrumentation_seconds": loop_seconds - timed_compute_seconds,
            "decisions_per_second": decisions / loop_seconds,
            "environment_steps_per_second": decisions / environment_seconds,
            "simulated_fleet_hours_per_wall_second": (
                simulated_fleet_hours / loop_seconds
            ),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(parse_args()), indent=2, sort_keys=True))
