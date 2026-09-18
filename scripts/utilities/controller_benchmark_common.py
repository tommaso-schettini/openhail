"""Shared configuration and controller construction for timing studies."""

import argparse
from pathlib import Path

from openhail.agents import NearestAgent, PeriodicPPOAgent, RandomAgent
from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils import arg_utilities

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FLEET_CONFIGS = {
    100: ("fleet_100", "horizon_1_2k"),
    500: ("fleet_500", "horizon_1_10k"),
    1000: ("fleet_1000", "horizon_1_20k"),
    2000: ("fleet_2000", "horizon_1_40k"),
}
CHARGING_LOCATIONS = {5: 1, 10: 2, 20: 5, 40: 10}
POLICIES = ("nearest", "random", "ppo_sampled")


def infrastructure_name(fleet_size: int, location_count: int) -> str:
    charging_locations = CHARGING_LOCATIONS[location_count]
    charging_posts = round(0.2 * fleet_size)
    if charging_posts % charging_locations:
        raise ValueError(
            f"Cannot distribute {charging_posts} posts over "
            f"{charging_locations} charging locations."
        )
    posts_per_location = charging_posts // charging_locations
    return f"S{location_count:02d}_{charging_locations}_{posts_per_location}"


def load_config(
    args: argparse.Namespace,
    fleet_size: int,
    location_count: int,
) -> tuple[dict, str]:
    fleet, horizon = FLEET_CONFIGS[fleet_size]
    infrastructure = infrastructure_name(fleet_size, location_count)
    parser = arg_utilities.get_parser(
        simulation=args.simulation,
        evaluation="S1",
        city=args.city,
        horizon=horizon,
        fleet=fleet,
        infrastructure=infrastructure,
        agent="periodic_ppo",
    )
    config, _ = arg_utilities.load_config(
        parser.parse_args([]),
        source_dir=PROJECT_ROOT / "data/config",
        data_root=PROJECT_ROOT,
    )
    simulation = config["simulation"]
    simulation["decision_clock"] = args.decision_clock
    simulation["strict_validation"] = args.strict_validation
    simulation["track_vehicles"] = args.tracking
    simulation["track_requests"] = args.tracking
    simulation["track_epochs"] = args.tracking
    return config, infrastructure


def make_agent(
    policy: str,
    instance: OpenhailInstance,
    checkpoint: Path,
    policy_seed: int,
):
    if policy == "nearest":
        return NearestAgent(instance)
    if policy == "random":
        return RandomAgent(
            instance,
            seed=policy_seed,
            periodic_repositioning=True,
        )
    return PeriodicPPOAgent(
        instance,
        training=False,
        sample_actions=True,
        checkpoint_path=str(checkpoint),
        seed=policy_seed,
    )
