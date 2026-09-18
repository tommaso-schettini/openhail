"""Evaluate a distributed PPO checkpoint on its recorded configuration."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from openhail.agents.periodic_ppo import PeriodicPPOAgent
from openhail.core.openhail_env import OpenhailEnv
from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils.arg_utilities import get_parser, load_config


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="fleet_100_S05_1_20")
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--policy-seed", type=int, default=200000)
    parser.add_argument("--greedy", action="store_true")
    args = parser.parse_args()
    records = json.loads((root / "checkpoints/manifest.json").read_text())
    matches = [r for r in records if Path(r["path"]).parent.name == args.checkpoint]
    if len(matches) != 1:
        parser.error("Unknown checkpoint; see checkpoints/manifest.json")
    record = matches[0]
    checkpoint = root / record["path"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != record["sha256"]:
        raise ValueError("Checkpoint does not match the distributed manifest")
    metadata = json.loads(checkpoint.with_name("config.json").read_text())
    config, _ = load_config(
        get_parser(
            simulation="debug_config_fast",
            city="nyc",
            evaluation="S1",
            fleet=record["fleet"],
            horizon=record["horizon"],
            infrastructure=record["infrastructure"],
            agent="periodic_ppo",
        ).parse_args([]),
        root / "data/config",
        data_root=root,
    )
    config["simulation"]["decision_clock"] = metadata["arguments"]["decision_clock"]
    torch.set_num_threads(1)
    instance = OpenhailInstance(config)
    options = dict(metadata["agent"])
    options.update(device="cpu", checkpoint_path=str(checkpoint))
    agent = PeriodicPPOAgent(instance, sample_actions=not args.greedy, **options)
    env = OpenhailEnv(instance, config["simulation"], {"render_mode": None})
    try:
        agent.begin_episode(args.policy_seed)
        state, _ = env.reset(seed=args.seed)
        reward_total, steps = 0.0, 0
        while True:
            state, reward, terminated, truncated, _ = env.step(
                agent.choose_action(state)
            )
            reward_total += float(reward)
            steps += 1
            if terminated or truncated:
                break
        print(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "reward": reward_total,
                    "steps": steps,
                    "seed": args.seed,
                    "policy_seed": args.policy_seed,
                    "greedy": args.greedy,
                }
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
