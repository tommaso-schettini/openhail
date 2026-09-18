# Numerical imports must follow process-wide thread-limit configuration.
# ruff: noqa: E402
"""Paired post-hoc timings using configuration-specific trained checkpoints."""

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Set before importing numerical libraries, then verify effective pools below.
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import torch
from controller_benchmark_common import POLICIES, PROJECT_ROOT, load_config, make_agent
from threadpoolctl import threadpool_info, threadpool_limits

from openhail.agents import PeriodicPPOAgent, RandomAgent
from openhail.core.openhail_env import OpenhailEnv
from openhail.core.openhail_instance import OpenhailInstance


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rollout(env, agent, seed, policy_seed):
    start = time.perf_counter()
    state, _ = env.reset(seed=seed)
    if isinstance(agent, RandomAgent):
        agent.seed = policy_seed
    agent.set_doy(env.request_manager.doy)
    if isinstance(agent, PeriodicPPOAgent):
        agent.begin_episode(policy_seed)
    reset_seconds = time.perf_counter() - start
    steps, agent_seconds, environment_seconds, reward_total = 0, 0.0, 0.0, 0.0
    started_utc = datetime.now(timezone.utc).isoformat()
    process_start = time.process_time()
    start = time.perf_counter()
    while True:
        before = time.perf_counter()
        action = agent.choose_action(state)
        after_agent = time.perf_counter()
        state, reward, terminated, truncated, _ = env.step(action)
        after_env = time.perf_counter()
        done = terminated or truncated
        if isinstance(agent, PeriodicPPOAgent):
            agent.observe(reward, state, done)
        after_observe = time.perf_counter()
        agent_seconds += after_agent - before + after_observe - after_env
        environment_seconds += after_env - after_agent
        steps += 1
        reward_total += reward
        if done:
            break
    wall = time.perf_counter() - start
    process_seconds = time.process_time() - process_start
    assert steps > 0 and np.isfinite([wall, reward_total]).all()
    return dict(
        seed=seed,
        policy_seed=policy_seed,
        day=int(env.request_manager.doy),
        started_utc=started_utc,
        process_seconds=process_seconds,
        requests=int(env.request_manager.dummy_request),
        steps=steps,
        reset_seconds=reset_seconds,
        wall_seconds=wall,
        agent_seconds=agent_seconds,
        environment_seconds=environment_seconds,
        overhead_seconds=wall - agent_seconds - environment_seconds,
        ms_per_decision_epoch=1000 * wall / steps,
        agent_ms_per_epoch=1000 * agent_seconds / steps,
        environment_ms_per_epoch=1000 * environment_seconds / steps,
        reward=reward_total,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--tasks", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory containing distributed fleet/infrastructure checkpoints",
    )
    args = parser.parse_args()
    if args.num_episodes < 1 or not set(args.tasks) <= set(range(1, 17)):
        raise ValueError("Invalid episode count or task selection")
    if (args.output_dir / "episodes.csv").exists():
        raise FileExistsError(
            "Use a fresh output directory to preserve prior measurements"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with (
        PROJECT_ROOT
        / (
            "scripts/instances/periodic_ppo_nyc_day1_4fleets_4location_counts_1seed_"
            "500ep_instances.tsv"
        )
    ).open() as handle:
        matrix = list(csv.DictReader(handle, delimiter="\t"))
    cpu = platform.processor()
    if sys.platform == "win32":
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    elif Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    configurations: list[dict[str, Any]] = []
    metadata = dict(
        started=datetime.now(timezone.utc).isoformat(),
        cpu=cpu,
        platform=platform.platform(),
        python=sys.version,
        numpy=np.__version__,
        torch=torch.__version__,
        torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(),
        git_head=(
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
            or None
        ),
        hostname=platform.node(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        cpu_affinity=(
            sorted(getattr(os, "sched_getaffinity")(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        source_sha256={
            str(p.relative_to(PROJECT_ROOT)): digest(p)
            for p in sorted((PROJECT_ROOT / "src").rglob("*.py"))
        },
        runner_sha256=digest(Path(__file__)),
        episodes_per_configuration=args.num_episodes,
        tasks=args.tasks,
        warmup=(
            "one unmeasured rollout per controller and configuration, seed 100000/"
            "200000"
        ),
        measurement=(
            "reset and model loading excluded; agent includes observe; no "
            "optimization; one process; cyclic controller order"
        ),
        configurations=configurations,
    )
    metadata_path = args.output_dir / "metadata.json"
    with (
        threadpool_limits(limits=1),
        (args.output_dir / "episodes.csv").open("w", newline="") as handle,
    ):
        metadata["threadpools"] = threadpool_info()
        writer = None
        for task in args.tasks:
            item = matrix[task - 1]
            fleet = int(item["fleet"].split("_")[1])
            locations = int(item["infrastructure"].split("_")[0][1:])
            name = (
                f"periodic_ppo_nyc_500ep_{item['fleet']}_{item['infrastructure']}_s321_"
                f"1455644_{task}"
            )
            checkpoint_root = args.checkpoint_dir or PROJECT_ROOT / "checkpoints"
            checkpoint = (
                checkpoint_root
                / f"{item['fleet']}_{item['infrastructure']}"
                / "checkpoint_best.pt"
            )
            if args.checkpoint_dir is None and not checkpoint.is_file():
                checkpoint = (
                    PROJECT_ROOT / item["output_dir"] / name / "checkpoint_best.pt"
                )
            train_config = json.loads(checkpoint.with_name("config.json").read_text())
            for key in ("fleet", "horizon", "infrastructure"):
                assert train_config["arguments"][key] == item[key]
            options = argparse.Namespace(
                simulation="performance",
                city="nyc",
                decision_clock=900.0,
                strict_validation=False,
                tracking=False,
            )
            config, infrastructure = load_config(options, fleet, locations)
            configurations.append(
                dict(
                    task=task,
                    config=config,
                    checkpoint=str(checkpoint.resolve()),
                    checkpoint_sha256=digest(checkpoint),
                )
            )
            metadata_path.write_text(json.dumps(metadata, indent=2))
            instance = OpenhailInstance(config)
            env = OpenhailEnv(instance, config["simulation"], {"render_mode": None})
            agents = {p: make_agent(p, instance, checkpoint, 200000) for p in POLICIES}
            for policy in POLICIES:
                rollout(env, agents[policy], 100000, 200000)
            print(f"task={task} N={fleet} L={locations} warm-up complete", flush=True)
            for offset in range(args.num_episodes):
                shift = (task + offset) % len(POLICIES)
                order = POLICIES[shift:] + POLICIES[:shift]
                for position, policy in enumerate(order):
                    row = dict(
                        task=task,
                        fleet_size=fleet,
                        locations=locations,
                        infrastructure=infrastructure,
                        policy=policy,
                        order=position,
                        **rollout(
                            env, agents[policy], 100000 + offset, 200000 + offset
                        ),
                    )
                    assert row["requests"] == 20 * fleet
                    if writer is None:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                    print(
                        (
                            f"task={task} {policy} seed={row['seed']} "
                            f"time={row['wall_seconds']:.2f}s "
                            f"epoch={row['ms_per_decision_epoch']:.3f}ms"
                        ),
                        flush=True,
                    )
            env.close()
        metadata["finished"] = datetime.now(timezone.utc).isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
