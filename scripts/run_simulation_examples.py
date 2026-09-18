"""Additional OpenHail examples with rendering kept explicitly opt-in."""

import argparse
import logging
from pathlib import Path

from openhail.utils.config_runner import ConfigurationRunner, quick_compare


def run_headless() -> None:
    runner = ConfigurationRunner(render=False)
    runner.run_single_config(
        simulation="debug_config_fast",
        evaluation="S1",
        city="nyc",
        agent="nearest",
        infrastructure="S05_1_4",
    )


def run_rendered() -> None:
    runner = ConfigurationRunner(render=True)
    runner.run_single_config(
        simulation="debug_config_fast",
        evaluation="S1",
        city="nyc",
        agent="nearest",
        infrastructure="S05_1_4",
    )


def save_gif(filename: str = "out/render/simulation.gif") -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    runner = ConfigurationRunner(
        render=True,
        save_gif=True,
        gif_filename=str(path),
    )
    runner.run_single_config(
        simulation="debug_config_fast",
        evaluation="S1",
        city="nyc",
        agent="nearest",
        horizon="horizon_1_2k",
        infrastructure="S05_4_3",
    )


def compare_baselines() -> None:
    quick_compare(
        config_variations={"agent": ["nearest", "random"]},
        simulation="debug_config_fast",
        evaluation="S1",
        city="nyc",
        infrastructure="S05_1_4",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "example",
        nargs="?",
        default="headless",
        choices=("headless", "render", "gif", "compare"),
        help="example to run",
    )
    parser.add_argument(
        "--gif-filename",
        default="out/render/simulation.gif",
        help="output path used by the gif example",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.example == "headless":
        run_headless()
    elif args.example == "render":
        run_rendered()
    elif args.example == "gif":
        save_gif(args.gif_filename)
    else:
        compare_baselines()
