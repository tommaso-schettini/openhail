"""Run OpenHail with optional interactive rendering or GIF capture."""

import argparse
import logging

from openhail.utils.config_runner import ConfigurationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", default="debug_config_fast")
    parser.add_argument("--evaluation", default="S1")
    parser.add_argument("--city", default="nyc")
    parser.add_argument("--horizon", default="day_2")
    parser.add_argument("--fleet", default="fleet_20")
    parser.add_argument("--infrastructure", default="S05_1_4")
    parser.add_argument("--agent", default="nearest")
    parser.add_argument("--config-dir", default="./data/config")
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--render",
        action="store_true",
        help="open the interactive Pygame renderer",
    )
    parser.add_argument(
        "--save-gif",
        nargs="?",
        const="out/render/simulation.gif",
        metavar="PATH",
        help=("capture the rendered run as a GIF; optionally provide an output path"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    render = args.render or args.save_gif is not None
    runner = ConfigurationRunner(
        render=render,
        save_gif=args.save_gif is not None,
        gif_filename=args.save_gif,
        config_dir=args.config_dir,
        data_root=args.data_root,
    )
    result = runner.run_single_config(
        simulation=args.simulation,
        evaluation=args.evaluation,
        city=args.city,
        horizon=args.horizon,
        fleet=args.fleet,
        agent=args.agent,
        infrastructure=args.infrastructure,
    )
    logging.info("Average reward: %.2f", result.average_reward())


if __name__ == "__main__":
    main()
