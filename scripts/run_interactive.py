"""Entry point for the interactive control panel.

Run from the repository root::

    python scripts/run_interactive.py

or with an explicit configuration, mirroring ``scripts/run_simulation.py``::

    python scripts/run_interactive.py --fleet fleet_100 --agent random

Press ``h`` inside the window for the key bindings.
"""

import argparse
import logging

from openhail.gui.session import run_interactive


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
        "--seed",
        type=int,
        default=None,
        help="evaluation seed; defaults to the evaluation config's first seed",
    )
    parser.add_argument(
        "--show-help",
        action="store_true",
        help="start with the key binding overlay visible",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    reward = run_interactive(
        simulation=args.simulation,
        evaluation=args.evaluation,
        city=args.city,
        horizon=args.horizon,
        fleet=args.fleet,
        infrastructure=args.infrastructure,
        agent=args.agent,
        config_dir=args.config_dir,
        data_root=args.data_root,
        seed=args.seed,
        show_help=args.show_help,
    )
    logging.info("Average reward: %.2f", reward)


if __name__ == "__main__":
    main()
