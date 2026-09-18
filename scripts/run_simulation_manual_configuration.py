"""Run OpenHail by assembling each public component explicitly."""

import logging

import openhail.agents.agent_initializer as agent_initializer
import openhail.utils.arg_utilities as arg_utilities
import openhail.utils.evaluation_utilities as evaluation_utilities
import openhail.utils.logging_utilities as log_utils
from openhail.core.openhail_instance import OpenhailInstance


def main() -> None:
    parser = arg_utilities.get_parser(
        simulation="debug_config_fast",
        evaluation="S1",
        city="nyc",
        horizon="day_2",
        fleet="fleet_20",
        infrastructure="S05_1_4",
        agent="nearest",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="open the interactive Pygame renderer",
    )
    parser.add_argument(
        "--save-gif",
        nargs="?",
        const="out/render/manual-simulation.gif",
        metavar="PATH",
        help=("capture the rendered run as a GIF; optionally provide an output path"),
    )
    args = parser.parse_args()
    render = args.render or args.save_gif is not None
    gif_filename = args.save_gif
    delattr(args, "render")
    delattr(args, "save_gif")
    instance_config, extra_config = arg_utilities.load_config(args)
    run_name = log_utils.get_run_name(instance_config, extra_config)
    log_utils.init_logging_stdout(level=logging.INFO)
    logging.info("Run name: %s", run_name)

    instance = OpenhailInstance(instance_config)
    agent = agent_initializer.initialize_agent(extra_config["agent"], instance)
    evaluator = evaluation_utilities.HailEvaluator(
        instance_config,
        agent,
        render=render,
        save_gif=gif_filename is not None,
        gif_filename=gif_filename or "envrender.gif",
    )
    result = evaluator.simulate()
    logging.info("Average reward: %.2f", result.average_reward())


if __name__ == "__main__":
    main()
