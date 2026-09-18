"""
Configuration runner utility for streamlined multi-configuration execution.

This module provides a high-level interface for running multiple configurations
of the ridehail simulator with minimal boilerplate code.
"""

import logging
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List, Optional

import openhail.agents.agent_initializer as agent_initializer
import openhail.utils.arg_utilities as arg_utilities
import openhail.utils.evaluation_utilities as evaluation_utilities
import openhail.utils.logging_utilities as log_utils
from openhail.core.openhail_instance import OpenhailInstance


@dataclass
class ConfigurationSet:
    """Defines a set of configurations to run."""

    simulation: List[str] | str | None = None
    evaluation: List[str] | str | None = None
    city: List[str] | str | None = None
    horizon: List[str] | str | None = None
    fleet: List[str] | str | None = None
    infrastructure: List[str] | str | None = None
    agent: List[str] | str | None = None

    def __post_init__(self):
        """Convert single values to lists for consistency."""
        for field_name in [
            "simulation",
            "evaluation",
            "city",
            "horizon",
            "fleet",
            "infrastructure",
            "agent",
        ]:
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, list):
                setattr(self, field_name, [value])

    def get_all_combinations(self) -> List[Dict[str, str]]:
        """Generate all possible combinations of the configuration parameters."""
        # Get non-None parameters
        params = {}
        for field_name in [
            "simulation",
            "evaluation",
            "city",
            "horizon",
            "fleet",
            "infrastructure",
            "agent",
        ]:
            value = getattr(self, field_name)
            if value is not None:
                params[field_name] = value

        if not params:
            return []

        # Generate all combinations
        keys = params.keys()
        combinations = []
        for values in product(*params.values()):
            combinations.append(dict(zip(keys, values)))

        return combinations


class ConfigurationRunner:
    """
    High-level interface for running multiple configurations with minimal boilerplate.
    """

    def __init__(
        self,
        render: bool = False,
        save_gif: bool = False,
        gif_filename: str | None = None,
        log_level: int = logging.INFO,
        stdout_logging: bool = True,
        config_dir: str = "./data/config",
        data_root: str | None = None,
    ):
        """Initialize the configuration runner.

        Args:
            render: Whether to enable rendering
            save_gif: Whether to save the rendering as a GIF
            gif_filename: Optional custom filename for the GIF (default: envrender.gif)
            log_level: Logging level to use
            stdout_logging: Whether to log to stdout
            config_dir: Directory containing the configuration subfolders
            data_root: Optional root for relative input and checkpoint paths
        """
        self.render = render
        self.save_gif = save_gif
        self.gif_filename = gif_filename if gif_filename else "envrender.gif"
        self.log_level = log_level
        self.stdout_logging = stdout_logging
        self.config_dir = config_dir
        self.data_root = data_root
        self.results = []

    def run_single_config(
        self,
        simulation: str = "debug_config_fast",
        evaluation: str = "S1",
        city: str = "nyc",
        horizon: str = "day_2",
        fleet: str = "fleet_20",
        infrastructure: str = "S05_1_4",
        agent: str = "nearest",
        custom_callback: Optional[Callable] = None,
    ) -> evaluation_utilities.SimulationResult:
        """Run a single configuration with the specified parameters.

        Args:
            simulation: Simulation configuration name
            evaluation: Evaluation configuration name
            city: City configuration name
            horizon: Horizon configuration name
            fleet: Fleet configuration name
            infrastructure: Infrastructure configuration name
            agent: Agent configuration name
            custom_callback: Optional callback function called with (config, result)

        Returns:
            SimulationResult from the evaluation
        """
        # Set up argument parser with the configuration
        parser = arg_utilities.get_parser(
            simulation=simulation,
            evaluation=evaluation,
            city=city,
            horizon=horizon,
            fleet=fleet,
            infrastructure=infrastructure,
            agent=agent,
        )

        args = parser.parse_args([])
        instance_config, extra_config = arg_utilities.load_config(
            args, self.config_dir, data_root=self.data_root
        )
        agent_config = extra_config["agent"]

        # Set up logging
        run_name = log_utils.get_run_name(instance_config, extra_config)
        if self.stdout_logging:
            log_utils.init_logging_stdout(level=self.log_level)

        # Create instance and agent
        instance = OpenhailInstance(instance_config)
        controller = agent_initializer.initialize_agent(agent_config, instance)

        # Create evaluator and run simulation
        evaluator = evaluation_utilities.HailEvaluator(
            instance_config,
            controller,
            render=self.render,
            save_gif=self.save_gif,
            gif_filename=self.gif_filename,
            instance=instance,
        )
        try:
            result = evaluator.simulate()
        finally:
            evaluator.env.close()

        # Store result with configuration info
        config_info = {
            "simulation": simulation,
            "evaluation": evaluation,
            "city": city,
            "horizon": horizon,
            "fleet": fleet,
            "infrastructure": infrastructure,
            "agent": agent,
            "run_name": run_name,
        }

        self.results.append((config_info, result))

        # Call custom callback if provided
        if custom_callback:
            custom_callback(config_info, result)

        return result

    def run_multiple_configs(
        self,
        config_set: ConfigurationSet,
        custom_callback: Optional[Callable] = None,
        print_progress: bool = True,
    ) -> List[evaluation_utilities.SimulationResult]:
        """Run multiple configurations defined by a ConfigurationSet.

        Args:
            config_set: Set of configurations to run
            custom_callback: Optional callback function called with (config, result) for
                each run
            print_progress: Whether to print progress information

        Returns:
            List of SimulationResult objects
        """
        combinations = config_set.get_all_combinations()

        if print_progress:
            logging.info(f"Running {len(combinations)} configuration combinations...")

        results = []
        for i, config in enumerate(combinations):
            if print_progress:
                logging.info(
                    f"Running configuration {i + 1}/{len(combinations)}: {config}"
                )

            result = self.run_single_config(
                simulation=config.get("simulation", "debug_config_fast"),
                evaluation=config.get("evaluation", "S1"),
                city=config.get("city", "nyc"),
                horizon=config.get("horizon", "day_2"),
                fleet=config.get("fleet", "fleet_20"),
                infrastructure=config.get("infrastructure", "S05_1_4"),
                agent=config.get("agent", "nearest"),
                custom_callback=custom_callback,
            )

            results.append(result)

            if print_progress:
                logging.info(f"Result: Average Reward = {result.average_reward():.2f}")

        return results

    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of all results run so far.

        Returns:
            Dictionary containing summary statistics
        """
        if not self.results:
            return {"message": "No results available"}

        rewards = [result.average_reward() for _, result in self.results]

        summary = {
            "num_configurations": len(self.results),
            "average_reward": sum(rewards) / len(rewards),
            "best_reward": max(rewards),
            "worst_reward": min(rewards),
            "best_config": None,
            "worst_config": None,
        }

        # Find best and worst configurations
        best_idx = rewards.index(max(rewards))
        worst_idx = rewards.index(min(rewards))

        summary["best_config"] = self.results[best_idx][0]
        summary["worst_config"] = self.results[worst_idx][0]

        return summary

    def print_results_summary(self):
        """Log a formatted summary of all results."""
        summary = self.get_results_summary()

        if "message" in summary:
            logging.info(summary["message"])
            return

        logging.info("\n" + "=" * 60)
        logging.info("CONFIGURATION RUNNER SUMMARY")
        logging.info("=" * 60)
        logging.info(f"Total configurations run: {summary['num_configurations']}")
        logging.info(
            f"Average reward across all configs: {summary['average_reward']:.2f}"
        )
        logging.info(f"Best reward: {summary['best_reward']:.2f}")
        logging.info(f"Worst reward: {summary['worst_reward']:.2f}")

        logging.info("\nBest configuration:")
        for key, value in summary["best_config"].items():
            if key != "run_name":
                logging.info(f"  {key}: {value}")

        logging.info("\nWorst configuration:")
        for key, value in summary["worst_config"].items():
            if key != "run_name":
                logging.info(f"  {key}: {value}")
        logging.info("=" * 60)


def quick_run(**kwargs) -> evaluation_utilities.SimulationResult:
    """Quick utility function to run a single configuration with minimal setup.

    Args:
        **kwargs: Configuration parameters (simulation, evaluation, city, etc.)

    Returns:
        SimulationResult from the evaluation
    """
    runner = ConfigurationRunner()
    return runner.run_single_config(**kwargs)


def quick_compare(config_variations: Dict[str, List[str]], **base_config) -> None:
    """Quick utility to compare variations of a single parameter.

    Args:
        config_variations: Dictionary mapping parameter names to lists of values to try
        **base_config: Base configuration parameters

    Example:
        quick_compare(
            {"agent": ["nearest", "random"], "infrastructure": ["S05_1_4", "S05_2_2"]},
            simulation="debug_config_fast",
            city="nyc"
        )
    """
    # Create ConfigurationSet from base config
    config_set = ConfigurationSet(**base_config)

    # Apply variations
    for param, values in config_variations.items():
        setattr(config_set, param, values)

    # Run comparisons
    runner = ConfigurationRunner()
    runner.run_multiple_configs(config_set)
    runner.print_results_summary()
