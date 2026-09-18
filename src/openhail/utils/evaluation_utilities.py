import logging
import statistics
import time
from pathlib import Path
from typing import Any, List, Tuple, cast

import gymnasium as gym
import numpy as np

from ..agents import Agent
from ..core.openhail_env import OpenhailEnv
from ..core.openhail_instance import OpenhailInstance
from .infrastructure_utilities import ChargingInfrastructure


class SimulationResult:
    """Container for simulation results and statistics."""

    def __init__(self, rewards: float, stats: List[Any], times: float):
        self.rewards = [rewards]
        self.times = [times]
        self.stats = stats

    def average_reward(self) -> float:
        """Calculate the average reward across all simulations."""
        return np.average(self.rewards)

    def average_time(self) -> float:
        """Calculate the average computation time across all simulations."""
        return np.average(self.times)

    def print_summary(self) -> None:
        """Print a summary of the simulation results."""
        logging.info(
            (
                f"AVERAGE REWARD: {self.average_reward():0.2f} -- TIME: "
                f"{self.average_time():0.2f}"
            )
        )

    def append_result(self, rewards: float, stats: List[Any], times: float) -> None:
        """Append a new simulation result to the existing results."""
        self.rewards.append(rewards)
        self.times.append(times)
        self.stats = [x + y for x, y in zip(self.stats, stats)]

    def get_entry(self) -> List[Any]:
        """
        Get a list of statistics including reward statistics and computation times.
        """
        variance = statistics.variance(self.rewards) if len(self.rewards) > 1 else 0
        min_reward = np.min(self.rewards)
        max_reward = np.max(self.rewards)
        return [
            self.average_reward(),
            variance,
            min_reward,
            max_reward,
            self.average_time(),
        ] + self.stats


class HailEvaluator:
    """Evaluator for ride-hailing simulation environments.

    This class manages the evaluation of agents in the ride-hailing environment,
    handling multiple simulation seeds, rendering, and result collection.
    """

    def __init__(
        self,
        instance_config: dict,
        agent: Agent,
        render: bool = False,
        render_factor: float = 0.1,
        sleep_enabled: bool = True,
        save_gif: bool = False,
        gif_filename: str = "envrender.gif",
        instance: OpenhailInstance | None = None,
    ):
        """Initialize the evaluator.

        Args:
            instance_config: Configuration for the simulation instance
            agent: The agent to evaluate
            render: Whether to render the simulation
            render_factor: Number of real minutes to render one simulated day (default
                20)
            sleep_enabled: Whether to actually sleep to control framerate (default True)
            save_gif: Whether to save the rendering as a GIF
            gif_filename: Filename for the saved GIF (default: envrender.gif)
            instance: Optional prebuilt instance shared with the agent
        """
        self.render_factor = render_factor
        self.sleep_enabled = sleep_enabled
        self.save_gif = save_gif
        self.gif_filename = gif_filename
        self._last_sim_time = 0.0
        self._last_wall_time = 0.0
        self._init_environment(
            instance_config,
            agent,
            render=render,
            save_gif=save_gif,
            instance=instance,
        )

    def _init_environment(
        self,
        instance_config: dict,
        agent: Agent,
        render: bool = False,
        save_gif: bool = False,
        instance: OpenhailInstance | None = None,
    ) -> None:
        """Initialize the environment and agent.

        Args:
            instance_config: Configuration for the simulation instance
            agent: The agent to evaluate
            render: Whether to render the simulation
            save_gif: Whether to save frames for GIF creation
            instance: Optional prebuilt instance shared with the agent
        """
        self.render = render
        self.agent = agent

        # Configure seeds based on render mode
        if render:
            logging.info(f"Rendering enabled with render_factor: {self.render_factor}")
            if save_gif:
                logging.info(f"GIF saving enabled - will save to {self.gif_filename}")
            if instance_config["evaluation"]["numeval"] > 1:
                logging.warning(
                    (
                        "WARNING - numeval > 1, however, when rendering is enabled, "
                        "only one seed will be evaluated!"
                    )
                )

            self.seeds = [instance_config["evaluation"]["firsteval"]]
            render_config = {"render_mode": "human", "save_frames": save_gif}
        else:
            self.seeds = list(
                range(
                    instance_config["evaluation"]["firsteval"],
                    instance_config["evaluation"]["firsteval"]
                    + instance_config["evaluation"]["numeval"],
                )
            )
            render_config = {}

        # Agents are initialized against an OpenhailInstance.  Reuse it for
        # the environment instead of rereading and preprocessing all city
        # inputs for the same run.
        self.instance = instance or getattr(agent, "instance", None)
        if self.instance is None:
            self.instance = OpenhailInstance(instance_config)
        simulation_config = instance_config["simulation"]

        # Use gym.make with instance as a parameter
        self.env = gym.make(
            "openhail-v0",
            instance=self.instance,
            simulation_config=simulation_config,
            render_config=render_config,
        )

    @property
    def simulator(self) -> OpenhailEnv:
        """Access our simulator through Gymnasium's wrapper boundary."""
        return cast(OpenhailEnv, self.env.unwrapped)

    def set_infrastructure(self, solution: ChargingInfrastructure) -> None:
        """Set the charging infrastructure for both environment and agent.

        Args:
            solution: The charging infrastructure solution to use
        """
        # Give the agent its normal update hook, then let the environment
        # rebuild all infrastructure-dependent managers and spaces.
        self.agent.set_infrastructure(solution)
        self.simulator.set_infrastructure(solution)

    def simulate(self) -> SimulationResult:
        """Run simulations for all configured seeds.

        Returns:
            SimulationResult containing the aggregated results
        """
        return self.simulate_all_seeds()

    def simulate_all_seeds(self) -> SimulationResult:
        """Run simulations for all configured seeds and aggregate results.

        Returns:
            SimulationResult containing the aggregated results
        """
        logging.info(f"Evaluating -- seeds: {self.seeds}")
        simulation_results = None

        for i, seed in enumerate(self.seeds):
            episode_reward, episode_stats, compute_time = self.simulate_seed(seed)

            if simulation_results is None:
                simulation_results = SimulationResult(
                    episode_reward, episode_stats, compute_time
                )
            else:
                simulation_results.append_result(
                    episode_reward, episode_stats, compute_time
                )

            logging.debug(
                f"  - seed: {seed} ({i}/{len(self.seeds)})"
                + f" -- REWARD: {episode_reward:0.2f}"
                + f" -- AVERAGE REWARD: {simulation_results.average_reward():0.2f}"
            )

        # Save GIF if requested
        if self.render and self.save_gif:
            self._save_gif()

        if simulation_results is None:
            raise ValueError("Evaluation requires at least one seed")
        return simulation_results

    def simulate_next_seed(
        self, simulation_results: SimulationResult
    ) -> SimulationResult:
        """Run simulation for the next seed in sequence.

        Args:
            simulation_results: Existing simulation results to append to

        Returns:
            Updated SimulationResult
        """
        next_index = len(simulation_results.rewards)
        if next_index >= len(self.seeds):
            return simulation_results
        next_seed = self.seeds[next_index]
        episode_reward, episode_stats, compute_time = self.simulate_seed(next_seed)
        simulation_results.append_result(episode_reward, episode_stats, compute_time)
        return simulation_results

    def simulate_seed(self, seed: int) -> Tuple[float, List[float], float]:
        """Run a single simulation for a specific seed.

        Args:
            seed: The random seed to use for the simulation

        Returns:
            Tuple of (episode_reward, episode_stats, compute_time)
        """
        t0 = time.time()
        episode_reward, episode_stats = self._simulate_episode(seed=seed)
        t1 = time.time()
        return episode_reward, episode_stats, t1 - t0

    def _simulate_episode(self, seed: int = 1234) -> Tuple[float, List[float]]:
        """Run a single episode of the simulation.

        Args:
            seed: The random seed to use for the simulation

        Returns:
            Tuple of (total_reward, episode_summary)
        """
        info, _ = self.env.reset(seed=seed)
        self.agent.set_doy(self.simulator.request_manager.doy)
        self.agent.begin_episode(seed)

        done = False
        total_reward = 0.0
        self._last_sim_time = self.simulator.time
        self._last_wall_time = time.time()

        while not done:
            action = self.agent.choose_action(info)
            info, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

            if self.render:
                self._handle_rendering()

        summary = self.simulator.get_episode_summary_list()
        return total_reward, summary

    def _handle_rendering(self) -> None:
        """Handle rendering and timing for visualization.
        The render_factor is the number of real minutes to render one simulated day.
        """
        current_sim_time = self.simulator.time
        current_wall_time = time.time()
        sim_time_delta = current_sim_time - self._last_sim_time
        wall_time_delta = current_wall_time - self._last_wall_time
        # 1 day = 24*60 sim minutes
        if self.render_factor > 0:
            target_real_time = (
                sim_time_delta / (24 * 60)
            ) * self.render_factor  # convert minutes to seconds
            sleep_time = max(0, target_real_time - wall_time_delta)
            if self.sleep_enabled and sleep_time > 0:
                time.sleep(sleep_time)
        self.env.render()
        self._last_sim_time = current_sim_time
        self._last_wall_time = time.time()

    def _save_gif(self) -> None:
        """
        Save captured frames as a GIF with uniform playback rate based on frame
        timestamps.
        """
        try:
            import imageio

            gif_path = Path(self.gif_filename)
            gif_path.parent.mkdir(parents=True, exist_ok=True)

            renderer = self.simulator.renderer
            if renderer and hasattr(renderer, "get_frames"):
                frames, frame_times = renderer.get_frames()
                if frames:
                    logging.info(f"Saving {len(frames)} frames to {gif_path}...")
                    # Convert frames to uint8 if needed
                    frames_uint8 = [
                        frame.astype(np.uint8) if frame.dtype != np.uint8 else frame
                        for frame in frames
                    ]

                    # Calculate durations between frames for uniform playback
                    if len(frame_times) > 1:
                        # Calculate time deltas and convert to frame durations
                        durations = []
                        for i in range(len(frame_times) - 1):
                            # Convert simulation minutes to real time (1 day = 24*60
                            # min, playback at 10 fps = 0.1s per frame)
                            # For uniform playback, use the average time delta
                            durations.append(
                                100
                            )  # 100ms per frame for consistent 10fps playback
                        durations.append(
                            durations[-1] if durations else 100
                        )  # Last frame duration
                    else:
                        durations = [100] * len(frames_uint8)

                    imageio.mimsave(gif_path, frames_uint8, duration=durations, loop=0)
                    logging.info(
                        (
                            f"GIF saved successfully to {gif_path} ({len(frames)} "
                            f"frames at 10 fps)"
                        )
                    )
                else:
                    logging.warning("No frames were captured for GIF creation")
            else:
                logging.warning("Renderer does not support frame capture")
        except ImportError:
            logging.error(
                "imageio package not found. Install it with: pip install imageio"
            )
        except Exception as e:
            logging.error(f"Error saving GIF: {e}")
