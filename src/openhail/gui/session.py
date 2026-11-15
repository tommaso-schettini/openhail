"""Interactive playback control for OpenHail simulations.

The evaluator in :mod:`openhail.utils.evaluation_utilities` owns the step loop
and exposes no seam for pausing, stepping or changing speed while a run is in
progress, so the only way to interact with a running simulation today is to
watch the Pygame window. This module provides a thin session wrapper that owns
the loop instead, handing control decisions (play/pause, single step, speed) to
a caller while keeping the environment, agent and renderer exactly as the
evaluator constructs them.

The wrapper deliberately does not modify the environment or the evaluator: it
reuses the same ``gym.make`` registration, the same agent initialisation and the
same ``RidehailRenderer``, so an interactive run and a headless run of the same
configuration produce the same trajectory for the same seed.

Typical use, driving the session by hand::

    session = SimulationSession.from_configuration(data_root=".")
    session.start()
    while not session.finished:
        session.advance()
        if session.playback.should_draw():
            session.draw()

or, in one call, with a keyboard-driven panel::

    run_interactive(simulation="debug_config_fast")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from openhail.agents.agent_initializer import initialize_agent
from openhail.core.openhail_env import OpenhailEnv
from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils import arg_utilities

LOGGER = logging.getLogger(__name__)

# Speeds are expressed as the number of real minutes used to render one
# simulated day. A larger number means a slower run. This is the same unit as
# ``HailEvaluator.render_factor``, so the defaults stay comparable with
# ``python scripts/run_simulation.py --render``.
SPEED_PRESETS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
DEFAULT_SPEED_INDEX = SPEED_PRESETS.index(0.1)


@dataclass
class PlaybackState:
    """Mutable playback state for an interactive run.

    Kept separate from the session so that controls (keyboard handlers, test
    code, or a future web front end) can manipulate it without holding a
    reference to the environment.
    """

    playing: bool = True
    speed_index: int = DEFAULT_SPEED_INDEX
    # Number of steps still to run while paused, used by "step" controls.
    pending_steps: int = 0
    finished: bool = False
    draw_interval: float = 1.0 / 30.0

    _last_draw: float = field(default=0.0, repr=False)
    _last_step: float = field(default=0.0, repr=False)

    @property
    def render_factor(self) -> float:
        """Real minutes used to render one simulated day."""
        return SPEED_PRESETS[self.speed_index]

    @property
    def speed_label(self) -> str:
        """Human-readable speed, e.g. ``"0.10 min/day"``."""
        return f"{self.render_factor:.2f} min/day"

    def faster(self) -> None:
        """Use a larger render factor, i.e. a slower run."""
        self.speed_index = min(self.speed_index + 1, len(SPEED_PRESETS) - 1)

    def slower(self) -> None:
        """Use a smaller render factor, i.e. a faster run."""
        self.speed_index = max(self.speed_index - 1, 0)

    def toggle_play(self) -> None:
        """Flip between playing and paused."""
        self.playing = not self.playing
        self.pending_steps = 0

    def step(self, count: int = 1) -> None:
        """Advance ``count`` steps, pausing playback first."""
        self.playing = False
        self.pending_steps += max(1, int(count))

    def should_step(self) -> bool:
        """Whether the loop may take a step right now.

        While playing, a step is always allowed; the caller is responsible for
        pacing with :meth:`throttle`. While paused, only an explicitly requested
        step is allowed.
        """
        if self.finished:
            return False
        if self.playing:
            return True
        if self.pending_steps > 0:
            self.pending_steps -= 1
            return True
        return False

    def should_draw(self, now: float | None = None) -> bool:
        """Rate-limit drawing so fast runs do not spend all their time rendering."""
        now = time.time() if now is None else now
        if now - self._last_draw < self.draw_interval:
            return False
        self._last_draw = now
        return True

    def throttle(self) -> None:
        """Sleep so the simulation advances at the configured speed."""
        now = time.time()
        target = self._last_step + _real_seconds_per_step(self.render_factor)
        delay = target - now
        self._last_step = now if delay <= 0 else target
        if delay > 0:
            time.sleep(delay)


def _real_seconds_per_step(render_factor: float) -> float:
    """Placeholder pacing helper.

    The session computes the real time budget from the simulated time advanced
    by each step, because decision epochs are not evenly spaced. This function
    exists for callers that want a rough constant-rate estimate and is not used
    by :class:`SimulationSession`.
    """
    return max(0.0, render_factor) / (24 * 60)


class SimulationSession:
    """Owns an environment, an agent and the step loop for one interactive run.

    Mirrors the evaluator's construction so that results are comparable, but
    exposes the individual steps so a caller can pause, step and throttle.
    """

    def __init__(
        self,
        instance: OpenhailInstance,
        agent_config: dict[str, Any],
        simulation_config: dict[str, Any],
        seed: int | None = None,
        playback: PlaybackState | None = None,
        render: bool = True,
    ) -> None:
        self.instance = instance
        self.agent = initialize_agent(agent_config, instance)
        self.agent_config = agent_config
        self.simulation_config = simulation_config
        self.seed = seed
        self.playback = playback or PlaybackState()

        render_config = {"render_mode": "human"} if render else {"render_mode": None}
        self.env = OpenhailEnv(
            instance=instance,
            simulation_config=simulation_config,
            render_config=render_config,
        )
        self.observation: dict[str, Any] | None = None
        self.total_reward = 0.0
        self.steps = 0
        self.started = False
        self.finished = False
        self._last_sim_time = instance.start_time
        self._last_wall_time = time.time()

    @classmethod
    def from_configuration(
        cls,
        simulation: str = "debug_config_fast",
        evaluation: str = "S1",
        city: str = "nyc",
        horizon: str = "day_2",
        fleet: str = "fleet_20",
        infrastructure: str = "S05_1_4",
        agent: str = "nearest",
        config_dir: str = "./data/config",
        data_root: str | None = None,
        seed: int | None = None,
        playback: PlaybackState | None = None,
        render: bool = True,
    ) -> "SimulationSession":
        """Build a session from the same configuration names the CLI accepts."""
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
            args, config_dir, data_root=data_root
        )
        instance = OpenhailInstance(instance_config)
        return cls(
            instance=instance,
            agent_config=extra_config["agent"],
            simulation_config=instance_config["simulation"],
            seed=seed,
            playback=playback,
            render=render,
        )

    @property
    def sim_time(self) -> float:
        """Current simulated time."""
        return self.env.time

    @property
    def day(self) -> int:
        """Zero-based simulated day relative to the instance start time."""
        return int((self.env.time - self.instance.start_time) // 86400)

    def start(self) -> dict[str, Any]:
        """Reset the environment and the agent, returning the first observation."""
        seed = self.seed if self.seed is not None else 0
        observation, _ = self.env.reset(seed=seed)
        self.agent.set_doy(self.env.request_manager.doy)
        self.agent.begin_episode(seed)

        self.observation = observation
        self.total_reward = 0.0
        self.steps = 0
        self.finished = False
        self.playback.finished = False
        self.started = True
        self._last_sim_time = self.env.time
        self._last_wall_time = time.time()
        return observation

    def advance(self) -> None:
        """Take exactly one step.

        Raises ``RuntimeError`` if :meth:`start` has not been called. A step past
        the end of the episode is a no-op so that callers may loop until
        :attr:`finished` without special-casing the final iteration.
        """
        if not self.started:
            raise RuntimeError("SimulationSession.advance() called before start()")
        if self.finished:
            return

        assert self.observation is not None
        action = self.agent.choose_action(self.observation)
        observation, reward, terminated, truncated, _ = self.env.step(action)

        self.observation = observation
        self.total_reward += float(reward)
        self.steps += 1
        self.finished = bool(terminated or truncated)
        self.playback.finished = self.finished

    def step_once(self, count: int = 1) -> None:
        """Advance ``count`` steps regardless of playback state."""
        for _ in range(max(0, count)):
            self.advance()

    def draw(self) -> None:
        """Render the current state through the environment's renderer."""
        self.env.render()

    def pump(self) -> list[Any]:
        """Process pending window events.

        Returns the raw Pygame event list so a caller can map keys to playback
        controls. Imported lazily so headless use does not require a display.
        """
        import pygame

        pygame.event.pump()
        return list(pygame.event.get())

    def run(
        self,
        until: Callable[["SimulationSession"], bool] | None = None,
        max_steps: int | None = None,
        draw: bool = True,
    ) -> float:
        """Run to completion, honouring playback state.

        Returns the accumulated reward. ``until`` allows early exit, and
        ``max_steps`` caps the run so tests do not need a full horizon.
        """
        if not self.started:
            self.start()

        taken = 0
        while not self.finished:
            if until is not None and until(self):
                break
            if max_steps is not None and taken >= max_steps:
                break
            if not self.playback.should_step():
                # Paused: yield to the caller rather than spinning.
                break

            self._pace_before_step()
            self.advance()
            taken += 1
            if draw and self.playback.should_draw():
                self.draw()

        return self.total_reward

    def _pace_before_step(self) -> None:
        """Sleep so the run advances at the configured speed.

        Pacing uses the simulated time advanced by the previous step, matching
        ``HailEvaluator._handle_rendering``, because decision epochs are not
        evenly spaced.
        """
        now = time.time()
        elapsed_sim = self.env.time - self._last_sim_time
        wall = now - self._last_wall_time
        factor = self.playback.render_factor
        if factor > 0 and elapsed_sim > 0:
            budget = (elapsed_sim / (24 * 60)) * factor
            delay = budget - wall
            if delay > 0:
                time.sleep(delay)
                now = time.time()
        self._last_sim_time = self.env.time
        self._last_wall_time = now

    def close(self) -> None:
        """Release the environment and its renderer."""
        self.env.close()

    def __enter__(self) -> "SimulationSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def run_interactive(**kwargs: Any) -> float:
    """Open the control panel for one configuration and return the reward.

    Any keyword accepted by :meth:`SimulationSession.from_configuration` is
    forwarded, plus the panel-specific keys handled below.
    """
    from .control_panel import ControlPanel

    session_keys: Iterable[str] = (
        "simulation",
        "evaluation",
        "city",
        "horizon",
        "fleet",
        "infrastructure",
        "agent",
        "config_dir",
        "data_root",
        "seed",
    )
    session_kwargs = {k: v for k, v in kwargs.items() if k in session_keys}
    panel_kwargs = {k: v for k, v in kwargs.items() if k not in session_keys}

    session = SimulationSession.from_configuration(**session_kwargs)
    try:
        panel = ControlPanel(session, **panel_kwargs)
        return panel.run()
    finally:
        session.close()
