import logging
import random
from typing import Any, Dict

import gymnasium as gym
import numpy as np

from ..utils import geometry
from ..utils.infrastructure_utilities import ChargingInfrastructure
from .constants import (
    CHARGER,
    CLOCK_EPOCH,
    END_OF_CHARGE,
    END_OF_HORIZON,
    END_OF_REPO,
    END_OF_SERVE,
    JOB_CHARGE,
    JOB_GO_CHARGE,
    JOB_IDLE,
    JOB_NULL,
    JOB_QUEUE,
    JOB_REPO,
    JOB_SERVE,
    LOC,
    NEW_REQUEST,
    NULL_EPOCH,
    OCCUPANCY,
    QUEUE,
    REQUESTED_EPOCH,
    T_DEST,
    TARGET_CHARGER,
    TIME,
    TYPE,
    Q,
)
from .env_components import (
    RenderableVehicleManager,
    RequestManager,
    SummaryManager,
    VehicleManager,
)
from .openhail_instance import OpenhailInstance
from .state_observer import StateObserver

TIME_TOLERANCE = 1e-8
LOGGER = logging.getLogger(__name__)

VALID_DECISION_EPOCH_TYPES = {
    END_OF_HORIZON,
    REQUESTED_EPOCH,
    CLOCK_EPOCH,
    END_OF_CHARGE,
    END_OF_REPO,
    END_OF_SERVE,
    NEW_REQUEST,
}

EPOCH_TYPE_NAMES = {
    NULL_EPOCH: "NULL_EPOCH",
    END_OF_SERVE: "END_OF_SERVE",
    END_OF_CHARGE: "END_OF_CHARGE",
    END_OF_REPO: "END_OF_REPO",
    NEW_REQUEST: "NEW_REQUEST",
    CLOCK_EPOCH: "CLOCK_EPOCH",
    REQUESTED_EPOCH: "REQUESTED_EPOCH",
    END_OF_HORIZON: "END_OF_HORIZON",
}

JOB_TYPE_NAMES = {
    JOB_SERVE: "JOB_SERVE",
    JOB_REPO: "JOB_REPO",
    JOB_GO_CHARGE: "JOB_GO_CHARGE",
    JOB_QUEUE: "JOB_QUEUE",
    JOB_CHARGE: "JOB_CHARGE",
    JOB_IDLE: "JOB_IDLE",
    JOB_NULL: "JOB_NULL",
}


class OpenhailEnv(gym.Env):
    action_space: gym.spaces.Tuple
    observation_space: gym.spaces.Dict

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        instance: OpenhailInstance,
        simulation_config: dict | None = None,
        render_config: dict | None = None,
    ):
        super().__init__()

        logging.info("RidehailEnv - init")

        # Store the instance directly
        self.instance = instance

        # Extract simulation_config if not provided separately
        if simulation_config is None:
            # This should not happen in normal usage, but provide a fallback
            raise ValueError("simulation_config is required")

        self._parse_simulation_config(simulation_config)

        # Set default render_config if not provided
        if render_config is None:
            render_config = {}
        self.render_config = render_config

        # Set render mode before initializing components
        self.render_mode = render_config.get("render_mode", None)

        # Initialize request manager
        self.request_manager = RequestManager(
            instance=self.instance,
            max_outstanding_requests=self.max_outstanding_requests,
            clear_requests=self.clear_requests,
            strict_validation=self.strict_validation,
        )

        # Initialize vehicle manager
        if self.render_mode == "human":
            logging.info("Using RenderableVehicleManager for rendering")
            self.vehicle_manager = RenderableVehicleManager(
                instance=self.instance, strict_validation=self.strict_validation
            )
        else:
            logging.info("Using basic VehicleManager (no rendering)")
            self.vehicle_manager = VehicleManager(
                instance=self.instance, strict_validation=self.strict_validation
            )

        # Initialize summary manager
        self.summary_manager = SummaryManager(
            track_vehicles=self.track_vehicles,
            track_requests=self.track_requests,
            track_epochs=self.track_epochs,
        )

        # Initialize state observer
        self.state_observer = StateObserver(
            self.instance, self.max_outstanding_requests
        )
        self.observation_space = self.state_observer.observation_space
        self.action_space = self._get_action_space()

        # rendering
        self.render_mode = render_config.get("render_mode", None)
        if self.render_mode == "human":
            from .env_renderer import RidehailRenderer

            self.renderer = RidehailRenderer(render_config, self.instance)
            self.vehicle_manager.initialize_render_data()
        else:
            self.renderer = None

        self.time = 0.0
        self.total_reward = 0.0
        self.decision_epoch_type = NULL_EPOCH

        return

    # Parse the simulation configuration (separated from instance creation)
    def _parse_simulation_config(self, simulation_config):
        logging.debug("RidehailEnv - parse_simulation_config")

        self.strict_validation = simulation_config.get("strict_validation", True)
        self.max_outstanding_requests = simulation_config.get(
            "max_outstanding_requests", 1
        )
        self.clear_requests = simulation_config.get("clear_requests", True)
        self.decision_clock = simulation_config.get("decision_clock", -1)
        self.max_interdecision_time = simulation_config.get(
            "max_interdecision_time", -1
        )
        self.end_of_charge_epoch = simulation_config.get("end_of_charge_epoch", True)
        self.end_of_service_epoch = simulation_config.get("end_of_service_epoch", True)
        self.request_epoch = simulation_config.get("request_epoch", True)
        self.track_vehicles = simulation_config.get("track_vehicles", True)
        self.track_requests = simulation_config.get("track_requests", True)
        self.track_epochs = simulation_config.get("track_epochs", True)

        boolean_fields = {
            "strict_validation": self.strict_validation,
            "clear_requests": self.clear_requests,
            "end_of_charge_epoch": self.end_of_charge_epoch,
            "end_of_service_epoch": self.end_of_service_epoch,
            "request_epoch": self.request_epoch,
            "track_vehicles": self.track_vehicles,
            "track_requests": self.track_requests,
            "track_epochs": self.track_epochs,
        }
        for name, value in boolean_fields.items():
            if not isinstance(value, bool):
                raise TypeError(f"simulation.{name} must be a boolean, got {value!r}.")

        if (
            isinstance(self.max_outstanding_requests, bool)
            or not isinstance(self.max_outstanding_requests, int)
            or self.max_outstanding_requests <= 0
        ):
            raise ValueError(
                "simulation.max_outstanding_requests must be a positive integer."
            )

        for name in ("decision_clock", "max_interdecision_time"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"simulation.{name} must be numeric.")
            value = float(value)
            if value != -1.0 and value <= 0.0:
                raise ValueError(
                    f"simulation.{name} must be -1 (disabled) or strictly positive."
                )
            setattr(self, name, value)

    def step(self, action) -> tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one step of the simulation with the given action.

        Args:
            action: Tuple of (serve_action, rnr_action, planned_epoch)
                   - serve_action: Vehicle assignments for requests
                   - rnr_action: Repositioning and recharging actions
                   - planned_epoch: Time for next decision epoch

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        LOGGER.debug("RidehailEnv - step - time: %.2f", self.time)
        step_start_time = self.time
        service_reward = 0.0
        reposition_reward = 0.0
        requests_served = 0

        serve_action, rnr_action, raw_planned_epoch = action
        requested_epoch = float(np.asarray(raw_planned_epoch).item())
        planned_epoch = (
            requested_epoch if requested_epoch > 0 else self.instance.max_time + 1
        )

        for r in range(self.max_outstanding_requests):
            vehicle_idx = int(serve_action[r])
            service_reward += self._process_serve_action(vehicle_idx, r)
            requests_served += int(vehicle_idx < self.instance.num_evs)
        rnr_action = np.asarray(rnr_action)
        self.vehicle_manager.process_reposition_actions(rnr_action, self.time)

        self.vehicle_manager.allocate_available_chargers(self.time)

        while True:
            sub_epoch_type, sub_epoch_reward = self._sub_step(planned_epoch)
            reposition_reward += sub_epoch_reward

            LOGGER.debug(
                "Sub-step - type: %s, reward: %.2f",
                sub_epoch_type,
                sub_epoch_reward,
            )

            if (
                self.strict_validation
                and sub_epoch_type not in VALID_DECISION_EPOCH_TYPES
            ):
                diagnostics = self._build_invalid_epoch_diagnostics(
                    sub_epoch_type=sub_epoch_type,
                    sub_epoch_reward=sub_epoch_reward,
                    raw_planned_epoch=raw_planned_epoch,
                    planned_epoch=planned_epoch,
                    serve_action=serve_action,
                    rnr_action=rnr_action,
                )
                logging.error(diagnostics)
                raise AssertionError(diagnostics)

            if sub_epoch_type in [END_OF_HORIZON, REQUESTED_EPOCH, CLOCK_EPOCH]:
                break
            if self.request_epoch and sub_epoch_type == NEW_REQUEST:
                break
            if self.end_of_charge_epoch and sub_epoch_type == END_OF_CHARGE:
                break
            if self.end_of_service_epoch and sub_epoch_type == END_OF_SERVE:
                break

        requests_added = self._advance_requests()
        self.vehicle_manager.advance_all_vehicle_states(self.time)

        epoch_reward = service_reward + reposition_reward
        self.total_reward += epoch_reward
        self.decision_epoch_type = int(sub_epoch_type)

        vehicle_states = self.vehicle_manager.get_vehicle_states()
        self.summary_manager.track_epoch_metrics(self.time, vehicle_states[Q])
        charger_states = self.vehicle_manager.get_charger_states()
        info = {
            "epoch_type": self.decision_epoch_type,
            "delta_time": float(self.time - step_start_time),
            "reward_components": {
                "service": float(service_reward),
                "reposition": float(reposition_reward),
            },
            "requests_added": int(requests_added),
            "requests_served": int(requests_served),
            "mean_soc": float(np.mean(vehicle_states[Q]) / self.instance.ev_max_Q),
            "charger_occupancy": charger_states[OCCUPANCY].copy(),
            "charger_queue": charger_states[QUEUE].copy(),
        }

        state = self.get_state(charger_state=charger_states)
        return (
            state,
            epoch_reward,
            self.time >= self.instance.max_time,
            False,
            info,
        )

    def _sub_step(self, planned_epoch: float) -> tuple[int, float]:
        LOGGER.debug("RidehailEnv - sub_step - BEGIN - time: %.2f", self.time)

        next_time, epoch_type = self._get_next_epoch(planned_epoch)
        job_result = self.vehicle_manager.advance_to_time(next_time, self.time)
        self.summary_manager.track_vehicle_time_metrics(
            job_result.repos_time,
            job_result.charge_time,
            job_result.idle_time,
            job_result.queue_time,
        )
        self._advance_time(next_time)

        LOGGER.debug(
            "RidehailEnv - sub_step - END - next epoch type: %s, reward %s",
            epoch_type,
            job_result.reward,
        )
        return epoch_type, job_result.reward

    def _get_next_epoch(self, planned_epoch: float) -> tuple[float, int]:
        # The horizon is a real event at max_time. Requested, vehicle, clock,
        # and request epochs at or beyond it are deliberately ignored.
        next_time = float(self.instance.max_time)
        epoch_type = END_OF_HORIZON

        if planned_epoch < next_time - TIME_TOLERANCE:
            next_time = planned_epoch
            epoch_type = REQUESTED_EPOCH

        vehicle_epochs = np.asarray(self.vehicle_manager.get_next_vehicle_epochs())
        vehicle_epoch_types = np.asarray(self.vehicle_manager.get_vehicle_epoch_types())
        valid_vehicle_events = (vehicle_epoch_types != NULL_EPOCH) & np.isfinite(
            vehicle_epochs
        )
        earlier_vehicle_events = valid_vehicle_events & (
            vehicle_epochs < next_time - TIME_TOLERANCE
        )
        if np.any(earlier_vehicle_events):
            next_time = float(np.min(vehicle_epochs[earlier_vehicle_events]))
            epoch_type = self._select_vehicle_epoch_type_at_time(
                vehicle_epochs,
                vehicle_epoch_types,
                next_time,
            )

        if self.decision_clock >= 0:
            next_decision_clock = (
                self.time - (self.time % self.decision_clock) + self.decision_clock
            )
            if next_time > next_decision_clock:
                next_time = next_decision_clock
                epoch_type = CLOCK_EPOCH

        if self.max_interdecision_time >= 0:
            next_decision_clock = self.time + self.max_interdecision_time
            if next_time > next_decision_clock:
                next_time = next_decision_clock
                epoch_type = CLOCK_EPOCH

        if self.request_epoch:
            next_request_time = self.request_manager.next_request_time()
            if next_time > next_request_time:
                next_time = next_request_time
                epoch_type = NEW_REQUEST

        if self.strict_validation and epoch_type == NULL_EPOCH:
            logging.error(
                self._build_null_epoch_selection_diagnostics(
                    next_time=next_time,
                    planned_epoch=planned_epoch,
                )
            )

        return next_time, epoch_type

    def _select_vehicle_epoch_type_at_time(
        self,
        vehicle_epochs: np.ndarray,
        vehicle_epoch_types: np.ndarray,
        next_time: float,
    ) -> int:
        tied_epoch_types = vehicle_epoch_types[
            (vehicle_epoch_types != NULL_EPOCH)
            & (np.abs(vehicle_epochs - next_time) <= TIME_TOLERANCE)
        ]
        tied_epoch_types = {int(epoch_type) for epoch_type in tied_epoch_types}

        if self.end_of_service_epoch and END_OF_SERVE in tied_epoch_types:
            return END_OF_SERVE
        if self.end_of_charge_epoch and END_OF_CHARGE in tied_epoch_types:
            return END_OF_CHARGE
        if END_OF_REPO in tied_epoch_types:
            return END_OF_REPO
        if END_OF_SERVE in tied_epoch_types:
            return END_OF_SERVE
        if END_OF_CHARGE in tied_epoch_types:
            return END_OF_CHARGE

        if not tied_epoch_types:
            return NULL_EPOCH
        return min(tied_epoch_types)

    def _build_null_epoch_selection_diagnostics(
        self,
        next_time: float,
        planned_epoch: float,
    ) -> str:
        vehicle_states = self.vehicle_manager.get_vehicle_states()
        vehicle_epochs = np.asarray(self.vehicle_manager.get_next_vehicle_epochs())
        vehicle_epoch_types = np.asarray(self.vehicle_manager.get_vehicle_epoch_types())
        vehicle_job_types = np.asarray(vehicle_states[TYPE])
        waitlist = np.asarray(self.vehicle_manager.WAITLIST)

        selected = np.flatnonzero(
            np.isclose(vehicle_epochs, next_time, atol=TIME_TOLERANCE)
            & (vehicle_epoch_types == NULL_EPOCH)
        )
        null_epochs = np.flatnonzero(vehicle_epoch_types == NULL_EPOCH)
        due_or_past = np.flatnonzero(vehicle_epochs <= self.time + TIME_TOLERANCE)

        lines = [
            (
                "OpenhailEnv._get_next_epoch selected NULL_EPOCH before vehicle "
                "advancement."
            ),
            (
                f"  time={self.time:.6f}, selected_next_time={next_time:.6f}, planned_"
                f"epoch={planned_epoch}"
            ),
            f"  next_request_time={self._safe_next_request_time()}",
            f"  selected_null_epoch_vehicles={selected.tolist()}",
            f"  all_null_epoch_vehicles={null_epochs.tolist()}",
            f"  due_or_past_vehicles={due_or_past.tolist()}",
            f"  vehicle_epoch_types={self._format_epoch_types(vehicle_epoch_types)}",
            f"  vehicle_next_epochs={self._format_array(vehicle_epochs)}",
            f"  vehicle_job_types={self._format_job_types(vehicle_job_types)}",
            f"  vehicle_waitlist={self._format_array(waitlist)}",
            "  selected vehicle snapshots:",
        ]

        snapshot_ids = selected.tolist()
        if not snapshot_ids:
            snapshot_ids = null_epochs[:10].tolist()

        for v_idx in snapshot_ids:
            lines.append(
                "    "
                f"v={v_idx}: "
                f"epoch={vehicle_epochs[v_idx]:.6f}, "
                f"epoch_type={vehicle_epoch_types[v_idx]} "
                f"({self._epoch_type_name(vehicle_epoch_types[v_idx])}), "
                f"job={vehicle_job_types[v_idx]} "
                f"({self._job_type_name(vehicle_job_types[v_idx])}), "
                f"time={vehicle_states[TIME][v_idx]:.6f}, "
                f"t_dest={vehicle_states[T_DEST][v_idx]:.6f}, "
                f"q={vehicle_states[Q][v_idx]:.6f}, "
                f"waitlist={waitlist[v_idx]:.6f}, "
                f"charger={vehicle_states[CHARGER][v_idx]}, "
                f"target_charger={vehicle_states[TARGET_CHARGER][v_idx]}"
            )

        return "\n".join(lines)

    def _build_invalid_epoch_diagnostics(
        self,
        sub_epoch_type: int,
        sub_epoch_reward: float,
        raw_planned_epoch: float,
        planned_epoch: float,
        serve_action,
        rnr_action,
    ) -> str:
        vehicle_states = self.vehicle_manager.get_vehicle_states()
        vehicle_epochs = np.asarray(self.vehicle_manager.get_next_vehicle_epochs())
        vehicle_epoch_types = np.asarray(self.vehicle_manager.get_vehicle_epoch_types())
        vehicle_job_types = np.asarray(vehicle_states[TYPE])
        waitlist = np.asarray(self.vehicle_manager.WAITLIST)

        interesting = set(
            np.flatnonzero(vehicle_epoch_types == sub_epoch_type).tolist()
        )
        interesting.update(np.flatnonzero(vehicle_epoch_types == NULL_EPOCH).tolist())
        if vehicle_epochs.size:
            min_epoch = np.nanmin(vehicle_epochs)
            interesting.update(
                np.flatnonzero(
                    np.isclose(vehicle_epochs, min_epoch, atol=TIME_TOLERANCE)
                ).tolist()
            )
            interesting.update(
                np.flatnonzero(vehicle_epochs <= self.time + TIME_TOLERANCE).tolist()
            )
            interesting.update(np.flatnonzero(~np.isfinite(vehicle_epochs)).tolist())

        next_request_time = self._safe_next_request_time()
        lines = [
            "Invalid sub-epoch type returned by OpenhailEnv._sub_step().",
            (
                f"  sub_epoch_type={sub_epoch_type} "
                f"({self._epoch_type_name(sub_epoch_type)})"
            ),
            f"  sub_epoch_reward={sub_epoch_reward:.6f}",
            (
                f"  time={self.time:.6f}, raw_planned_epoch={raw_planned_epoch}, "
                f"planned_epoch={planned_epoch}"
            ),
            f"  horizon=[{self.instance.start_time}, {self.instance.max_time}]",
            (
                "  config: "
                f"strict_validation={self.strict_validation}, "
                f"request_epoch={self.request_epoch}, "
                f"end_of_charge_epoch={self.end_of_charge_epoch}, "
                f"end_of_service_epoch={self.end_of_service_epoch}, "
                f"decision_clock={self.decision_clock}, "
                f"max_interdecision_time={self.max_interdecision_time}"
            ),
            f"  next_request_time={next_request_time}",
            f"  vehicle_epoch_types={self._format_epoch_types(vehicle_epoch_types)}",
            f"  vehicle_next_epochs={self._format_array(vehicle_epochs)}",
            f"  vehicle_job_types={self._format_job_types(vehicle_job_types)}",
            f"  vehicle_waitlist={self._format_array(waitlist)}",
            f"  serve_action={self._summarize_action_array(serve_action)}",
            f"  rnr_action={self._summarize_action_array(rnr_action)}",
            "  vehicle snapshots:",
        ]

        if not interesting:
            interesting = set(range(min(self.instance.num_evs, 10)))

        for v_idx in sorted(interesting):
            lines.append(
                "    "
                f"v={v_idx}: "
                f"epoch={vehicle_epochs[v_idx]:.6f}, "
                f"epoch_type={vehicle_epoch_types[v_idx]} "
                f"({self._epoch_type_name(vehicle_epoch_types[v_idx])}), "
                f"job={vehicle_job_types[v_idx]} "
                f"({self._job_type_name(vehicle_job_types[v_idx])}), "
                f"time={vehicle_states[TIME][v_idx]:.6f}, "
                f"t_dest={vehicle_states[T_DEST][v_idx]:.6f}, "
                f"q={vehicle_states[Q][v_idx]:.6f}, "
                f"waitlist={waitlist[v_idx]:.6f}, "
                f"charger={vehicle_states[CHARGER][v_idx]}, "
                f"target_charger={vehicle_states[TARGET_CHARGER][v_idx]}"
            )

        return "\n".join(lines)

    def _safe_next_request_time(self):
        try:
            return self.request_manager.next_request_time()
        except Exception as exc:
            return f"<error reading next request time: {exc!r}>"

    @staticmethod
    def _format_array(values: np.ndarray) -> str:
        return np.array2string(values, precision=6, separator=", ", threshold=80)

    def _format_epoch_types(self, values: np.ndarray) -> str:
        return (
            "["
            + ", ".join(f"{int(v)}:{self._epoch_type_name(v)}" for v in values)
            + "]"
        )

    def _format_job_types(self, values: np.ndarray) -> str:
        return (
            "[" + ", ".join(f"{int(v)}:{self._job_type_name(v)}" for v in values) + "]"
        )

    @staticmethod
    def _summarize_action_array(action) -> str:
        try:
            arr = np.asarray(action)
            if arr.size == 0:
                return "shape=(), empty"
            return (
                f"shape={arr.shape}, dtype={arr.dtype}, "
                f"min={np.nanmin(arr)}, max={np.nanmax(arr)}, "
                f"values={np.array2string(arr, threshold=40)}"
            )
        except Exception as exc:
            return f"<unavailable: {exc!r}>"

    @staticmethod
    def _epoch_type_name(epoch_type) -> str:
        return EPOCH_TYPE_NAMES.get(int(epoch_type), "UNKNOWN_EPOCH_TYPE")

    @staticmethod
    def _job_type_name(job_type) -> str:
        return JOB_TYPE_NAMES.get(int(job_type), "UNKNOWN_JOB_TYPE")

    # Reset the environment with the given seed
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        super().reset(seed=seed)
        self._seed_substreams(seed)
        self._last_reset_seed = seed
        self._has_reset = True

        self.time = self.instance.start_time
        self.total_reward = 0.0
        self.decision_epoch_type = NULL_EPOCH
        self.request_manager.initialize_requests(self.requests_seed)
        self.vehicle_manager.initialize_vehicles(self.vehicles_seed)

        self.summary_manager.initialize_summary(
            self.instance.start_time, self.instance.max_time
        )

        return (self.get_state(), {})

    # render the current state of the environment
    def render(self):
        """Render the environment according to the render mode.

        Returns:
            If render_mode is:
                - "human": None
                - "ansi": str containing terminal-based rendering
        """
        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            if self.renderer is None:
                from .env_renderer import RidehailRenderer

                self.renderer = RidehailRenderer(self.render_config, self.instance)
                self.vehicle_manager.initialize_render_data()

            render_data = self.vehicle_manager.get_render_data()
            self.renderer.render(render_data, self.time, self.total_reward)
            return None

        elif self.render_mode == "ansi":
            return self._render_ansi()

        else:
            raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def _render_ansi(self):
        """Render a simple ASCII representation of the environment."""
        output = []
        output.append(f"Time: {self.time:.2f}")
        output.append(f"Total Reward: {self.total_reward:.2f}")

        # Add vehicle positions
        vehicle_states = self.vehicle_manager.get_vehicle_states()
        for v in range(len(vehicle_states[LOC])):
            loc = vehicle_states[LOC][v]
            charge = vehicle_states[Q][v] / self.instance.ev_max_Q
            output.append(
                f"Vehicle {v}: pos=({loc[0]:.2f}, {loc[1]:.2f}), charge={charge:.2f}"
            )

        return "\n".join(output)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()

    # generate the gym action space
    def _get_action_space(self):
        action_space = gym.spaces.Tuple(
            (
                gym.spaces.MultiDiscrete(
                    [self.instance.num_evs + 1] * self.max_outstanding_requests
                ),
                gym.spaces.MultiDiscrete(
                    [2 * self.instance.D_repo + 1] * self.instance.num_evs
                ),
                gym.spaces.Box(low=-1, high=np.inf, shape=(), dtype=np.float64),
            )
        )
        return action_space

    # ------------------------------------------------------------------------------

    # get the state of the environment
    def get_state(
        self, charger_state: Dict[str, np.ndarray] | None = None
    ) -> Dict[str, Any]:
        """Get the current state of the environment.

        Returns:
            Dictionary containing the current state observation
        """
        if charger_state is None:
            charger_state = self.vehicle_manager.get_charger_states()
        return self.state_observer.get_state(
            time=self.time,
            V=self.vehicle_manager.get_vehicle_states(),
            request_manager=self.request_manager,
            charger_state=charger_state,
            epoch_type=self.decision_epoch_type,
        )

    def _check_state(self, state: Dict[str, Any]) -> None:
        """Validate state consistency and correctness.

        This is a validation utility that performs comprehensive checks.
        Should only be used during development/testing as it can be expensive.

        Args:
            state: The state to validate
        """
        self.state_observer.check_state(state, self.strict_validation)

    # advance the environment to the next time step and verify that it is valid
    def _advance_time(self, next_time):
        # check feasibility
        if self.strict_validation and next_time < self.time - TIME_TOLERANCE:
            # Error .1 - time not advancing
            vehicle_epochs = self.vehicle_manager.get_next_vehicle_epochs()
            if min(vehicle_epochs) < self.time - TIME_TOLERANCE:
                min_idx = np.argmin(vehicle_epochs)
                raise ValueError(
                    (
                        f"Error preparing next step - next epoch of {min_idx} is "
                        f"before current epoch.\n"
                    )
                    + (
                        f"This is probably due to vehicle {min_idx} triggering an "
                        f"epoch but not receiving a repositioning/serve action from "
                        f"the agent"
                    )
                )
            if self.request_manager.next_request_time() < self.time - TIME_TOLERANCE:
                raise ValueError(
                    (
                        "Error preparing next step - next request time is before "
                        "current epoch."
                    )
                )
            else:
                raise ValueError(
                    (
                        f"Error preparing next step - agent planned epoch is before "
                        f"current epoch: {next_time} vs {self.time}"
                    )
                )

        self.time = next_time

    def _advance_requests(self):
        added_requests = self.request_manager.advance_requests(self.time)
        self.summary_manager.track_requests_added(self.time, added_requests)
        return added_requests

    # process the serve action and update the state of the serving vehicle
    def _process_serve_action(self, v_idx, r_idx):
        if v_idx == self.instance.num_evs:
            return 0
        request_info = self.request_manager.consume_request(r_idx)
        vehicle_states = self.vehicle_manager.get_vehicle_states()
        start_t = float(vehicle_states[TIME][v_idx])
        start_loc = vehicle_states[LOC][v_idx].copy()

        delta_reward = self.vehicle_manager.process_serve_action(
            v_idx, request_info, self.time
        )

        orig = request_info.orig
        process_dist = request_info.process_time
        process_t = process_dist / self.instance.ev_speed
        preprocess_dist = geometry.get_distance(start_loc, orig)
        preprocess_t = preprocess_dist / self.instance.ev_speed
        self.summary_manager.track_serve_job(
            start_t,
            process_t,
            preprocess_t,
            request_info.time,
        )

        return delta_reward

    def _seed_substreams(self, seed: int | None = None) -> None:
        """Seed request and vehicle initialization streams.

        Args:
            seed: Random seed for the simulation
        """
        if seed is None:
            self.vehicles_seed = int(self.np_random.integers(0, 2**16))
            self.requests_seed = int(self.np_random.integers(0, 2**16))
        else:
            seed_rng = random.Random(seed)
            self.vehicles_seed = seed_rng.getrandbits(16)
            self.requests_seed = seed_rng.getrandbits(16)

    def set_infrastructure(self, solution: ChargingInfrastructure) -> None:
        """Set the charging infrastructure for the simulation.

        Args:
            solution: Charging infrastructure configuration
        """
        self.instance.set_charging_infrastructure(solution)

        manager_type = (
            RenderableVehicleManager if self.render_mode == "human" else VehicleManager
        )
        self.vehicle_manager = manager_type(
            instance=self.instance,
            strict_validation=self.strict_validation,
        )
        self.state_observer = StateObserver(
            self.instance,
            self.max_outstanding_requests,
        )
        self.observation_space = self.state_observer.observation_space
        self.action_space = self._get_action_space()

        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.set_instance(self.instance)
            self.vehicle_manager.initialize_render_data()

        # Infrastructure indices are embedded in vehicle jobs. Restart an
        # initialized episode rather than carrying stale indices into the new
        # infrastructure.
        if getattr(self, "_has_reset", False):
            self.reset(seed=getattr(self, "_last_reset_seed", None))

    def get_episode_summary_list(self) -> list:
        """Get episode summary statistics.

        Returns:
            List of episode summary statistics
        """
        return self.summary_manager.get_episode_summary_list(self.instance.num_evs)

    def get_episode_summary(self) -> Dict[str, Any]:
        """Return named episode statistics for analysis and training logs."""
        return self.summary_manager.get_summary_data()
