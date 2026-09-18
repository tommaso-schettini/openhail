"""Vehicle management for the OpenHail ridehail simulation."""

import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

from ...utils import geometry
from ..constants import (
    CAPACITY,
    CHARGER,
    DEST,
    END_OF_CHARGE,
    END_OF_REPO,
    END_OF_SERVE,
    EPS,
    JOB_CHARGE,
    JOB_GO_CHARGE,
    JOB_IDLE,
    JOB_NULL,
    JOB_QUEUE,
    JOB_REPO,
    JOB_SERVE,
    LOC,
    NEXT_EPOCH,
    NULL_EPOCH,
    OCCUPANCY,
    Q_DEST,
    QUEUE,
    T_DEST,
    TARGET_CHARGER,
    TIME,
    TYPE,
    WAYPOINT,
    Q,
)

if TYPE_CHECKING:
    from ..openhail_instance import OpenhailInstance
    from .request_manager import RequestInfo


@dataclass
class JobResult:
    """Result of a job operation including reward and time metrics."""

    reward: float
    repos_time: float = 0.0
    charge_time: float = 0.0
    idle_time: float = 0.0
    queue_time: float = 0.0


class VehicleManager:
    """Manages all vehicle state and operations in the ridehail simulation."""

    def __init__(self, instance: "OpenhailInstance", strict_validation: bool = True):
        """Initialize the vehicle manager.

        Args:
            instance: The simulation instance containing vehicle and infrastructure data
            strict_validation: Whether to enable strict validation checks
        """
        self.instance = instance
        self.strict_validation = strict_validation

        # Vehicle state arrays - will be initialized in initialize_vehicles()
        self.V: Dict[str, np.ndarray]
        self.WAITLIST: np.ndarray
        self.epoch_type: np.ndarray

        # Charger occupancy tracking
        self.current_occupancy: np.ndarray
        self._charger_locations = np.asarray(
            self.instance.repo_coords, dtype=np.float64
        )
        self._charger_capacity = np.asarray(self.instance.charger_count, dtype=np.int32)

    def initialize_vehicles(self, vehicles_seed: int) -> None:
        """Initialize all vehicles with random starting positions and charges.

        Args:
            vehicles_seed: Random seed for vehicle initialization
        """
        # Initialize vehicle state arrays
        self.V = {
            LOC: np.zeros((self.instance.num_evs, 2), dtype=np.float64),
            TIME: np.zeros(self.instance.num_evs, dtype=np.float64),
            Q: np.zeros(self.instance.num_evs, dtype=np.float64),
            DEST: np.zeros((self.instance.num_evs, 2), dtype=np.float64),
            T_DEST: np.zeros(self.instance.num_evs, dtype=np.float64),
            Q_DEST: np.zeros(self.instance.num_evs, dtype=np.float64),
            CHARGER: np.zeros(self.instance.num_evs, dtype=np.int32),
            TARGET_CHARGER: np.zeros(self.instance.num_evs, dtype=np.int32),
            TYPE: np.zeros(self.instance.num_evs, dtype=np.int32),
            NEXT_EPOCH: np.zeros(self.instance.num_evs, dtype=np.float64),
        }

        self.WAITLIST = -1.0 * np.ones(self.instance.num_evs, dtype=np.float64)
        self.epoch_type = np.zeros(self.instance.num_evs, dtype=np.int32)

        # Initialize charger occupancy
        self.current_occupancy = np.zeros(self.instance.D_repo, dtype=np.int32)

        # Set random starting states for vehicles
        v_rng = random.Random(vehicles_seed)
        init_weights = self.instance.parking_lot_weights

        for v_idx in range(self.instance.num_evs):
            initial_Q = v_rng.uniform(0.5, 1) * self.instance.ev_max_Q
            p_lot = v_rng.choices(range(self.instance.D), weights=init_weights, k=1)[0]
            initial_LOC = self.instance.lot_coords[p_lot]

            self.V[CHARGER][v_idx] = -1
            self.V[TARGET_CHARGER][v_idx] = -1
            self.V[TYPE][v_idx] = JOB_IDLE
            self.V[NEXT_EPOCH][v_idx] = self.instance.max_time + 1
            self.V[DEST][v_idx] = self.instance.midpoint

            self.V[LOC][v_idx] = initial_LOC
            self.V[TIME][v_idx] = self.instance.start_time
            self.V[Q][v_idx] = initial_Q

    def initialize_render_data(self) -> None:
        """Initialize render data structures for visualization.
        Base implementation does nothing - override in subclasses if needed.
        """
        pass

    def get_vehicle_states(self) -> Dict[str, np.ndarray]:
        """Get current vehicle states.

        Returns:
            Dictionary containing all vehicle state arrays
        """
        return self.V

    def get_charger_states(self) -> Dict[str, np.ndarray]:
        """Return public per-location charging capacity, occupancy, and queues."""
        queue = np.zeros(self.instance.D_repo, dtype=np.int32)
        queued = self.V[TYPE] == JOB_QUEUE
        queue_targets = self.V[TARGET_CHARGER][queued]
        queue_targets = queue_targets[
            (queue_targets >= 0) & (queue_targets < self.instance.D_repo)
        ]
        if queue_targets.size:
            np.add.at(queue, queue_targets, 1)

        return {
            LOC: self._charger_locations.copy(),
            CAPACITY: self._charger_capacity.copy(),
            OCCUPANCY: np.asarray(self.current_occupancy, dtype=np.int32).copy(),
            QUEUE: queue,
        }

    def get_render_data(self) -> Optional[Dict[str, Any]]:
        """Get render data for visualization.

        For the base VehicleManager, this provides approximated render data
        based on current vehicle states.

        Returns:
            Dictionary containing approximated render data
        """
        if not hasattr(self, "V"):
            return None

        # Create approximated render data from current vehicle states
        render_data = {
            LOC: self.V[LOC].copy(),
            Q: self.V[Q] / self.instance.ev_max_Q,  # Normalize charge for rendering
            WAYPOINT: [],
            TIME: [],
        }

        # For each vehicle, create a simple waypoint path
        for v_idx in range(self.instance.num_evs):
            waypoints = deque([self.V[LOC][v_idx].copy()])
            times = deque([self.V[TIME][v_idx]])

            # If vehicle has a destination different from current location, add it
            if not np.allclose(self.V[DEST][v_idx], self.V[LOC][v_idx]):
                waypoints.append(self.V[DEST][v_idx].copy())
                times.append(self.V[T_DEST][v_idx])

            render_data[WAYPOINT].append(waypoints)
            render_data[TIME].append(times)

        return render_data

    def get_next_vehicle_epochs(self) -> np.ndarray:
        """Get the next epoch times for all vehicles.

        Returns:
            Array of next epoch times for each vehicle
        """
        return self.V[NEXT_EPOCH]

    def get_vehicle_epoch_types(self) -> np.ndarray:
        """Get the epoch types for all vehicles.

        Returns:
            Array of epoch types for each vehicle
        """
        return self.epoch_type

    def process_serve_action(
        self, v_idx: int, request_info: "RequestInfo", current_time: float
    ) -> float:
        """Process a serve action for a vehicle.

        Args:
            v_idx: Vehicle index
            request_info: Information about the request to serve
            current_time: Current simulation time

        Returns:
            Reward delta from serving the request
        """
        # Vehicle data
        start_t = self.V[TIME][v_idx]
        start_q = self.V[Q][v_idx]
        start_loc = self.V[LOC][v_idx].copy()

        # Request data
        request_time = request_info.time
        orig = request_info.orig
        dest = request_info.dest
        process_dist = request_info.process_time

        # Operation costs
        process_t = process_dist / self.instance.ev_speed
        process_q = process_dist * self.instance.ev_discharge_rate
        preprocess_dist = geometry.get_distance(start_loc, orig)
        preprocess_t = preprocess_dist / self.instance.ev_speed
        preprocess_q = preprocess_dist * self.instance.ev_discharge_rate

        # Waypoints
        mid_t = start_t + preprocess_t
        end_t = start_t + preprocess_t + process_t
        end_q = start_q - preprocess_q - process_q

        # Validate feasibility
        self._validate_serve_job_feasibility(v_idx, request_time, mid_t, end_q)

        # Release vehicle from charger if currently charging
        if self.V[TYPE][v_idx] == JOB_CHARGE:
            self._release_vehicle_from_charger(v_idx)

        # Assign vehicle to serve job
        self._assign_serve_job(v_idx, dest, end_t, end_q)

        # Calculate reward
        delta_reward = self._calculate_serve_job_reward(
            orig, dest, preprocess_t, process_t
        )

        # Update render data - override in subclasses if needed
        self._update_serve_render_data(
            v_idx, start_t, start_loc, mid_t, end_t, orig, dest
        )

        return delta_reward

    def process_reposition_action(
        self, v_idx: int, action_value: int, current_time: float
    ) -> None:
        """Process a repositioning action for a vehicle.

        Args:
            v_idx: Vehicle index
            action_value: The repositioning action value
            current_time: Current simulation time
        """
        # No-Op
        if action_value == 0:
            return

        # Parse action parameters
        should_charge = action_value % 2 == 0
        target_charger = self._get_loc_id(action_value)

        if self.V[TARGET_CHARGER][v_idx] != target_charger:
            # Vehicle needs to move to a new location
            orig = self.V[LOC][v_idx]
            dest = self.instance.repo_coords[target_charger]

            # Idle and queued vehicles can retain the time at which their current
            # state began. A new repositioning request starts no earlier than the
            # current decision epoch. Vehicles finishing a future job retain that
            # later availability time so the repositioning remains sequential.
            start_t = max(self.V[TIME][v_idx], current_time)
            start_q = self.V[Q][v_idx]
            repo_t = self.instance.time(orig, dest)
            repo_q = self.instance.charge(orig, dest)
            end_t = start_t + repo_t
            end_q = start_q - repo_q

            # Validate feasibility
            self._validate_repo_job_feasibility(v_idx, target_charger, end_q)

            # Release vehicle from charger if currently charging
            if self.V[TYPE][v_idx] == JOB_CHARGE:
                self._release_vehicle_from_charger(v_idx)

            # Assign vehicle to repositioning job
            self._assign_repo_job(
                v_idx,
                dest,
                start_t,
                end_t,
                end_q,
                should_charge,
                target_charger,
            )

            # Update render data - override in subclasses if needed
            self._update_repo_render_data(v_idx, start_t, orig, end_t, dest)

        else:
            # Vehicle is already at target location, just change charging state
            self._handle_vehicle_charging_state_change(
                v_idx, should_charge, current_time
            )

    def process_reposition_actions(
        self,
        action_values: np.ndarray,
        current_time: float,
    ) -> None:
        """Process only actions that can change the corresponding vehicle."""
        action_values = np.asarray(action_values)
        active = np.flatnonzero(action_values)
        if active.size == 0:
            return

        selected = np.asarray(action_values[active], dtype=np.int64)
        needs_processing = np.ones(active.size, dtype=bool)
        valid = (selected > 0) & (selected <= 2 * self.instance.D_repo)
        if np.any(valid):
            vehicles = active[valid]
            valid_actions = selected[valid]
            targets = (valid_actions - 1) // 2
            same_target = self.V[TARGET_CHARGER][vehicles] == targets
            job_types = self.V[TYPE][vehicles]
            should_charge = valid_actions % 2 == 0

            charging_change = should_charge & (
                (job_types == JOB_REPO)
                | (
                    (job_types == JOB_IDLE)
                    & (self.V[Q][vehicles] < self.instance.ev_max_Q - 0.1)
                )
            )
            noncharging_change = (~should_charge) & (
                (job_types == JOB_GO_CHARGE)
                | (job_types == JOB_QUEUE)
                | (job_types == JOB_CHARGE)
            )
            needs_processing[valid] = (
                ~same_target | charging_change | noncharging_change
            )

        for position in np.flatnonzero(needs_processing):
            self.process_reposition_action(
                int(active[position]),
                int(selected[position]),
                current_time,
            )

    def allocate_charger(self, v_idx: int, current_time: float) -> None:
        """Try to allocate a charger to a queuing vehicle.

        Args:
            v_idx: Vehicle index
            current_time: Current simulation time
        """
        # Only queuing vehicles are processed here
        if self.V[TYPE][v_idx] != JOB_QUEUE:
            return

        # Verify consistency
        if self.strict_validation:
            if self.V[TARGET_CHARGER][v_idx] == -1:
                raise RuntimeError(f"Queued vehicle {v_idx} has no target charger.")
            if self.V[NEXT_EPOCH][v_idx] != self.instance.max_time + 1:
                raise RuntimeError(
                    f"Queued vehicle {v_idx} has an active vehicle epoch."
                )
            if self.WAITLIST[v_idx] == -1:
                raise RuntimeError(
                    f"Queued vehicle {v_idx} is absent from the waitlist."
                )

        lot_id = self.V[TARGET_CHARGER][v_idx]
        station_capacity = self.instance.charger_count[lot_id]

        # Check if there is residual capacity on the charger
        if self.current_occupancy[lot_id] >= station_capacity:
            return

        # Calculate charging parameters and assign vehicle to charging job
        q_start = self.V[Q][v_idx]
        t_start = current_time
        t_end = t_start + self._get_charging_time(q_start)

        self._assign_charge_job(v_idx, lot_id, t_start, t_end)

    def allocate_available_chargers(self, current_time: float) -> None:
        """Allocate available posts to queued vehicles in FIFO order.

        Only queued vehicles participate in the ordering.  The former event
        loop sorted the complete fleet before calling ``allocate_charger`` on
        every vehicle, even though the method immediately returns for all
        non-queued vehicles.
        """
        residual_capacity = np.maximum(
            self.instance.charger_count - self.current_occupancy,
            0,
        )
        if not np.any(residual_capacity > 0):
            return

        queued = np.flatnonzero(self.V[TYPE] == JOB_QUEUE)
        if queued.size == 0:
            return

        targets = self.V[TARGET_CHARGER][queued]
        valid_targets = (targets >= 0) & (targets < self.instance.D_repo)
        if not np.all(valid_targets):
            # Preserve the former validation and failure behavior for malformed
            # queue states rather than hiding them in the optimized path.
            order = np.argsort(self.WAITLIST[queued], kind="stable")
            for v_idx in queued[order]:
                self.allocate_charger(int(v_idx), current_time)
            return

        can_allocate = residual_capacity[targets] > 0
        queued = queued[can_allocate]
        targets = targets[can_allocate]
        if queued.size == 0:
            return

        queued_per_station = np.bincount(
            targets,
            minlength=self.instance.D_repo,
        )
        allocations_remaining = int(
            np.minimum(residual_capacity, queued_per_station).sum()
        )
        order = np.argsort(self.WAITLIST[queued], kind="stable")
        for position in order:
            lot_id = int(targets[position])
            if residual_capacity[lot_id] <= 0:
                continue
            self.allocate_charger(int(queued[position]), current_time)
            residual_capacity[lot_id] -= 1
            allocations_remaining -= 1
            if allocations_remaining == 0:
                break

    def advance_to_time(self, next_time: float, current_time: float) -> JobResult:
        """Advance aggregate activity accounting and complete due jobs.

        Between consecutive simulator events no vehicle changes job type.
        Activity times and operating costs can therefore be accumulated once
        from vectorized state counts.  Only vehicles whose jobs complete at
        ``next_time`` and vehicles waiting for chargers require individual
        processing.
        """
        duration = next_time - current_time
        if self.strict_validation and duration < -EPS:
            raise RuntimeError(
                f"Cannot advance vehicles backwards from {current_time} to {next_time}."
            )
        duration = max(0.0, duration)

        job_types = self.V[TYPE]
        repos_time = duration * np.count_nonzero(
            (job_types == JOB_REPO) | (job_types == JOB_GO_CHARGE)
        )
        charge_time = duration * np.count_nonzero(job_types == JOB_CHARGE)
        idle_time = duration * np.count_nonzero(job_types == JOB_IDLE)
        queue_time = duration * np.count_nonzero(job_types == JOB_QUEUE)

        travel_cost = repos_time * self.instance.ev_speed * self.instance.cost_travel
        charging_cost = (
            charge_time * self.instance.ev_charge_rate * self.instance.cost_charge
        )
        result = JobResult(
            reward=-(travel_cost + charging_cost),
            repos_time=float(repos_time),
            charge_time=float(charge_time),
            idle_time=float(idle_time),
            queue_time=float(queue_time),
        )

        # JOB_NULL marks a service completion that was not exposed as a
        # decision epoch.  The previous loop converted it to idle on the next
        # internal event, so retain that transition without scanning all EVs.
        stale_service_completions = np.flatnonzero(job_types == JOB_NULL)

        active_job = (
            (job_types == JOB_REPO)
            | (job_types == JOB_GO_CHARGE)
            | (job_types == JOB_CHARGE)
            | (job_types == JOB_SERVE)
        )
        due = np.flatnonzero(active_job & (self.V[NEXT_EPOCH] <= next_time + EPS))

        # Complete all due jobs before allocating newly available charging
        # posts.  This ensures simultaneous charge completions release their
        # capacity before the FIFO queue is considered.
        for v_idx in due:
            v_idx = int(v_idx)
            job_type = int(self.V[TYPE][v_idx])
            self._validate_job_completion_time(v_idx, self.V[T_DEST][v_idx], next_time)

            if job_type == JOB_SERVE:
                self.V[TYPE][v_idx] = JOB_NULL
                self.V[NEXT_EPOCH][v_idx] = self.instance.max_time + 1
                self.epoch_type[v_idx] = NULL_EPOCH
                continue

            self._complete_vehicle_job(v_idx)
            if self.WAITLIST[v_idx] != -1:
                # First make every arriving EV a queue candidate.  Allocation
                # happens globally below so older waiters cannot be bypassed.
                self._assign_queue_job(v_idx, next_time)
            else:
                self._assign_idle_job(v_idx, next_time)

        for v_idx in stale_service_completions:
            self._assign_idle_job(int(v_idx), next_time)

        self.allocate_available_chargers(next_time)

        # Preserve the public-state convention that stationary vehicle
        # timestamps equal the most recently processed simulator time.
        stationary = (self.V[TYPE] == JOB_IDLE) | (self.V[TYPE] == JOB_QUEUE)
        self.V[TIME][stationary] = next_time
        self.V[T_DEST][stationary] = next_time

        return result

    def advance_job_queue(
        self, v_idx: int, next_time: float, current_time: float
    ) -> JobResult:
        """Advance the job queue for a vehicle and return metrics.

        Args:
            v_idx: Vehicle index
            next_time: The next simulation time
            current_time: Current simulation time

        Returns:
            JobResult containing reward and time metrics
        """
        result = JobResult(reward=0.0)

        if self.V[T_DEST][v_idx] <= next_time + EPS:
            # Current job is finishing - complete it and assign next job
            arrive_time = self.V[T_DEST][v_idx]

            # Calculate time metrics for the completed job
            if self.V[TYPE][v_idx] == JOB_REPO or self.V[TYPE][v_idx] == JOB_GO_CHARGE:
                self._validate_job_completion_time(v_idx, arrive_time, next_time)
                result.repos_time = arrive_time - current_time
            elif self.V[TYPE][v_idx] == JOB_CHARGE:
                self._validate_job_completion_time(v_idx, arrive_time, next_time)
                result.charge_time = arrive_time - current_time
            elif self.V[TYPE][v_idx] == JOB_QUEUE:
                self._validate_job_completion_time(v_idx, arrive_time, current_time)
                result.queue_time = arrive_time - current_time
            elif self.V[TYPE][v_idx] == JOB_IDLE:
                self._validate_job_completion_time(v_idx, arrive_time, current_time)
                result.idle_time = arrive_time - current_time
            elif self.V[TYPE][v_idx] == JOB_SERVE:
                self._validate_job_completion_time(v_idx, arrive_time, next_time)
                self.V[TYPE][v_idx] = JOB_NULL
                self.V[NEXT_EPOCH][v_idx] = self.instance.max_time + 1
                self.epoch_type[v_idx] = NULL_EPOCH
                return result

            # Complete the current job
            self._complete_vehicle_job(v_idx)

            # Assign next job based on waitlist status
            if self.WAITLIST[v_idx] != -1:
                # Vehicle wants to charge - try to assign to charger
                self._try_assign_vehicle_to_charger(v_idx, next_time)
            else:
                # Vehicle doesn't want to charge - assign to idle
                if self.V[CHARGER][v_idx] != -1:
                    self.current_occupancy[self.V[TARGET_CHARGER][v_idx]] -= 1
                self._assign_idle_job(v_idx, next_time)

        else:
            # Job is still in progress - calculate partial time metrics
            repos_time, charge_time, idle_time, queue_time = (
                self._calculate_job_time_metrics(v_idx, next_time, current_time)
            )
            result.repos_time = repos_time
            result.charge_time = charge_time
            result.idle_time = idle_time
            result.queue_time = queue_time

        # Travel cost is distance-based; charging cost is energy-based. With
        # charge_rate expressed in kWh/s, charge_time * charge_rate is the
        # energy acquired during this interval.
        travel_cost = (
            result.repos_time * self.instance.ev_speed * self.instance.cost_travel
        )
        charging_cost = (
            result.charge_time
            * self.instance.ev_charge_rate
            * self.instance.cost_charge
        )
        result.reward = -(travel_cost + charging_cost)
        return result

    def _validate_job_completion_time(
        self,
        v_idx: int,
        actual_time: float,
        expected_time: float,
    ) -> None:
        if self.strict_validation and not np.isclose(actual_time, expected_time):
            raise RuntimeError(
                f"Vehicle {v_idx} job completes at {actual_time}, "
                f"expected {expected_time}."
            )

    def advance_vehicle_state(self, v_idx: int, current_time: float) -> None:
        """Advance the state of a single vehicle to the current time.

        Args:
            v_idx: Vehicle index
            current_time: Current simulation time
        """
        if self.V[TIME][v_idx] < current_time:
            if self.V[TYPE][v_idx] == JOB_REPO or self.V[TYPE][v_idx] == JOB_GO_CHARGE:
                self.V[LOC][v_idx] = self._interpolate(
                    current_time,
                    self.V[TIME][v_idx],
                    self.V[LOC][v_idx],
                    self.V[T_DEST][v_idx],
                    self.V[DEST][v_idx],
                )
            if self.V[TYPE][v_idx] in [JOB_REPO, JOB_GO_CHARGE, JOB_CHARGE]:
                self.V[Q][v_idx] = self._interpolate(
                    current_time,
                    self.V[TIME][v_idx],
                    self.V[Q][v_idx],
                    self.V[T_DEST][v_idx],
                    self.V[Q_DEST][v_idx],
                )
                self.V[TIME][v_idx] = current_time

        # Update render data - override in subclasses if needed
        self._update_render_state(v_idx, current_time)

    def advance_all_vehicle_states(self, current_time: float) -> None:
        """Vectorize interpolation of all moving and charging vehicles."""
        before_current = self.V[TIME] < current_time
        moving = before_current & (
            (self.V[TYPE] == JOB_REPO) | (self.V[TYPE] == JOB_GO_CHARGE)
        )
        charge_changing = before_current & (moving | (self.V[TYPE] == JOB_CHARGE))

        if np.any(moving):
            start_time = self.V[TIME][moving]
            end_time = self.V[T_DEST][moving]
            fraction = (current_time - start_time) / (end_time - start_time)
            fraction = np.clip(fraction, 0.0, 1.0)
            self.V[LOC][moving] += (
                self.V[DEST][moving] - self.V[LOC][moving]
            ) * fraction[:, None]

        if np.any(charge_changing):
            start_time = self.V[TIME][charge_changing]
            end_time = self.V[T_DEST][charge_changing]
            fraction = (current_time - start_time) / (end_time - start_time)
            fraction = np.clip(fraction, 0.0, 1.0)
            self.V[Q][charge_changing] += (
                self.V[Q_DEST][charge_changing] - self.V[Q][charge_changing]
            ) * fraction
            self.V[TIME][charge_changing] = current_time

        self._update_all_render_states(current_time)

    # ============================
    # == PRIVATE HELPER METHODS ==
    # ============================

    def _interpolate(
        self, next_time: float, t1: float, x1: np.ndarray, t2: float, x2: np.ndarray
    ) -> np.ndarray:
        """Interpolate between two states at given times.

        Args:
            next_time: Target time for interpolation
            t1: Start time
            x1: Start state
            t2: End time
            x2: End state

        Returns:
            Interpolated state at next_time
        """
        return x1 + (x2 - x1) * (next_time - t1) / (t2 - t1)

    def _get_loc_id(self, action_value: int) -> int:
        """Convert action value to location ID.

        Args:
            action_value: The action value

        Returns:
            Location ID
        """
        return (action_value - 1) // 2

    def _get_charging_time(self, initial_charge: float) -> float:
        """Calculate time needed to charge from initial_charge to full.

        Args:
            initial_charge: Starting charge level

        Returns:
            Time needed to reach full charge
        """
        return (self.instance.ev_max_Q - initial_charge) / self.instance.ev_charge_rate

    def _release_vehicle_from_charger(self, v_idx: int) -> None:
        """Release a vehicle from its current charger and update occupancy.

        Args:
            v_idx: Vehicle index
        """
        if self.V[CHARGER][v_idx] != -1:
            self.current_occupancy[self.V[CHARGER][v_idx]] -= 1
            self.V[CHARGER][v_idx] = -1

    def _assign_serve_job(
        self, v_idx: int, dest: np.ndarray, end_time: float, end_charge: float
    ) -> None:
        """Assign a vehicle to a serve job.

        Args:
            v_idx: Vehicle index
            dest: Destination location
            end_time: Job completion time
            end_charge: Final charge level
        """
        self.V[TYPE][v_idx] = JOB_SERVE
        self.WAITLIST[v_idx] = -1

        self.V[LOC][v_idx] = dest
        self.V[Q][v_idx] = end_charge
        self.V[TIME][v_idx] = end_time
        self.V[DEST][v_idx] = dest
        self.V[Q_DEST][v_idx] = end_charge
        self.V[T_DEST][v_idx] = end_time

        self.V[NEXT_EPOCH][v_idx] = end_time
        self.epoch_type[v_idx] = END_OF_SERVE

        self.V[CHARGER][v_idx] = -1
        self.V[TARGET_CHARGER][v_idx] = -1

    def _assign_repo_job(
        self,
        v_idx: int,
        dest: np.ndarray,
        start_time: float,
        end_time: float,
        end_charge: float,
        should_charge: bool,
        target_charger: int,
    ) -> None:
        """Assign a vehicle to a repositioning job.

        Args:
            v_idx: Vehicle index
            dest: Destination location
            start_time: Job dispatch time
            end_time: Job completion time
            end_charge: Final charge level
            should_charge: Whether vehicle should charge after repositioning
            target_charger: Target charger ID
        """
        self.V[TYPE][v_idx] = JOB_GO_CHARGE if should_charge else JOB_REPO
        self.WAITLIST[v_idx] = end_time if should_charge else -1

        # LOC and Q describe the state at the beginning of this leg. TIME must
        # therefore be the actual dispatch time, particularly when an idle or
        # queued vehicle is redirected after retaining an older state timestamp.
        self.V[TIME][v_idx] = start_time
        self.V[DEST][v_idx] = dest
        self.V[Q_DEST][v_idx] = end_charge
        self.V[T_DEST][v_idx] = end_time

        self.V[NEXT_EPOCH][v_idx] = end_time
        self.epoch_type[v_idx] = END_OF_REPO

        self.V[CHARGER][v_idx] = -1
        self.V[TARGET_CHARGER][v_idx] = target_charger

    def _assign_charge_job(
        self, v_idx: int, lot_id: int, start_time: float, end_time: float
    ) -> None:
        """Assign a vehicle to a charging job.

        Args:
            v_idx: Vehicle index
            lot_id: Charger location ID
            start_time: Charging start time
            end_time: Charging completion time
        """
        self.V[TYPE][v_idx] = JOB_CHARGE
        self.V[TIME][v_idx] = start_time
        self.V[Q_DEST][v_idx] = self.instance.ev_max_Q
        self.V[T_DEST][v_idx] = end_time

        self.V[NEXT_EPOCH][v_idx] = end_time
        self.epoch_type[v_idx] = END_OF_CHARGE
        self.WAITLIST[v_idx] = -1

        self.V[CHARGER][v_idx] = lot_id
        self.current_occupancy[lot_id] += 1

    def _assign_queue_job(self, v_idx: int, current_time: float) -> None:
        """Assign a vehicle to queue for charging.

        Args:
            v_idx: Vehicle index
            current_time: Current simulation time
        """
        self.V[TYPE][v_idx] = JOB_QUEUE
        self.V[TIME][v_idx] = current_time
        self.V[T_DEST][v_idx] = current_time
        self.V[NEXT_EPOCH][v_idx] = self.instance.max_time + 1
        self.epoch_type[v_idx] = NULL_EPOCH

    def _assign_idle_job(self, v_idx: int, current_time: float) -> None:
        """Assign a vehicle to idle state.

        Args:
            v_idx: Vehicle index
            current_time: Current simulation time
        """
        self.V[TYPE][v_idx] = JOB_IDLE
        self.V[TIME][v_idx] = current_time
        self.V[T_DEST][v_idx] = current_time
        self.V[NEXT_EPOCH][v_idx] = self.instance.max_time + 1
        self.epoch_type[v_idx] = NULL_EPOCH

    def _complete_vehicle_job(self, v_idx: int) -> None:
        """Complete the current job for a vehicle and update its state.

        Args:
            v_idx: Vehicle index
        """
        if self.V[TYPE][v_idx] in [JOB_REPO, JOB_GO_CHARGE]:
            self.V[LOC][v_idx] = self.V[DEST][v_idx]
            self.V[Q][v_idx] = self.V[Q_DEST][v_idx]
        elif self.V[TYPE][v_idx] == JOB_CHARGE:
            self._release_vehicle_from_charger(v_idx)
            self.V[Q][v_idx] = self.V[Q_DEST][v_idx]
        elif self.V[TYPE][v_idx] == JOB_SERVE:
            self.V[TYPE][v_idx] = JOB_NULL
            self.V[NEXT_EPOCH][v_idx] = self.instance.max_time + 1
            self.epoch_type[v_idx] = NULL_EPOCH

    def _validate_serve_job_feasibility(
        self, v_idx: int, request_time: float, mid_t: float, end_q: float
    ) -> None:
        """Validate that a serve job is feasible.

        Args:
            v_idx: Vehicle index
            request_time: Time when the request was made
            mid_t: Time when vehicle arrives at pickup location
            end_q: Vehicle charge after completing the job

        Raises:
            ValueError: If the job is not feasible and strict_validation is enabled
        """
        if not self.strict_validation:
            return

        # Check if vehicle arrives within passenger wait time
        if mid_t > request_time + self.instance.passenger_max_wait:
            raise ValueError(
                f"Vehicle {v_idx} cannot serve request: arrives at {mid_t:.2f}, "
                + (
                    f"but passenger max wait time expires at "
                    f"{request_time + self.instance.passenger_max_wait:.2f}"
                )
            )

        # Check if vehicle has sufficient charge
        if end_q < 0:
            raise ValueError(
                f"Vehicle {v_idx} cannot serve request: insufficient charge "
                + f"(would end with {end_q:.2f} charge units)"
            )

    def _validate_repo_job_feasibility(
        self, v_idx: int, target_charger: int, end_q: float
    ) -> None:
        """Validate that a repositioning job is feasible.

        Args:
            v_idx: Vehicle index
            target_charger: Target charger location ID
            end_q: Vehicle charge after repositioning

        Raises:
            ValueError: If the repositioning is not feasible and strict_validation is
                enabled
        """
        if not self.strict_validation:
            return

        if end_q < 0:
            raise ValueError(
                (
                    f"Vehicle {v_idx} cannot reach charger {target_charger}: "
                    f"insufficient charge "
                )
                + f"(would end with {end_q:.2f} charge units)"
            )

    def _calculate_serve_job_reward(
        self, orig: np.ndarray, dest: np.ndarray, preprocess_t: float, process_t: float
    ) -> float:
        """Calculate the reward for a serve job.

        Args:
            orig: Origin location
            dest: Destination location
            preprocess_t: Time to reach pickup
            process_t: Time to complete service

        Returns:
            Net reward for the serve job
        """
        delta_reward = self.instance.request_value(orig, dest)
        delta_reward -= (
            (preprocess_t + process_t)
            * self.instance.ev_speed
            * self.instance.cost_travel
        )
        return delta_reward

    def _handle_vehicle_charging_state_change(
        self, v_idx: int, should_charge: bool, current_time: float
    ) -> None:
        """Handle changes to a vehicle's charging intention.

        Args:
            v_idx: Vehicle index
            should_charge: Whether the vehicle should charge
            current_time: Current simulation time
        """
        if should_charge:
            if self.V[TYPE][v_idx] == JOB_REPO:
                self.V[TYPE][v_idx] = JOB_GO_CHARGE
                self.WAITLIST[v_idx] = self.V[T_DEST][v_idx]
            elif (
                self.V[TYPE][v_idx] == JOB_IDLE
                and self.V[Q][v_idx] < self.instance.ev_max_Q - 0.1
            ):
                self._assign_queue_job(v_idx, current_time)
                self.WAITLIST[v_idx] = current_time
        else:
            if self.V[TYPE][v_idx] == JOB_GO_CHARGE:
                self.V[TYPE][v_idx] = JOB_REPO
                self.WAITLIST[v_idx] = -1
            elif self.V[TYPE][v_idx] == JOB_QUEUE:
                self.WAITLIST[v_idx] = -1
                self._assign_idle_job(v_idx, current_time)
            elif self.V[TYPE][v_idx] == JOB_CHARGE:
                self._release_vehicle_from_charger(v_idx)
                self.WAITLIST[v_idx] = -1
                self._assign_idle_job(v_idx, current_time)

    def _try_assign_vehicle_to_charger(self, v_idx: int, next_time: float) -> None:
        """Try to assign a vehicle to a charger if capacity is available.

        Args:
            v_idx: Vehicle index
            next_time: Time of assignment
        """
        lot_id = self.V[TARGET_CHARGER][v_idx]
        station_capacity = self.instance.charger_count[lot_id]

        if self.current_occupancy[lot_id] < station_capacity:
            # Charger available - assign to charging
            q_start = self.V[Q][v_idx]
            t_start = next_time
            t_end = t_start + self._get_charging_time(q_start)
            self._assign_charge_job(v_idx, lot_id, t_start, t_end)
        else:
            # No charger available - assign to queue
            self._assign_queue_job(v_idx, next_time)

    def _calculate_job_time_metrics(
        self, v_idx: int, next_time: float, current_time: float
    ) -> Tuple[float, float, float, float]:
        """Calculate time metrics for different job types.

        Args:
            v_idx: Vehicle index
            next_time: Next simulation time
            current_time: Current simulation time

        Returns:
            Tuple of (repos_time, charge_time, idle_time, queue_time)
        """
        repos_time = 0.0
        charge_time = 0.0
        idle_time = 0.0
        queue_time = 0.0

        if self.V[TYPE][v_idx] == JOB_REPO or self.V[TYPE][v_idx] == JOB_GO_CHARGE:
            repos_time = next_time - current_time
        elif self.V[TYPE][v_idx] == JOB_CHARGE:
            charge_time = next_time - current_time
        elif self.V[TYPE][v_idx] == JOB_QUEUE:
            queue_time = next_time - current_time
        elif self.V[TYPE][v_idx] == JOB_IDLE:
            idle_time = next_time - current_time

        return repos_time, charge_time, idle_time, queue_time

    def _update_repo_render_data(
        self,
        v_idx: int,
        start_t: float,
        orig: np.ndarray,
        end_t: float,
        dest: np.ndarray,
    ) -> None:
        """Update render data for a repositioning job.
        Base implementation does nothing - override in subclasses if needed.

        Args:
            v_idx: Vehicle index
            start_t: Start time
            orig: Origin location
            end_t: End time
            dest: Destination location
        """
        pass

    def _update_serve_render_data(
        self,
        v_idx: int,
        start_t: float,
        start_loc: np.ndarray,
        mid_t: float,
        end_t: float,
        orig: np.ndarray,
        dest: np.ndarray,
    ) -> None:
        """Update render data for a serve job.
        Base implementation does nothing - override in subclasses if needed.

        Args:
            v_idx: Vehicle index
            start_t: Start time
            start_loc: Starting location
            mid_t: Pickup time
            end_t: Dropoff time
            orig: Pickup location
            dest: Dropoff location
        """
        pass

    def _update_render_state(self, v_idx: int, current_time: float) -> None:
        """Update render state for a vehicle at current time.
        Base implementation does nothing - override in subclasses if needed.

        Args:
            v_idx: Vehicle index
            current_time: Current simulation time
        """
        pass

    def _update_all_render_states(self, current_time: float) -> None:
        """Update render-only state after the vectorized physical update."""
        pass
