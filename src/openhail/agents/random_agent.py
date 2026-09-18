"""Random-feasible baseline with closest feasible request assignment."""

from typing import Any, Dict, Tuple

import numpy as np

from ..core.constants import CLOCK_EPOCH, EPOCH_TYPE, JOB_NULL, TYPE
from ..utils import state_utilities
from .agent import Agent


class RandomAgent(Agent):
    """Choose uniformly among every feasible native repositioning action."""

    def __init__(
        self,
        instance,
        seed: int = 0,
        periodic_repositioning: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(instance, **kwargs)
        self.seed = seed
        self.periodic_repositioning = periodic_repositioning
        self.rng = np.random.default_rng(seed)

    def set_doy(self, doy: int) -> None:
        """Reset exploration reproducibly for a newly initialized episode."""
        self.rng = np.random.default_rng(self.seed + int(doy))

    def choose_action(
        self, state: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        V = state["V"]
        request = state["request"]

        serve_action = state_utilities.compute_nearest_assignment(
            self.instance, request, V
        )
        post_assignment_V = state_utilities.update_state(
            self.instance, V, request, serve_action
        )

        if self.periodic_repositioning and int(state[EPOCH_TYPE]) != CLOCK_EPOCH:
            rnr_action = self._non_periodic_action(
                serve_action,
                post_assignment_V,
            )
        else:
            rnr_action = self._random_feasible_action(post_assignment_V)

        planned_epoch = np.asarray(-1.0, dtype=np.float64)
        return serve_action, rnr_action, planned_epoch

    def _random_feasible_action(
        self,
        vehicles: Dict[str, np.ndarray],
    ) -> np.ndarray:
        rnr_mask, _ = state_utilities.compute_rnr_mask(self.instance, vehicles)

        rnr_action = np.zeros(self.instance.num_evs, dtype=np.int32)
        for vehicle_idx in range(self.instance.num_evs):
            feasible_actions = []
            if rnr_mask[vehicle_idx, 0]:
                feasible_actions.append(0)

            for destination_idx in np.flatnonzero(rnr_mask[vehicle_idx, 1:]):
                location = int(destination_idx) + 1
                feasible_actions.append(2 * location - 1)
                if self.instance.charger_count[destination_idx] > 0:
                    feasible_actions.append(2 * location)

            if not feasible_actions:
                raise RuntimeError(
                    f"Vehicle {vehicle_idx} has no feasible repositioning action."
                )
            rnr_action[vehicle_idx] = self.rng.choice(feasible_actions)
        return rnr_action

    def _non_periodic_action(
        self,
        serve_action: np.ndarray,
        vehicles: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """No-op except for vehicles that must receive a simulator job."""
        actions = np.zeros(self.instance.num_evs, dtype=np.int32)
        assigned_vehicles = {
            int(vehicle)
            for vehicle in np.asarray(serve_action).tolist()
            if int(vehicle) < self.instance.num_evs
        }
        must_reposition = np.flatnonzero(vehicles[TYPE] == JOB_NULL)
        fallback_vehicles = np.asarray(
            [
                int(vehicle)
                for vehicle in must_reposition
                if int(vehicle) not in assigned_vehicles
            ],
            dtype=np.intp,
        )
        if fallback_vehicles.size == 0:
            return actions

        feasibility, distances = state_utilities.compute_rnr_feasibility(
            self.instance,
            vehicles,
            fallback_vehicles,
        )
        for position, vehicle in enumerate(fallback_vehicles):
            feasible_destinations = np.flatnonzero(feasibility[position])
            if feasible_destinations.size == 0:
                raise RuntimeError(
                    f"Vehicle {vehicle} must reposition but has no feasible "
                    "destination."
                )
            destination = int(
                feasible_destinations[
                    np.argmin(distances[position, feasible_destinations])
                ]
            )
            actions[vehicle] = 2 * destination + 1
        return actions
