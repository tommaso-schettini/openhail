from typing import Any, Dict, Tuple

import numpy as np

from ..core.constants import JOB_NULL, TYPE, Q
from ..core.openhail_instance import OpenhailInstance
from ..utils import state_utilities as state_utilities
from . import Agent


class NearestAgent(Agent):
    """A simple agent that, in general, uses vehicles' proximity to make decisions.

    Assigns the nearest feasible EV to a request.
    Tells vehicles to charge at the nearest CS when their
    battery dips below a certain threshold.
    Instructs vehicles to wait at nearest idle zone if they must be
    relocated.
    """

    def __init__(
        self,
        instance_config: OpenhailInstance,
        soc_threshold: float = 0.2,
        immediate_reposition: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the NearestAgent.

        Args:
            instance_config: The OpenhailInstance used by the environment
            soc_threshold: State of charge threshold below which vehicles should
                recharge (0-1)
            immediate_reposition: Whether to update state immediately with serve actions
            **kwargs: Additional parameters passed to parent Agent class
        """
        Agent.__init__(self, instance_config, **kwargs)

        if not 0 <= soc_threshold <= 1:
            raise ValueError(
                "soc_threshold must be a relative charge value between 0 and 1."
            )
        self.q_thresh = soc_threshold * self.instance.ev_max_Q
        self.immediate_reposition = immediate_reposition

        return

    def choose_action(
        self, state: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Choose action based on nearest vehicle assignment strategy.

        Args:
            state: Environment state containing vehicle states, requests, and time

        Returns:
            Tuple of (serve_action, rnr_action, rebal_action)
            - serve_action: Vehicle assignments for active requests
            - rnr_action: Repositioning and recharging actions for vehicles
            - rebal_action: Planned epoch time for next decision
        """
        V = state["V"]
        request = state["request"]

        serve_action = state_utilities.compute_nearest_assignment(
            self.instance, request, V
        )
        post_assignment_V = state_utilities.update_state(
            self.instance, V, request, serve_action
        )
        rnr_action = self.choose_sparse_rnr(post_assignment_V)
        rebal_action = np.asarray(-1.0, dtype=np.float64)

        return serve_action, rnr_action, rebal_action

    def choose_sparse_rnr(self, V: Dict[str, np.ndarray]) -> np.ndarray:
        """Choose moves without constructing a full-fleet destination mask."""
        do_move = V[TYPE] == JOB_NULL
        do_recharge = V[Q] <= self.q_thresh
        candidates = np.flatnonzero(do_move | do_recharge)
        result = np.zeros(self.instance.num_evs, dtype=int)

        if candidates.size == 0:
            return result

        feasibility, distances = state_utilities.compute_rnr_feasibility(
            self.instance,
            V,
            candidates,
        )

        move_positions = np.flatnonzero(do_move[candidates])
        if move_positions.size:
            destinations = np.argmin(distances[move_positions], axis=1) + 1
            result[candidates[move_positions]] = 2 * destinations - 1

        recharge_positions = np.flatnonzero(do_recharge[candidates])
        if recharge_positions.size == 0:
            return result

        charger_destinations = np.flatnonzero(self.instance.charger_count > 0)
        if charger_destinations.size == 0:
            raise RuntimeError("NearestAgent requires at least one charging location.")

        charger_feasibility = feasibility[recharge_positions][:, charger_destinations]
        unreachable_positions = np.flatnonzero(~np.any(charger_feasibility, axis=1))
        if unreachable_positions.size:
            vehicles = candidates[recharge_positions[unreachable_positions]].tolist()
            raise RuntimeError(
                f"Low-SOC vehicles cannot reach any charging location: {vehicles}."
            )

        charger_distances = np.where(
            charger_feasibility,
            distances[recharge_positions][:, charger_destinations],
            np.inf,
        )
        nearest_chargers = (
            charger_destinations[np.argmin(charger_distances, axis=1)] + 1
        )
        result[candidates[recharge_positions]] = 2 * nearest_chargers
        return result

    def choose_rnr(
        self, V: Dict[str, np.ndarray], mask_rnr: np.ndarray, rnr_dist: np.ndarray
    ) -> np.ndarray:
        """Choose repositioning and recharging actions for vehicles.

        Args:
            V: Vehicle states dictionary
            mask_rnr: Boolean mask indicating valid repositioning/recharging options
            rnr_dist: Distance matrix to repositioning/recharging locations

        Returns:
            Array of repositioning/recharging actions for each vehicle
        """
        do_move = ~mask_rnr[:, 0]
        do_recharge = V[Q] <= self.q_thresh

        res = np.zeros((self.instance.num_evs), dtype=int)
        repos_location = np.argmin(rnr_dist, axis=1) + 1
        res[do_move] = 2 * repos_location[do_move] - 1

        charger_destinations = np.flatnonzero(self.instance.charger_count > 0)
        if charger_destinations.size == 0:
            raise RuntimeError("NearestAgent requires at least one charging location.")

        charger_feasibility = mask_rnr[:, 1:][:, charger_destinations]
        unreachable = do_recharge & ~np.any(charger_feasibility, axis=1)
        if np.any(unreachable):
            vehicles = np.flatnonzero(unreachable).tolist()
            raise RuntimeError(
                f"Low-SOC vehicles cannot reach any charging location: {vehicles}."
            )

        charger_distances = np.where(
            charger_feasibility,
            rnr_dist[:, charger_destinations],
            np.inf,
        )
        nearest_charger = charger_destinations[np.argmin(charger_distances, axis=1)] + 1
        res[do_recharge] = 2 * nearest_charger[do_recharge]

        return res
