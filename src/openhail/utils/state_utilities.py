"""
A collection of utility functions for processing simulator state information.

These utilities help agents make decisions by:
- Computing masks and distances for vehicle assignments to requests
- Determining which vehicles are eligible to serve requests based on:
  - Battery levels
  - Wait time constraints
  - Distance/travel time calculations
- Updating vehicle states after assignments
- Computing recharge needs and masks for vehicles
"""

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core.constants import DEST, EPS, JOB_NULL, LOC, ORIG, PROC_TIME, TIME, TYPE, Q
from . import geometry

if TYPE_CHECKING:
    from ..core.openhail_instance import OpenhailInstance


def compute_assignment_mask(
    instance: "OpenhailInstance",
    request: Optional[Dict[str, np.ndarray]],
    V: Dict[str, np.ndarray],
    time: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute assignment mask for vehicles to serve requests.

    Args:
        instance: The simulation instance
        request: Request information (None if no active requests)
        V: Current vehicle states
        time: Current simulation time

    Returns:
        Tuple of (assignment_mask, preprocess_distances)
        - assignment_mask: Boolean mask indicating which vehicles can serve the request
        - preprocess_distances: Distances from vehicles to request origin
    """
    if request is None:
        return np.array([0 for _ in range(instance.num_evs)] + [1]), np.array(
            [0 for _ in range(instance.num_evs)]
        )

    # -> compute minimum distance associated to serving the request
    # i.e.: the distance of the request itself + the distance to the closest charger
    minimum_request_distance = request[PROC_TIME] + np.min(
        geometry.vector_distance(request[DEST], instance.cs_coords)
    )

    # -> compute travel times
    preprocess_distance = geometry.vector_distance(V[LOC], request[ORIG])
    wait_time = V[TIME] + (preprocess_distance / instance.ev_speed) - time
    assignment_mask = wait_time <= instance.passenger_max_wait
    charge_required = (
        preprocess_distance + minimum_request_distance
    ) * instance.ev_discharge_rate
    assignment_mask = assignment_mask & (V[Q] >= charge_required + EPS)
    assignment_mask = np.append(assignment_mask, True)

    return assignment_mask, preprocess_distance


def compute_nearest_assignment(
    instance: "OpenhailInstance",
    request: Dict[str, np.ndarray],
    V: Dict[str, np.ndarray],
) -> np.ndarray:
    """Compute nearest vehicle assignment for active requests using optimization.

    Args:
        instance: The simulation instance
        request: Request information arrays
        V: Current vehicle states

    Returns:
        Array of vehicle assignments for each request (instance.num_evs means no
            assignment)
    """
    active_requests = request[PROC_TIME] != -1
    active_request_indices = np.flatnonzero(active_requests)
    num_active_requests = active_request_indices.size

    if num_active_requests == 0:
        return np.array([instance.num_evs] * len(active_requests), dtype=np.int32)

    dest_2_cs = np.min(
        geometry.distance_matrix(request[DEST][active_requests], instance.cs_coords),
        axis=1,
    )
    orig_2_cs = request[PROC_TIME][active_requests] + dest_2_cs
    loc_2_orig = geometry.distance_matrix(V[LOC], request[ORIG][active_requests])

    total_dist = loc_2_orig + orig_2_cs[np.newaxis, :]
    total_charge_required = total_dist * instance.ev_discharge_rate + EPS

    eligible_assignments = V[Q][:, np.newaxis] >= total_charge_required
    wait_time = (
        V[TIME][:, np.newaxis]
        + (loc_2_orig / instance.ev_speed)
        - request[TIME][np.newaxis, active_requests]
    )
    eligible_assignments = eligible_assignments & (
        wait_time <= instance.passenger_max_wait
    )
    wait_time[~eligible_assignments] = instance.passenger_max_wait + 1

    if num_active_requests == 1:
        res = np.array([instance.num_evs] * len(active_requests), dtype=np.int32)
        best_vehicle = int(np.argmin(wait_time[:, 0]))
        if wait_time[best_vehicle, 0] <= instance.passenger_max_wait:
            res[active_request_indices[0]] = best_vehicle
        return res

    res = np.array([instance.num_evs] * len(active_requests), dtype=np.int32)
    vehicle_indices, request_indices = linear_sum_assignment(wait_time)
    for vehicle_idx, request_idx in zip(vehicle_indices, request_indices):
        if wait_time[vehicle_idx, request_idx] <= instance.passenger_max_wait:
            res[active_request_indices[request_idx]] = vehicle_idx
    return res


def update_state(
    instance: "OpenhailInstance",
    V: Dict[str, np.ndarray],
    request: Dict[str, np.ndarray],
    serve_action: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Update vehicle states after serving requests.

    Args:
        instance: The simulation instance
        V: Current vehicle states
        request: Request information arrays
        serve_action: Vehicle assignments for requests

    Returns:
        Updated vehicle states after processing serve actions
    """
    # Copy only arrays mutated by the hypothetical assignments.  Observations
    # contain several additional vehicle arrays that can remain shared because
    # this helper never writes to them.
    V = V.copy()
    for key in (TYPE, TIME, Q, LOC):
        V[key] = V[key].copy()
    for idx, action in enumerate(serve_action):
        if action < instance.num_evs:
            orig = request[ORIG][idx]
            dest = request[DEST][idx]
            process_dist = request[PROC_TIME][idx]

            process_t = process_dist / instance.ev_speed
            process_q = process_dist * instance.ev_discharge_rate
            preprocess_dist = geometry.get_distance(V[LOC][action], orig)
            preprocess_t = preprocess_dist / instance.ev_speed
            preprocess_q = preprocess_dist * instance.ev_discharge_rate

            V[TYPE][action] = JOB_NULL
            V[TIME][action] = V[TIME][action] + preprocess_t + process_t
            V[Q][action] = V[Q][action] - preprocess_q - process_q
            V[LOC][action] = dest
    return V


def compute_rnr_mask(
    instance: "OpenhailInstance",
    V: Dict[str, np.ndarray],
    request: Optional[Dict[str, np.ndarray]] = None,
    serve_action: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the repositioning mask for vehicles.

    Args:
        instance: The simulation instance
        V: Current vehicle states
        request: Current requests (optional, for simulating serve actions)
        serve_action: Serve actions that will be processed (optional)

    Returns:
        Tuple of (rnr_mask, rnr_distances)
    """
    # If serve actions are provided, simulate their effect on vehicle states
    if request is not None and serve_action is not None:
        V = update_state(instance, V, request, serve_action)

    rnr_mask = np.zeros((instance.num_evs, (1 + instance.D_repo)), dtype=bool)
    # -> null jobs have to reposition
    rnr_mask[:, 0] = V[TYPE] != JOB_NULL
    # The full-mask API is a convenience wrapper. Feasibility itself is
    # evaluated through the selected-row API below, so agents can request any
    # subset of vehicles at any decision epoch.
    free_vehicle_indices = np.flatnonzero(V[TYPE] < 10)
    sufficient_charge, selected_distances = compute_rnr_feasibility(
        instance,
        V,
        free_vehicle_indices,
    )
    rnr_distances = np.full(
        (instance.num_evs, instance.D_repo),
        100000.0,
        dtype=np.float64,
    )
    rnr_distances[free_vehicle_indices] = selected_distances
    rnr_mask[free_vehicle_indices, 1:] = sufficient_charge

    return rnr_mask, rnr_distances


def compute_rnr_feasibility(
    instance: "OpenhailInstance",
    V: Dict[str, np.ndarray],
    vehicle_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute destination feasibility for agent-selected vehicle rows.

    Selection is independent of simulator epoch type. An agent may request
    one vehicle, the full fleet, or any intermediate subset whenever its
    decision rule considers repositioning or charging those vehicles.

    Returns:
        A tuple containing a boolean feasibility matrix and a distance matrix,
        both with shape ``(len(vehicle_indices), instance.D_repo)``. As in the
        full-mask API, infeasible distances are replaced by ``100000``.
    """
    indices = np.asarray(vehicle_indices, dtype=np.intp)
    # ``repository_distances`` owns shape and bounds validation for this same
    # index array. Repeating those reductions here added work to every sparse
    # policy decision without providing a second layer of protection.
    cached_distances = instance.repository_distances(V[LOC], indices)
    distances = cached_distances[indices]
    free_vehicles = V[TYPE][indices] < 10
    sufficient_charge = free_vehicles[:, None] & (
        V[Q][indices, None]
        >= (distances + instance.repo_to_cs[None, :]) * instance.ev_discharge_rate + EPS
    )
    return sufficient_charge, np.where(
        sufficient_charge,
        distances,
        100000.0,
    )
