"""Normalized vehicle, charging-location, and time features."""

from typing import Any, Dict, Tuple

import numpy as np

from ...core.constants import (
    CAPACITY,
    CHARGER,
    CHARGERS,
    DEST,
    JOB_CHARGE,
    JOB_GO_CHARGE,
    JOB_IDLE,
    JOB_NULL,
    JOB_QUEUE,
    JOB_REPO,
    JOB_SERVE,
    LOC,
    OCCUPANCY,
    Q_DEST,
    QUEUE,
    T_DEST,
    TARGET_CHARGER,
    TIME,
    TYPE,
    Q,
)
from ...core.openhail_instance import OpenhailInstance

JOB_TYPES = (
    JOB_NULL,
    JOB_IDLE,
    JOB_CHARGE,
    JOB_QUEUE,
    JOB_GO_CHARGE,
    JOB_REPO,
    JOB_SERVE,
)


class PeriodicFeatureExtractor:
    """Create normalized features independent of fleet and location counts."""

    def __init__(self, instance: OpenhailInstance) -> None:
        self.instance = instance
        self.num_vehicles = instance.num_evs
        self.num_locations = instance.D_repo
        self.midpoint = np.asarray(instance.midpoint, dtype=np.float64)
        self.spatial_scale = max(
            instance.x_bounds[1] - instance.x_bounds[0],
            instance.y_bounds[1] - instance.y_bounds[0],
            1.0,
        )
        self.horizon = max(float(instance.max_time - instance.start_time), 1.0)

        # loc(2), capacity, occupancy, queue, positive-capacity indicator.
        self.location_feature_dim = 6
        # loc(2), SOC, availability, destination(2), destination SOC/time,
        # job one-hot(7), has-target indicator, target-location features(6),
        # using-charger indicator.
        self.vehicle_feature_dim = (
            8 + len(JOB_TYPES) + 1 + self.location_feature_dim + 1
        )
        self.time_feature_dim = 3

    def extract(
        self,
        state: Dict[str, Any],
        vehicles: Dict[str, np.ndarray] | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return vehicle, location, and periodic time features."""
        current_time = float(state["time"])
        V = state["V"] if vehicles is None else vehicles
        chargers = state[CHARGERS]

        location_features = np.zeros(
            (self.num_locations, self.location_feature_dim), dtype=np.float32
        )
        location_features[:, 0:2] = self._normalize_locations(chargers[LOC])
        scale = max(self.num_vehicles, 1)
        capacity = np.asarray(chargers[CAPACITY], dtype=np.float64)
        location_features[:, 2] = np.clip(capacity / scale, 0.0, 1.0)
        location_features[:, 3] = np.clip(
            np.asarray(chargers[OCCUPANCY], dtype=np.float64) / scale,
            0.0,
            1.0,
        )
        location_features[:, 4] = np.clip(
            np.asarray(chargers[QUEUE], dtype=np.float64) / scale,
            0.0,
            1.0,
        )
        location_features[:, 5] = capacity > 0

        vehicle_features = np.zeros(
            (self.num_vehicles, self.vehicle_feature_dim), dtype=np.float32
        )
        vehicle_features[:, 0:2] = self._normalize_locations(V[LOC])
        vehicle_features[:, 2] = np.clip(
            np.asarray(V[Q], dtype=np.float64) / self.instance.ev_max_Q,
            0.0,
            1.0,
        )
        vehicle_features[:, 3] = np.clip(
            (np.asarray(V[TIME], dtype=np.float64) - current_time) / self.horizon,
            0.0,
            1.0,
        )
        vehicle_features[:, 4:6] = self._normalize_locations(V[DEST])
        vehicle_features[:, 6] = np.clip(
            np.asarray(V[Q_DEST], dtype=np.float64) / self.instance.ev_max_Q,
            -0.05,
            1.0,
        )
        vehicle_features[:, 7] = np.clip(
            (np.asarray(V[T_DEST], dtype=np.float64) - current_time) / self.horizon,
            0.0,
            1.0,
        )

        offset = 8
        vehicle_types = np.asarray(V[TYPE], dtype=np.int32)
        for job_offset, job_type in enumerate(JOB_TYPES):
            vehicle_features[:, offset + job_offset] = vehicle_types == job_type
        offset += len(JOB_TYPES)

        target_chargers = np.asarray(V[TARGET_CHARGER], dtype=np.int32)
        valid_target = (target_chargers >= 0) & (target_chargers < self.num_locations)
        vehicle_features[valid_target, offset] = 1.0
        vehicle_features[
            valid_target, offset + 1 : offset + 1 + self.location_feature_dim
        ] = location_features[target_chargers[valid_target]]
        offset += 1 + self.location_feature_dim
        vehicle_features[:, offset] = np.asarray(V[CHARGER], dtype=np.int32) >= 0

        fraction = np.clip(
            (current_time - self.instance.start_time) / self.horizon,
            0.0,
            1.0,
        )
        angle = 2.0 * np.pi * fraction
        time_features = np.asarray(
            [fraction, np.sin(angle), np.cos(angle)], dtype=np.float32
        )
        return vehicle_features, location_features, time_features

    def _normalize_locations(self, locations: np.ndarray) -> np.ndarray:
        return np.asarray(
            (np.asarray(locations, dtype=np.float64) - self.midpoint)
            / self.spatial_scale,
            dtype=np.float32,
        )
