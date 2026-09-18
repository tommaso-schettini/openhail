"""
Renderable Vehicle Manager for OpenHail simulation.

This module contains the RenderableVehicleManager class which extends
the base VehicleManager with detailed render tracking capabilities.
"""

from collections import deque
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from ..constants import LOC, TIME, WAYPOINT, Q
from .vehicle_manager import VehicleManager

if TYPE_CHECKING:
    from ..openhail_instance import OpenhailInstance


class RenderableVehicleManager(VehicleManager):
    """Extended VehicleManager that tracks detailed render data for visualization."""

    def __init__(self, instance: "OpenhailInstance", strict_validation: bool = True):
        """Initialize the renderable vehicle manager.

        Args:
            instance: The simulation instance containing vehicle and infrastructure data
            strict_validation: Whether to enable strict validation checks
        """
        super().__init__(instance, strict_validation)

        # Render data - will be initialized when needed
        self.V_render = None

    def initialize_render_data(self) -> None:
        """Initialize render data structures for detailed visualization tracking."""
        self.V_render = {
            LOC: np.zeros((self.instance.num_evs, 2), dtype=np.float64),
            Q: np.zeros(self.instance.num_evs, dtype=np.float64),
            WAYPOINT: [
                deque([self.instance.midpoint]) for _ in range(self.instance.num_evs)
            ],
            TIME: [
                deque([self.instance.max_time + 1])
                for _ in range(self.instance.num_evs)
            ],
        }

    def get_render_data(self) -> Optional[Dict[str, Any]]:
        """Get detailed render data for visualization.

        Returns:
            Dictionary containing detailed render data, or approximated data if render
                tracking is not initialized
        """
        if self.V_render is not None:
            return self.V_render
        else:
            # Fall back to approximated render data from base class
            return super().get_render_data()

    def _update_repo_render_data(
        self,
        v_idx: int,
        start_t: float,
        orig: np.ndarray,
        end_t: float,
        dest: np.ndarray,
    ) -> None:
        """Update render data for a repositioning job.

        Args:
            v_idx: Vehicle index
            start_t: Start time
            orig: Origin location
            end_t: End time
            dest: Destination location
        """
        if self.V_render is None:
            return

        if self.V_render[TIME][v_idx][-1] > start_t:
            self.V_render[TIME][v_idx].pop()
            self.V_render[WAYPOINT][v_idx].pop()
        if len(self.V_render[WAYPOINT][v_idx]) == 1:
            self.V_render[TIME][v_idx][0] = start_t
            self.V_render[WAYPOINT][v_idx][0] = orig.copy()

        self.V_render[TIME][v_idx].append(end_t)
        self.V_render[WAYPOINT][v_idx].append(dest.copy())

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

        Args:
            v_idx: Vehicle index
            start_t: Start time
            start_loc: Starting location
            mid_t: Pickup time
            end_t: Dropoff time
            orig: Pickup location
            dest: Dropoff location
        """
        if self.V_render is None:
            return

        if self.V_render[TIME][v_idx][-1] > start_t:
            self.V_render[TIME][v_idx].pop()
            self.V_render[WAYPOINT][v_idx].pop()
        if len(self.V_render[WAYPOINT][v_idx]) == 1:
            self.V_render[TIME][v_idx][0] = start_t
            self.V_render[WAYPOINT][v_idx][0] = start_loc.copy()
        self.V_render[TIME][v_idx].append(mid_t)
        self.V_render[TIME][v_idx].append(end_t)
        self.V_render[WAYPOINT][v_idx].append(orig.copy())
        self.V_render[WAYPOINT][v_idx].append(dest.copy())

    def _update_render_state(self, v_idx: int, current_time: float) -> None:
        """Update render state for a vehicle at current time.

        Args:
            v_idx: Vehicle index
            current_time: Current simulation time
        """
        if self.V_render is None:
            return

        while True:
            if len(self.V_render[TIME][v_idx]) <= 1:
                break
            if self.V_render[TIME][v_idx][1] > current_time:
                break
            self.V_render[TIME][v_idx].popleft()
            self.V_render[WAYPOINT][v_idx].popleft()

        if len(self.V_render[WAYPOINT][v_idx]) == 1:
            self.V_render[LOC][v_idx] = self.V_render[WAYPOINT][v_idx][0]
        else:
            self.V_render[LOC][v_idx] = self._interpolate(
                current_time,
                self.V_render[TIME][v_idx][0],
                self.V_render[WAYPOINT][v_idx][0],
                self.V_render[TIME][v_idx][1],
                self.V_render[WAYPOINT][v_idx][1],
            )
        self.V_render[Q][v_idx] = self.V[Q][v_idx] / self.instance.ev_max_Q

    def _update_all_render_states(self, current_time: float) -> None:
        if self.V_render is None:
            return
        for v_idx in range(self.instance.num_evs):
            self._update_render_state(v_idx, current_time)
