import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..constants import DEST, ORIG, PROC_TIME, TIME


@dataclass
class RequestInfo:
    """Information about a single request."""

    request_id: int
    time: float
    orig: np.ndarray
    dest: np.ndarray
    process_time: float


class RequestManager:
    """Manages all request-related operations for the ridehail environment."""

    def __init__(
        self,
        instance,
        max_outstanding_requests: int,
        clear_requests: bool,
        strict_validation: bool,
    ):
        """Initialize the RequestManager.

        Args:
            instance: The OpenhailInstance containing request generation logic
            max_outstanding_requests: Maximum number of outstanding requests to track
            clear_requests: Whether to clear unserved requests each epoch
            strict_validation: Whether to perform strict validation
        """
        self.instance = instance
        self.max_outstanding_requests = max_outstanding_requests
        self.clear_requests = clear_requests
        self.strict_validation = strict_validation

        # Request data arrays
        self.request_time: np.ndarray
        self.request_orig: np.ndarray
        self.request_dest: np.ndarray
        self.request_proc: np.ndarray

        # Request state
        self.request_idx = 0
        self.dummy_request = -1
        self.active_requests: np.ndarray

        # Day of year for request generation
        self.doy: int

    def initialize_requests(self, requests_seed: int) -> None:
        """Generate and initialize all requests for the episode.

        Args:
            requests_seed: Random seed for request generation
        """
        doy, requests = self.instance.generate_requests(requests_seed)
        self.doy = doy

        self.request_time = requests[TIME].values
        self.request_orig = np.stack(requests[ORIG].values)
        self.request_dest = np.stack(requests[DEST].values)
        self.request_proc = requests[PROC_TIME].values

        self.dummy_request = len(self.request_time) - 1
        self.request_idx = 0
        self.active_requests = self.dummy_request * np.ones(
            self.max_outstanding_requests, dtype=int
        )

    def get_current_requests(self) -> Dict[str, np.ndarray]:
        """Get the current active requests.

        Returns:
            Dictionary with keys TIME, ORIG, DEST, and PROC_TIME containing request data
        """
        return {
            TIME: self.request_time[self.active_requests],
            ORIG: self.request_orig[self.active_requests],
            DEST: self.request_dest[self.active_requests],
            PROC_TIME: self.request_proc[self.active_requests],
        }

    def get_request_info(self, request_idx: int) -> RequestInfo:
        """Get information about a specific active request.

        Args:
            request_idx: Index in the active_requests array

        Returns:
            RequestInfo object with request details

        Raises:
            ValueError: If no active request at the given index
        """
        if (
            self.strict_validation
            and self.active_requests[request_idx] == self.dummy_request
        ):
            raise ValueError(f"No active request at index {request_idx}")

        request_id = self.active_requests[request_idx]
        return RequestInfo(
            request_id=request_id,
            time=float(self.request_time[request_id]),
            orig=self.request_orig[request_id].copy(),
            dest=self.request_dest[request_id].copy(),
            process_time=float(self.request_proc[request_id]),
        )

    def consume_request(self, request_idx: int) -> RequestInfo:
        """Get and remove a request from the active requests.

        Args:
            request_idx: Index in the active_requests array

        Returns:
            RequestInfo object with request details
        """
        request_info = self.get_request_info(request_idx)
        self.active_requests[request_idx] = self.dummy_request
        return request_info

    def advance_requests(self, current_time: float) -> int:
        """Advance request state: remove expired requests and add new ones.

        Args:
            current_time: Current simulation time

        Returns:
            Number of new requests added
        """
        # Remove unserved requests
        if self.clear_requests:
            # Remove all requests that are not immediately served
            self.active_requests[:] = self.dummy_request
        else:
            # Remove requests that have passed their deadline
            for i in range(self.max_outstanding_requests):
                idx = self.active_requests[i]
                if (
                    idx != self.dummy_request
                    and self.request_time[idx] + self.instance.passenger_max_wait
                    <= current_time
                ):
                    self.active_requests[i] = self.dummy_request
                else:
                    break

        self.active_requests.sort()
        current_active_requests = self.num_active_requests()
        added_requests = 0

        # Add new requests that have arrived
        while (
            self.request_idx < self.dummy_request
            and self.request_time[self.request_idx] <= current_time
        ):
            idx = current_active_requests % self.max_outstanding_requests

            if current_active_requests >= self.max_outstanding_requests:
                overwritten_request = self.active_requests[idx]
                logging.warning(
                    "Warning: attempting to load more than "
                    f"{self.max_outstanding_requests} "
                    f"requests at time {current_time}. "
                    "Excess requests will be removed in FIFO manner. "
                    "To prevent this, increase max_outstanding_requests in config "
                    "or reduce epoch clock. "
                    f"Overwriting slot {idx}; "
                    f"overwritten={self._format_request_debug(overwritten_request)}, "
                    f"incoming={self._format_request_debug(self.request_idx)}, "
                    f"active_before_overwrite={self._format_active_requests_debug()}."
                )

            self.active_requests[
                current_active_requests % self.max_outstanding_requests
            ] = self.request_idx
            self.request_idx += 1
            current_active_requests += 1
            added_requests += 1

        self.active_requests.sort()
        return added_requests

    def num_active_requests(self) -> int:
        """Get the number of currently active requests.

        Returns:
            Number of active requests
        """
        return np.sum(self.active_requests != self.dummy_request)

    def _format_active_requests_debug(self) -> list[dict[str, Any]]:
        """Format active requests for overflow diagnostics."""
        return [
            self._request_debug_dict(request_id)
            for request_id in self.active_requests
            if request_id != self.dummy_request
        ]

    def _format_request_debug(self, request_id: int) -> dict[str, Any]:
        """Format a single request for overflow diagnostics."""
        if request_id == self.dummy_request:
            return {"request_id": int(request_id), "dummy": True}
        return self._request_debug_dict(request_id)

    def _request_debug_dict(self, request_id: int) -> dict[str, Any]:
        return {
            "request_id": int(request_id),
            "time": float(self.request_time[request_id]),
            "orig": self.request_orig[request_id].tolist(),
            "dest": self.request_dest[request_id].tolist(),
            "process_time": float(self.request_proc[request_id]),
        }

    def next_request_time(self) -> float:
        """Get the time of the next request to arrive.

        Returns:
            Time of next request, or infinity if no more requests
        """
        if self.request_idx >= self.dummy_request:
            return float("inf")
        return self.request_time[self.request_idx]

    def has_active_request(self, request_idx: int) -> bool:
        """Check if there's an active request at the given index.

        Args:
            request_idx: Index in the active_requests array

        Returns:
            True if there's an active request at the index
        """
        return (
            request_idx < len(self.active_requests)
            and self.active_requests[request_idx] != self.dummy_request
        )
