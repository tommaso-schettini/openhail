"""
A class for tracking and summarizing episode-level statistics during simulation.
Statistics are organized into three categories: vehicle activities, request statistics,
    and epochs.
"""

# Vehicle time allocation metrics
VEHICLE_ACTIVITY_KEYS = {
    "time_repos": 0.0,  # Time spent repositioning
    "time_charging": 0.0,  # Time spent charging
    "time_idle": 0.0,  # Time spent idle
    "time_queue": 0.0,  # Time spent queuing for charger
    "time_serving": 0.0,  # Time spent serving customers (total)
    "time_preprocess": 0.0,  # Time traveling to pickup
    "time_process": 0.0,  # Time driving customer to destination
}

# Request service metrics
REQUEST_STATISTICS_KEYS = {
    "requests_total": 0,  # Total requests generated
    "requests_served": 0,  # Total requests served
    "cust_waiting_time": 0.0,  # Total customer waiting time
}

# Basic episode information
EPOCH_STATISTICS_KEYS = {
    "epochs": 0,  # Total number of epochs
    "total_time": 0.0,  # Total simulation time
}

# Combined default keys
DEFAULT_KEYS = {
    **VEHICLE_ACTIVITY_KEYS,
    **REQUEST_STATISTICS_KEYS,
    **EPOCH_STATISTICS_KEYS,
}


class Summary:
    """Summary class for tracking episode statistics organized by category."""

    def __init__(self, extra_keys=None):
        """Initialize summary with default keys plus any extra keys.

        Args:
            extra_keys: Additional keys to track (e.g., time-bucketed request metrics)
        """
        self.data = {}
        if extra_keys is None:
            extra_keys = {}

        # Combine default keys with any extra keys
        all_keys = {**DEFAULT_KEYS, **extra_keys}
        for key, value in all_keys.items():
            self.data[key] = value

    def accumulate(self, data: dict):
        """Add values to existing metrics.

        Args:
            data: Dictionary of metric updates to accumulate
        """
        for key, value in data.items():
            if key not in self.data:
                raise KeyError(f"Unknown key {key!r} in summary.")
            self.data[key] += value

    def track_maximum(self, data: dict):
        """Track maximum values for specified metrics.

        Args:
            data: Dictionary of metrics to track maximums for
        """
        for key, value in data.items():
            if key not in self.data:
                raise KeyError(f"Unknown key {key!r} in summary.")
            self.data[key] = max(self.data[key], value)

    def track_minimum(self, data: dict):
        """Track minimum values for specified metrics.

        Args:
            data: Dictionary of metrics to track minimums for
        """
        for key, value in data.items():
            if key not in self.data:
                raise KeyError(f"Unknown key {key!r} in summary.")
            self.data[key] = min(self.data[key], value)

    def get_vehicle_activities(self) -> dict:
        """Get vehicle activity metrics.

        Returns:
            Dictionary containing vehicle time allocation metrics
        """
        return {key: self.data[key] for key in VEHICLE_ACTIVITY_KEYS.keys()}

    def get_request_statistics(self) -> dict:
        """Get request service metrics.

        Returns:
            Dictionary containing request statistics
        """
        return {key: self.data[key] for key in REQUEST_STATISTICS_KEYS.keys()}

    def get_epoch_statistics(self) -> dict:
        """Get epoch-level metrics.

        Returns:
            Dictionary containing epoch statistics
        """
        return {key: self.data[key] for key in EPOCH_STATISTICS_KEYS.keys()}
