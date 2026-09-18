import logging
from typing import Any, Dict, List, Optional

from numpy.typing import ArrayLike

from ...utils.episode_summary import Summary


class SummaryManager:
    """Manages episode summary statistics collection and reporting.

    Statistics are organized into three categories:
    - Vehicle Activities: Time allocation metrics for vehicles
    - Request Statistics: Service quality and request metrics
    - Epoch Statistics: Basic episode information
    """

    def __init__(
        self,
        track_vehicles: bool = True,
        track_requests: bool = True,
        track_epochs: bool = True,
    ):
        """Initialize the SummaryManager.

        Args:
            track_vehicles: Whether to track vehicle activity metrics
            track_requests: Whether to track request statistics
            track_epochs: Whether to track epoch statistics
        """
        self.track_vehicles = track_vehicles
        self.track_requests = track_requests
        self.track_epochs = track_epochs

        # Summary object
        self.summary: Optional[Summary] = None

        # Episode timing
        self.start_time: float = 0.0
        self.max_time: float = 0.0

    def initialize_summary(self, start_time: float, max_time: float) -> None:
        """Initialize summary tracking for a new episode.

        Args:
            start_time: Episode start time
            max_time: Episode end time
        """
        self.start_time = start_time
        self.max_time = max_time

        if not self.is_tracking():
            self.summary = None
            return

        logging.info("Summary tracking initialized")
        self.summary = Summary({})

    def is_tracking(self) -> bool:
        """Check if any summary tracking is enabled.

        Returns:
            True if any tracking is active
        """
        return self.track_vehicles or self.track_requests or self.track_epochs

    # =========================
    # REQUEST STATISTICS TRACKING
    # =========================

    def track_requests_added(self, current_time: float, num_requests: int) -> None:
        """Track requests added to the system.

        Args:
            current_time: Current simulation time
            num_requests: Number of requests added
        """
        if not self.track_requests or not self.summary:
            return

        # Update total requests
        self.summary.accumulate({"requests_total": num_requests})

    def track_request_served(self, current_time: float) -> None:
        """Track a request being served.

        Args:
            current_time: Current simulation time
        """
        if not self.track_requests or not self.summary:
            return

        # Update total requests served
        self.summary.accumulate({"requests_served": 1})

    def track_serve_job(
        self,
        start_time: float,
        process_time: float,
        preprocess_time: float,
        current_time: float,
    ) -> None:
        """Track statistics for a serve job.

        Args:
            start_time: Job start time
            process_time: Time spent processing the request
            preprocess_time: Time spent traveling to pickup
            current_time: Current simulation time
        """
        if not self.summary:
            return

        # Calculate total serving time
        total_serving_time = process_time + preprocess_time

        # Update vehicle activity metrics
        if self.track_vehicles:
            self.summary.accumulate(
                {
                    "time_serving": total_serving_time,
                    "time_process": process_time,
                    "time_preprocess": preprocess_time,
                }
            )

        # Update request statistics
        if self.track_requests:
            waiting_time = start_time + preprocess_time - current_time
            self.summary.accumulate(
                {
                    "requests_served": 1,
                    "cust_waiting_time": waiting_time,
                }
            )

    # =========================
    # VEHICLE ACTIVITIES TRACKING
    # =========================

    def track_vehicle_time_metrics(
        self, repos_time: float, charge_time: float, idle_time: float, queue_time: float
    ) -> None:
        """Track vehicle time allocation metrics.

        Args:
            repos_time: Time spent repositioning
            charge_time: Time spent charging
            idle_time: Time spent idle
            queue_time: Time spent queuing for charger
        """
        if not self.track_vehicles or not self.summary:
            return

        self.summary.accumulate(
            {
                "time_repos": repos_time,
                "time_charging": charge_time,
                "time_idle": idle_time,
                "time_queue": queue_time,
            }
        )

    # =========================
    # EPOCH STATISTICS TRACKING
    # =========================

    def track_epoch_metrics(
        self, current_time: float, vehicle_charges: ArrayLike
    ) -> None:
        """Track epoch-level metrics.

        Args:
            current_time: Current simulation time
            vehicle_charges: Current vehicle charge levels
        """
        if not self.track_epochs or not self.summary:
            return

        self.summary.accumulate(
            {
                "epochs": 1,
            }
        )
        self.summary.track_maximum({"total_time": current_time - self.start_time})

    # =========================
    # SUMMARY RETRIEVAL
    # =========================

    def get_episode_summary_list(self, num_vehicles: int) -> List[float]:
        """Get episode summary as a list of values.

        Args:
            num_vehicles: Number of vehicles in the simulation

        Returns:
            List of summary statistics values
        """
        if not self.is_tracking() or not self.summary:
            return []

        # Get all statistics organized by category
        vehicle_activities = self.get_vehicle_activities()
        request_statistics = self.get_request_statistics()
        epoch_statistics = self.get_epoch_statistics()

        # Combine all statistics in a consistent order
        summary_list = []

        # Vehicle Activities (7 metrics) - only if tracking vehicles
        if self.track_vehicles:
            summary_list.extend(
                [
                    vehicle_activities.get("time_repos", 0.0),
                    vehicle_activities.get("time_charging", 0.0),
                    vehicle_activities.get("time_idle", 0.0),
                    vehicle_activities.get("time_queue", 0.0),
                    vehicle_activities.get("time_serving", 0.0),
                    vehicle_activities.get("time_preprocess", 0.0),
                    vehicle_activities.get("time_process", 0.0),
                ]
            )

        # Request Statistics (3 metrics) - only if tracking requests
        if self.track_requests:
            summary_list.extend(
                [
                    request_statistics.get("requests_total", 0.0),
                    request_statistics.get("requests_served", 0.0),
                    request_statistics.get("cust_waiting_time", 0.0),
                ]
            )

        # Epoch Statistics (2 metrics) - only if tracking epochs
        if self.track_epochs:
            summary_list.extend(
                [
                    epoch_statistics.get("epochs", 0.0),
                    epoch_statistics.get("total_time", 0.0),
                ]
            )

        return summary_list

    def get_summary_data(self) -> Dict[str, Any]:
        """Get complete summary data organized by category.

        Returns:
            Dictionary with categorized summary statistics
        """
        if not self.is_tracking() or not self.summary:
            return {}

        result = {}
        if self.track_vehicles:
            result["vehicle_activities"] = self.get_vehicle_activities()
        if self.track_requests:
            result["request_statistics"] = self.get_request_statistics()
        if self.track_epochs:
            result["epoch_statistics"] = self.get_epoch_statistics()

        return result

    def get_vehicle_activities(self) -> Dict[str, float]:
        """Get vehicle activity metrics.

        Returns:
            Dictionary of vehicle time allocation metrics
        """
        if not self.summary:
            return {}

        return self.summary.get_vehicle_activities()

    def get_request_statistics(self) -> Dict[str, float]:
        """Get request service metrics.

        Returns:
            Dictionary of request statistics
        """
        if not self.summary:
            return {}

        return self.summary.get_request_statistics()

    def get_epoch_statistics(self) -> Dict[str, float]:
        """Get basic episode information.

        Returns:
            Dictionary of epoch statistics
        """
        if not self.summary:
            return {}

        return self.summary.get_epoch_statistics()
