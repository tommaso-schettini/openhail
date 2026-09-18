from typing import Any, Dict

import numpy as np

from ..core.openhail_instance import OpenhailInstance
from ..utils.infrastructure_utilities import ChargingInfrastructure


class Agent(object):
    """Base class for agents for the ridehail environment.

    This class provides the interface that all ridehail agents must implement.
    Agents are responsible for making decisions about vehicle assignments,
    repositioning, and charging in the simulation environment.
    """

    def __init__(self, instance: OpenhailInstance, **kwargs) -> None:
        """Initialize the agent with a ridehail instance.

        Args:
            instance: The OpenhailInstance containing simulation parameters
            **kwargs: Additional agent-specific parameters
        """
        self.instance = instance
        return

    # Backward compatibility constructor
    @classmethod
    def from_config(cls, instance_config: dict, **kwargs):
        """Create agent from config dictionary (backward compatibility).

        Args:
            instance_config: Configuration for creating the OpenhailInstance
            **kwargs: Additional agent-specific parameters

        Returns:
            Agent instance
        """
        instance = OpenhailInstance(instance_config)
        return cls(instance, **kwargs)

    def set_infrastructure(self, solution: ChargingInfrastructure) -> None:
        """Set the charging infrastructure for the simulation.

        Args:
            solution: Charging infrastructure configuration
        """
        self.instance.set_charging_infrastructure(solution)

    def set_doy(self, doy: int) -> None:
        """Set the day of year for request generation.

        Args:
            doy: Day of year (1-365)
        """
        pass

    def begin_episode(self, seed: int | None = None) -> None:
        """Reset optional episodic policy state before evaluation."""

    def choose_action(self, state: Dict[str, Any]) -> tuple[Any, Any, np.ndarray]:
        """Choose an action given the current state.

        Args:
            state: Current environment state containing vehicle and request information

        Returns:
            Tuple of (serve_action, rnr_action, planned_epoch)

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("choose_action must be implemented by subclasses")
