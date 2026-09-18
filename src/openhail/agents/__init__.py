from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .periodic_ppo import PeriodicPPOAgent

from .agent import Agent
from .nearest import NearestAgent
from .random_agent import RandomAgent

__all__ = [
    "Agent",
    "NearestAgent",
    "PeriodicPPOAgent",
    "RandomAgent",
]


def __getattr__(name):
    """Load optional agents only when they are explicitly requested."""
    if name == "PeriodicPPOAgent":
        from .periodic_ppo import PeriodicPPOAgent

        return PeriodicPPOAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
