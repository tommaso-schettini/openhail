from ..core.openhail_instance import OpenhailInstance
from . import Agent, NearestAgent, RandomAgent


def initialize_agent(agent_config: dict, instance: OpenhailInstance) -> Agent:
    agent_str = agent_config["name"].split("_")[0]

    if agent_str == "nearest":
        return NearestAgent(instance, **agent_config)
    if agent_str == "periodic":
        from .periodic_ppo import PeriodicPPOAgent

        return PeriodicPPOAgent(instance, **agent_config)
    if agent_str == "random":
        return RandomAgent(instance, **agent_config)
    else:
        raise ValueError(f"Agent type {agent_str} not supported.")


# Backward compatibility function
def initialize_agent_from_config(agent_config: dict, instance_config: dict) -> Agent:
    """Create agent from config dictionaries (backward compatibility).

    Args:
        agent_config: Agent-specific configuration
        instance_config: Configuration for creating the OpenhailInstance

    Returns:
        Agent instance
    """
    instance = OpenhailInstance(instance_config)
    return initialize_agent(agent_config, instance)
