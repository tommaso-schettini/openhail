from gymnasium.envs.registration import register

from .openhail_env import OpenhailEnv as OpenhailEnv

register(
    id="openhail-v0",
    entry_point="openhail.core.openhail_env:OpenhailEnv",
)
