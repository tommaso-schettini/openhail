"""Configuration for the periodic repositioning PPO agent."""

from dataclasses import dataclass


@dataclass
class PeriodicPPOConfig:
    """Hyperparameters for the periodic actor and attentive fleet critic."""

    learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    reward_scale: float = 0.01
    update_epochs: int = 4
    minibatch_decisions: int = 32

    vehicle_hidden_dim: int = 64
    location_hidden_dim: int = 32
    context_hidden_dim: int = 64
    policy_hidden_dim: int = 64
    critic_hidden_dim: int = 64
    critic_attention_heads: int = 4
    critic_attention_layers: int = 2
    critic_feedforward_dim: int = 128

    device: str = "cpu"
    seed: int = 0
    checkpoint_path: str = ""
