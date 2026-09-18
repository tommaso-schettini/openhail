"""Permutation-aware actor and independent attentive fleet critic."""

from collections.abc import Iterator

import torch
import torch.nn as nn

from .config import PeriodicPPOConfig


def _actor_mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Retain the actor architecture used by the initial PPO experiments."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
        nn.Tanh(),
    )


def _critic_token_encoder(input_dim: int, hidden_dim: int) -> nn.Sequential:
    """Encode one critic token without bounded, saturating activations."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.SiLU(),
    )


class CriticAttentionBlock(nn.Module):
    """Pre-normalized self-attention over an unordered fleet-state set."""

    def __init__(self, hidden_dim: int, num_heads: int, feedforward_dim: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(hidden_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.feedforward(self.feedforward_norm(tokens))


class AttentiveFleetCritic(nn.Module):
    """Estimate state value from time, vehicle, and infrastructure tokens."""

    _GLOBAL_TOKEN = 0
    _VEHICLE_TOKEN = 1
    _LOCATION_TOKEN = 2

    def __init__(
        self,
        config: PeriodicPPOConfig,
        vehicle_feature_dim: int,
        location_feature_dim: int,
        time_feature_dim: int,
    ) -> None:
        super().__init__()
        hidden_dim = config.critic_hidden_dim
        if hidden_dim % config.critic_attention_heads != 0:
            raise ValueError(
                "critic_hidden_dim must be divisible by critic_attention_heads"
            )

        self.vehicle_encoder = _critic_token_encoder(vehicle_feature_dim, hidden_dim)
        self.location_encoder = _critic_token_encoder(location_feature_dim, hidden_dim)
        self.time_encoder = _critic_token_encoder(time_feature_dim, hidden_dim)
        self.global_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.token_type_embeddings = nn.Parameter(torch.empty(3, hidden_dim))
        nn.init.normal_(self.token_type_embeddings, mean=0.0, std=0.02)

        self.attention_blocks = nn.ModuleList(
            CriticAttentionBlock(
                hidden_dim,
                config.critic_attention_heads,
                config.critic_feedforward_dim,
            )
            for _ in range(config.critic_attention_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        vehicle_features: torch.Tensor,
        location_features: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        vehicle_tokens = self.vehicle_encoder(vehicle_features)
        vehicle_tokens = (
            vehicle_tokens + self.token_type_embeddings[self._VEHICLE_TOKEN]
        )
        location_tokens = self.location_encoder(location_features)
        location_tokens = (
            location_tokens + self.token_type_embeddings[self._LOCATION_TOKEN]
        )

        batch_size = vehicle_features.shape[0]
        global_token = self.global_token.expand(batch_size, -1, -1)
        global_token = global_token + self.time_encoder(time_features).unsqueeze(1)
        global_token = global_token + self.token_type_embeddings[self._GLOBAL_TOKEN]
        tokens = torch.cat((global_token, vehicle_tokens, location_tokens), dim=1)
        for block in self.attention_blocks:
            tokens = block(tokens)

        fleet_embedding = self.output_norm(tokens[:, 0])
        return self.value_head(fleet_embedding).squeeze(-1)


class PeriodicActorCritic(nn.Module):
    """Score fleet actions and estimate value with fully separate parameters."""

    def __init__(
        self,
        config: PeriodicPPOConfig,
        vehicle_feature_dim: int,
        location_feature_dim: int,
        time_feature_dim: int,
    ) -> None:
        super().__init__()

        # Actor modules intentionally retain the architecture of the successful
        # sampled-policy experiment. The critic has no access to these parameters.
        self.vehicle_encoder = _actor_mlp(
            vehicle_feature_dim,
            config.vehicle_hidden_dim,
            config.vehicle_hidden_dim,
        )
        self.location_encoder = _actor_mlp(
            location_feature_dim,
            config.location_hidden_dim,
            config.location_hidden_dim,
        )

        context_input = (
            config.vehicle_hidden_dim + config.location_hidden_dim + time_feature_dim
        )
        self.context_encoder = _actor_mlp(
            context_input,
            config.context_hidden_dim,
            config.context_hidden_dim,
        )

        noop_input = config.vehicle_hidden_dim + config.context_hidden_dim
        self.noop_head = nn.Sequential(
            nn.Linear(noop_input, config.policy_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.policy_hidden_dim, 1),
        )

        pair_input = (
            config.vehicle_hidden_dim
            + config.location_hidden_dim
            + config.context_hidden_dim
        )
        self.location_head = nn.Sequential(
            nn.Linear(pair_input, config.policy_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.policy_hidden_dim, 2),
        )
        self.critic = AttentiveFleetCritic(
            config,
            vehicle_feature_dim,
            location_feature_dim,
            time_feature_dim,
        )

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate only over action-policy parameters."""
        actor_modules = (
            self.vehicle_encoder,
            self.location_encoder,
            self.context_encoder,
            self.noop_head,
            self.location_head,
        )
        for module in actor_modules:
            yield from module.parameters()

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate only over value-function parameters."""
        yield from self.critic.parameters()

    def forward_actor(
        self,
        vehicle_features: torch.Tensor,
        location_features: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        vehicle_embeddings = self.vehicle_encoder(vehicle_features)
        location_embeddings = self.location_encoder(location_features)

        fleet_context = vehicle_embeddings.mean(dim=1)
        infrastructure_context = location_embeddings.mean(dim=1)
        context = self.context_encoder(
            torch.cat((fleet_context, infrastructure_context, time_features), dim=-1)
        )

        batch_size, num_vehicles, _ = vehicle_embeddings.shape
        num_locations = location_embeddings.shape[1]
        vehicle_context = context.unsqueeze(1).expand(-1, num_vehicles, -1)
        noop_logits = self.noop_head(
            torch.cat((vehicle_embeddings, vehicle_context), dim=-1)
        )

        pair_vehicle = vehicle_embeddings.unsqueeze(2).expand(-1, -1, num_locations, -1)
        pair_location = location_embeddings.unsqueeze(1).expand(
            -1, num_vehicles, -1, -1
        )
        pair_context = context[:, None, None, :].expand(
            -1, num_vehicles, num_locations, -1
        )
        mode_logits = self.location_head(
            torch.cat((pair_vehicle, pair_location, pair_context), dim=-1)
        )
        mode_logits = mode_logits.reshape(batch_size, num_vehicles, -1)
        return torch.cat((noop_logits, mode_logits), dim=-1)

    def forward_value(
        self,
        vehicle_features: torch.Tensor,
        location_features: torch.Tensor,
        time_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.critic(vehicle_features, location_features, time_features)

    def forward(
        self,
        vehicle_features: torch.Tensor,
        location_features: torch.Tensor,
        time_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return native action logits and one global value per state."""
        logits = self.forward_actor(vehicle_features, location_features, time_features)
        values = self.forward_value(vehicle_features, location_features, time_features)
        return logits, values
