"""Periodic PPO agent with deterministic closest request assignment."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from ...core.constants import CLOCK_EPOCH, EPOCH_TYPE, JOB_NULL, TYPE
from ...core.openhail_instance import OpenhailInstance
from ...utils import state_utilities
from ..agent import Agent
from .config import PeriodicPPOConfig
from .feature_extractor import PeriodicFeatureExtractor
from .model import PeriodicActorCritic


class PeriodicPPOAgent(Agent):
    """Learn fleet repositioning at clock epochs and assign requests by distance."""

    def __init__(
        self,
        instance: OpenhailInstance,
        training: bool = False,
        sample_actions: bool = True,
        checkpoint_path: str = "",
        **kwargs,
    ) -> None:
        super().__init__(instance)
        config_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in PeriodicPPOConfig.__dataclass_fields__
        }
        self.config = PeriodicPPOConfig(**config_kwargs)
        if checkpoint_path:
            self.config.checkpoint_path = checkpoint_path

        self.training = training
        self.sample_actions = sample_actions
        self.device = torch.device(self.config.device)
        self.num_actions = 1 + 2 * instance.D_repo
        self.feature_extractor = PeriodicFeatureExtractor(instance)
        self.network = PeriodicActorCritic(
            self.config,
            self.feature_extractor.vehicle_feature_dim,
            self.feature_extractor.location_feature_dim,
            self.feature_extractor.time_feature_dim,
        ).to(self.device)
        self._actor_parameters = tuple(self.network.actor_parameters())
        self._critic_parameters = tuple(self.network.critic_parameters())
        self.actor_optimizer = torch.optim.Adam(
            self._actor_parameters, lr=self.config.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self._critic_parameters, lr=self.config.critic_learning_rate
        )

        generator_device = "cuda" if self.device.type == "cuda" else "cpu"
        self.generator = torch.Generator(device=generator_device)
        self.generator.manual_seed(self.config.seed)

        self._active_decision: dict[str, Any] | None = None
        self._rollout: list[dict[str, Any]] = []
        self._latest_training_stats = self._empty_training_stats()
        self.total_policy_decisions = 0
        self.total_updates = 0

        self.last_policy_decision = False
        self.last_empty_mask_count = 0
        self.last_policy_entropy = 0.0

        if self.config.checkpoint_path:
            self.load(self.config.checkpoint_path)

    def begin_episode(self, seed: int | None = None) -> None:
        """Clear episodic state and optionally seed stochastic action selection."""
        self._active_decision = None
        self._rollout = []
        self.last_policy_decision = False
        self.last_empty_mask_count = 0
        self.last_policy_entropy = 0.0
        if seed is not None:
            self.generator.manual_seed(int(seed))

    def set_infrastructure(self, solution) -> None:
        """Refresh feature dimensions when reposition locations change."""
        super().set_infrastructure(solution)
        self.num_actions = 1 + 2 * self.instance.D_repo
        self.feature_extractor = PeriodicFeatureExtractor(self.instance)
        self.begin_episode()

    def set_mode(self, *, training: bool, sample_actions: bool) -> None:
        """Set learning and action-selection modes independently."""
        self.training = training
        self.sample_actions = sample_actions

    def choose_action(self, state: Dict[str, Any]):
        """Return closest assignment and periodic fleet actions."""
        vehicles = state["V"]
        request = state["request"]
        serve_action = state_utilities.compute_nearest_assignment(
            self.instance, request, vehicles
        )
        post_assignment_vehicles = state_utilities.update_state(
            self.instance, vehicles, request, serve_action
        )

        self.last_policy_decision = int(state[EPOCH_TYPE]) == CLOCK_EPOCH
        self.last_empty_mask_count = 0
        self.last_policy_entropy = 0.0

        if self.last_policy_decision:
            # This particular policy opts the eligible fleet into feasibility
            # at its periodic decisions. The feasibility API itself is not
            # clock-gated and other agents may opt in at any epoch.
            rnr_mask, _ = state_utilities.compute_rnr_mask(
                self.instance, post_assignment_vehicles
            )
            rnr_action = self._choose_policy_action(
                state,
                post_assignment_vehicles,
                rnr_mask,
            )
        else:
            rnr_action = self._non_policy_action(
                serve_action,
                post_assignment_vehicles,
            )

        planned_epoch = np.asarray(-1.0, dtype=np.float64)
        return serve_action, rnr_action, planned_epoch

    def observe(
        self,
        reward: float,
        next_state: Dict[str, Any],
        done: bool,
    ) -> None:
        """Accumulate rewards between decisions and train at episode end."""
        if not self.training or self._active_decision is None:
            return

        self._active_decision["reward"] += float(reward) * self.config.reward_scale
        reached_next_decision = int(next_state[EPOCH_TYPE]) == CLOCK_EPOCH
        if reached_next_decision or done:
            self._active_decision["done"] = bool(done)
            self._rollout.append(self._active_decision)
            self._active_decision = None

        if done and self._rollout:
            self._latest_training_stats = self._update_policy()
            self._rollout = []

    def pop_training_stats(self) -> dict[str, float]:
        """Return the latest episode's optimizer diagnostics and clear them."""
        stats = self._latest_training_stats
        self._latest_training_stats = self._empty_training_stats()
        return stats

    def _choose_policy_action(
        self,
        state: Dict[str, Any],
        post_assignment_vehicles: Dict[str, np.ndarray],
        rnr_mask: np.ndarray,
    ) -> np.ndarray:
        if self.training and self._active_decision is not None:
            raise RuntimeError(
                "A new periodic action was requested before the previous reward "
                "interval was finalized."
            )

        action_mask = self._native_action_mask(rnr_mask)
        feasible_destinations = np.any(rnr_mask[:, 1:], axis=1)
        # Every feasible destination supplies at least the odd repositioning
        # action. Only vehicles that cannot no-op therefore need an explicit
        # nonempty-destination check; rescanning the wider native mask is
        # equivalent but redundant.
        empty = (~rnr_mask[:, 0]) & (~feasible_destinations)
        self.last_empty_mask_count = int(empty.sum())
        if np.any(empty):
            vehicles = np.flatnonzero(empty).tolist()
            raise RuntimeError(f"Vehicles have no feasible native action: {vehicles}")

        features = self.feature_extractor.extract(
            state, vehicles=post_assignment_vehicles
        )
        vehicle_tensor, location_tensor, time_tensor = self._feature_tensors(features)
        with torch.no_grad():
            logits, value = self.network(vehicle_tensor, location_tensor, time_tensor)
            masked_logits = logits.masked_fill(
                ~torch.as_tensor(action_mask, device=self.device).unsqueeze(0),
                -1e9,
            )
            log_probabilities = F.log_softmax(masked_logits, dim=-1)
            probabilities = log_probabilities.exp()
            if self.sample_actions:
                actions = torch.multinomial(
                    probabilities.squeeze(0),
                    num_samples=1,
                    generator=self.generator,
                ).squeeze(-1)
            else:
                actions = masked_logits.squeeze(0).argmax(dim=-1)
            selected_log_probabilities = (
                log_probabilities.squeeze(0)
                .gather(1, actions.unsqueeze(-1))
                .squeeze(-1)
            )
            entropies = -(probabilities.squeeze(0) * log_probabilities.squeeze(0)).sum(
                dim=-1
            )

        if np.any(feasible_destinations):
            self.last_policy_entropy = float(
                entropies[torch.as_tensor(feasible_destinations, device=self.device)]
                .mean()
                .cpu()
                .item()
            )

        actions_array = actions.cpu().numpy().astype(np.int32)
        self.total_policy_decisions += 1
        if self.training:
            self._active_decision = {
                "vehicle_features": features[0],
                "location_features": features[1],
                "time_features": features[2],
                "action_mask": action_mask,
                "eligible": feasible_destinations,
                "actions": actions_array,
                "old_log_probabilities": selected_log_probabilities.cpu().numpy(),
                "value": float(value.item()),
                "reward": 0.0,
                "done": False,
            }
        return actions_array

    def _non_policy_action(
        self,
        serve_action: np.ndarray,
        vehicles: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """No-op except for vehicles that must receive a simulator job."""
        actions = np.zeros(self.instance.num_evs, dtype=np.int32)
        assigned_vehicles = {
            int(vehicle)
            for vehicle in np.asarray(serve_action).tolist()
            if int(vehicle) < self.instance.num_evs
        }
        must_reposition = np.flatnonzero(vehicles[TYPE] == JOB_NULL)
        fallback_vehicles = np.asarray(
            [
                int(vehicle)
                for vehicle in must_reposition
                if int(vehicle) not in assigned_vehicles
            ],
            dtype=np.intp,
        )
        if fallback_vehicles.size == 0:
            return actions

        # Nonperiodic service completions are also decision opportunities.
        # Request feasibility for precisely the vehicles this branch intends
        # to reposition, regardless of the epoch that triggered the call.
        feasibility, distances = state_utilities.compute_rnr_feasibility(
            self.instance,
            vehicles,
            fallback_vehicles,
        )
        for position, vehicle in enumerate(fallback_vehicles):
            feasible_destinations = np.flatnonzero(feasibility[position])
            if feasible_destinations.size == 0:
                self.last_empty_mask_count += 1
                raise RuntimeError(
                    f"Vehicle {vehicle} must reposition but has no feasible "
                    "destination."
                )
            destination = int(
                feasible_destinations[
                    np.argmin(distances[position, feasible_destinations])
                ]
            )
            actions[vehicle] = 2 * destination + 1
        return actions

    def _native_action_mask(self, rnr_mask: np.ndarray) -> np.ndarray:
        action_mask = np.zeros((self.instance.num_evs, self.num_actions), dtype=bool)
        action_mask[:, 0] = rnr_mask[:, 0]
        feasible = rnr_mask[:, 1:]
        action_mask[:, 1::2] = feasible
        action_mask[:, 2::2] = feasible & (self.instance.charger_count[None, :] > 0)
        return action_mask

    def _feature_tensors(self, features):
        return tuple(
            torch.as_tensor(feature, dtype=torch.float32, device=self.device).unsqueeze(
                0
            )
            for feature in features
        )

    def _update_policy(self) -> dict[str, float]:
        rewards = np.asarray(
            [transition["reward"] for transition in self._rollout],
            dtype=np.float32,
        )
        values = np.asarray(
            [transition["value"] for transition in self._rollout],
            dtype=np.float32,
        )
        dones = np.asarray(
            [transition["done"] for transition in self._rollout],
            dtype=np.float32,
        )
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for index in reversed(range(len(self._rollout))):
            next_value = 0.0 if index == len(values) - 1 else values[index + 1]
            continuation = 1.0 - dones[index]
            delta = (
                rewards[index]
                + self.config.gamma * next_value * continuation
                - values[index]
            )
            gae = (
                delta + self.config.gamma * self.config.gae_lambda * continuation * gae
            )
            advantages[index] = gae
        returns = advantages + values
        if len(advantages) > 1 and float(advantages.std()) > 1e-8:
            normalized_advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )
        else:
            normalized_advantages = advantages.copy()

        update_stats: list[dict[str, float]] = []
        decision_count = len(self._rollout)
        batch_size = max(1, self.config.minibatch_decisions)
        for _ in range(self.config.update_epochs):
            permutation = (
                torch.randperm(
                    decision_count, generator=self.generator, device=self.device
                )
                .cpu()
                .numpy()
            )
            for start in range(0, decision_count, batch_size):
                indices = permutation[start : start + batch_size]
                stats = self._ppo_minibatch(
                    indices,
                    normalized_advantages,
                    returns,
                )
                update_stats.append(stats)
                self.total_updates += 1

        with torch.no_grad():
            predicted_values = self._predict_rollout_values()
        target_variance = float(np.var(returns))
        explained_variance = (
            0.0
            if target_variance < 1e-8
            else 1.0 - float(np.var(returns - predicted_values)) / target_variance
        )
        return {
            "updates": float(len(update_stats)),
            "policy_entropy": float(
                np.mean([stats["policy_entropy"] for stats in update_stats])
            ),
            "policy_loss": float(
                np.mean([stats["policy_loss"] for stats in update_stats])
            ),
            "approximate_kl": float(
                np.mean([stats["approximate_kl"] for stats in update_stats])
            ),
            "clip_fraction": float(
                np.mean([stats["clip_fraction"] for stats in update_stats])
            ),
            "value_loss": float(
                np.mean([stats["value_loss"] for stats in update_stats])
            ),
            "explained_variance": float(explained_variance),
            "gradient_norm": float(
                np.mean([stats["gradient_norm"] for stats in update_stats])
            ),
            "policy_gradient_norm": float(
                np.mean([stats["policy_gradient_norm"] for stats in update_stats])
            ),
            "value_gradient_norm": float(
                np.mean([stats["value_gradient_norm"] for stats in update_stats])
            ),
        }

    def _ppo_minibatch(
        self,
        indices: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        transitions = [self._rollout[int(index)] for index in indices]
        vehicle_features = torch.as_tensor(
            np.stack([item["vehicle_features"] for item in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        location_features = torch.as_tensor(
            np.stack([item["location_features"] for item in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        time_features = torch.as_tensor(
            np.stack([item["time_features"] for item in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        action_mask = torch.as_tensor(
            np.stack([item["action_mask"] for item in transitions]),
            dtype=torch.bool,
            device=self.device,
        )
        eligible = torch.as_tensor(
            np.stack([item["eligible"] for item in transitions]),
            dtype=torch.bool,
            device=self.device,
        )
        actions = torch.as_tensor(
            np.stack([item["actions"] for item in transitions]),
            dtype=torch.long,
            device=self.device,
        )
        old_log_probabilities = torch.as_tensor(
            np.stack([item["old_log_probabilities"] for item in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        advantages_tensor = torch.as_tensor(
            advantages[indices], dtype=torch.float32, device=self.device
        )
        returns_tensor = torch.as_tensor(
            returns[indices], dtype=torch.float32, device=self.device
        )

        logits, predicted_values = self.network(
            vehicle_features, location_features, time_features
        )
        masked_logits = logits.masked_fill(~action_mask, -1e9)
        log_probabilities = F.log_softmax(masked_logits, dim=-1)
        probabilities = log_probabilities.exp()
        selected_log_probabilities = log_probabilities.gather(
            2, actions.unsqueeze(-1)
        ).squeeze(-1)
        new_joint_log_probabilities = self._joint_log_probabilities(
            selected_log_probabilities, eligible
        )
        old_joint_log_probabilities = self._joint_log_probabilities(
            old_log_probabilities, eligible
        )
        joint_log_ratios = new_joint_log_probabilities - old_joint_log_probabilities
        joint_ratios = torch.exp(joint_log_ratios)

        unclipped = joint_ratios * advantages_tensor
        clipped = (
            torch.clamp(
                joint_ratios,
                1.0 - self.config.clip_ratio,
                1.0 + self.config.clip_ratio,
            )
            * advantages_tensor
        )
        value_loss = 0.5 * F.mse_loss(predicted_values, returns_tensor)
        entropy_by_vehicle = -(probabilities * log_probabilities).sum(dim=-1)
        valid_decisions = torch.any(eligible, dim=1)
        if torch.any(valid_decisions):
            policy_loss = -torch.minimum(unclipped, clipped)[valid_decisions].mean()
            policy_entropy = entropy_by_vehicle[eligible].mean()
            approximate_kl = ((joint_ratios - 1.0) - joint_log_ratios)[
                valid_decisions
            ].mean()
            clip_fraction = (
                (torch.abs(joint_ratios - 1.0) > self.config.clip_ratio)[
                    valid_decisions
                ]
                .float()
                .mean()
            )
        else:
            # A decision with only forced no-ops can still train the critic, but
            # it does not provide an actor-learning signal.
            policy_loss = logits.sum() * 0.0
            policy_entropy = entropy_by_vehicle.sum() * 0.0
            approximate_kl = joint_log_ratios.sum() * 0.0
            clip_fraction = joint_log_ratios.sum() * 0.0
        actor_loss = policy_loss - self.config.entropy_coefficient * policy_entropy
        critic_loss = self.config.value_coefficient * value_loss

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.critic_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        critic_loss.backward()
        policy_gradient_norm = torch.nn.utils.clip_grad_norm_(
            self._actor_parameters, self.config.max_gradient_norm
        )
        value_gradient_norm = torch.nn.utils.clip_grad_norm_(
            self._critic_parameters, self.config.max_gradient_norm
        )
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        gradient_norm = torch.maximum(policy_gradient_norm, value_gradient_norm)
        return {
            "policy_entropy": float(policy_entropy.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "approximate_kl": float(approximate_kl.detach().cpu().item()),
            "clip_fraction": float(clip_fraction.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "gradient_norm": float(gradient_norm.detach().cpu().item()),
            "policy_gradient_norm": float(policy_gradient_norm.detach().cpu().item()),
            "value_gradient_norm": float(value_gradient_norm.detach().cpu().item()),
        }

    @staticmethod
    def _joint_log_probabilities(
        selected_log_probabilities: torch.Tensor,
        eligible: torch.Tensor,
    ) -> torch.Tensor:
        """Return one log probability for each factorized fleet action."""
        return (
            selected_log_probabilities * eligible.to(selected_log_probabilities.dtype)
        ).sum(dim=-1)

    def _predict_rollout_values(self) -> np.ndarray:
        vehicle_features = torch.as_tensor(
            np.stack([item["vehicle_features"] for item in self._rollout]),
            dtype=torch.float32,
            device=self.device,
        )
        location_features = torch.as_tensor(
            np.stack([item["location_features"] for item in self._rollout]),
            dtype=torch.float32,
            device=self.device,
        )
        time_features = torch.as_tensor(
            np.stack([item["time_features"] for item in self._rollout]),
            dtype=torch.float32,
            device=self.device,
        )
        values = self.network.forward_value(
            vehicle_features, location_features, time_features
        )
        return values.cpu().numpy()

    @staticmethod
    def _empty_training_stats() -> dict[str, float]:
        return {
            "updates": 0.0,
            "policy_entropy": 0.0,
            "policy_loss": 0.0,
            "approximate_kl": 0.0,
            "clip_fraction": 0.0,
            "value_loss": 0.0,
            "explained_variance": 0.0,
            "gradient_norm": 0.0,
            "policy_gradient_norm": 0.0,
            "value_gradient_norm": 0.0,
        }

    def save(self, path: str | Path) -> None:
        checkpoint = {
            "checkpoint_version": 3,
            "feature_schema": "target_location_features_v1",
            "config": asdict(self.config),
            "network": self.network.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_policy_decisions": self.total_policy_decisions,
            "total_updates": self.total_updates,
        }
        torch.save(checkpoint, Path(path))
        logging.info("PeriodicPPO: saved checkpoint to %s", path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path), map_location=self.device, weights_only=True)
        checkpoint_version = int(checkpoint.get("checkpoint_version", 1))
        if checkpoint_version != 3:
            raise ValueError(
                "This checkpoint predates the location-count-independent target "
                "feature schema. Start a fresh run or load a version-3 checkpoint."
            )
        if checkpoint.get("feature_schema") != "target_location_features_v1":
            raise ValueError("Unsupported PPO checkpoint feature schema")
        self.network.load_state_dict(checkpoint["network"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.total_policy_decisions = int(checkpoint.get("total_policy_decisions", 0))
        self.total_updates = int(checkpoint.get("total_updates", 0))
        logging.info("PeriodicPPO: loaded checkpoint from %s", path)
