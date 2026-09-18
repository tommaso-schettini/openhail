# Skip collection before importing the optional PPO modules when torch is absent.
# ruff: noqa: E402
import argparse
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from openhail.agents import PeriodicPPOAgent
from openhail.agents.agent_initializer import initialize_agent
from openhail.core.constants import (
    CLOCK_EPOCH,
    DEST,
    END_OF_HORIZON,
    EPOCH_TYPE,
    JOB_NULL,
    LOC,
    NEW_REQUEST,
    ORIG,
    PROC_TIME,
    TIME,
    TYPE,
    Q,
)
from openhail.core.openhail_env import OpenhailEnv
from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils import arg_utilities, state_utilities

pytestmark = [pytest.mark.integration, pytest.mark.ppo]


def build_environment(infrastructure="S05_1_4", num_vehicles=None):
    args = argparse.Namespace(
        simulation="debug_config_fast",
        evaluation="S1",
        city="nyc",
        horizon="horizon_1_2k",
        fleet="fleet_20",
        infrastructure=infrastructure,
        agent="periodic_ppo",
    )
    config, _ = arg_utilities.load_config(args)
    config["simulation"]["decision_clock"] = 900.0
    config["simulation"]["strict_validation"] = True
    if num_vehicles is not None:
        config["fleet"]["num_vehicles"] = num_vehicles
    instance = OpenhailInstance(config)
    return OpenhailEnv(instance, config["simulation"], {"render_mode": None})


def test_periodic_features_are_finite_and_have_declared_shapes():
    env = build_environment()
    state, _ = env.reset(seed=123)
    agent = PeriodicPPOAgent(env.instance)

    vehicle, locations, time = agent.feature_extractor.extract(state)

    assert vehicle.shape == (
        env.instance.num_evs,
        agent.feature_extractor.vehicle_feature_dim,
    )
    assert locations.shape == (
        env.instance.D_repo,
        agent.feature_extractor.location_feature_dim,
    )
    assert time.shape == (agent.feature_extractor.time_feature_dim,)
    assert np.all(np.isfinite(vehicle))
    assert np.all(np.isfinite(locations))
    assert np.all(np.isfinite(time))


def test_vehicle_feature_dimension_is_independent_of_location_count():
    five_location_env = build_environment("S05_1_4")
    ten_location_env = build_environment("S10_2_5")

    five_location_agent = PeriodicPPOAgent(five_location_env.instance)
    ten_location_agent = PeriodicPPOAgent(ten_location_env.instance)

    assert five_location_env.instance.D_repo == 5
    assert ten_location_env.instance.D_repo == 10
    assert (
        five_location_agent.feature_extractor.vehicle_feature_dim
        == ten_location_agent.feature_extractor.vehicle_feature_dim
    )


def test_closest_assignment_is_used_outside_policy_epochs():
    env = build_environment()
    state, _ = env.reset(seed=123)
    agent = PeriodicPPOAgent(env.instance)

    while int(state[EPOCH_TYPE]) != NEW_REQUEST:
        action = agent.choose_action(state)
        assert not agent.last_policy_decision
        state, _, terminated, _, _ = env.step(action)
        assert not terminated

    expected = state_utilities.compute_nearest_assignment(
        env.instance, state["request"], state["V"]
    )
    action = agent.choose_action(state)

    np.testing.assert_array_equal(action[0], expected)
    assert not agent.last_policy_decision
    assert env.action_space.contains(action)


def test_sparse_non_policy_fallback_matches_dense_feasibility_mask():
    env = build_environment()
    state, _ = env.reset(seed=123)
    agent = PeriodicPPOAgent(env.instance)
    vehicles = state["V"]
    request = state["request"]

    request[ORIG][0] = vehicles[LOC][0]
    request[DEST][0] = vehicles[LOC][0]
    request[PROC_TIME][0] = 0.0
    request[TIME][0] = float(state["time"])
    vehicles[TYPE][1] = JOB_NULL
    vehicles[Q][1] = env.instance.ev_max_Q

    serve_action = state_utilities.compute_nearest_assignment(
        env.instance,
        request,
        vehicles,
    )
    post_assignment = state_utilities.update_state(
        env.instance,
        vehicles,
        request,
        serve_action,
    )
    dense_mask, dense_distances = state_utilities.compute_rnr_mask(
        env.instance,
        post_assignment,
    )
    expected = np.zeros(env.instance.num_evs, dtype=np.int32)
    assigned = {
        int(vehicle) for vehicle in serve_action if int(vehicle) < env.instance.num_evs
    }
    for vehicle in np.flatnonzero(~dense_mask[:, 0]):
        if int(vehicle) in assigned:
            continue
        feasible = np.flatnonzero(dense_mask[vehicle, 1:])
        destination = int(feasible[np.argmin(dense_distances[vehicle, feasible])])
        expected[vehicle] = 2 * destination + 1

    _, actual, _ = agent.choose_action(state)

    assert not agent.last_policy_decision
    np.testing.assert_array_equal(actual, expected)


def test_policy_acts_only_at_clock_and_respects_native_mask():
    env = build_environment()
    state, _ = env.reset(seed=123)
    agent = PeriodicPPOAgent(env.instance)

    while int(state[EPOCH_TYPE]) != CLOCK_EPOCH:
        action = agent.choose_action(state)
        state, _, terminated, _, _ = env.step(action)
        assert not terminated

    action = agent.choose_action(state)
    assert agent.last_policy_decision
    assert env.action_space.contains(action)

    post_assignment = state_utilities.update_state(
        env.instance, state["V"], state["request"], action[0]
    )
    rnr_mask, _ = state_utilities.compute_rnr_mask(env.instance, post_assignment)
    native_mask = agent._native_action_mask(rnr_mask)
    for vehicle, native_action in enumerate(action[1]):
        assert native_mask[vehicle, native_action]


def test_charge_actions_are_masked_at_zero_capacity_locations():
    env = build_environment()
    state, _ = env.reset(seed=123)
    agent = PeriodicPPOAgent(env.instance)
    rnr_mask, _ = state_utilities.compute_rnr_mask(env.instance, state["V"])

    native_mask = agent._native_action_mask(rnr_mask)

    for destination, capacity in enumerate(env.instance.charger_count):
        charge_action = 2 * destination + 2
        if capacity > 0:
            np.testing.assert_array_equal(
                native_mask[:, charge_action], rnr_mask[:, destination + 1]
            )
        else:
            assert not np.any(native_mask[:, charge_action])


def test_sampled_evaluation_is_reproducible_and_does_not_train():
    env = build_environment()
    state, _ = env.reset(seed=123)
    state[EPOCH_TYPE] = np.asarray(CLOCK_EPOCH, dtype=np.int32)
    agent = PeriodicPPOAgent(
        env.instance,
        training=False,
        sample_actions=True,
        seed=456,
    )

    agent.begin_episode(789)
    first_action = agent.choose_action(state)
    agent.observe(10.0, state, True)
    agent.begin_episode(789)
    second_action = agent.choose_action(state)

    for first, second in zip(first_action, second_action):
        np.testing.assert_array_equal(first, second)
    assert agent.total_updates == 0
    assert agent._active_decision is None
    assert agent._rollout == []


def test_two_decision_rollout_performs_finite_ppo_update():
    env = build_environment()
    state, _ = env.reset(seed=123)
    state[EPOCH_TYPE] = np.asarray(CLOCK_EPOCH, dtype=np.int32)
    agent = PeriodicPPOAgent(
        env.instance,
        training=True,
        update_epochs=1,
        minibatch_decisions=2,
        seed=123,
    )
    agent.begin_episode(123)

    agent.choose_action(state)
    next_state = copy.deepcopy(state)
    next_state["time"] = np.asarray(900.0)
    next_state[EPOCH_TYPE] = np.asarray(CLOCK_EPOCH, dtype=np.int32)
    agent.observe(10.0, next_state, False)

    agent.choose_action(next_state)
    final_state = copy.deepcopy(next_state)
    final_state["time"] = np.asarray(env.instance.max_time)
    final_state[EPOCH_TYPE] = np.asarray(END_OF_HORIZON, dtype=np.int32)
    agent.observe(5.0, final_state, True)
    stats = agent.pop_training_stats()

    assert stats["updates"] == 1
    for metric in (
        "policy_entropy",
        "policy_loss",
        "approximate_kl",
        "clip_fraction",
        "value_loss",
        "explained_variance",
        "gradient_norm",
        "policy_gradient_norm",
        "value_gradient_norm",
    ):
        assert np.isfinite(stats[metric])
    assert 0.0 <= stats["clip_fraction"] <= 1.0


def test_critic_is_independent_attentive_and_permutation_invariant():
    env = build_environment()
    state, _ = env.reset(seed=123)
    agent = PeriodicPPOAgent(env.instance)
    vehicle, locations, time = agent.feature_extractor.extract(state)
    vehicle_tensor, location_tensor, time_tensor = agent._feature_tensors(
        (vehicle, locations, time)
    )

    actor_parameters = tuple(agent.network.actor_parameters())
    critic_parameters = tuple(agent.network.critic_parameters())
    assert set(map(id, actor_parameters)).isdisjoint(map(id, critic_parameters))
    assert any(
        isinstance(module, torch.nn.MultiheadAttention)
        for module in agent.network.critic.modules()
    )
    assert not any(
        isinstance(module, torch.nn.Tanh) for module in agent.network.critic.modules()
    )

    vehicle_permutation = torch.randperm(vehicle_tensor.shape[1])
    location_permutation = torch.randperm(location_tensor.shape[1])
    with torch.no_grad():
        value = agent.network.forward_value(
            vehicle_tensor, location_tensor, time_tensor
        )
        permuted_value = agent.network.forward_value(
            vehicle_tensor[:, vehicle_permutation],
            location_tensor[:, location_permutation],
            time_tensor,
        )
    torch.testing.assert_close(value, permuted_value, atol=1e-6, rtol=1e-6)

    agent.network.zero_grad(set_to_none=True)
    agent.network.forward_value(
        vehicle_tensor, location_tensor, time_tensor
    ).sum().backward()
    assert all(parameter.grad is None for parameter in actor_parameters)
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in critic_parameters
    )


def test_joint_ratio_compounds_all_eligible_vehicle_probabilities():
    num_vehicles = 20
    old_probabilities = torch.full((1, num_vehicles), 0.5)
    new_probabilities = torch.full((1, num_vehicles), 0.6)
    eligible = torch.ones((1, num_vehicles), dtype=torch.bool)

    old_joint = PeriodicPPOAgent._joint_log_probabilities(
        old_probabilities.log(), eligible
    )
    new_joint = PeriodicPPOAgent._joint_log_probabilities(
        new_probabilities.log(), eligible
    )
    joint_ratio = torch.exp(new_joint - old_joint)

    assert torch.isclose(joint_ratio, torch.tensor([1.2**num_vehicles]))
    assert joint_ratio.item() > 1.2


def test_joint_probability_excludes_forced_vehicle_actions():
    selected_log_probabilities = torch.tensor(
        [[-0.2, -0.3, -100.0], [-0.4, -100.0, -100.0]]
    )
    eligible = torch.tensor(
        [[True, True, False], [True, False, False]], dtype=torch.bool
    )

    joint = PeriodicPPOAgent._joint_log_probabilities(
        selected_log_probabilities, eligible
    )

    torch.testing.assert_close(joint, torch.tensor([-0.5, -0.4]))


def test_checkpoint_round_trip_preserves_greedy_action(tmp_path):
    env = build_environment()
    state, _ = env.reset(seed=123)
    state[EPOCH_TYPE] = np.asarray(CLOCK_EPOCH, dtype=np.int32)
    agent = PeriodicPPOAgent(
        env.instance,
        training=False,
        sample_actions=False,
    )
    expected_action = agent.choose_action(state)

    checkpoint = tmp_path / "periodic_ppo.pt"
    agent.save(checkpoint)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["checkpoint_version"] == 3
    assert saved["feature_schema"] == "target_location_features_v1"
    assert "actor_optimizer" in saved
    assert "critic_optimizer" in saved
    loaded = PeriodicPPOAgent(
        env.instance,
        training=False,
        sample_actions=False,
        checkpoint_path=str(checkpoint),
    )
    actual_action = loaded.choose_action(state)

    for expected, actual in zip(expected_action, actual_action):
        np.testing.assert_array_equal(expected, actual)


def test_checkpoint_loads_with_different_fleet_and_location_counts(tmp_path):
    training_env = build_environment("S05_1_4")
    checkpoint = tmp_path / "periodic_ppo.pt"
    PeriodicPPOAgent(training_env.instance).save(checkpoint)

    evaluation_env = build_environment("S40_10_2", num_vehicles=100)
    state, _ = evaluation_env.reset(seed=123)
    state[EPOCH_TYPE] = np.asarray(CLOCK_EPOCH, dtype=np.int32)
    loaded = PeriodicPPOAgent(
        evaluation_env.instance,
        training=False,
        sample_actions=False,
        checkpoint_path=str(checkpoint),
    )
    action = loaded.choose_action(state)

    assert evaluation_env.instance.D_repo == 40
    assert loaded.feature_extractor.vehicle_feature_dim == 23
    assert action[1].shape == (100,)
    assert evaluation_env.action_space.contains(action)


def test_agent_initializer_supports_periodic_ppo():
    env = build_environment()
    agent = initialize_agent({"name": "periodic_ppo", "training": False}, env.instance)
    assert isinstance(agent, PeriodicPPOAgent)
    assert agent.sample_actions


def test_ppo_refreshes_features_after_infrastructure_change():
    env = build_environment()
    agent = PeriodicPPOAgent(env.instance, sample_actions=False)
    charging = np.zeros(env.instance.D, dtype=np.int32)
    reposition = np.zeros_like(charging)
    charging[:2] = 4
    reposition[:10] = 1
    agent.set_infrastructure((charging, reposition))
    env.set_infrastructure((charging, reposition))
    try:
        state, _ = env.reset(seed=123)
        state[EPOCH_TYPE] = np.asarray(CLOCK_EPOCH, dtype=np.int32)
        assert agent.feature_extractor.num_locations == 10
        assert env.action_space.contains(agent.choose_action(state))
    finally:
        env.close()
