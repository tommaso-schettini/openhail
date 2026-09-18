import argparse

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from openhail.agents import NearestAgent, RandomAgent
from openhail.core.constants import (
    CAPACITY,
    CHARGER,
    CHARGERS,
    CLOCK_EPOCH,
    DEST,
    END_OF_CHARGE,
    END_OF_HORIZON,
    END_OF_REPO,
    END_OF_SERVE,
    EPOCH_TYPE,
    JOB_CHARGE,
    JOB_GO_CHARGE,
    JOB_IDLE,
    JOB_NULL,
    JOB_QUEUE,
    JOB_REPO,
    LOC,
    NEW_REQUEST,
    NEXT_EPOCH,
    NULL_EPOCH,
    OCCUPANCY,
    ORIG,
    PROC_TIME,
    QUEUE,
    T_DEST,
    TARGET_CHARGER,
    TIME,
    TYPE,
    Q,
)
from openhail.core.openhail_env import OpenhailEnv
from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils import arg_utilities, state_utilities

pytestmark = pytest.mark.integration


ANALYSIS_INFRASTRUCTURES = (
    "S05_1_1",
    "S05_1_2",
    "S05_1_4",
    "S05_1_6",
    "S05_2_1",
    "S05_2_2",
    "S05_2_3",
    "S05_2_4",
    "S05_3_2",
)


def build_environment(infrastructure="S05_1_4", strict_validation=True):
    args = argparse.Namespace(
        simulation="debug_config_fast",
        evaluation="S1",
        city="nyc",
        horizon="horizon_1_2k",
        fleet="fleet_20",
        infrastructure=infrastructure,
        agent="nearest",
    )
    config, _ = arg_utilities.load_config(args)
    config["simulation"]["strict_validation"] = strict_validation
    config["simulation"]["track_vehicles"] = True
    config["simulation"]["track_requests"] = True
    config["simulation"]["track_epochs"] = True
    instance = OpenhailInstance(config)
    return OpenhailEnv(instance, config["simulation"], {"render_mode": None})


# Requested epochs have no finite upper cap (post-horizon requests are ignored),
# and request processing distance has no declared finite cap. Keep that public
# contract; suppress only the checker's two advisory bound warnings here.
@pytest.mark.filterwarnings(
    "ignore:.*A Box action space maximum value is infinity.*:"
    "UserWarning:gymnasium.utils.env_checker"
)
@pytest.mark.filterwarnings(
    "ignore:.*A Box observation space maximum value is infinity.*:"
    "UserWarning:gymnasium.utils.env_checker"
)
def test_environment_passes_gymnasium_checker_in_normal_mode():
    env = build_environment(strict_validation=False)
    check_env(env, skip_render_check=True)


def test_requested_epochs_at_or_after_horizon_are_ignored():
    env = build_environment()
    env.reset(seed=123)
    env.time = env.instance.max_time - 1
    env.vehicle_manager.V[NEXT_EPOCH][:] = env.instance.max_time + 1
    env.vehicle_manager.epoch_type[:] = NULL_EPOCH
    env.request_manager.request_idx = env.request_manager.dummy_request

    next_time, epoch_type = env._get_next_epoch(env.instance.max_time + 100)

    assert next_time == env.instance.max_time
    assert epoch_type == END_OF_HORIZON


def test_observations_do_not_alias_internal_vehicle_state():
    env = build_environment()
    state, _ = env.reset(seed=123)
    internal_location = env.vehicle_manager.V[LOC][0].copy()

    state["V"][LOC][0] += 1000

    np.testing.assert_array_equal(env.vehicle_manager.V[LOC][0], internal_location)


def test_set_infrastructure_rebuilds_dependent_state_and_spaces():
    env = build_environment()
    env.reset(seed=123)
    chargers = np.zeros(env.instance.D, dtype=np.int32)
    repositories = np.zeros(env.instance.D, dtype=np.int32)
    repositories[:3] = 100
    chargers[0] = 2

    env.set_infrastructure((chargers, repositories))
    state = env.get_state()

    assert env.instance.D_repo == 3
    assert state[CHARGERS][OCCUPANCY].shape == (3,)
    reposition_space = env.action_space[1]
    assert isinstance(reposition_space, spaces.MultiDiscrete)
    assert np.all(reposition_space.nvec == 7)
    assert env.observation_space.contains(state)


def test_repository_distances_remain_correct_after_vehicle_movement():
    env = build_environment()
    try:
        state, _ = env.reset(seed=123)
        instance = env.instance
        locations = state["V"][LOC].copy()
        indices = np.arange(instance.num_evs)
        for movement in (0.0, 0.0, 1.0, -2.0):
            locations[3] += np.array([movement, -movement])
            expected = np.linalg.norm(
                locations[:, None, :] - instance.repo_coords[None, :, :], axis=2
            )
            actual = instance.repository_distances(locations, indices)
            np.testing.assert_allclose(actual, expected)
    finally:
        env.close()


def test_sparse_rnr_matches_full_mask_nearest_actions():
    env = build_environment()
    state, _ = env.reset(seed=123)
    agent = NearestAgent(env.instance)

    for _ in range(100):
        serve_action = state_utilities.compute_nearest_assignment(
            env.instance,
            state["request"],
            state["V"],
        )
        post_assignment = state_utilities.update_state(
            env.instance,
            state["V"],
            state["request"],
            serve_action,
        )
        full_mask, full_distances = state_utilities.compute_rnr_mask(
            env.instance,
            post_assignment,
        )
        expected = agent.choose_rnr(
            post_assignment,
            full_mask,
            full_distances,
        )
        actual = agent.choose_sparse_rnr(post_assignment)
        np.testing.assert_array_equal(actual, expected)

        state, _, terminated, _, _ = env.step((serve_action, actual, np.asarray(-1.0)))
        if terminated:
            break


def test_single_active_request_assignment_preserves_request_slot():
    env = build_environment()
    state, _ = env.reset(seed=123)
    midpoint = env.instance.midpoint
    origin = state["V"][LOC][0]
    request = {
        ORIG: np.stack((midpoint, origin, midpoint)),
        DEST: np.stack((midpoint, origin, midpoint)),
        PROC_TIME: np.array((-1.0, 0.0, -1.0)),
        TIME: np.array(
            (
                env.instance.max_time + 1,
                state["V"][TIME][0],
                env.instance.max_time + 1,
            )
        ),
    }

    assignment = state_utilities.compute_nearest_assignment(
        env.instance,
        request,
        state["V"],
    )

    assert assignment[0] == env.instance.num_evs
    assert assignment[1] < env.instance.num_evs
    assert assignment[2] == env.instance.num_evs


def test_vehicle_epoch_ties_use_only_absolute_time_tolerance():
    env = build_environment()
    next_time = 100_000.0
    vehicle_epochs = np.full(env.instance.num_evs, env.instance.max_time + 1.0)
    vehicle_epoch_types = np.full(env.instance.num_evs, NULL_EPOCH)
    vehicle_epochs[:2] = (next_time + 0.5, next_time)
    vehicle_epoch_types[:2] = (END_OF_SERVE, END_OF_REPO)

    selected = env._select_vehicle_epoch_type_at_time(
        vehicle_epochs,
        vehicle_epoch_types,
        next_time,
    )

    assert selected == END_OF_REPO


def test_charger_state_is_public_and_valid():
    env = build_environment()
    state, _ = env.reset(seed=123)

    assert CHARGERS in state
    assert set(state[CHARGERS]) == {LOC, CAPACITY, OCCUPANCY, QUEUE}
    assert state[CHARGERS][LOC].shape == (env.instance.D_repo, 2)
    assert state[CHARGERS][CAPACITY].shape == (env.instance.D_repo,)
    assert state[CHARGERS][OCCUPANCY].shape == (env.instance.D_repo,)
    assert state[CHARGERS][QUEUE].shape == (env.instance.D_repo,)
    assert env.observation_space.contains(state)


@pytest.mark.parametrize("infrastructure", ANALYSIS_INFRASTRUCTURES)
def test_analysis_matrix_configurations_have_valid_initial_contract(infrastructure):
    env = build_environment(infrastructure)
    state, _ = env.reset(seed=123)
    agent = RandomAgent(env.instance, seed=123)

    assert env.observation_space.contains(state)
    assert env.action_space.contains(agent.choose_action(state))


def test_declared_action_space_contains_real_agent_actions():
    env = build_environment()
    state, _ = env.reset(seed=123)

    nearest = NearestAgent(env.instance)
    nearest.set_doy(env.request_manager.doy)
    assert nearest.q_thresh == 0.2 * env.instance.ev_max_Q
    nearest_action = nearest.choose_action(state)
    assert nearest_action[2].shape == ()
    assert env.action_space.contains(nearest_action)

    random = RandomAgent(env.instance, seed=123)
    random.set_doy(env.request_manager.doy)
    action = random.choose_action(state)
    assert action[2].shape == ()
    assert env.action_space.contains(action)

    serve_action, rnr_action, _ = action
    post_assignment = state_utilities.update_state(
        env.instance, state["V"], state["request"], serve_action
    )
    mask, _ = state_utilities.compute_rnr_mask(env.instance, post_assignment)
    for vehicle_idx, native_action in enumerate(rnr_action):
        if native_action == 0:
            assert mask[vehicle_idx, 0]
        else:
            destination = int(np.ceil(native_action / 2))
            assert mask[vehicle_idx, destination]


def test_random_agent_excludes_charging_at_zero_capacity_locations(monkeypatch):
    env = build_environment("S05_3_2")
    state, _ = env.reset(seed=123)
    zero_capacity_destination = int(np.flatnonzero(env.instance.charger_count == 0)[0])
    state["V"][LOC][:] = env.instance.repo_coords[zero_capacity_destination]
    state["V"][Q][:] = env.instance.ev_max_Q

    offered_actions = []

    class RecordingChoice:
        def choice(self, values):
            offered_actions.append(tuple(values))
            return values[0]

    agent = RandomAgent(env.instance, seed=123)
    monkeypatch.setattr(agent, "rng", RecordingChoice())
    agent.choose_action(state)

    invalid_charge_action = 2 * zero_capacity_destination + 2
    assert offered_actions
    assert all(invalid_charge_action not in actions for actions in offered_actions)
    for actions in offered_actions:
        charge_actions = np.asarray(
            [action for action in actions if action > 0 and action % 2 == 0],
            dtype=np.int32,
        )
        destinations = charge_actions // 2 - 1
        assert np.all(env.instance.charger_count[destinations] > 0)


def test_periodic_random_samples_repositioning_only_at_clock_epochs(monkeypatch):
    env = build_environment()
    state, _ = env.reset(seed=123)
    state["V"][TYPE][:] = JOB_IDLE

    offered_actions = []

    class RecordingChoice:
        def choice(self, values):
            offered_actions.append(tuple(values))
            return values[0]

    agent = RandomAgent(
        env.instance,
        seed=123,
        periodic_repositioning=True,
    )
    monkeypatch.setattr(agent, "rng", RecordingChoice())

    state[EPOCH_TYPE] = np.asarray(NEW_REQUEST, dtype=np.int32)
    _, nonperiodic_action, _ = agent.choose_action(state)
    assert not offered_actions
    assert np.all(nonperiodic_action == 0)

    state[EPOCH_TYPE] = np.asarray(CLOCK_EPOCH, dtype=np.int32)
    _, periodic_action, _ = agent.choose_action(state)
    assert offered_actions
    assert env.action_space[1].contains(periodic_action)


def test_periodic_random_repositions_jobless_vehicles_between_clocks(monkeypatch):
    env = build_environment()
    state, _ = env.reset(seed=123)
    state[EPOCH_TYPE] = np.asarray(NEW_REQUEST, dtype=np.int32)
    state["V"][TYPE][:] = JOB_IDLE
    state["V"][TYPE][0] = JOB_NULL
    monkeypatch.setattr(
        state_utilities,
        "compute_nearest_assignment",
        lambda instance, request, vehicles: np.full(
            len(request[ORIG]),
            instance.num_evs,
            dtype=np.int32,
        ),
    )

    agent = RandomAgent(
        env.instance,
        seed=123,
        periodic_repositioning=True,
    )
    _, rnr_action, _ = agent.choose_action(state)

    assert rnr_action[0] > 0
    assert rnr_action[0] % 2 == 1
    assert np.all(rnr_action[1:] == 0)


def test_nearest_low_soc_actions_target_real_chargers():
    env = build_environment("S05_3_2")
    state, _ = env.reset(seed=123)
    agent = NearestAgent(env.instance)
    charger_destinations = np.flatnonzero(env.instance.charger_count > 0)

    for vehicle_idx in range(env.instance.num_evs):
        destination = charger_destinations[vehicle_idx % len(charger_destinations)]
        state["V"][LOC][vehicle_idx] = env.instance.repo_coords[destination]
    state["V"][Q][:] = agent.q_thresh

    _, rnr_action, _ = agent.choose_action(state)
    assert np.all((rnr_action > 0) & (rnr_action % 2 == 0))

    selected_destinations = rnr_action // 2 - 1
    assert np.all(env.instance.charger_count[selected_destinations] > 0)

    charger_distances = state_utilities.geometry.distance_matrix(
        state["V"][LOC], env.instance.repo_coords[charger_destinations]
    )
    expected_destinations = charger_destinations[np.argmin(charger_distances, axis=1)]
    np.testing.assert_array_equal(selected_destinations, expected_destinations)


def test_step_info_and_request_summary_are_consistent():
    env = build_environment()
    state, _ = env.reset(seed=123)
    agent = NearestAgent(env.instance)
    agent.set_doy(env.request_manager.doy)

    observed_service = False
    for _ in range(20):
        action = agent.choose_action(state)
        state, reward, terminated, _, info = env.step(action)
        assert np.isclose(
            reward,
            info["reward_components"]["service"]
            + info["reward_components"]["reposition"],
        )
        assert info["delta_time"] >= 0
        if info["requests_served"]:
            observed_service = True
            break
        assert not terminated

    assert observed_service
    request_summary = env.get_episode_summary()["request_statistics"]
    assert request_summary["requests_served"] == 1


def test_service_summary_uses_pre_assignment_vehicle_state():
    env = build_environment()
    state, _ = env.reset(seed=123)
    agent = NearestAgent(env.instance)
    agent.set_doy(env.request_manager.doy)

    for _ in range(20):
        action = agent.choose_action(state)
        assigned = np.flatnonzero(action[0] < env.instance.num_evs)
        if assigned.size:
            request_idx = int(assigned[0])
            vehicle_idx = int(action[0][request_idx])
            preprocess_time = (
                state_utilities.geometry.get_distance(
                    state["V"][LOC][vehicle_idx],
                    state["request"]["orig"][request_idx],
                )
                / env.instance.ev_speed
            )
            expected_wait = (
                state["V"][TIME][vehicle_idx]
                + preprocess_time
                - state["request"][TIME][request_idx]
            )
            env.step(action)
            summary = env.get_episode_summary()["request_statistics"]
            assert np.isclose(summary["cust_waiting_time"], expected_wait)
            return
        state, _, terminated, _, _ = env.step(action)
        assert not terminated

    pytest.fail("No request was served within 20 decisions.")


def test_charging_cost_is_applied_to_charging_energy():
    env = build_environment()
    env.reset(seed=123)
    vehicle_idx = 0
    env.instance.cost_charge = 2.5
    env.vehicle_manager.V[TYPE][vehicle_idx] = JOB_CHARGE
    env.vehicle_manager.V[TIME][vehicle_idx] = 0.0
    env.vehicle_manager.V[T_DEST][vehicle_idx] = 100.0

    result = env.vehicle_manager.advance_job_queue(vehicle_idx, 10.0, 0.0)

    expected_cost = 10.0 * env.instance.ev_charge_rate * env.instance.cost_charge
    assert result.charge_time == 10.0
    assert np.isclose(result.reward, -expected_cost)


def test_idle_vehicle_repositioning_starts_at_current_epoch():
    env = build_environment()
    env.reset(seed=123)
    vehicle_idx = 0
    current_time = 10_000.0
    env.vehicle_manager.V[TYPE][vehicle_idx] = JOB_IDLE
    env.vehicle_manager.V[TIME][vehicle_idx] = 0.0

    env.vehicle_manager.process_reposition_action(vehicle_idx, 1, current_time)

    assert env.vehicle_manager.V[TIME][vehicle_idx] == current_time
    assert env.vehicle_manager.V[NEXT_EPOCH][vehicle_idx] >= current_time


def test_charging_intention_change_resets_stationary_job_timestamps():
    env = build_environment()
    env.reset(seed=123)
    vehicle_idx = 0
    current_time = 10_000.0
    vehicles = env.vehicle_manager.V
    vehicles[TARGET_CHARGER][vehicle_idx] = 0
    vehicles[TYPE][vehicle_idx] = JOB_QUEUE
    vehicles[TIME][vehicle_idx] = 1_000.0
    vehicles[T_DEST][vehicle_idx] = 1_000.0

    # Odd native action: stay at this destination without charging.
    env.vehicle_manager.process_reposition_action(vehicle_idx, 1, current_time)

    assert vehicles[TYPE][vehicle_idx] == JOB_IDLE
    assert vehicles[TIME][vehicle_idx] == current_time
    assert vehicles[T_DEST][vehicle_idx] == current_time


def test_batched_reposition_processing_matches_scalar_processing():
    batch_env = build_environment("S05_3_2")
    scalar_env = build_environment("S05_3_2")
    batch_env.reset(seed=123)
    scalar_env.reset(seed=123)

    target = int(np.flatnonzero(batch_env.instance.charger_count > 0)[0])
    other_target = (target + 1) % batch_env.instance.D_repo
    odd_action = 2 * target + 1
    even_action = odd_action + 1
    actions = np.zeros(batch_env.instance.num_evs, dtype=np.int64)
    actions[:9] = (
        even_action,
        odd_action,
        even_action,
        even_action,
        odd_action,
        odd_action,
        even_action,
        odd_action,
        2 * other_target + 1,
    )

    def configure(manager):
        vehicles = manager.V
        vehicles[TARGET_CHARGER][:8] = target
        vehicles[TYPE][:8] = (
            JOB_REPO,
            JOB_GO_CHARGE,
            JOB_IDLE,
            JOB_IDLE,
            JOB_QUEUE,
            JOB_CHARGE,
            JOB_QUEUE,
            JOB_REPO,
        )
        vehicles[Q][2] = manager.instance.ev_max_Q / 2
        vehicles[Q][3] = manager.instance.ev_max_Q
        vehicles[CHARGER][5] = target
        manager.current_occupancy[target] = 1
        manager.WAITLIST[:8] = np.arange(8, dtype=float)
        vehicles[Q][8] = manager.instance.ev_max_Q

    configure(batch_env.vehicle_manager)
    configure(scalar_env.vehicle_manager)

    current_time = 10_000.0
    batch_env.vehicle_manager.process_reposition_actions(actions, current_time)
    for vehicle_idx in np.flatnonzero(actions):
        scalar_env.vehicle_manager.process_reposition_action(
            int(vehicle_idx),
            int(actions[vehicle_idx]),
            current_time,
        )

    for key in batch_env.vehicle_manager.V:
        np.testing.assert_array_equal(
            batch_env.vehicle_manager.V[key],
            scalar_env.vehicle_manager.V[key],
        )
    np.testing.assert_array_equal(
        batch_env.vehicle_manager.WAITLIST,
        scalar_env.vehicle_manager.WAITLIST,
    )
    np.testing.assert_array_equal(
        batch_env.vehicle_manager.epoch_type,
        scalar_env.vehicle_manager.epoch_type,
    )
    np.testing.assert_array_equal(
        batch_env.vehicle_manager.current_occupancy,
        scalar_env.vehicle_manager.current_occupancy,
    )


def test_advance_to_time_aggregates_vehicle_activity():
    env = build_environment()
    env.reset(seed=123)
    vehicles = env.vehicle_manager.V
    vehicles[TYPE][:] = JOB_IDLE
    vehicles[TYPE][0] = JOB_REPO
    vehicles[TYPE][1] = JOB_GO_CHARGE
    vehicles[TYPE][2] = JOB_CHARGE
    vehicles[TYPE][3] = JOB_QUEUE
    vehicles[T_DEST][:4] = 100.0
    vehicles[NEXT_EPOCH][:4] = 100.0
    charger_idx = int(np.flatnonzero(env.instance.charger_count > 0)[0])
    vehicles[TARGET_CHARGER][3] = charger_idx
    vehicles[NEXT_EPOCH][3] = env.instance.max_time + 1
    env.vehicle_manager.WAITLIST[3] = 0.0
    env.vehicle_manager.current_occupancy[charger_idx] = int(
        env.instance.charger_count[charger_idx]
    )

    result = env.vehicle_manager.advance_to_time(10.0, 0.0)

    assert result.repos_time == 20.0
    assert result.charge_time == 10.0
    assert result.queue_time == 10.0
    assert result.idle_time == 10.0 * (env.instance.num_evs - 4)
    expected_reward = -(
        result.repos_time * env.instance.ev_speed * env.instance.cost_travel
        + result.charge_time * env.instance.ev_charge_rate * env.instance.cost_charge
    )
    assert np.isclose(result.reward, expected_reward)


def test_simultaneous_charge_completion_preserves_fifo_queue_order():
    env = build_environment()
    env.reset(seed=123)
    manager = env.vehicle_manager
    vehicles = manager.V
    charger_idx = int(np.flatnonzero(env.instance.charger_count > 0)[0])
    capacity = int(env.instance.charger_count[charger_idx])

    charging_vehicle = 0
    first_waiter = 1
    second_waiter = 2
    manager.current_occupancy[charger_idx] = capacity

    vehicles[TYPE][charging_vehicle] = JOB_CHARGE
    vehicles[CHARGER][charging_vehicle] = charger_idx
    vehicles[TARGET_CHARGER][charging_vehicle] = charger_idx
    vehicles[TIME][charging_vehicle] = 0.0
    vehicles[T_DEST][charging_vehicle] = 10.0
    vehicles[NEXT_EPOCH][charging_vehicle] = 10.0
    vehicles[Q][charging_vehicle] = env.instance.ev_max_Q / 2
    manager.epoch_type[charging_vehicle] = END_OF_CHARGE

    for vehicle_idx, wait_time in ((first_waiter, 1.0), (second_waiter, 2.0)):
        vehicles[TYPE][vehicle_idx] = JOB_QUEUE
        vehicles[TARGET_CHARGER][vehicle_idx] = charger_idx
        vehicles[TIME][vehicle_idx] = 0.0
        vehicles[T_DEST][vehicle_idx] = 0.0
        vehicles[NEXT_EPOCH][vehicle_idx] = env.instance.max_time + 1
        manager.epoch_type[vehicle_idx] = NULL_EPOCH
        manager.WAITLIST[vehicle_idx] = wait_time

    manager.advance_to_time(10.0, 0.0)

    assert vehicles[TYPE][charging_vehicle] == JOB_IDLE
    assert vehicles[TYPE][first_waiter] == JOB_CHARGE
    assert vehicles[TYPE][second_waiter] == JOB_QUEUE
    assert manager.current_occupancy[charger_idx] == capacity


def test_registered_environment_reaches_horizon_after_ten_thousand_steps():
    import gymnasium as gym

    config, _ = arg_utilities.load_config(arg_utilities.get_parser().parse_args([]))
    config["fleet"]["num_vehicles"] = 1
    config["horizon"].update(start_time=0, max_time=10010, num_requests=1)
    config["simulation"]["decision_clock"] = 1.0
    instance = OpenhailInstance(config)
    env = gym.make(
        "openhail-v0", instance=instance, simulation_config=config["simulation"]
    )
    agent = NearestAgent(instance)
    try:
        state, _ = env.reset(seed=123)
        for steps in range(1, 10020):
            state, _, terminated, truncated, _ = env.step(agent.choose_action(state))
            if terminated or truncated:
                break
        else:
            pytest.fail("Episode did not finish at its configured horizon")
        assert terminated and not truncated
        assert steps > 10000
        assert float(state["time"]) == pytest.approx(instance.max_time)
    finally:
        env.close()
