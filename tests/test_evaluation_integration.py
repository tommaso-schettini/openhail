"""Evaluation workflows and headless rendering."""

import numpy as np
import pytest

from openhail.agents.agent_initializer import initialize_agent
from openhail.core.openhail_env import OpenhailEnv
from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils import arg_utilities
from openhail.utils.config_runner import ConfigurationRunner
from openhail.utils.evaluation_utilities import HailEvaluator, SimulationResult

pytestmark = pytest.mark.integration


def test_incremental_evaluation_uses_seed_index_and_stops(monkeypatch):
    evaluator = HailEvaluator.__new__(HailEvaluator)
    evaluator.seeds = [100, 101]
    calls = []

    def simulate(seed):
        calls.append(seed)
        return 2.0, [], 0.1

    monkeypatch.setattr(evaluator, "simulate_seed", simulate)
    results = SimulationResult(1.0, [], 0.1)
    evaluator.simulate_next_seed(results)
    evaluator.simulate_next_seed(results)
    assert calls == [101]
    assert results.rewards == [1.0, 2.0]


def test_configuration_runner_executes_nearest():
    runner = ConfigurationRunner(render=False, stdout_logging=False)
    result = runner.run_single_config(
        simulation="debug_config_fast",
        evaluation="S1",
        city="nyc",
        horizon="horizon_1_2k",
        fleet="fleet_20",
        infrastructure="S05_1_4",
        agent="nearest",
    )
    assert np.isfinite(result.average_reward())
    assert result.average_reward() > 0


@pytest.fixture
def configured_instance():
    args = arg_utilities.get_parser(
        evaluation="S1",
        city="nyc",
        horizon="day_2",
        fleet="fleet_20",
        infrastructure="S05_1_4",
        agent="nearest",
    ).parse_args([])
    config, _ = arg_utilities.load_config(args)
    return config, OpenhailInstance(config)


@pytest.mark.parametrize("name", ["nearest", "random"])
def test_baseline_completes_episode(configured_instance, name, monkeypatch):
    config, instance = configured_instance
    agent = initialize_agent({"name": name, "seed": 123}, instance)
    episode_seeds = []
    monkeypatch.setattr(agent, "begin_episode", episode_seeds.append)
    evaluator = HailEvaluator(config, agent, render=False, sleep_enabled=False)
    assert evaluator.instance is instance
    try:
        result = evaluator.simulate()
        assert episode_seeds == evaluator.seeds
        assert np.isfinite(result.average_reward())
    finally:
        evaluator.env.close()


@pytest.mark.render
# Pygame imports this deprecated setuptools API while loading font resources.
# Scope the exception to that upstream message in this rendering check.
@pytest.mark.filterwarnings(
    "ignore:pkg_resources is deprecated as an API.*:UserWarning:pygame.pkgdata"
)
def test_offscreen_render_preserves_transition(configured_instance):
    config, instance = configured_instance
    env = OpenhailEnv(instance, config["simulation"], {"render_mode": "human"})
    agent = initialize_agent({"name": "nearest"}, instance)
    try:
        state, _ = env.reset(seed=123)
        action = agent.choose_action(state)
        # Render the same state repeatedly, then compare against an unrendered
        # transition after resetting to the same seed.
        env.render()
        env.render()
        rendered = env.step(action)
        state, _ = env.reset(seed=123)
        reference = env.step(agent.choose_action(state))
        assert rendered[1:4] == reference[1:4]
        for key in rendered[0]:
            np.testing.assert_equal(rendered[0][key], reference[0][key])
    finally:
        env.close()


def test_evaluator_rejects_empty_seed_configuration(configured_instance):
    config, instance = configured_instance
    config["evaluation"]["numeval"] = 0
    agent = initialize_agent({"name": "nearest"}, instance)
    evaluator = HailEvaluator(config, agent, sleep_enabled=False)
    try:
        with pytest.raises(ValueError, match="at least one seed"):
            evaluator.simulate()
    finally:
        evaluator.env.close()
