import unittest

import numpy as np
import pandas as pd
import pytest

from openhail.core.openhail_env import OpenhailEnv
from openhail.core.openhail_instance import OpenhailInstance
from openhail.core.request.request_generator import request_generator
from openhail.core.request.request_nyc import request_nyc
from openhail.utils import arg_utilities


@pytest.mark.integration
class TestRequestGeneration(unittest.TestCase):
    """Request sampling, reproducibility, and time boundaries."""

    def setUp(self):
        """Set up test fixtures with a smaller instance for faster testing."""

        # Create a test configuration
        parser = arg_utilities.get_parser(
            evaluation="S1",
            city="nyc",
            horizon="horizon_1_2k",
            fleet="fleet_20",
            infrastructure="S05_1_4",
            agent="nearest",
        )

        args = parser.parse_args([])
        self.instance_config, self.extra_config = arg_utilities.load_config(args)

        # Create instance
        self.instance = OpenhailInstance(self.instance_config)
        self.test_doy = int(self.instance.request_generator.doys[0])

    def test_request_generation_deterministic(self):
        """Test that the same seed always produces identical requests."""

        seed = 123
        num_requests = 10

        # Generate requests twice with same seed
        requests1 = self.instance.generate_requests_for_T(
            doy=self.test_doy, seed=seed, num_requests=num_requests
        )
        requests2 = self.instance.generate_requests_for_T(
            doy=self.test_doy, seed=seed, num_requests=num_requests
        )

        # Should be identical
        self.assertEqual(len(requests1), len(requests2))

        for field in ["time", "orig"]:
            if field in requests1.columns:
                if field == "orig":
                    # For coordinate arrays
                    orig1 = np.array(requests1[field].tolist())
                    orig2 = np.array(requests2[field].tolist())
                    np.testing.assert_array_equal(
                        orig1, orig2, f"Field {field} should be identical"
                    )
                else:
                    # For scalar fields
                    np.testing.assert_array_equal(
                        requests1[field].to_numpy(),
                        requests2[field].to_numpy(),
                        f"Field {field} should be identical",
                    )

    def test_request_generation_different_seeds(self):
        """Test that different seeds produce different requests."""

        num_requests = 10
        requests_seed1 = self.instance.generate_requests_for_T(
            doy=self.test_doy, seed=111, num_requests=num_requests
        )
        requests_seed2 = self.instance.generate_requests_for_T(
            doy=self.test_doy, seed=222, num_requests=num_requests
        )

        # Should have same structure but different content
        self.assertEqual(len(requests_seed1), len(requests_seed2))

        # Times should be different (very unlikely to be identical with different seeds)
        times1 = requests_seed1["time"].to_numpy()
        times2 = requests_seed2["time"].to_numpy()

        self.assertFalse(
            np.array_equal(times1, times2),
            "Different seeds should produce different request times",
        )

    def test_request_properties_validation(self):
        """Test that generated requests have valid properties."""

        requests = self.instance.generate_requests_for_T(
            doy=self.test_doy, seed=456, num_requests=20
        )

        # Test required fields exist
        required_fields = ["time", "orig"]
        for field in required_fields:
            self.assertIn(field, requests.columns, f"Missing required field: {field}")

        # Test time bounds
        times = requests["time"].to_numpy()
        self.assertTrue(
            np.all(times >= self.instance.start_time),
            "All times should be >= start_time",
        )
        self.assertTrue(
            np.all(times < self.instance.max_time), "All times should be < max_time"
        )

        # Test coordinate validity
        origins = np.array(requests["orig"].tolist())

        self.assertEqual(origins.shape[1], 2, "Origins should be 2D coordinates")
        self.assertTrue(np.all(origins > 0), "All coordinates should be positive")

    def test_nyc_filter_uses_day_and_half_open_time_interval(self):
        """Requests must come from the requested day and [start, end) interval."""
        generator = self.instance.request_generator
        assert isinstance(generator, request_nyc)
        original = generator.requests_df
        generator.requests_df = pd.DataFrame(
            {
                "doy": [8, 8, 8, 9],
                "episode_time": [99.0, 100.0, 200.0, 150.0],
            }
        )
        try:
            requests = generator.get_df(doy=8, start_time=100.0, end_time=200.0)
        finally:
            generator.requests_df = original

        self.assertEqual(requests["episode_time"].tolist(), [100.0])
        self.assertEqual(requests["doy"].tolist(), [8])

    def test_requests_arrive_before_episode_horizon(self):
        """Every real request must allow a decision strictly before termination."""
        simulation_config = self.instance_config["simulation"]
        env = OpenhailEnv(
            instance=self.instance,
            simulation_config=simulation_config,
            render_config={"render_mode": None},
        )

        for seed in (100002, 100017):
            with self.subTest(seed=seed):
                env.reset(seed=seed)
                real_request_times = env.request_manager.request_time[:-1]
                self.assertTrue(np.all(real_request_times < self.instance.max_time))
                self.assertEqual(
                    env.request_manager.request_time[-1], self.instance.max_time + 1
                )

    def test_overlapping_request_timestamps_are_rerolled(self):
        """Simultaneous sampled requests must be replaced deterministically."""
        simulation_config = self.instance_config["simulation"]
        env = OpenhailEnv(
            instance=self.instance,
            simulation_config=simulation_config,
            render_config={"render_mode": None},
        )

        env.reset(seed=100009)
        real_request_times = env.request_manager.request_time[:-1]
        self.assertEqual(len(np.unique(real_request_times)), len(real_request_times))

    def test_environment_integration(self):
        """Test that environment properly integrates with request generation."""

        simulation_config = {
            "strict_validation": True,
            "max_outstanding_requests": 1,
            "clear_requests": True,
            "decision_clock": -1,
            "max_interdecision_time": -1,
            "end_of_charge_epoch": True,
            "end_of_service_epoch": True,
            "request_epoch": True,
            "track_vehicles": True,
            "track_requests": True,
            "track_epochs": True,
        }

        env = OpenhailEnv(instance=self.instance, simulation_config=simulation_config)

        # Test environment initialization
        self.assertIsNotNone(env.request_manager)
        self.assertIsNotNone(env.vehicle_manager)
        self.assertIsNotNone(env.summary_manager)

        # Test state structure
        state, _ = env.reset(seed=42)
        self.assertIsNotNone(state)
        self.assertIn("V", state)
        self.assertIn("request", state)
        self.assertIn("time", state)

        # Test that request manager has been initialized with requests
        # Note: At reset time, no requests are "active" yet, but they should be loaded
        self.assertIsNotNone(env.request_manager.request_time)
        self.assertGreater(len(env.request_manager.request_time), 0)

        # Test that next request time is reasonable
        next_request_time = env.request_manager.next_request_time()
        self.assertGreater(next_request_time, env.time)
        self.assertLess(next_request_time, env.instance.max_time)


def test_destination_randomization_uses_destination_probability():
    generator = request_generator(
        name="synthetic",
        doys=[1],
        zone_ids=[3],
        tris={},
        start_time=0,
        max_time=100,
        orig_randomization=0.0,
        dest_randomization=1.0,
        time_randomization=0.0,
        midpoint=np.zeros(2),
    )
    generator._rand_pt_in_zone = lambda zone_id, rng: np.array([zone_id, 0.0])
    source = pd.DataFrame({"episode_time": [1.0], "puzone": [1], "dozone": [2]})

    requests = generator._generate_from_df(source, seed=1, num_requests=1)

    np.testing.assert_array_equal(requests.iloc[0]["orig"], [1.0, 0.0])
    np.testing.assert_array_equal(requests.iloc[0]["dest"], [3.0, 0.0])


def test_oversampled_requests_receive_unique_reproducible_times():
    generator = request_generator(
        name="synthetic",
        doys=[1],
        zone_ids=[1, 2],
        tris={},
        start_time=0,
        max_time=100,
        orig_randomization=0.0,
        dest_randomization=0.0,
        time_randomization=0.0,
        midpoint=np.zeros(2),
    )
    generator._rand_pt_in_zone = lambda zone_id, rng: np.array([zone_id, 0.0])
    source = pd.DataFrame(
        {
            "episode_time": [10.0, 20.0],
            "puzone": [1, 2],
            "dozone": [2, 1],
        }
    )

    first = generator._generate_from_df(source, seed=7, num_requests=5)
    second = generator._generate_from_df(source, seed=7, num_requests=5)

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 5
    assert first["time"].is_unique
    assert first["time"].is_monotonic_increasing
    assert first["time"].between(0.0, 100.0, inclusive="left").all()


def test_request_cache_name_changes_with_source_content(tmp_path):
    source = tmp_path / "requests.csv"
    source.write_text("first", encoding="utf-8")
    generator = request_generator(
        name=source.name,
        doys=[1],
        zone_ids=[],
        tris={},
        start_time=0,
        max_time=100,
        orig_randomization=0.0,
        dest_randomization=0.0,
        time_randomization=0.0,
        midpoint=np.zeros(2),
    )
    generator.request_path = str(source)
    first = generator._cache_name(1, 1, 1, 0, 100)

    source.write_text("second", encoding="utf-8")
    generator._source_hashes.clear()
    second = generator._cache_name(1, 1, 1, 0, 100)

    assert first != second
