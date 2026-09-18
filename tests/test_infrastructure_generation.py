import unittest

import numpy as np
import pytest

from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils import arg_utilities as arg_utilities
from openhail.utils import infrastructure_utilities

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("name", ["S05_1_4", "S05_2_2"])
def test_seeded_infrastructure_is_reproducible_and_respects_capacity(name):
    config, _ = arg_utilities.load_config(
        arg_utilities.get_parser(infrastructure=name, infrastructure_seed=1).parse_args(
            []
        )
    )
    settings = config["infrastructure"]
    for candidate_count in (103, 302, 1000):
        first = infrastructure_utilities.parse_infrastructure(config, candidate_count)
        second = infrastructure_utilities.parse_infrastructure(config, candidate_count)
        for expected, actual in zip(first, second):
            np.testing.assert_array_equal(expected, actual)
        chargers, locations = first
        assert chargers.shape == locations.shape == (candidate_count,)
        assert np.count_nonzero(locations) == settings["locations"]
        assert np.count_nonzero(chargers) == settings["selection"]
        assert np.all(locations[chargers > 0] > 0)
        assert np.all(chargers[chargers > 0] == settings["count"])


class Test_Infrastructure(unittest.TestCase):
    """Test infrastructure generation, initialization, and configuration."""

    def setUp(self):
        """Set up test fixtures for infrastructure tests."""

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

        # Create instance for infrastructure tests
        self.instance = OpenhailInstance(self.instance_config)

    def test_charging_infrastructure_initialization(self):
        """Test that charging infrastructure initializes correctly and consistently."""

        # Test 1: Check initial infrastructure state
        self.assertIsNotNone(self.instance.cs_idxs)
        self.assertIsNotNone(self.instance.charger_count)
        self.assertIsNotNone(self.instance.repo_idxs)
        self.assertIsInstance(self.instance.cs_idxs, np.ndarray)
        self.assertIsInstance(self.instance.charger_count, np.ndarray)

        # Test 2: Infrastructure should be within valid parking lot bounds
        max_lot_id = self.instance.D - 1
        self.assertTrue(np.all(self.instance.cs_idxs >= 0))
        self.assertTrue(np.all(self.instance.cs_idxs <= max_lot_id))
        self.assertTrue(np.all(self.instance.repo_idxs >= 0))
        self.assertTrue(np.all(self.instance.repo_idxs <= max_lot_id))

        # Test 3: Number of charging stations should be reasonable
        num_cs = len(self.instance.cs_idxs)
        num_repo = len(self.instance.repo_idxs)
        self.assertGreater(num_cs, 0, "Should have at least one charging station")
        self.assertGreater(num_repo, 0, "Should have at least one repository")
        self.assertLessEqual(
            num_cs, self.instance.D, "CS count should not exceed total parking lots"
        )
        self.assertLessEqual(
            num_repo, self.instance.D, "Repo count should not exceed total parking lots"
        )

    def test_custom_infrastructure_setting(self):
        """Test setting custom charging infrastructure."""

        # Get original infrastructure
        original_cs = self.instance.cs_idxs.copy()
        original_repo = self.instance.repo_idxs.copy()

        # Create a custom infrastructure solution
        infrastructure_config = self.instance_config.get("infrastructure", {})
        custom_solution = infrastructure_utilities.get_random_infrastructure(
            num_lots=self.instance.D,  # Use parking lots, not zones
            charger_config=infrastructure_config,
            seed=12345,
        )

        # Apply custom infrastructure
        self.instance.set_charging_infrastructure(custom_solution)

        # Verify infrastructure changed (at least one should be different)
        cs_changed = not np.array_equal(original_cs, self.instance.cs_idxs)
        repo_changed = not np.array_equal(original_repo, self.instance.repo_idxs)
        self.assertTrue(
            cs_changed or repo_changed,
            "Infrastructure should change when custom solution is applied",
        )

        # Verify new infrastructure is valid
        max_lot_id = self.instance.D - 1
        self.assertTrue(np.all(self.instance.cs_idxs >= 0))
        self.assertTrue(np.all(self.instance.cs_idxs <= max_lot_id))
        self.assertTrue(np.all(self.instance.repo_idxs >= 0))
        self.assertTrue(np.all(self.instance.repo_idxs <= max_lot_id))
