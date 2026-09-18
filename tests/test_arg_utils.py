import unittest

import pytest

from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils import arg_utilities as arg_utilities
from openhail.utils import logging_utilities as log_utils

pytestmark = pytest.mark.integration


class Test_ArgUtils(unittest.TestCase):
    def test_parse_args(self):
        log_utils.init_logging_stdout()
        parser = arg_utilities.get_parser(
            infrastructure="S05_1_4",
            infrastructure_seed=1,
        )
        args = parser.parse_args([])
        instance_config, extra_config = arg_utilities.load_config(args)

        self.assertEqual(instance_config["evaluation"]["name"], "S1")
        self.assertEqual(instance_config["horizon"]["name"], "horizon_1_2k")
        self.assertEqual(instance_config["city"]["name"], "nyc")
        self.assertEqual(instance_config["fleet"]["name"], "F20")
        self.assertEqual(instance_config["infrastructure"]["name"], "S05_1_4")
        self.assertEqual(instance_config["infrastructure"]["seed"], 1)
        self.assertEqual(extra_config["agent"]["name"], "nearest")


if __name__ == "__main__":
    unittest.main()


def test_overrides_keep_underscored_field_names_and_types():
    parser = arg_utilities.get_parser(
        simulation_decision_clock=900.0,
        simulation_clear_requests=True,
        fleet_num_vehicles=20,
        horizon_request_destination_randomization=0.25,
    )
    args = parser.parse_args(
        [
            "--simulation_decision_clock",
            "450.5",
            "--simulation_clear_requests",
            "false",
            "--fleet_num_vehicles",
            "7",
        ]
    )
    config, _ = arg_utilities.load_config(args)
    assert config["simulation"]["decision_clock"] == 450.5
    assert config["simulation"]["clear_requests"] is False
    assert config["fleet"]["num_vehicles"] == 7
    assert config["horizon"]["request_destination_randomization"] == 0.25
    assert OpenhailInstance(config).request_destination_randomization == 0.25


def test_explicit_resource_roots_work_outside_checkout(tmp_path, monkeypatch):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    config, _ = arg_utilities.load_config(
        arg_utilities.get_parser().parse_args([]),
        root / "data/config",
        data_root=root,
    )
    assert Path(config["city"]["request_file"]).is_file()
