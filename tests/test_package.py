"""Installed-package imports must work outside the repository and without PPO."""

import subprocess
import sys


def test_core_imports_without_optional_dependencies(tmp_path):
    program = """
import importlib.abc
import sys

class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'gurobipy'}:
            raise ModuleNotFoundError(f'Optional dependency requested: {fullname}')

sys.meta_path.insert(0, BlockOptionalDependencies())
from openhail.agents import Agent, NearestAgent, RandomAgent
from openhail.agents.agent_initializer import initialize_agent
from openhail.core.openhail_env import OpenhailEnv
from openhail.utils.config_runner import ConfigurationRunner

try:
    initialize_agent({'name': 'unsupported'}, None)
except ValueError as exc:
    assert 'not supported' in str(exc)
else:
    raise AssertionError('Unknown controller should be rejected')
assert 'torch' not in sys.modules
assert 'gurobipy' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", program],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
