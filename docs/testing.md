# Development checks

From the repository root:

```sh
python -m pip install -e ".[dev,ppo]"
python -m pytest -q
python -m ruff check src scripts tests
python -m ruff format --check src scripts tests
python -m pyright
```

PPO tests require the `ppo` extra. Install `.[dev]` to run the core suite without
PyTorch; PPO tests will be skipped.

## Test coverage

The suite covers the Gymnasium interface, request generation, charging queues,
energy accounting, action feasibility, configuration loading, baseline agents,
PPO updates and checkpoints, rendering, and package imports.

Integration tests use the bundled NYC data and configurations. Tests run
headlessly and do not require trained checkpoints or downloaded experiments.
Temporary test files are written under `tests/.tmp/`.

To select a group of tests:

```sh
python -m pytest -m integration -q
python -m pytest -m render -q
python -m pytest -m ppo -q
python -m pytest -m "not integration" -q
```

## Continuous integration

GitHub Actions builds and tests the wheel on Windows and Linux with Python 3.10
and 3.13. A Linux job also checks PPO and types. Lint and formatting checks run
alongside the tests.

See [type checking](typechecking.md) for editor setup.
