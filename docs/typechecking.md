# Type checking

Install the developer and PPO dependencies, then run Pyright from the repository
root:

```sh
python -m pip install -e ".[dev,ppo]"
python -m pyright
```

`pyrightconfig.json` checks `src/`, `scripts/`, and `tests/` in basic mode using
the project's `.venv` and Python 3.10 syntax compatibility.

In VS Code, select `.venv` as the Python interpreter so Pylance resolves the same
dependencies. Restart the language server if diagnostics persist after changing
the interpreter.
