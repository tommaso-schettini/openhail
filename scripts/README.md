# Scripts

Run scripts from the repository root after installing OpenHail.

| Script | Purpose |
| --- | --- |
| `run_simulation.py` | Run a simulation; add `--render` for visualization. |
| `run_simulation_examples.py` | Rendering, GIF, and baseline-comparison examples. |
| `run_simulation_manual_configuration.py` | Assemble an instance, agent, and evaluator directly. |
| `evaluate_pretrained.py` | Evaluate a bundled PPO checkpoint. |
| `benchmark_performance.py` | Measure headless simulator performance. |
| `utilities/train_periodic_ppo.py` | Train the example PPO agent. |
| `utilities/analyze_simulator.py` | Summarize simulator and baseline behavior. |
| `utilities/benchmark_trained_controllers.py` | Compare controller runtimes. |
| `utilities/analyze_posthoc_timings.py` | Generate timing summaries and figures. |

`instances/` contains experiment matrices. `arrays/` contains Slurm launchers;
adapt their allocation and environment settings before submitting a job.

See the [training guide](../docs/training_periodic_ppo.md) and
[reproduction commands](../docs/public-reproduction.md) for examples.
