# Pretrained PPO agents

This directory contains the 16 checkpoints used in the paper's controller
benchmark, with their training configurations. `manifest.json` lists file
checksums and source experiments.

Install the PPO extra and run an example from the repository root:

```sh
python -m pip install -e ".[ppo]"
python scripts/evaluate_pretrained.py
```

The default example uses 100 vehicles and five repositioning locations. Select
a different checkpoint with:

```sh
python scripts/evaluate_pretrained.py --checkpoint fleet_500_S10_2_50
```

Evaluation uses fixed environment and policy seeds. Use `--seed` and
`--policy-seed` to change them, or `--greedy` for deterministic action selection.
The default seeds were also used for checkpoint selection.

See the [reproduction guide](../docs/reproducibility.md) for experiment settings.
