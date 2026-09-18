# Reproduction commands

Install from a repository checkout and run the commands below from its root:

```sh
python -m pip install -e ".[dev,ppo]"
```

## Evaluate a pretrained agent

```sh
python scripts/evaluate_pretrained.py
```

See [pretrained agents](../checkpoints/README.md) for checkpoint selection and
seed options.

## Generate the paper's timing figures

```sh
python scripts/utilities/analyze_posthoc_timings.py --input-dir paper/reproducibility/posthoc-20260909 --output-dir out/reproduced-figures
```

This plots the recorded measurements and updates `summary.csv` beside the input
records. Figures are written to `out/reproduced-figures/`.

## Run new timing experiments

```sh
python scripts/utilities/benchmark_trained_controllers.py --output-dir out/new-timings
```

The runner evaluates all 16 bundled checkpoints. Add `--tasks 1 --num-episodes 1`
for a short trial. Choose a fresh output directory for each run. Timings depend
on the machine and software environment.

## Train an agent

```sh
python scripts/utilities/train_periodic_ppo.py --num-episodes 1 --eval-interval 1 --eval-episodes 1 --update-epochs 1 --output-dir out/smoke-training
```

This runs one training episode and one evaluation episode. See the
[training guide](training_periodic_ppo.md) for a full run and the
[experiment reference](reproducibility.md) for the paper's configuration matrix,
seeds, and training-cost calculation.
