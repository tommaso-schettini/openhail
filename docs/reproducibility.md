# Reproducing the paper's experiments

See [reproduction commands](public-reproduction.md) to evaluate a pretrained
agent, regenerate timing figures, run benchmarks, or train a new policy.

## Experiment settings

The [training matrix](../scripts/instances/periodic_ppo_nyc_day1_4fleets_4location_counts_1seed_500ep_instances.tsv)
contains 16 fleet and infrastructure configurations. Each policy was trained
for 500 episodes, starting from environment seed 321. Checkpoints were evaluated
every 25 episodes using environment seeds 100000–100004 and policy seeds
200000–200004.

The timing study compares nearest, random feasible, and sampled PPO controllers
on those five seed pairs for each configuration. Each controller receives one
unmeasured warm-up; controller order rotates between repetitions. Model loading,
reset, and optimization are outside the measured rollout loop. Evaluation seeds
were also used for checkpoint selection, so these runs are not held-out tests
of policy quality.

## Timing figures

```sh
python scripts/utilities/analyze_posthoc_timings.py --input-dir paper/reproducibility/posthoc-20260909 --output-dir out/reproduced-figures
```

The analyzer reads `episodes.csv` and `metadata.json`, checks paired runs, and
writes episode-time and decision-epoch-time figures as PDF and PNG files. It
also updates `summary.csv` and prints the largest configuration's runtime
breakdown. Points show medians over five rollouts; error bars show interquartile
ranges.

## Training costs

[durations.csv](../paper/reproducibility/training-1455644/durations.csv) records
iteration times for all 16 configurations. To recover training hours:

```python
import pandas as pd

durations = pd.read_csv("paper/reproducibility/training-1455644/durations.csv")
hours = durations.groupby(["fleet_size", "locations"])["seconds"].sum() / 3600
print(hours.unstack("locations").round(2).to_string())
```

These times include simulation, optimization, logging, and scheduled evaluation.

## Source and environment

| Study | Source | Environment |
| --- | --- | --- |
| Training batch `1455644` | Revision `d532ad10ac84fbf3d442e4316fb9e09ddfaa71b7` | CPU execution; four cores and 32 GB requested per task. |
| Timing run `2730959` | [Archived source](../paper/reproducibility/posthoc-20260909/source.tar.gz) | Linux, Python 3.13.2, NumPy 2.4.2, PyTorch 2.13.0; one AMD EPYC 7532 core and one numerical-library thread. |

The [source manifest](../paper/reproducibility/posthoc-20260909/source_manifest.json)
identifies the files used for the timing study. Use this archive when repeating
the original workload; the checkout revision in the execution metadata alone
does not identify the bundled source.

All 16 selected weights and their training configurations are in
[checkpoints/](../checkpoints/). The current benchmark runner loads them directly.
When using the archived runner, place the weights at the paths in its
[manifest](../paper/reproducibility/posthoc-20260909/manifest.json).

Complete dependency locks were not recorded for the original experiments.
Retraining may produce different weights, and timings depend on hardware and
software versions.

The archived source expects NYC inputs under `data/.raw/`. When running that
snapshot, copy the files from `data/nyc/` to its `data/.raw/` directory.
