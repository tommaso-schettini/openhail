# Simulator Characterization

Run the closest/nearest and random-feasible baselines before training a learned
policy:

```bash
python scripts/utilities/analyze_simulator.py
```

The default matrix is `scripts/instances/simulator_analysis_instances.tsv`. It contains
the nine infrastructure configurations used by the preliminary simulator study,
with 20 common holdout seeds and a 15-minute periodic decision clock. Use
`--config-index N` to run one row by its one-based index, or submit
`scripts/arrays/run_simulator_analysis_array.sh` on the cluster. On Compute Canada,
create the shared Python 3.13 environment first:

```bash
bash scripts/utilities/setup_compute_canada.sh
sbatch scripts/arrays/run_simulator_analysis_array.sh
```

For a local smoke run:

```bash
python scripts/utilities/analyze_simulator.py \
  --config-index 1 \
  --num-episodes 1 \
  --policies nearest random
```

Each configuration writes `episodes.csv` and `summary.json` beneath
`out/simulator_analysis/`. The logs cover rewards by component, request
acceptance, action modes, decision timing, state ranges, SOC, charger
occupancy and queues, reposition feasibility, empty masks, invalid actions,
and decision-epoch types. Action destinations are emitted as one count column
per reposition location so configurations with different location counts remain
self-describing.
