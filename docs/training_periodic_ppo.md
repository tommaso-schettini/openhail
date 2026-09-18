# Training the Periodic PPO Agent

`periodic_ppo` is an example agent that combines nearest-request assignment
with learned repositioning and charging:

- every request is assigned to the closest feasible vehicle;
- the neural policy acts only at periodic `CLOCK_EPOCH` events (15 minutes by
  default);
- every vehicle that can accept a repositioning action is controlled;
- each controlled vehicle chooses among no-op, reposition to a destination, or
  travel to a destination in charging mode;
- the simulator's native feasibility mask is applied before sampling an action.
- charging mode is available only at destinations with positive charger
  capacity.

The policy does not change the simulator's global event behavior. At events
between clock epochs, it emits no-op actions unless a vehicle is required by the
simulator to receive a new job; that exceptional action is the closest feasible
repositioning move.

## Network

The actor shares its encoders across vehicles so it does not create a separate
network for every vehicle:

1. A vehicle encoder processes location, state of charge, availability,
   destination/job state, charging state, and the normalized state of the
   vehicle's current target location. The target representation has a fixed
   dimension and does not depend on the number of repositioning locations.
2. A location encoder processes charging capacity, occupancy, queue length, and
   location identity for each repositioning destination.
3. Mean-pooled actor embeddings, together with time-of-day, produce the actor's
   fleet-level context vector.
4. The actor scores no-op per vehicle and both native modes for every
   vehicle-destination pair. Infeasible scores are masked.

The critic is an independent set-attention network. It has its own SiLU and
LayerNorm vehicle, location, and time encoders; it does not reuse or update the
actor's representation. The time embedding initializes a global token. Two
self-attention blocks assemble that token from every vehicle and infrastructure
token, and the value head reads the resulting fleet embedding. No vehicle or
location ordering embeddings are used, so the value is invariant to the
arbitrary order of either set. The critic contains no `tanh` activations.

Actor and critic parameters use separate Adam optimizers and are gradient
clipped separately. This prevents the much larger value loss from dominating
the actor's gradient clipping.

PPO treats the periodic fleet decision as one transition. Rewards from all
simulator events until the next clock epoch are accumulated into that
transition. Generalized advantage estimation is computed over the 95 periodic
decisions in a full simulated day.

Although vehicle actions are sampled from factorized categorical distributions,
the simulator returns one reward for the complete fleet action. PPO therefore
sums the selected vehicle log probabilities into one joint fleet-action log
probability and clips one joint likelihood ratio per periodic decision. It does
not clip vehicle likelihood ratios independently.

## Installation

From the repository root, use the project environment and install the PPO
extra:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ppo]"
```

On Linux or a cluster:

```bash
python -m pip install -e '.[ppo]'
```

On Compute Canada, create the persistent Python 3.13 environment from the
repository root. The cluster pip configuration prefers the Alliance wheelhouse
and falls back to PyPI for Python 3.13 dependencies that are not mirrored
there. Run setup on a login node with network access. It also creates the
`logs/` and `err/` directories Slurm needs before submission:

```bash
bash scripts/utilities/setup_compute_canada.sh
sbatch scripts/arrays/run_periodic_ppo_preliminary_array.sh
```

For Python 3.13, the setup uses the Alliance `pyproj` wheel together with the
PROJ database supplied by the `proj/9.4.1` module. Both array launchers validate
`proj.db` and export `PROJ_DATA` before starting OpenHail. Rerun setup to verify
an environment that raises `pyproj.exceptions.DataDirError`.

## Local Smoke Run

This runs one training episode and one reproducibly sampled evaluation episode:

```powershell
.\.venv\Scripts\python.exe scripts\utilities\train_periodic_ppo.py `
  --num-episodes 1 `
  --eval-interval 1 `
  --eval-episodes 1 `
  --update-epochs 1 `
  --run-name smoke
```

A first substantive run can use the defaults:

```powershell
.\.venv\Scripts\python.exe scripts\utilities\train_periodic_ppo.py `
  --num-episodes 500 `
  --infrastructure S05_1_4 `
  --run-name periodic_ppo_S05_1_4
```

Training and evaluation seeds are separate. Primary evaluation samples from the
policy without collecting rollouts or updating its weights. It always uses the
same environment/policy seed pairs (`100000`/`200000`, `100001`/`200001`, and
so on), enabling reproducible comparisons with the random and nearest
baselines. The default is 20 evaluation seed pairs.

To record an additional deterministic diagnostic, pass `--greedy-eval`. These
rows are labelled `evaluation_greedy`; they are not used to select the best
checkpoint because independent argmax actions can cause similar vehicles to
move together.

## Multiple Infrastructure Configurations

The supplied Slurm array covers the same nine infrastructure configurations as
the simulator analysis:

```bash
sbatch scripts/arrays/run_periodic_ppo_preliminary_array.sh
```

Edit `scripts/instances/periodic_ppo_preliminary_instances.tsv` to change the
Python module, episode counts, hyperparameters, devices, or output locations.
The table header is not an array entry; Slurm task 1 consumes its first data
row. The supplied matrix uses `python/3.13` throughout, and the launcher rejects
a virtual environment created with another Python minor version.

An additional experiment uses three independent training seeds for all
nine configurations (27 tasks total, with at most nine running concurrently):

```bash
sbatch scripts/arrays/run_periodic_ppo_nyc_day1_fleet20_9infra_3seeds_500ep_array.sh
```

Its complete matrix is stored in
`scripts/instances/periodic_ppo_nyc_day1_fleet20_9infra_3seeds_500ep_instances.tsv`.
Run directories include the experiment dimensions, infrastructure, training
seed, Slurm array job ID, and task ID, for example:

```text
periodic_ppo_nyc_day1_fleet20_9infra_3seeds_500ep_S05_1_4_s654_<job>_<task>
```

The paper's fleet--location scaling experiment trains one policy for every
combination of $N \in \{100,500,1000,2000\}$ vehicles and
$L \in \{5,10,20,40\}$ reposition locations. Demand remains fixed at 20
requests per vehicle-day and charging capacity remains fixed at 0.2 posts per
vehicle. Consequently, every learned-policy benchmark uses a checkpoint trained
at the same fleet size, demand intensity, and infrastructure scale. Training
replication is omitted because these runs produce computational-performance
checkpoints rather than estimates of policy quality. Submit the 16-task array
with:

Checkpoint selection evaluates five fixed seed pairs every 25 episodes. This
schedule provides 20 evaluation points over the 500 training episodes.

```bash
sbatch scripts/arrays/run_periodic_ppo_nyc_large_scale_500ep_array.sh
```

The corresponding matrix is stored in
`scripts/instances/periodic_ppo_nyc_day1_4fleets_4location_counts_1seed_500ep_instances.tsv`.

## Outputs and Monitoring

Every run creates:

```text
out/periodic_ppo_training/<run_name>/
|-- config.json
|-- metrics.csv
|-- checkpoint_best.pt
|-- checkpoint_final.pt
`-- checkpoint_ep<N>.pt       # only when --save-interval is nonzero
```

`metrics.csv` uses one stable schema for every phase. The `phase` column is
`train`, `evaluation_sampled`, or the optional `evaluation_greedy`. It includes:

- total, service, and repositioning reward;
- requests seen/served and acceptance rate;
- no-op, reposition, and charging action counts;
- zero-capacity charging-command count, which must remain zero;
- separate JSON arrays of repositioning and charging destination counts;
- policy-decision count and mean time between policy decisions;
- entropy, policy/value losses, joint approximate KL, joint clipping fraction,
  critic explained variance, and separate policy/value gradient norms;
- optimizer update count and separate actor/critic learning rates;
- invalid-action and empty-mask counts;
- mean state of charge, charger occupancy, queue length, simulator event time,
  and step count.

For a full day with a 900-second decision clock, there are normally about 95
policy decisions and the mean policy interval is 900 seconds. Simultaneous
events can reduce the count slightly. Invalid-action, empty-mask, and
zero-capacity charging counts should remain zero. Use the mean fixed-seed
sampled-evaluation trend, rather than a single episode or the greedy diagnostic,
to judge learning.

## Loading a Checkpoint

The agent loads version-3 checkpoints, including the pretrained weights
distributed with this repository.

The actor scores vehicle--location pairs with shared weights, while the critic
uses unordered vehicle and location tokens. A version-3 checkpoint can therefore
be loaded with different fleet sizes and numbers of repositioning locations.
Charging capacity, occupancy, and queue features are normalized by fleet size;
preserving their relative scale remains advisable when transferring a policy.

```python
from openhail.agents import PeriodicPPOAgent

agent = PeriodicPPOAgent(
    instance,
    training=False,
    sample_actions=True,
    checkpoint_path="out/periodic_ppo_training/my_run/checkpoint_best.pt",
)
action = agent.choose_action(observation)
```

`training=False` prevents rollout collection and optimizer updates.
`sample_actions=True` retains the stochastic policy used during training. Set
it to `False` only when a deterministic greedy diagnostic is explicitly wanted.
