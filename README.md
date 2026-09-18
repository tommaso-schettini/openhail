# OpenHail: Electric Ride-Hailing Fleet Control

OpenHail is a discrete-event [Gymnasium](https://gymnasium.farama.org/) environment for joint request assignment, repositioning, and charging control in electric ride-hailing fleets. It represents battery dynamics and finite-capacity charging queues, with configurable decision epochs for policy experiments. Travel times use spatial distances and a constant speed; network congestion and route choice are outside the current model.

## Documentation and research use

- [Operational model and decision interface](docs/operational_model.md)
- [Package verification and test commands](docs/testing.md)
- [Reproducing the paper's computational results](docs/reproducibility.md)
- [PPO training](docs/training_periodic_ppo.md)
- [Run a pretrained PPO example](checkpoints/README.md)
- [Reproduction commands](docs/public-reproduction.md)

OpenHail is distributed under the [MIT license](LICENSE). The examples below run from a repository checkout, which supplies the sample data and configurations; the installable wheel contains the Python package.

Install the package from the repository root with `python -m pip install -e .`. To use the PPO controller, install `python -m pip install -e ".[ppo]"`.

When using OpenHail from another working directory, pass explicit resource roots:

```python
from openhail.utils.config_runner import ConfigurationRunner

runner = ConfigurationRunner(
    config_dir="/path/to/openhail/data/config",
    data_root="/path/to/openhail",
)
```

The bundled NYC inputs are described in [data/README.md](data/README.md).

## Quick Start

The following examples show how to run simulations using the high-level configuration interface.

### Example 1: Single Configuration Run

```python
from openhail.utils.config_runner import ConfigurationRunner

# Create a runner instance
runner = ConfigurationRunner(render=False)  # Set render=True for visualization

# Run a single configuration
result = runner.run_single_config(
    simulation="debug_config_fast",
    evaluation="S1", 
    city="nyc",
    agent="nearest",
    infrastructure="S05_1_4"
)

print(f"Average reward: {result.average_reward():.2f}")
```

### Example 2: Compare Multiple Configurations

```python
from openhail.utils.config_runner import quick_compare

# Compare different agents with the same setup
quick_compare(
    config_variations={"agent": ["nearest", "random"]},
    simulation="debug_config_fast",
    evaluation="S1",
    city="nyc", 
    infrastructure="S05_1_4"
)
```

### Example 3: Rendered Simulation

The public runner is headless by default. Add `--render` to open the Pygame
window, or `--save-gif` to render and save the run:

```bash
python scripts/run_simulation.py --render
python scripts/run_simulation.py --save-gif out/render/simulation.gif
```

```python
from openhail.utils.config_runner import ConfigurationRunner

# Run with visualization enabled
runner = ConfigurationRunner(render=True)
result = runner.run_single_config(
    simulation="debug_config_fast",
    evaluation="S1",
    city="nyc", 
    agent="nearest",
    infrastructure="S05_1_4"
)

print(f"Rendered result: {result.average_reward():.2f}")
```

### Advanced Usage: Direct Gymnasium Interface

For advanced users who want to work directly with the Gymnasium environment:

```python
import gymnasium as gym
from openhail.core.openhail_instance import OpenhailInstance
from openhail.core.openhail_env import OpenhailEnv
from openhail.agents.agent_initializer import initialize_agent
from openhail.utils.arg_utilities import load_config, get_parser

# Load configuration
parser = get_parser(
    evaluation="S1",
    city="nyc", 
    horizon="day_2",
    fleet="fleet_20",
    infrastructure="S05_1_4",
    agent="nearest"
)
args = parser.parse_args([])
instance_config, extra_config = load_config(args)

# Create instance and environment
instance = OpenhailInstance(instance_config)
env = OpenhailEnv(
    instance=instance,
    simulation_config=instance_config["simulation"],
    render_config={"render_mode": None}  # or "human" for visualization
)

# Initialize agent
agent = initialize_agent(extra_config["agent"], instance)

# Run simulation loop
obs, info = env.reset()
total_reward = 0
done = False

while not done:
    action = agent.choose_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

print(f"Total reward: {total_reward:.2f}")
env.close()
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

### Installation

#### Option 1: Install from Source (Recommended for Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/tommaso-schettini/openhail.git
   cd openhail
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the package in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

4. Run the tests (optional):
   ```bash
   python -m pytest tests/
   ```

#### Option 2: Install from PyPI (Future Release)

Once published to PyPI, you can install directly:
```bash
pip install openhail
```

#### Dependencies

The package requires Python 3.10+ and installs the following main dependencies:
- gymnasium
- numpy, pandas, geopandas
- pygame (for visualization)
- matplotlib, folium (for plotting)
- networkx, scikit-learn

Install `openhail[ppo]` for the trainable PPO agent. The supported controllers
are nearest, random feasible, and periodic PPO; no solver licence is required.

### Observation Space

The observation space of the environment is expressed as a `gym.spaces.Dict` with the following keys:
1. `request` - the current active requests.
2. `V` - the state of the vehicles.
3. `chargers` - charging-location coordinates, capacity, occupancy, and queue length.
4. `time` - the current simulation time.
5. `epoch_type` - the event type that triggered the current decision epoch.

#### Observation Space: request

The `request` portion of the observation space is a `gym.spaces.Dict` with the following keys:

1. `time` - a 1-d array with length `max_outstanding_requests` - issue time of the active requests
2. `orig` - a 2-d array with size `(max_outstanding_requests, 2)` - origin of the active requests
    1. the first dimension corresponds to the index of the active request
    2. the second dimension indexes the coordinates of the request origin
3. `dest` - a 2-d array with size `(max_outstanding_requests, 2)` - destination of the active requests
    1. the first dimension corresponds to the index of the active request
    2. the second dimension indexes the coordinates of the request destination
4. `proc_time` - a 1-d array with length `max_outstanding_requests` - distance between origin and destination of the active requests

If there are fewer than `max_outstanding_requests` currently active, the data associated with active requests will be placed at the start of their corresponding vectors.
The remaining entries will be filled with dummy entries.
Dummy entries may be identified by their `time`, with an issue time that is `end_t + 1`, i.e., after the end of the planning horizon.

#### Observation Space: Vehicles

The `V` portion of the observation space is a `gym.spaces.Dict` with the following keys:
1. `time` - a 1-d array with length `num_evs` - ending time of the last uninterruptible job in the evs' job queue.
2. `loc` - a 2-d array with shape `(num_evs, 2)` - the location of the evs' `time`
    1. the first dimension corresponds to the index of the vehicle
    2. the second dimension indexes the coordinates of the vehicle location
3. `Q` - a 1-d array with length `num_evs` - level of charge at the evs' `time`
4. `t_dest` - a 1-d array with length `num_evs` - ending time of the last job in the evs' job queue
5. `dest` - a 2-d array with shape `(num_evs, 2)` - the location of the evs' `t_dest`
    1. the first dimension corresponds to the index of the vehicle
    2. the second dimension indexes the coordinates of the vehicle location
6. `q_dest` - a 1-d array with length `num_evs` - level of charge at the evs' `t_dest`
7. `target_charger` - a 1-d array with length `num_evs` - ID of the charger/repositioning station that the evs are heading towards
8. `charger` - a 1-d array with length `num_evs` - ID of the charger that is being used by the evs
9. `type` - a 1-d array with length `num_evs` - type of the last job in the evs' job queue
10. `next_epoch` - a 1-d array with length `num_evs` - time of the next epoch triggered by the evs

#### Observation Space: Chargers

The `chargers` portion is a `gym.spaces.Dict` indexed by reposition location:

1. `loc` - coordinates with shape `(num_repo_locations, 2)`.
2. `capacity` - number of chargers at each location.
3. `occupancy` - number of chargers currently occupied at each location.
4. `queue` - number of vehicles waiting to charge at each location.

### Action Space

The action space expected by the code is a `gym.spaces.Tuple` with three entries: `serve`, `reposition`, `epoch`.

1. `serve` - a 1-d vector with length `max_outstanding_requests` - ID of the vehicles assigned to the active requests.
2. `reposition` - a 1-d vector with length `num_evs` - repositioning actions (more details in the next section).
3. `epoch` - scalar - requested epoch by the operator.

The declared Gymnasium action space includes the full vector shapes and the
unassigned-request sentinel described below, so callers can validate actions
with `env.action_space.contains(action)`.

#### Action Space: Serve

`serve` specifies the ID of the vehicles to be assigned to the active requests.
A value of `v` in the `i`-th entry of the array indicates that vehicle `v` should be assigned to the `i`-th active request.
The ID of the vehicles are 0-indexed.
If a request is to be left unassigned, a value of `num_evs` is used.
The operator is responsible for ensuring the feasibility of the assignment, a helper method is available to determine the feasible assignment mask (see section Helper methods)

#### Action Space: Reposition

`reposition` specifies one action for each vehicle. Action `0` is no-op. For
zero-based destination `j`, action `2*j + 1` relocates there without charging,
while action `2*j + 2` relocates there and requests charging. Both modes use
the same destination-feasibility mask.

#### Action Space: Epoch

`epoch` is a scalar value for the requested epoch by the operator. Requested
epochs at or beyond the episode horizon are ignored; the environment terminates
at the horizon with an `END_OF_HORIZON` epoch.

### Helper Methods

By default, the environment does not provide feasible allocation and repositioning masks.
However, the code for doing so is available in
`src/openhail/utils/state_utilities.py`.
Two helper functions are provided:

`compute_assignment_mask(instance, request, V, time)` - computes the feasible assignment mask given the state.
`compute_rnr_mask(instance, V)` - computes the feasible repositioning mask given the state.

Both methods provide both binary masks and distances.

An additional helper function is provided in the form of `compute_nearest_assignment(instance, request, V)`, which will compute the `serve` action associated to the nearest assignment.

### Additional Notes

To develop your own agent, it is best to override the base class `agent.py`.
The agent itself is not aware of its own score, but the environment does output the score during the simulation, in case it is useful for training.
The environment itself operates like a normal Gymnasium environment, but it can
also be wrapped in the evaluator class for configuration-driven experiments.
`scripts/run_simulation.py` provides the smallest headless example, while
`scripts/run_simulation_examples.py` contains opt-in rendering, GIF, and
baseline-comparison examples.

The repository includes `nearest`, `random`, and `periodic_ppo` agents.
The `random` baseline uses closest feasible request assignment and samples a
feasible reposition/charging action for every vehicle.
The trainable `periodic_ppo` agent keeps closest feasible request assignment
deterministic and learns a stochastic policy for periodic vehicle repositioning
and charging. Charging actions are restricted to locations with actual charger
capacity. See
[docs/training_periodic_ppo.md](docs/training_periodic_ppo.md) for its
architecture, training commands, output metrics, and checkpoint usage.

Agents are normally instantiated by the `agent_initializer` class, but that is optional.

### Tests

The test suite covers request generation, the Gymnasium environment contract,
infrastructure generation, baseline agents, training metrics, and PPO
checkpointing. Run `python -m pytest` from the repository root after installing
`pip install -e ".[dev,ppo]"`. All tests are headless and rendering checks use a
bounded number of transitions. Without the PPO extra, its tests are skipped.
See [docs/testing.md](docs/testing.md) for test categories and release checks.

Before starting a learned-policy run, use the multi-configuration simulator
characterization workflow in [docs/simulator_analysis.md](docs/simulator_analysis.md).

### Performance baseline

Run the reproducible large-scale, headless benchmark with:

```powershell
python scripts/benchmark_performance.py
```

The default workload represents one simulated NYC day with 2,101 EVs,
40,000 requests, 67 reposition and charging locations, and four charging
posts per location. This is approximately 19 requests per vehicle-day and
268 charging posts. Seed 0 selects a source day containing enough records to
draw all 40,000 requests without replacement. The benchmark disables
validation, rendering, and detailed trajectory tracking, and reports agent
time separately from environment time.

For a quick smoke run of the same benchmark contract, reduce both dimensions:

```powershell
python scripts/benchmark_performance.py --fleet-size 100 --requests-per-day 1000
```

The smaller invocation is intended for development checks, not as the
reported performance baseline. In horizon configuration files,
`num_requests` denotes requests per simulated day.

### Inputs

The simulator is configured using 6 files

#### Inputs: replication

Describe the meta-parameters of the simulation - number of simulations to run, seed of the simulation

- `name` - name of the file - used for logging purposes
- `firsteval` - first seed to be simulated
- `numeval` - number of seeds to be simulated

Seeds will be simulated in ascending order, using increments of one.

#### Inputs: city

- `name` - name of the file - used for logging purposes
- `geojson_file` - geojson file describing the area delimitations
- `parking_file` - json file describing the available parking locations from which the infrastructure is selected
- `request_file` - request data
- `max_wait` - maximum waiting time before a request is lost
- `charge_cost` - cost per kWh charged
- `travel_cost` - cost of travelling per km
- `fixed_reward` - fixed reward per accepted request
- `variable_reward_distance` - variable reward per accepted request per km

#### Input: horizon

- `name` - name of the file - used for logging purposes
- `num_requests` - number of requests per day
- `start_time` - starting time of the simulation in seconds
- `max_time` - total duration of the simulation in seconds


#### Input: fleet

- `name` - name of the file - used for logging purposes
- `Q` - max charge of the vehicles expressed in kwh
- `charge_rate` - charge rate expressed in kwh/s
- `discharge_rate` - discharge rate expressed in kwh/km
- `speed` - speed of the vehicles
- `num_vehicles` - number of vehciles


#### Input: infrastructure

- `name` - name of the file - used for logging purposes
- `mode` - randominzation mode - `select` is currently the only option
- `locations` - number of locations with chargers / parking spots
- `selection` - number of locations with chargers
- `count` - number of chargers installed in each location with chargers
- `seed` - randominzation seed

#### Input: agent

- `name` - name of the file - used for logging purposes
Other inputs depend on the specific agent being used

#### Input: simulator

- `name` - name of the file - used for logging purposes
- `decision_clock` - if set to a postitive number, decision epochs are triggered every `decision_clock`
- `max_outstanding_requests` - maxumum number of active requests - active requests beyond this threshold are processed in a FIFO fashion
- `clear_requests` - Boolean. If true, active requests not immediately accepted are rejected and removed; if false, they remain active until their deadline.
- `max_interdecision_time` - if set to a postitive number, decision epochs are automatically triggered if more than `max_interdecision_time` would elapse before the next epoch
- `end_of_service_epoch` - if set to true, the epoch triggered by the end of a serving event is exposed to the operator, otherwise, the epoch is internally processed and not exposed to the operator 
- `end_of_charge_epoch` - if set to true, the epoch triggered by the end of a charging event is exposed to the operator, otherwise, the epoch is internally processed and not exposed to the operator
- `request_epoch` - whether request arrivals are exposed as decision epochs
- `strict_validation` - whether action and simulator consistency checks raise explicit runtime errors
- `track_vehicles` - whether vehicle activity totals are included in the episode summary
- `track_requests` - whether request totals and waiting times are included in the episode summary
- `track_epochs` - whether decision-epoch totals and elapsed time are included in the episode summary

