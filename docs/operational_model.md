# Operational model and decision interface

OpenHail models a homogeneous electric ride-hailing fleet through a centralized
Gymnasium policy interface.

## Inputs and state

`OpenhailInstance` combines geography, trip records, fleet and charging
infrastructure, and service and economic parameters. Fleet size `N`, request
buffer size `M`, and the number of reposition locations `D` are fixed when an
environment is constructed. Demand, vehicle states, charger occupancy, queue
lengths, and time evolve during an episode.

Travel times use spatial distances and a configured constant speed. Energy
consumption and charging follow configured linear rates. The current model
abstracts road-network congestion, route choice, heterogeneous vehicle
technologies, and ride pooling.

Request generation and vehicle initialization use separate random streams
derived from the episode seed. Holding the demand inputs and seed fixed allows
policies to face common request realizations; their actions can still produce
different vehicle trajectories and decision counts.

## Actions and observations

The joint action contains request-to-vehicle assignments, one repositioning or
charging instruction per vehicle, and an optional absolute future decision
time. Service assignments are processed before repositioning and charging
instructions. The declared action space includes no-assignment and no-movement
choices; use its encoding and the provided feasibility utilities when
implementing a policy.

Assignments must respect pickup and energy constraints. Repositioning
destinations must be reachable with the available energy, and charging actions
must target a location with charging capacity. Feasible-action utilities help a
policy construct actions; strict validation additionally checks actions and
state consistency as `step` runs. A violation raises an exception, rather
than returning a normal episode-termination signal.

Observations expose the fixed-size request buffer, vehicle states, reposition
locations, charging-facility states, current time, and triggering epoch type.
Returned arrays do not alias the mutable internal vehicle state.

## Operational events and policy decisions

Vehicle jobs include passenger service, repositioning, travel to a charger,
waiting for a charger, and charging. Upon arrival at a charging location, a
vehicle either occupies an available post or waits in a first-in--first-out
queue. The vehicle manager processes completions and allocates available posts.
Queue waiting is therefore determined by fleet actions and station capacity.

The simulator processes internal events even when they do not trigger a
policy interaction. The decision schedule is controlled through:

| Setting | Effect |
| --- | --- |
| `request_epoch` | Exposes request arrivals as decision opportunities. |
| `end_of_service_epoch` | Exposes service completions. |
| `end_of_charge_epoch` | Exposes charging completions. |
| `decision_clock` | Adds periodic decision opportunities. |
| `max_interdecision_time` | Bounds intervals without interaction. |
| The action's requested epoch | Requests an absolute future decision time. |

Repositioning completions remain internal transitions. When request epochs are
disabled, arrivals are released to the request buffer at the next exposed
decision, subject to buffer retention and pickup limits. This makes buffer
settings part of the experimental configuration when changing decision timing.

The next applicable event or clock determines how far the simulator advances.
The configured planning horizon ends the episode. Requested epochs at or after
the horizon do not extend it. Selection among simultaneous events and FIFO
allocation are covered by regression tests in the environment-contract suite.

## Reward and measurement

The returned reward accumulates operating value between exposed decisions.
Service contributes fixed and distance-dependent revenue less pickup and
passenger travel costs; repositioning and charging contribute their configured
travel and energy costs. The information dictionary reports reward components,
elapsed simulation time, the epoch type, request counts, mean state of charge,
and charger occupancy and queue lengths.

Detailed request, vehicle-activity, and epoch summaries depend on the enabled
tracking settings. Policies using a discount factor should account for the
fact that successive `step` calls may represent unequal simulation durations;
the information dictionary exposes this duration as `delta_time`.

## Implementation and verification

- [Environment coordination, event selection, and Gymnasium methods](../src/openhail/core/openhail_env.py)
- [Instance construction and configuration](../src/openhail/core/openhail_instance.py)
- [Observation construction](../src/openhail/core/state_observer.py)
- [Environment-contract regression tests](../tests/test_environment_contract.py)
- [Verification scope and execution commands](testing.md)
- [PPO controller's aggregation between clock decisions](training_periodic_ppo.md)
