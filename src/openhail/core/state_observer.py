# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING, Any, Dict, cast

import gymnasium as gym
import numpy as np

from .constants import (
    CAPACITY,
    CHARGER,
    CHARGERS,
    DEST,
    END_OF_HORIZON,
    EPOCH_TYPE,
    LOC,
    NEXT_EPOCH,
    NULL_EPOCH,
    OCCUPANCY,
    ORIG,
    PROC_TIME,
    Q_DEST,
    QUEUE,
    T_DEST,
    TARGET_CHARGER,
    TIME,
    TYPE,
    Q,
)
from .openhail_instance import OpenhailInstance

if TYPE_CHECKING:
    from .env_components.request_manager import RequestManager


class StateObserver:
    """Handles observation space definition and state compilation for the ride-hailing
        environment.

    This class is responsible for:
    1. Defining the observation space structure
    2. Compiling the current state into an observation
    3. Validating state observations
    """

    def __init__(self, instance: OpenhailInstance, max_outstanding_requests: int):
        """Initialize the state observer.

        Args:
            instance: The ride-hailing instance containing environment parameters
            max_outstanding_requests: Maximum number of outstanding requests to track
        """
        self.instance = instance
        self.max_outstanding_requests = max_outstanding_requests
        self.observation_space = self._create_observation_space()

    def _create_observation_space(self) -> gym.spaces.Dict:
        """Create the observation space for the environment.

        Returns:
            A gym.spaces.Dict containing the observation space definition
        """
        # Define location bounds
        loc_low = np.array(
            (self.instance.x_bounds[0], self.instance.y_bounds[0]), dtype=np.float64
        )
        loc_high = np.array(
            (self.instance.x_bounds[1], self.instance.y_bounds[1]), dtype=np.float64
        )
        ev_loc_low = np.tile(loc_low, (self.instance.num_evs, 1))
        ev_loc_high = np.tile(loc_high, (self.instance.num_evs, 1))

        # Time bounds
        t_start = self.instance.start_time
        t_end = self.instance.max_time

        # Vehicle state space
        V_space = gym.spaces.Dict(
            {
                LOC: gym.spaces.Box(
                    low=ev_loc_low,
                    high=ev_loc_high,
                    shape=(self.instance.num_evs, 2),
                    dtype=np.float64,
                ),
                TIME: gym.spaces.Box(
                    low=t_start,
                    high=t_end + 10000,
                    shape=(self.instance.num_evs,),
                    dtype=np.float64,
                ),
                Q: gym.spaces.Box(
                    low=0,
                    high=self.instance.ev_max_Q,
                    shape=(self.instance.num_evs,),
                    dtype=np.float64,
                ),
                DEST: gym.spaces.Box(
                    low=ev_loc_low,
                    high=ev_loc_high,
                    shape=(self.instance.num_evs, 2),
                    dtype=np.float64,
                ),
                T_DEST: gym.spaces.Box(
                    low=t_start,
                    high=t_end + 10000,
                    shape=(self.instance.num_evs,),
                    dtype=np.float64,
                ),
                Q_DEST: gym.spaces.Box(
                    low=-1,
                    high=self.instance.ev_max_Q,
                    shape=(self.instance.num_evs,),
                    dtype=np.float64,
                ),
                CHARGER: gym.spaces.Box(
                    low=-1,
                    high=self.instance.D,
                    shape=(self.instance.num_evs,),
                    dtype=np.int32,
                ),
                TARGET_CHARGER: gym.spaces.Box(
                    low=-1,
                    high=self.instance.D,
                    shape=(self.instance.num_evs,),
                    dtype=np.int32,
                ),
                TYPE: gym.spaces.Box(
                    low=-1,
                    high=10,
                    shape=(self.instance.num_evs,),
                    dtype=np.int32,
                ),
                NEXT_EPOCH: gym.spaces.Box(
                    low=t_start,
                    high=t_end + 10000,
                    shape=(self.instance.num_evs,),
                    dtype=np.float64,
                ),
            }
        )

        # Request state space
        request_loc_low = np.tile(loc_low, (self.max_outstanding_requests, 1))
        request_loc_high = np.tile(loc_high, (self.max_outstanding_requests, 1))

        request_space = gym.spaces.Dict(
            {
                TIME: gym.spaces.Box(
                    low=-1,
                    high=t_end + 1,
                    shape=(self.max_outstanding_requests,),
                    dtype=np.float64,
                ),
                ORIG: gym.spaces.Box(
                    low=request_loc_low,
                    high=request_loc_high,
                    shape=(self.max_outstanding_requests, 2),
                    dtype=np.float64,
                ),
                DEST: gym.spaces.Box(
                    low=request_loc_low,
                    high=request_loc_high,
                    shape=(self.max_outstanding_requests, 2),
                    dtype=np.float64,
                ),
                PROC_TIME: gym.spaces.Box(
                    low=-2,
                    high=np.inf,
                    shape=(self.max_outstanding_requests,),
                    dtype=np.float64,
                ),
            }
        )

        # Time space
        time_space = gym.spaces.Box(
            low=0,
            high=t_end + 2,
            shape=(),
            dtype=np.float64,
        )

        charger_loc_low = np.tile(loc_low, (self.instance.D_repo, 1))
        charger_loc_high = np.tile(loc_high, (self.instance.D_repo, 1))
        charger_space = gym.spaces.Dict(
            {
                LOC: gym.spaces.Box(
                    low=charger_loc_low,
                    high=charger_loc_high,
                    shape=(self.instance.D_repo, 2),
                    dtype=np.float64,
                ),
                CAPACITY: gym.spaces.Box(
                    low=0,
                    high=np.iinfo(np.int32).max,
                    shape=(self.instance.D_repo,),
                    dtype=np.int32,
                ),
                OCCUPANCY: gym.spaces.Box(
                    low=0,
                    high=self.instance.num_evs,
                    shape=(self.instance.D_repo,),
                    dtype=np.int32,
                ),
                QUEUE: gym.spaces.Box(
                    low=0,
                    high=self.instance.num_evs,
                    shape=(self.instance.D_repo,),
                    dtype=np.int32,
                ),
            }
        )

        epoch_type_space = gym.spaces.Box(
            low=NULL_EPOCH,
            high=END_OF_HORIZON,
            shape=(),
            dtype=np.int32,
        )

        # Complete observation space
        return gym.spaces.Dict(
            {
                "request": request_space,
                "V": V_space,
                CHARGERS: charger_space,
                "time": time_space,
                EPOCH_TYPE: epoch_type_space,
            }
        )

    def get_state(
        self,
        time: float,
        V: Dict[str, np.ndarray],
        request_manager: "RequestManager",
        charger_state: Dict[str, np.ndarray],
        epoch_type: int,
    ) -> Dict[str, Any]:
        """Compile the current state into an observation.

        Args:
            V: Dictionary containing vehicle states
            time: Current simulation time
            request_manager: The request manager containing current request state

        Returns:
            Dictionary containing the current state observation
        """
        return {
            "request": self._get_requests_from_manager(request_manager),
            "time": np.array(time),
            "V": {key: value.copy() for key, value in V.items()},
            CHARGERS: {key: value.copy() for key, value in charger_state.items()},
            EPOCH_TYPE: np.array(epoch_type, dtype=np.int32),
        }

    def _get_requests_from_manager(
        self, request_manager: "RequestManager"
    ) -> Dict[str, np.ndarray]:
        """Get the current request state from the request manager.

        Args:
            request_manager: The request manager containing current request state

        Returns:
            Dictionary containing the current request state
        """
        # NumPy advanced indexing in ``get_current_requests`` already creates
        # independent arrays, so copying them again only adds per-step work.
        return request_manager.get_current_requests()

    def check_state(self, state: Dict[str, Any], assertions: bool = True) -> None:
        """Validate that a state is within the observation space.

        Args:
            state: The state to validate
            assertions: Whether to raise assertions on invalid states

        Raises:
            AssertionError: If assertions is True and the state is invalid
        """
        if not assertions:
            return

        # Check request state
        request_space = cast(gym.spaces.Dict, self.observation_space["request"])
        request_state = state["request"]
        for key in request_state:
            if key not in request_space.spaces:
                raise AssertionError(f"Invalid key in request state: {key}")
            if not request_space.spaces[key].contains(request_state[key]):
                raise AssertionError(
                    (
                        f"Invalid value for request key {key}: {request_state[key]} vs "
                        f"{request_space.spaces[key]}"
                    )
                )

        # Check vehicle state
        V_space = cast(gym.spaces.Dict, self.observation_space["V"])
        V_state = state["V"]
        for key in V_state:
            if key not in V_space.spaces:
                raise AssertionError(f"Invalid key in vehicle state: {key}")
            if not V_space.spaces[key].contains(V_state[key]):
                raise AssertionError(
                    (
                        f"Invalid value for vehicle key {key}: {V_state[key]} vs "
                        f"{V_space.spaces[key]}"
                    )
                )

        charger_space = cast(gym.spaces.Dict, self.observation_space[CHARGERS])
        charger_state = state[CHARGERS]
        for key in charger_state:
            if key not in charger_space.spaces:
                raise AssertionError(f"Invalid key in charger state: {key}")
            if not charger_space.spaces[key].contains(charger_state[key]):
                raise AssertionError(
                    f"Invalid value for charger key {key}: "
                    f"{charger_state[key]} vs {charger_space.spaces[key]}"
                )
