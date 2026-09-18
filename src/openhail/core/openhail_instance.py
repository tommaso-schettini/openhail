import hashlib
import json
import os
import pickle
import tempfile
import warnings

# Suppress pyogrio shapely.geos deprecation warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyogrio")
    import geopandas as gpd
import numpy as np

from ..utils import geometry
from ..utils import triangulation as tri
from ..utils.infrastructure_utilities import (
    ChargingInfrastructure,
    parse_infrastructure,
)
from .request.request_chicago import request_chicago
from .request.request_nyc import request_nyc


class OpenhailInstance:
    def __init__(self, instance_config: dict):
        self._parse_config(instance_config)
        self._load_zones(instance_config)
        self._load_requests()
        self._load_parking_lots()
        self._parse_infrastructure(instance_config)

    def plot_lots(self):
        """Plot the location and id of the parking lots."""

        import matplotlib.pyplot as plt

        _, ax = plt.subplots(1, 1, figsize=(8, 8))
        self.zones_gdf.plot(ax=ax, color="none", edgecolor="grey")

        plt.scatter(
            self.lot_coords[:, 0],
            self.lot_coords[:, 1],
            c="blue",
            s=10,
            marker="x",
            alpha=0.5,
        )
        for idx in range(len(self.lot_coords)):
            plt.text(self.lot_coords[idx, 0], self.lot_coords[idx, 1] + 10, str(idx))

        ax.set_title("Zones")
        plt.show()

    def _parse_config(self, instance_config):
        self.parking_file = instance_config["city"]["parking_file"]
        self.request_file = instance_config["city"]["request_file"]

        self.gdf_meters_per_unit = instance_config["city"].get("meters_per_unit", 1.0)
        self.cost_charge = instance_config["city"]["charge_cost"]
        self.reward_fixed = instance_config["city"]["fixed_reward"]
        self.passenger_max_wait = instance_config["city"]["max_wait"]

        cost_travel_meters = instance_config["city"]["travel_cost"]
        reward_variable_dist_meters = instance_config["city"][
            "variable_reward_distance"
        ]
        self.reward_variable_dist = (
            reward_variable_dist_meters / self.gdf_meters_per_unit
        )
        self.cost_travel = cost_travel_meters / self.gdf_meters_per_unit

        # -> HORIZON
        self.num_requests = instance_config["horizon"]["num_requests"]
        self.max_time = instance_config["horizon"]["max_time"]
        self.start_time = instance_config["horizon"]["start_time"]
        # -> request randomization
        horizon = instance_config["horizon"]
        self.request_origin_randomization = self._randomization_value(
            horizon,
            "request_origin_randomization",
            "request_origin_radomization",
        )
        self.request_destination_randomization = self._randomization_value(
            horizon,
            "request_destination_randomization",
            "request_destination_radomization",
        )
        self.request_time_randomization = self._randomization_value(
            horizon,
            "request_time_randomization",
            "request_time_radomization",
        )

        # -> FLEET
        self.num_evs = instance_config["fleet"]["num_vehicles"]
        self.ev_max_Q = instance_config["fleet"]["Q"]
        self.ev_charge_rate = instance_config["fleet"]["charge_rate"]
        self.initialization_mode = instance_config["fleet"].get(
            "initialization_mode", "legacy"
        )
        # some parameters need to be adjusted to the unit of the grid
        ev_speed_meters = instance_config["fleet"]["speed"]
        ev_discharge_rate_meters = instance_config["fleet"]["discharge_rate"]
        self.ev_discharge_rate = ev_discharge_rate_meters / self.gdf_meters_per_unit
        self.ev_speed = ev_speed_meters * self.gdf_meters_per_unit

        # compute compuound parameters
        self.max_distance_from_request = self.ev_speed * self.passenger_max_wait

        if self.cost_charge < 0:
            raise ValueError("city.charge_cost must be non-negative.")
        if self.ev_max_Q <= 0:
            raise ValueError("fleet.Q must be strictly positive.")
        if self.ev_charge_rate <= 0:
            raise ValueError("fleet.charge_rate must be strictly positive.")
        if self.ev_speed <= 0:
            raise ValueError("fleet.speed must be strictly positive.")
        if self.num_evs <= 0:
            raise ValueError("fleet.num_vehicles must be strictly positive.")
        if self.max_time <= self.start_time:
            raise ValueError(
                "horizon.max_time must be greater than horizon.start_time."
            )

    @staticmethod
    def _randomization_value(config, key, legacy_key):
        """Read a probability while accepting the former misspelled key."""
        if key in config and legacy_key in config and config[key] != config[legacy_key]:
            raise ValueError(
                f"Conflicting values for {key!r} and legacy {legacy_key!r}."
            )
        if key in config:
            value = config[key]
        elif legacy_key in config:
            warnings.warn(
                f"{legacy_key!r} is deprecated; use {key!r}.",
                DeprecationWarning,
                stacklevel=3,
            )
            value = config[legacy_key]
        else:
            value = 0.0
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"horizon.{key} must be between 0 and 1.")
        return value

    def _load_zones(self, config):
        # load the zone file
        self.city = config["city"]["name"]
        self.zones_file = config["city"]["geojson_file"]
        self.key_EPSG = config["city"].get("grs", "EPSG:4326")
        self.key_zone_id = config["city"].get("key_zone_id", "zone_id")
        self.zones_gdf = gpd.read_file(self.zones_file)
        self.zones_gdf = self.zones_gdf.to_crs(self.key_EPSG)
        self.zones_gdf["zone_id"] = self.zones_gdf[self.key_zone_id].astype(int)
        # extract the relevant information from the zones file
        self.Z = len(self.zones_gdf)
        self.zone_ids = [
            self.zones_gdf.at[i, "zone_id"]
            for i, _ in enumerate(self.zones_gdf["geometry"])
        ]
        self._load_or_calculate_triangulation()
        # bounds - only used for plotting
        bounds = self.zones_gdf.total_bounds
        self.x_bounds = [bounds[0], bounds[2]]
        self.y_bounds = [bounds[1], bounds[3]]
        self.midpoint = np.array(
            [(bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2]
        )

    def _load_or_calculate_triangulation(self):
        """Load or calculate the triangulation for each zone."""

        target_folder = "./out/storage/"
        os.makedirs(target_folder, exist_ok=True)
        digest = hashlib.sha256()
        digest.update(str(self.zones_gdf.crs).encode("utf-8"))
        for zone_id, zone_geometry in zip(self.zone_ids, self.zones_gdf.geometry):
            digest.update(str(zone_id).encode("ascii") + b"\0")
            digest.update(zone_geometry.wkb)
        triangulation_file = f"{target_folder}TRIANGULATION-v2-{digest.hexdigest()}.npy"

        if os.path.isfile(triangulation_file):
            # the triangulation is available, so we load it
            with open(triangulation_file, "rb") as f:
                self.zone_tris = pickle.load(f)
                return

        # the triangulation is not available, so we need to calculate it
        self.zone_tris = {}
        for i, zone_geometry in enumerate(self.zones_gdf["geometry"]):
            tris = tri.get_zone_triangulation(zone_geometry)
            self.zone_tris[self.zones_gdf.at[i, "zone_id"]] = tris
        fd, tmp_file = tempfile.mkstemp(
            prefix=f".{os.path.basename(triangulation_file)}.",
            suffix=".tmp",
            dir=target_folder,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self.zone_tris, f)
            os.replace(tmp_file, triangulation_file)
        finally:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)

    def _load_requests(self):
        if self.city == "nyc":
            self.request_generator = request_nyc(
                self.request_file,
                self.zone_ids,
                self.zone_tris,
                self.start_time,
                self.max_time,
                self.request_origin_randomization,
                self.request_destination_randomization,
                self.request_time_randomization,
                self.midpoint,
            )
        elif self.city == "chicago":
            self.request_generator = request_chicago(
                self.request_file,
                self.zone_ids,
                self.zone_tris,
                self.start_time,
                self.max_time,
                self.request_origin_randomization,
                self.request_destination_randomization,
                self.request_time_randomization,
                self.midpoint,
            )
        else:
            raise NotImplementedError(f"City {self.city} not supported")

    def _load_parking_lots(self):
        with open(self.parking_file) as f:
            parkings = json.load(f)
            parking_lots = parkings.get("parking_lots", [])
        self.D = len(parking_lots)

        self.parking_lot_weights = self._get_lot_weights(parking_lots)
        self.lot_coords = np.array([[c["x"], c["y"]] for c in parking_lots])

    def _parse_infrastructure(self, config: dict):
        infrastructure = parse_infrastructure(config, self.D)
        self.set_charging_infrastructure(infrastructure)

    def set_charging_infrastructure(self, solution: ChargingInfrastructure):
        if len(solution) != 2:
            raise ValueError(
                "Charging infrastructure must contain charging and reposition counts."
            )

        self.full_charger_count = np.asarray(solution[0], dtype=np.int32)
        self.full_repo_count = np.asarray(solution[1], dtype=np.int32)
        expected_shape = (self.D,)
        if self.full_charger_count.shape != expected_shape:
            raise ValueError(
                f"Charger counts must have shape {expected_shape}, "
                f"got {self.full_charger_count.shape}."
            )
        if self.full_repo_count.shape != expected_shape:
            raise ValueError(
                f"Reposition counts must have shape {expected_shape}, "
                f"got {self.full_repo_count.shape}."
            )
        if np.any(self.full_charger_count < 0) or np.any(self.full_repo_count < 0):
            raise ValueError("Infrastructure counts cannot be negative.")
        if not np.any(self.full_repo_count > 0):
            raise ValueError(
                "Infrastructure must contain at least one reposition location."
            )
        if not np.any(self.full_charger_count > 0):
            raise ValueError(
                "Infrastructure must contain at least one charging location."
            )
        chargers_without_repository = (self.full_charger_count > 0) & (
            self.full_repo_count <= 0
        )
        if np.any(chargers_without_repository):
            indices = np.flatnonzero(chargers_without_repository).tolist()
            raise ValueError(
                "Every charging location must also be a reposition location; "
                f"invalid candidate indices: {indices}."
            )

        self.D_repo: int = int(np.sum(self.full_repo_count > 0))
        self.cs_idxs = np.nonzero(self.full_charger_count > 0)[0]
        self.repo_idxs = np.nonzero(self.full_repo_count > 0)[0]

        self.repo_coords = self.lot_coords[self.repo_idxs]
        self.cs_coords = self.lot_coords[self.cs_idxs]
        self.repo_count = self.full_repo_count[self.repo_idxs]
        self.charger_count = self.full_charger_count[self.repo_idxs]

        self.repo_to_cs = np.min(
            geometry.distance_matrix(self.repo_coords, self.cs_coords),
            axis=1,
        )
        self._reset_repository_distance_cache()

    def _reset_repository_distance_cache(self) -> None:
        """Invalidate cached vehicle-to-reposition-location distance rows."""
        self._repo_distance_cache_locations = np.empty(
            (self.num_evs, 2), dtype=np.float64
        )
        self._repo_distance_cache = np.full(
            (self.num_evs, self.D_repo), np.inf, dtype=np.float64
        )
        self._repo_distance_cache_valid = np.zeros(self.num_evs, dtype=bool)

    def repository_distances(
        self,
        vehicle_locations: np.ndarray,
        vehicle_indices: np.ndarray,
    ) -> np.ndarray:
        """Update and return cached vehicle-to-reposition-location distances.

        Reposition coordinates remain fixed between infrastructure updates,
        while most vehicles remain stationary across consecutive decision
        epochs.  Cache rows by vehicle identity and exact location so only
        vehicles whose locations changed incur a new distance calculation.
        Only the rows named by ``vehicle_indices`` are guaranteed to match
        ``vehicle_locations``.  Callers must treat the returned cache as
        read-only.
        """
        locations = np.asarray(vehicle_locations, dtype=np.float64)
        indices = np.asarray(vehicle_indices, dtype=np.intp)

        expected_shape = (self.num_evs, 2)
        if locations.shape != expected_shape:
            raise ValueError(
                f"Vehicle locations must have shape {expected_shape}, "
                f"got {locations.shape}."
            )
        if indices.ndim != 1:
            raise ValueError("Vehicle indices must be one-dimensional.")
        if indices.size == 0:
            return self._repo_distance_cache
        if np.any(indices < 0) or np.any(indices >= self.num_evs):
            raise IndexError("Vehicle index is outside the configured fleet.")

        selected_locations = locations[indices]
        unchanged = self._repo_distance_cache_valid[indices] & np.all(
            self._repo_distance_cache_locations[indices] == selected_locations,
            axis=1,
        )
        changed_indices = indices[~unchanged]
        if changed_indices.size:
            self._repo_distance_cache[changed_indices] = geometry.distance_matrix(
                locations[changed_indices], self.repo_coords
            )
            self._repo_distance_cache_locations[changed_indices] = locations[
                changed_indices
            ]
            self._repo_distance_cache_valid[changed_indices] = True

        return self._repo_distance_cache

    def _get_lot_weights(self, parking_lots):
        """Lot weights are the probability that they will be selected as the
        starting point for a vehicle at the beginning of an episode. Episodes begin
        in the 3am hour, so the weights are the frequency of zones' dropoffs during
        the 2am hour.
        """

        request_df = self.request_generator.get_df(
            self.request_generator.doys[0], 0, 86400
        )
        # request_df = request_df[request_df[self.request_generator.time_key] < 7200]
        do_hits = request_df.groupby(self.request_generator.origin_key).size()
        do_frequency = do_hits / sum(do_hits)  # type: ignore

        cs_zone_ct = {
            lot["zone"]: sum(1 for lot_ in parking_lots if lot_["zone"] == lot["zone"])
            for lot in parking_lots
        }
        return [
            do_frequency[lot["zone"]] / cs_zone_ct[lot["zone"]] for lot in parking_lots
        ]

    def generate_requests(self, seed=1234, num_requests=-1):
        num_generated = num_requests if num_requests > 0 else self.num_requests
        return self.request_generator.generate(seed, num_requests=num_generated)

    def generate_requests_for_T(self, doy, seed=1234, num_requests=10000):
        return self.request_generator._load_or_generate_doy(
            doy,
            seed,
            num_requests=num_requests,
            start_time=0,
            end_time=86400,
            generate_dropoff=False,
            generate_proctime=False,
        )

    def time(self, o, d):
        """Returns the constant-velocity travel time between o and d."""
        return geometry.get_distance(o, d) / self.ev_speed

    def charge(self, o, d):
        """
        Returns the charge consumed to travel between o and d when consuming charge in r
        kWh/km.
        """
        return geometry.get_distance(o, d) * self.ev_discharge_rate

    def request_value(self, o, d):
        """Revenue associted to serving a request with origin `o` and destination `d`"""
        return (
            self.reward_fixed
            + self.reward_variable_dist * geometry.element_distance(o, d)
        )
