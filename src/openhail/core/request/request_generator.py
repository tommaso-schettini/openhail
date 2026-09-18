import hashlib
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from ...utils import triangulation as tri
from ...utils.geometry import get_distance
from ..constants import DEST, ORIG, PROC_TIME, TIME

DAY_SECONDS = 86400
CACHE_VERSION = 5


class request_generator:
    def __init__(
        self,
        name,
        doys,
        zone_ids,
        tris,
        start_time,
        max_time,
        orig_randomization,
        dest_randomization,
        time_randomization,
        midpoint,
        time_key="episode_time",
        origin_key="puzone",
        destination_key="dozone",
        time_discretization=-1,
    ):
        self.name = name
        self.tris = tris
        self.zone_ids = zone_ids
        self.doys = doys

        self.orig_randomization = orig_randomization
        self.dest_randomization = dest_randomization
        self.time_randomization = time_randomization

        self.start_time = start_time
        self.max_time = max_time
        self.midpoint = midpoint

        self.time_key = time_key
        self.origin_key = origin_key
        self.destination_key = destination_key

        self.time_discretization = time_discretization
        self.request_path: str | None = None
        self._source_hashes = {}
        geometry_digest = hashlib.sha256()
        for zone_id in zone_ids:
            geometry_digest.update(str(zone_id).encode("ascii") + b"\0")
            if zone_id not in tris:
                continue
            for field in (tri.TRIANGLES, tri.REL_AREAS):
                geometry_digest.update(
                    np.asarray(tris[zone_id][field], dtype="<f8").tobytes()
                )
        self._geometry_hash = geometry_digest.hexdigest()[:16]

    def generate(
        self,
        seed: int,
        num_requests: int,
        load: bool = True,
        generate_dummy: bool = True,
    ) -> tuple[int, pd.DataFrame]:
        if self.max_time <= DAY_SECONDS:
            return self.generate_single(
                seed,
                num_requests,
                self.start_time,
                self.max_time,
                load,
                generate_dummy,
            )

        rng = np.random.RandomState(seed=seed)
        shuffled_doys = rng.permutation(self.doys)
        if not len(shuffled_doys):
            raise ValueError("Request source contains no days")
        required_days = int(np.ceil(self.max_time / DAY_SECONDS))
        if required_days > len(shuffled_doys):
            raise ValueError(
                "Episode horizon exceeds the number of available source days"
            )

        current_start_time = self.start_time
        current_max_time = self.max_time

        full_requests = None
        for idx, doy in enumerate(shuffled_doys):
            next_max_time = min(DAY_SECONDS, current_max_time)

            if load:
                requests = self._load_or_generate_doy(
                    doy, seed, num_requests, current_start_time, next_max_time
                )
            else:
                requests = self._generate_doy(
                    doy, seed, num_requests, current_start_time, next_max_time
                )

            requests[TIME] = requests[TIME] + DAY_SECONDS * idx

            current_max_time = current_max_time - next_max_time
            current_start_time = 0

            if full_requests is None:
                full_requests = requests
            else:
                full_requests = pd.concat([full_requests, requests], ignore_index=True)
            if current_max_time == 0:
                break

        if full_requests is None:
            raise RuntimeError("Request generation produced no day of requests.")
        full_requests.reset_index(inplace=True)

        if generate_dummy:
            # request is a dataframe
            full_requests.loc[len(full_requests)] = {
                TIME: self.max_time + 1,
                ORIG: self.midpoint,
                DEST: self.midpoint,
                PROC_TIME: -1,
            }

        return shuffled_doys[0], full_requests

    def generate_single(
        self,
        seed: int,
        num_requests: int,
        start_time: int,
        end_time: int,
        load: bool = True,
        generate_dummy: bool = True,
    ) -> tuple[int, pd.DataFrame]:
        rng = np.random.RandomState(seed=seed)
        doy = rng.choice(self.doys)

        if load:
            requests = self._load_or_generate_doy(
                doy, seed, num_requests, start_time, end_time
            )
        else:
            requests = self._generate_doy(doy, seed, num_requests, start_time, end_time)

        if generate_dummy:
            dummy_request = pd.DataFrame(
                [
                    {
                        TIME: self.max_time + 1,
                        ORIG: self.midpoint,
                        DEST: self.midpoint,
                        PROC_TIME: -1,
                    }
                ]
            )
            requests = pd.concat([requests, dummy_request], ignore_index=True)

        return doy, requests

    def _load_or_generate_doy(
        self,
        doy: int,
        seed: int,
        num_requests: int,
        start_time: int,
        end_time: int,
        generate_pickup: bool = True,
        generate_dropoff: bool = True,
        generate_proctime: bool = True,
    ) -> pd.DataFrame:
        target = self._cache_name(
            doy,
            seed,
            num_requests,
            start_time,
            end_time,
            generate_pickup,
            generate_dropoff,
            generate_proctime,
        )
        if os.path.isfile(target):
            with open(target, "rb") as f:
                cached = pd.read_pickle(f)
                if not isinstance(cached, pd.DataFrame):
                    raise ValueError("Request cache must contain a DataFrame")
                return cached

        requests = self._generate_doy(
            doy,
            seed,
            num_requests,
            start_time,
            end_time,
            generate_pickup,
            generate_dropoff,
            generate_proctime,
        )

        target_dir = os.path.dirname(target)
        os.makedirs(target_dir, exist_ok=True)
        fd, tmp_file = tempfile.mkstemp(
            prefix=f".{os.path.basename(target)}.",
            suffix=".tmp",
            dir=target_dir,
        )
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(requests, f)
            os.replace(tmp_file, target)
        finally:
            if os.path.exists(tmp_file):
                os.unlink(tmp_file)
        return requests

    def _generate_doy(
        self,
        doy,
        seed,
        num_requests,
        start_time,
        end_time,
        generate_pickup=True,
        generate_dropoff=True,
        generate_proctime=True,
    ) -> pd.DataFrame:
        raise NotImplementedError("_generate_doy not implemented!")

    def _generate_from_df(
        self,
        doy_requests_df,
        seed,
        num_requests,
        generate_pickup=True,
        generate_dropoff=True,
        generate_proctime=True,
    ) -> pd.DataFrame:
        rng = np.random.RandomState(seed=seed)
        requests_in_doy = doy_requests_df.shape[0]
        replace = num_requests > requests_in_doy
        if num_requests == 0:
            return pd.DataFrame(columns=[TIME, ORIG, DEST, PROC_TIME])
        if requests_in_doy == 0:
            raise ValueError("No source requests exist in the selected day/time window")
        req_idxs = rng.choice(requests_in_doy, replace=replace, size=num_requests)
        req_idxs = self._reroll_duplicate_request_times(
            doy_requests_df, req_idxs, rng, replace
        )
        reqs_df = doy_requests_df.iloc[req_idxs]

        request_list = []
        for _, row in reqs_df.iterrows():
            req = {}

            time = row[self.time_key]
            if self.time_discretization > 0:
                time = time + rng.randint(0, self.time_discretization)
            if (self.time_randomization > 0.0) and (
                rng.random() < self.time_randomization
            ):
                time = self.start_time + (
                    (self.max_time - self.start_time) * rng.random()
                )
            req[TIME] = time

            if generate_pickup:
                puzone = row[self.origin_key]
                if (self.orig_randomization > 0.0) and (
                    rng.random() < self.orig_randomization
                ):
                    puzone = rng.choice(self.zone_ids)
                req[ORIG] = self._rand_pt_in_zone(puzone, rng)

            if generate_dropoff:
                dozone = row[self.destination_key]
                if (self.dest_randomization > 0.0) and (
                    rng.random() < self.dest_randomization
                ):
                    dozone = rng.choice(self.zone_ids)
                req[DEST] = self._rand_pt_in_zone(dozone, rng)

            if generate_proctime and generate_pickup and generate_dropoff:
                req[PROC_TIME] = get_distance(req[ORIG], req[DEST])

            request_list.append(req)

        requests = pd.DataFrame(request_list)
        requests[TIME] = self._make_request_times_unique(
            requests[TIME].to_numpy(dtype=float)
        )
        return requests.sort_values(by=TIME).reset_index(drop=True)

    def _reroll_duplicate_request_times(
        self,
        requests_df,
        request_indices,
        rng,
        replace,
    ):
        """Replace sampled rows whose source timestamp is already represented."""
        request_indices = np.asarray(request_indices, dtype=int).copy()
        unique_source_times = requests_df[self.time_key].nunique()
        if len(request_indices) > unique_source_times:
            # Oversampled data cannot retain unique source timestamps. Keep the
            # empirical row sample and separate coincident events by adjacent
            # floating-point timestamps after request randomization.
            return request_indices
        sampled_times = requests_df.iloc[request_indices][self.time_key].to_numpy()

        time_counts = {}
        duplicate_positions = []
        for position, request_time in enumerate(sampled_times):
            if time_counts.get(request_time, 0):
                duplicate_positions.append(position)
            time_counts[request_time] = time_counts.get(request_time, 0) + 1

        if not duplicate_positions:
            return request_indices

        reserved_indices = set(request_indices.tolist()) if not replace else set()
        source_times = requests_df[self.time_key].to_numpy()

        for position in duplicate_positions:
            old_index = int(request_indices[position])
            old_time = source_times[old_index]
            time_counts[old_time] -= 1
            if not replace:
                reserved_indices.remove(old_index)

            candidates = [
                candidate
                for candidate, candidate_time in enumerate(source_times)
                if (replace or candidate not in reserved_indices)
                and time_counts.get(candidate_time, 0) == 0
            ]
            if not candidates:
                raise ValueError(
                    f"Cannot draw {len(request_indices)} requests with unique "
                    "timestamps from the selected day and horizon."
                )

            replacement = int(rng.choice(candidates))
            replacement_time = source_times[replacement]
            request_indices[position] = replacement
            time_counts[replacement_time] = 1
            if not replace:
                reserved_indices.add(replacement)

        return request_indices

    @staticmethod
    def _make_request_times_unique(request_times):
        """Separate coincident events without changing their demand period."""
        result = np.asarray(request_times, dtype=float).copy()
        used = set()
        next_candidates = {}
        for position, request_time in enumerate(result):
            value = float(request_time)
            if value not in used:
                used.add(value)
                continue

            candidate = next_candidates.get(value, np.nextafter(value, np.inf))
            while candidate in used:
                candidate = np.nextafter(candidate, np.inf)
            result[position] = candidate
            used.add(candidate)
            next_candidates[value] = np.nextafter(candidate, np.inf)
        return result

    def _rand_pt_in_zone(self, zone_id, rng):
        return tri.sample_random_pt(rng, self.tris[zone_id])

    def _cache_name(
        self,
        doy,
        seed,
        num_requests,
        start_time,
        end_time,
        generate_pickup=True,
        generate_dropoff=True,
        generate_proctime=True,
    ) -> str:

        filename = self.name
        source_hash = self._source_hash(doy)
        randomizations = (
            f"{self.orig_randomization}-{self.dest_randomization}-"
            f"{self.time_randomization}"
        )
        flag_dict = {
            "P": generate_pickup,
            "D": generate_dropoff,
            "T": generate_proctime,
        }
        flags = "".join([key if value else "_" for key, value in flag_dict.items()])

        res = (
            f"{filename}-{source_hash}-{self._geometry_hash}-"
            f"{doy}-{seed}-{start_time}-{end_time}-"
            f"{randomizations}[{num_requests}]-{flags}"
        )
        return f"./out/storage/REQ-v{CACHE_VERSION}-{res}.npy"

    def _source_hash(self, doy) -> str:
        """Return a cached content hash for the source data used by a day."""
        if doy in self._source_hashes:
            return self._source_hashes[doy]
        if self.request_path is None:
            raise RuntimeError("Request source path was not configured.")

        source = Path(self.request_path)
        if source.is_dir():
            day_file = source / f"{int(doy):02}.csv"
            files = [day_file] if day_file.is_file() else sorted(source.glob("*.csv"))
        else:
            files = [source]
        if not files or any(not path.is_file() for path in files):
            raise FileNotFoundError(f"Cannot fingerprint request source {source}.")

        digest = hashlib.sha256()
        for path in files:
            digest.update(path.name.encode("utf-8"))
            with path.open("rb") as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
        value = digest.hexdigest()[:16]
        self._source_hashes[doy] = value
        return value

    def get_df(
        self, doy, start_time: float = 0, end_time: float = 86400
    ) -> pd.DataFrame:
        raise NotImplementedError("get_df not implemented!")
