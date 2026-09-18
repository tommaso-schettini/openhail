from pathlib import Path

import pandas as pd

from .request_generator import request_generator


class request_nyc(request_generator):
    def __init__(
        self,
        request_file,
        zone_ids,
        tris,
        start_time,
        max_time,
        orig_randomization,
        dest_randomization,
        time_randomization,
        midpoint,
    ):
        super().__init__(
            Path(request_file).name,
            [],
            zone_ids,
            tris,
            start_time,
            max_time,
            orig_randomization,
            dest_randomization,
            time_randomization,
            midpoint,
        )

        self.request_path = request_file
        self.load_requests(request_file)

    def load_requests(self, request_file):
        self.requests_df = pd.read_csv(request_file, dtype=int)

        self.requests_df["episode_time"] = (
            ((self.requests_df["hr"] - 3) % 24) * 3600.0
            + self.requests_df["min"] * 60.0
            + self.requests_df["sec"]
            + self.requests_df["msec"] / 1000.0
        )

        self.requests_df = self.requests_df.dropna(
            subset=["doy", "episode_time", "dozone", "puzone"]
        )
        self.requests_df.reset_index(drop=True, inplace=True)
        self.doys = self.requests_df["doy"].unique()

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
    ):

        return self._generate_from_df(
            self.get_df(doy, start_time, end_time),
            seed,
            num_requests,
            generate_pickup,
            generate_dropoff,
            generate_proctime,
        )

    def get_df(
        self, doy, start_time: float = 0, end_time: float = 86400
    ) -> pd.DataFrame:
        in_day = self.requests_df["doy"] == doy
        after_start = self.requests_df["episode_time"] >= start_time
        before_end = self.requests_df["episode_time"] < end_time
        return self.requests_df.loc[in_day & after_start & before_end].reset_index(
            drop=True
        )
