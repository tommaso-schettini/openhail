import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .request_generator import request_generator

DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"

CHICAGO_COLUMNS = [
    "Trip ID",
    "Taxi ID",
    "Trip Start Timestamp",
    "Trip End Timestamp",
    "Trip Seconds",
    "Trip Miles",
    "Pickup Census Tract",
    "Dropoff Census Tract",
    "Pickup Community Area",
    "Dropoff Community Area",
    "Fare",
    "Tips",
    "Tolls",
    "Extras",
    "Trip Total",
    "Payment Type",
    "Company",
    "Pickup Centroid Latitude",
    "Pickup Centroid Longitude",
    "Pickup Centroid Location",
    "Dropoff Centroid Latitude",
    "Dropoff Centroid Longitude",
    "Dropoff Centroid Location",
]


class request_chicago(request_generator):
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
            (np.arange(31) + 1),
            zone_ids,
            tris,
            start_time,
            max_time,
            orig_randomization,
            dest_randomization,
            time_randomization,
            midpoint,
            time_key="time",
            origin_key="Pickup Community Area",
            destination_key="Dropoff Community Area",
            time_discretization=(15 * 60),
        )

        self.request_path = request_file

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

        request_df = self.get_df(doy, start_time, end_time)
        return self._generate_from_df(
            request_df,
            seed,
            num_requests,
            generate_pickup,
            generate_dropoff,
            generate_proctime,
        )

    def get_df(
        self, doy, start_time: float = 0, end_time: float = 86400
    ) -> pd.DataFrame:
        def convert_to_timestamp(value):
            return datetime.datetime.strptime(value, DATE_FORMAT).timestamp()

        assert self.request_path is not None
        source_name = self.request_path + f"/{doy:02}.csv"
        request_df = pd.read_csv(source_name, header=None, names=CHICAGO_COLUMNS)
        request_df = request_df.dropna(
            subset=[
                "Pickup Community Area",
                "Dropoff Community Area",
                "Trip Start Timestamp",
            ]
        )
        request_df = request_df.reset_index(drop=True)
        request_df["time"] = request_df["Trip Start Timestamp"].apply(
            convert_to_timestamp
        )

        sampled_timestamp = (
            datetime.datetime.strptime(
                request_df.iloc[1]["Trip Start Timestamp"], DATE_FORMAT
            )
            .replace(hour=0, minute=0)
            .timestamp()
        )

        request_df["time"] = request_df["time"] - sampled_timestamp
        request_df = request_df.loc[request_df["time"] >= start_time]
        request_df = request_df.loc[request_df["time"] < end_time]
        request_df.reset_index(inplace=True, drop=True)

        return request_df
