"""Invalid cached and geometric inputs should fail at their boundary."""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Polygon

from openhail.core.request.request_generator import request_generator
from openhail.utils import triangulation as tri
from openhail.utils.triangulation import triangulate_polygon


def test_empty_multipolygon_is_rejected():
    with pytest.raises(ValueError, match="empty MultiPolygon"):
        triangulate_polygon(MultiPolygon([]))


def test_request_cache_rejects_series(tmp_path, monkeypatch):
    path = tmp_path / "requests.pkl"
    pd.Series([1, 2]).to_pickle(path)
    generator = request_generator(
        name="synthetic",
        doys=[1],
        zone_ids=[],
        tris={},
        start_time=0,
        max_time=100,
        orig_randomization=0,
        dest_randomization=0,
        time_randomization=0,
        midpoint=np.zeros(2),
    )
    monkeypatch.setattr(generator, "_cache_name", lambda *args: str(path))
    with pytest.raises(ValueError, match="must contain a DataFrame"):
        generator._load_or_generate_doy(1, 1, 2, 0, 100)


def test_nonzero_start_uses_absolute_end_time(monkeypatch):
    generator = request_generator(
        "synthetic", [1], [], {}, 100, 200, 0, 0, 0, np.zeros(2)
    )
    intervals = []

    def generate(doy, seed, count, start, end):
        intervals.append((start, end))
        return pd.DataFrame({"time": [start]})

    monkeypatch.setattr(generator, "_generate_doy", generate)
    generator.generate(0, 1, load=False, generate_dummy=False)
    assert intervals == [(100, 200)]


def test_horizon_does_not_silently_exhaust_source_days():
    generator = request_generator(
        "synthetic", [1], [], {}, 0, 172800, 0, 0, 0, np.zeros(2)
    )
    with pytest.raises(ValueError, match="available source days"):
        generator.generate(0, 1, load=False)


def test_request_cache_tracks_geometry(tmp_path):
    source = tmp_path / "requests.csv"
    source.write_text("source", encoding="utf-8")
    names = []
    for width in (1, 3):
        triangles = tri.get_zone_triangulation(
            Polygon([(0, 0), (width, 0), (width, 1), (0, 1)])
        )
        generator = request_generator(
            "synthetic", [1], [1], {1: triangles}, 0, 100, 0, 0, 0, np.zeros(2)
        )
        generator.request_path = str(source)
        names.append(generator._cache_name(1, 0, 1, 0, 100))
    assert names[0] != names[1]


def test_request_sampling_accepts_noncontiguous_dataframe_index():
    generator = request_generator(
        "synthetic", [1], [], {}, 0, 100, 0, 0, 0, np.zeros(2)
    )
    source = pd.DataFrame({"episode_time": [10, 20]}, index=[50, 90])
    requests = generator._generate_from_df(
        source,
        0,
        2,
        generate_pickup=False,
        generate_dropoff=False,
        generate_proctime=False,
    )
    assert requests["time"].tolist() == [10, 20]
    assert generator._generate_from_df(source, 0, 0).empty
