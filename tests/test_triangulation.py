"""Polygon coverage, sampling geometry, and triangulation caching."""

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon

from openhail.core.openhail_instance import OpenhailInstance
from openhail.utils import triangulation as tri
from openhail.utils.triangulation import Triangulation, triangulate_polygon

pytestmark = pytest.mark.filterwarnings("error:Arrays of 2-dimensional vectors")


def test_multipolygon_triangles_cover_each_component():
    polygon = MultiPolygon(
        [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(10, 0), (12, 0), (12, 2), (10, 2)]),
        ]
    )
    points, triangles = triangulate_polygon(polygon)
    pieces = [Polygon(points[indices]) for indices in triangles]
    for component in polygon.geoms:
        assert sum(p.area for p in pieces if component.covers(p)) == pytest.approx(
            component.area
        )


def test_polygon_hole_is_not_sampled():
    polygon = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )
    points, triangles = triangulate_polygon(polygon)
    pieces = [Polygon(points[indices]) for indices in triangles]
    assert all(polygon.covers(piece) for piece in pieces)
    assert sum(piece.area for piece in pieces) == pytest.approx(polygon.area)


@pytest.mark.parametrize("height, orientation", [(1, -1), (-1, 1), (0, 0)])
def test_planar_orientation(height, orientation):
    triangulation = Triangulation(
        np.array([[2, 3], [4, 3], [3, 3 + height]]),
        np.array([[0, 1], [1, 2], [2, 0]]),
    )
    assert triangulation._orientation((0, 1), 2) == orientation
    assert bool(triangulation._iscounterclockwise(0, 1, 2)) == (height > 0)


@pytest.mark.parametrize("reverse", [False, True])
def test_concave_polygon_triangulation_preserves_geometry(reverse):
    vertices = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 3), (0, 3)]
    polygon = Polygon(vertices[::-1] if reverse else vertices)
    points, triangles = triangulate_polygon(polygon)
    pieces = [Polygon(points[indices]) for indices in triangles]
    assert pieces
    assert all(polygon.covers(piece) for piece in pieces)
    assert sum(piece.area for piece in pieces) == pytest.approx(polygon.area)


def test_triangulation_cache_tracks_geometry(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    instance = OpenhailInstance.__new__(OpenhailInstance)
    instance.zones_file = "same-name.geojson"
    instance.zone_ids = [1]
    for width in (1, 3):
        instance.zones_gdf = gpd.GeoDataFrame(
            {"zone_id": [1]},
            geometry=[Polygon([(0, 0), (width, 0), (width, 1), (0, 1)])],
            crs="EPSG:26918",
        )
        instance._load_or_calculate_triangulation()
        assert sum(instance.zone_tris[1][tri.AREAS]) == pytest.approx(width)
