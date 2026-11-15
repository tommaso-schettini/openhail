"""Tests for the two defects fixed alongside the interactive playback controls.

Both are small, but both are real:

* the request generator used the deprecated ``reset_index(inplace=True)``, which
  pandas 3 emits a ``Pandas4Warning`` for and plans to remove (PDEP-8);
* the renderer read one ``ev_color`` key for three different colours, so an
  unconfigured run silently drew vehicles, lines and dots in the same colour.
"""

import warnings

import pandas as pd
import pytest

pygame = pytest.importorskip("pygame")

from openhail.core.env_renderer import (  # noqa: E402
    DARK_GREEN,
    DARK_RED,
    DARK_YELLOW,
    RidehailRenderer,
)


class TestResetIndexIsNotInPlace:
    """``inplace=True`` is deprecated in pandas 3 and slated for removal."""

    def test_no_reset_index_inplace_in_request_modules(self):
        import pathlib

        import openhail

        root = next(
            (pathlib.Path(p) / "core" / "request" for p in openhail.__path__),
            None,
        )
        assert root is not None and root.is_dir(), "request package not found"
        offenders = []
        for path in sorted(root.glob("*.py")):
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1
            ):
                if "reset_index" in line and "inplace" in line:
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
        assert offenders == [], "reset_index(inplace=...) is deprecated:\n" + "\n".join(
            offenders
        )

    def test_reset_index_stays_equivalent(self):
        """The non-inplace form must produce the same frame as the inplace one."""
        frame = pd.DataFrame({"a": [1, 2, 3]}, index=[10, 11, 12])
        expected = frame.reset_index(drop=True)
        assert list(expected["a"]) == [1, 2, 3]
        assert list(expected.index) == [0, 1, 2]

    def test_assignment_suppresses_the_pandas4_warning(self):
        frame = pd.DataFrame({"a": [1]}, index=[7])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            frame = frame.reset_index(drop=True)
        deprecations = [
            str(w.message)
            for w in caught
            if "inplace" in str(w.message) and "reset_index" in str(w.message)
        ]
        assert deprecations == []


class TestRendererColourConfig:
    """Each colour must read its own key so configs can set them separately."""

    @staticmethod
    def _colours(config):
        """Read the three colour attributes without building a full instance."""
        renderer = RidehailRenderer.__new__(RidehailRenderer)
        renderer.full_ev_color = config.get("ev_color", DARK_YELLOW)
        renderer.empty_ev_color = config.get("empty_ev_color", DARK_RED)
        renderer.line_color = config.get("line_color", DARK_GREEN)
        return renderer

    def test_defaults_are_three_distinct_colours(self):
        renderer = self._colours({})
        assert renderer.full_ev_color == DARK_YELLOW
        assert renderer.empty_ev_color == DARK_RED
        assert renderer.line_color == DARK_GREEN
        assert (
            len(
                {
                    renderer.full_ev_color,
                    renderer.empty_ev_color,
                    renderer.line_color,
                }
            )
            == 3
        )

    def test_keys_are_independent(self):
        renderer = self._colours(
            {
                "ev_color": (1, 2, 3),
                "empty_ev_color": (4, 5, 6),
                "line_color": (7, 8, 9),
            }
        )
        assert renderer.full_ev_color == (1, 2, 3)
        assert renderer.empty_ev_color == (4, 5, 6)
        assert renderer.line_color == (7, 8, 9)

    def test_setting_one_key_does_not_move_the_others(self):
        renderer = self._colours({"line_color": (7, 8, 9)})
        assert renderer.line_color == (7, 8, 9)
        assert renderer.empty_ev_color == DARK_RED

    def test_ev_color_still_controls_the_vehicle_colour(self):
        """The documented key keeps working for existing configs."""
        renderer = self._colours({"ev_color": (10, 20, 30)})
        assert renderer.full_ev_color == (10, 20, 30)
