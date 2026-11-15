"""Interactive playback controls for rendered simulations."""

from typing import Protocol, cast

import pytest

pygame = pytest.importorskip("pygame")

from openhail.gui.control_panel import KEY_BINDINGS, ControlPanel  # noqa: E402
from openhail.gui.session import (  # noqa: E402
    DEFAULT_SPEED_INDEX,
    SPEED_PRESETS,
    PlaybackState,
    SimulationSession,
    _real_seconds_per_step,
)


class TestPlaybackState:
    def test_defaults_play_and_use_the_documented_default_speed(self):
        state = PlaybackState()
        assert state.playing is True
        assert state.render_factor == SPEED_PRESETS[DEFAULT_SPEED_INDEX]
        assert state.pending_steps == 0

    def test_toggle_play_flips_and_clears_pending_steps(self):
        state = PlaybackState()
        state.step(3)
        assert state.playing is False
        assert state.pending_steps == 3

        state.toggle_play()
        assert state.playing is True
        assert state.pending_steps == 0

    def test_speed_changes_clamp_at_both_ends(self):
        state = PlaybackState()
        for _ in range(len(SPEED_PRESETS) * 2):
            state.faster()
        assert state.speed_index == len(SPEED_PRESETS) - 1
        assert state.render_factor == max(SPEED_PRESETS)

        for _ in range(len(SPEED_PRESETS) * 2):
            state.slower()
        assert state.speed_index == 0
        assert state.render_factor == min(SPEED_PRESETS)

    def test_up_is_slower_so_faster_raises_the_render_factor(self):
        state = PlaybackState()
        before = state.render_factor
        state.faster()
        assert state.render_factor > before

    def test_step_pauses_and_reserves_exactly_the_requested_steps(self):
        state = PlaybackState()
        state.step(10)
        assert state.playing is False
        allowed = sum(1 for _ in range(20) if state.should_step())
        assert allowed == 10
        assert state.pending_steps == 0

    def test_step_never_reserves_fewer_than_one(self):
        state = PlaybackState()
        state.step(0)
        assert state.pending_steps == 1
        state2 = PlaybackState()
        state2.step(-5)
        assert state2.pending_steps == 1

    def test_while_playing_every_step_is_allowed(self):
        state = PlaybackState()
        assert all(state.should_step() for _ in range(5))

    def test_finished_blocks_all_steps_even_when_a_step_is_pending(self):
        state = PlaybackState(finished=True)
        state.step(5)
        assert state.should_step() is False
        assert state.pending_steps == 5

    def test_should_draw_is_rate_limited(self):
        state = PlaybackState(draw_interval=1.0)
        assert state.should_draw(now=100.0) is True
        assert state.should_draw(now=100.5) is False
        assert state.should_draw(now=101.1) is True

    def test_speed_label_reports_the_render_factor(self):
        state = PlaybackState()
        assert state.speed_label == f"{state.render_factor:.2f} min/day"


class TestKeyHandling:
    @staticmethod
    def _key(key, mod=0):
        """A lightweight stand-in for a Pygame KEYDOWN event."""
        return pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": mod})

    def test_every_documented_binding_is_exercised(self):
        """Fail loudly if a binding is added to the overlay but not handled."""
        handled = {"space", "right", "shift+right", "up", "down", "h", "q", "esc"}
        assert handled == set(KEY_BINDINGS)

    def test_space_pauses_and_resumes(self):
        panel = _panel(PlaybackState())
        panel.handle_events([self._key(pygame.K_SPACE)])
        assert panel.playback.playing is False
        panel.handle_events([self._key(pygame.K_SPACE)])
        assert panel.playback.playing is True

    def test_right_steps_once_and_shift_right_steps_ten(self):
        panel = _panel(PlaybackState())
        panel.handle_events([self._key(pygame.K_RIGHT)])
        assert panel.playback.pending_steps == 1
        assert panel.playback.playing is False

        panel2 = _panel(PlaybackState())
        panel2.handle_events([self._key(pygame.K_RIGHT, pygame.KMOD_SHIFT)])
        assert panel2.playback.pending_steps == 10

    def test_up_is_slower_and_down_is_faster(self):
        panel = _panel(PlaybackState())
        start = panel.playback.render_factor
        panel.handle_events([self._key(pygame.K_UP)])
        assert panel.playback.render_factor > start
        panel.handle_events([self._key(pygame.K_DOWN)])
        assert panel.playback.render_factor == start

    def test_h_toggles_help(self):
        panel = _panel(PlaybackState())
        assert panel.show_help is False
        panel.handle_events([self._key(pygame.K_h)])
        assert panel.show_help is True
        panel.handle_events([self._key(pygame.K_h)])
        assert panel.show_help is False

    def test_q_and_escape_request_quit(self):
        for key in (pygame.K_q, pygame.K_ESCAPE):
            panel = _panel(PlaybackState())
            panel.handle_events([self._key(key)])
            assert panel.quit_requested is True

    def test_non_key_events_are_ignored(self):
        panel = _panel(PlaybackState())
        panel.handle_events([pygame.event.Event(pygame.MOUSEMOTION, {"pos": (1, 1)})])
        assert panel.quit_requested is False
        assert panel.playback.playing is True


class TestPacing:
    def test_zero_and_negative_render_factors_never_sleep(self):
        assert _real_seconds_per_step(0.0) == 0.0
        assert _real_seconds_per_step(-1.0) == 0.0

    def test_pacing_scales_with_the_render_factor(self):
        assert _real_seconds_per_step(2.0) > _real_seconds_per_step(1.0) > 0.0


class _Controllable(Protocol):
    """The surface ControlPanel actually needs from a session.

    Mirrors :class:`openhail.gui.control_panel.ControllableSession`; restated
    here so the fake is checked against the panel's real requirements.
    """

    playback: PlaybackState
    finished: bool
    steps: int
    total_reward: float
    sim_time: float
    instance: object
    env: object

    def start(self) -> object: ...

    def pump(self) -> list[object]: ...

    def advance(self) -> None: ...

    def draw(self) -> None: ...

    def _pace_before_step(self) -> None: ...


class _FakeSession:
    """Minimal stand-in so controls can be tested without an environment."""

    def __init__(self, playback: PlaybackState) -> None:
        self.playback = playback
        self.finished = False
        self.steps = 0
        self.total_reward = 0.0
        self.sim_time = 0.0
        self.instance = object()
        self.env = object()

    def start(self) -> dict:
        return {}

    def pump(self) -> list:
        return []

    def advance(self) -> None:
        self.steps += 1

    def draw(self) -> None:
        pass

    def _pace_before_step(self) -> None:
        pass


def _panel(playback: PlaybackState) -> ControlPanel:
    """Build a panel over the fake session for control-only tests."""
    return ControlPanel(cast("_Controllable", _FakeSession(playback)))


class TestSessionWithoutDisplay:
    """The session itself is exercised in the integration test; these cover the
    parts that need no environment."""

    def test_advance_before_start_is_an_error(self):
        session = SimulationSession.__new__(SimulationSession)
        session.started = False
        session.finished = False
        with pytest.raises(RuntimeError):
            SimulationSession.advance(session)
