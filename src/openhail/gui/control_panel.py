"""Keyboard-driven control panel for an interactive OpenHail run.

The panel adds playback controls on top of :class:`SimulationSession`: pause and
resume, single-step, speed up and slow down, and a quit key. Key bindings are
chosen to be mnemonic and to avoid the keys the renderer already uses, and are
exported as a constant so they can be documented and tested.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

from .session import PlaybackState

if TYPE_CHECKING:
    from .session import SimulationSession

LOGGER = logging.getLogger(__name__)

# Reserved so the README, the help overlay and the tests agree on one source.
KEY_BINDINGS: dict[str, str] = {
    "space": "pause / resume",
    "right": "step one decision epoch",
    "shift+right": "step ten decision epochs",
    "up": "slower (more real minutes per simulated day)",
    "down": "faster (fewer real minutes per simulated day)",
    "h": "toggle this help overlay",
    "q": "quit",
    "esc": "quit",
}


class ControllableSession(Protocol):
    """The part of a session the panel drives.

    Declared as a protocol rather than a concrete class so that callers may
    substitute a stand-in (for example in tests, or for a future replay that
    reads a recorded trajectory) without subclassing the real session.
    """

    playback: PlaybackState
    finished: bool
    steps: int
    total_reward: float
    sim_time: float
    instance: Any
    env: Any

    def start(self) -> Any: ...

    def pump(self) -> list[Any]: ...

    def advance(self) -> None: ...

    def draw(self) -> None: ...

    def _pace_before_step(self) -> None: ...


class ControlPanel:
    """Drive a :class:`SimulationSession` from keyboard input.

    Args:
        session: The session to drive. It need not be started.
        show_help: Whether the help overlay starts visible.
        on_quit: Optional callback invoked with the accumulated reward just
            before the panel returns.
    """

    def __init__(
        self,
        session: "SimulationSession | ControllableSession",
        show_help: bool = False,
        on_quit: Callable[[float], None] | None = None,
    ) -> None:
        self.session = cast(ControllableSession, session)
        self.playback: PlaybackState = session.playback
        self.show_help = show_help
        self.on_quit = on_quit
        self.quit_requested = False

    def handle_events(self, events: list[Any]) -> None:
        """Map Pygame key events to playback actions.

        Separated from :meth:`run` so the mapping can be exercised without a
        display.
        """
        import pygame

        for event in events:
            if getattr(event, "type", None) != pygame.KEYDOWN:
                continue
            key = event.key
            mods = getattr(event, "mod", 0)

            if key in (pygame.K_ESCAPE, pygame.K_q):
                self.quit_requested = True
            elif key == pygame.K_SPACE:
                self.playback.toggle_play()
            elif key == pygame.K_h:
                self.show_help = not self.show_help
            elif key == pygame.K_RIGHT:
                step_count = 10 if mods & pygame.KMOD_SHIFT else 1
                self.playback.step(step_count)
            elif key == pygame.K_UP:
                # A larger render factor is a slower run.
                self.playback.faster()
            elif key == pygame.K_DOWN:
                self.playback.slower()

    def run(self) -> float:
        """Run until the episode ends or the user quits; return the reward."""
        self.session.start()
        self.session.draw()

        while not self.session.finished and not self.quit_requested:
            self.handle_events(self.session.pump())
            if self.quit_requested:
                break

            if self.playback.should_step():
                self.session._pace_before_step()
                self.session.advance()

            if self.playback.should_draw():
                self.session.draw()
                self._draw_overlay()

        if self.on_quit is not None:
            self.on_quit(self.session.total_reward)
        LOGGER.info(
            "Interactive run ended: steps=%d time=%.2f reward=%.2f",
            self.session.steps,
            self.session.sim_time - self.session.instance.start_time,
            self.session.total_reward,
        )
        return self.session.total_reward

    def _draw_overlay(self) -> None:
        """Draw the pause banner and, when toggled, the help overlay.

        Draws onto the renderer's window so the panel adds no second surface and
        no extra dependency.
        """
        renderer = getattr(self.session.env, "renderer", None)
        window = getattr(renderer, "window", None)
        if window is None:
            return

        import pygame

        font = pygame.font.SysFont("Consolas", 18, bold=True)
        lines: list[str] = []
        if not self.playback.playing:
            lines.append("PAUSED  space=resume  right=step")
        lines.append(f"speed {self.playback.speed_label}")
        if self.show_help:
            lines.extend(f"{k:<14}{v}" for k, v in KEY_BINDINGS.items())

        y = 8
        for line in lines:
            text = font.render(line, True, (0, 0, 0))
            background = pygame.Surface(
                (text.get_width() + 8, text.get_height() + 4), pygame.SRCALPHA
            )
            background.fill((255, 255, 255, 200))
            window.blit(background, (6, y))
            window.blit(text, (10, y + 2))
            y += text.get_height() + 6

        pygame.display.flip()
