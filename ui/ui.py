"""
ui.py — Heads-up display, info panel, and time controls.

Draws on top of the renderer each frame:
  • Top bar  : time warp controls + sim time elapsed
  • Right panel : selected body stats
  • Bottom bar  : controls cheat-sheet
  • Notifications: collision messages that fade out
"""
from __future__ import annotations
import math
import time as _time

import pygame

from simulation import Simulation, TIME_WARPS
from bodies.body import CelestialBody
from bodies.star import Star
from bodies.planet import Planet
from bodies.spacecraft import Spacecraft


def _fmt_time(seconds: float) -> str:
    """Format simulation time in a human-readable way."""
    if seconds < 3600:
        return f"{seconds:.0f} s"
    if seconds < 86400:
        return f"{seconds/3600:.1f} h"
    if seconds < 86400 * 365.25:
        return f"{seconds/86400:.1f} d"
    years = seconds / (365.25 * 86400)
    if years < 1000:
        return f"{years:.2f} yr"
    return f"{years:.2e} yr"


def _fmt_si(value: float, unit: str) -> str:
    """Format a value with SI prefix."""
    for prefix, exp in [("G", 9), ("M", 6), ("k", 3)]:
        if abs(value) >= 10 ** exp:
            return f"{value / 10**exp:.3g} {prefix}{unit}"
    return f"{value:.4g} {unit}"


class Notification:
    def __init__(self, msg: str, duration: float = 3.0, color: tuple = (255, 220, 80)) -> None:
        self.msg = msg
        self.color = color
        self.expire_at = _time.time() + duration


class HUD:
    """
    Draws the entire UI overlay.

    Parameters
    ----------
    screen   : pygame.Surface
    sim      : Simulation
    font_sm  : small font (for labels)
    font_md  : medium font (for stats)
    font_lg  : large font (for top bar)
    """

    PANEL_W = 260
    TOP_H   = 36
    BOT_H   = 24

    def __init__(self, screen: pygame.Surface, sim: Simulation) -> None:
        self.screen = screen
        self.sim = sim
        self.W = screen.get_width()
        self.H = screen.get_height()

        pygame.font.init()
        self.font_sm = pygame.font.SysFont("Consolas", 11)
        self.font_md = pygame.font.SysFont("Consolas", 13)
        self.font_lg = pygame.font.SysFont("Consolas", 15, bold=True)

        self._notifications: list[Notification] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def push_notification(self, msg: str, color: tuple = (255, 220, 80), duration: float = 3.0) -> None:
        self._notifications.append(Notification(msg, duration, color))

    def draw(self) -> None:
        self._draw_top_bar()
        self._draw_body_panel()
        self._draw_bottom_bar()
        self._draw_notifications()

    # ------------------------------------------------------------------ #
    #  Top bar: time warp + elapsed time                                   #
    # ------------------------------------------------------------------ #

    def _draw_top_bar(self) -> None:
        bar = pygame.Surface((self.W, self.TOP_H), pygame.SRCALPHA)
        bar.fill((10, 10, 30, 200))
        self.screen.blit(bar, (0, 0))

        sim = self.sim
        warp = sim.time_warp
        elapsed = _fmt_time(sim.sim_time)

        # Time warp indicator
        warp_str = "PAUSED" if sim.paused else (f"◀◀" if warp < 1 else f"▶▶" if warp > 1 else "▶")
        warp_label = f"  {warp_str}  ×{warp:g}    T: {elapsed}"
        surf = self.font_lg.render(warp_label, True, (255, 220, 80))
        self.screen.blit(surf, (10, 8))

        # Warp bar — highlight current level
        for i, w in enumerate(TIME_WARPS):
            if w == 0.0:
                continue
            col = (255, 180, 0) if w == warp else (60, 60, 80)
            px = 300 + i * 28
            label = f"×{w:g}" if w <= 100 else f"×{w:.0e}"
            s = self.font_sm.render(label, True, col)
            self.screen.blit(s, (px, 10))

    # ------------------------------------------------------------------ #
    #  Right panel: selected body info                                     #
    # ------------------------------------------------------------------ #

    def _draw_body_panel(self) -> None:
        body = self.sim.selected_body()
        if body is None:
            return

        panel_x = self.W - self.PANEL_W - 5
        panel_y = self.TOP_H + 5
        panel_h = 280

        panel = pygame.Surface((self.PANEL_W, panel_h), pygame.SRCALPHA)
        panel.fill((5, 10, 30, 210))
        self.screen.blit(panel, (panel_x, panel_y))

        lines: list[tuple[str, tuple]] = []
        lines.append((f"◉ {body.name}", (255, 230, 100)))
        lines.append((f"  Type: {body.body_type}", (180, 200, 255)))
        lines.append((f"  Mass: {_fmt_si(body.mass, 'kg')}", (200, 220, 200)))
        lines.append((f"  Radius: {_fmt_si(body.radius, 'm')}", (200, 220, 200)))
        lines.append((f"  Speed: {_fmt_si(body.speed, 'm/s')}", (200, 220, 200)))
        lines.append((f"  Surf-g: {body.surface_gravity():.2f} m/s²", (200, 220, 200)))

        if isinstance(body, Star):
            lines.append((f"  Temp: {body.temperature:.0f} K", (255, 200, 100)))
            lines.append((f"  Class: {body.spectral_class}", (255, 200, 100)))
            hz_i, hz_o = body.habitable_zone()
            au = 1.496e11
            lines.append((f"  HZ: {hz_i/au:.2f}–{hz_o/au:.2f} AU", (80, 230, 100)))

        if isinstance(body, Planet):
            lines.append((f"  Planet type: {body.planet_type}", (180, 200, 255)))

        if isinstance(body, Spacecraft):
            lines.append((f"  Fuel: {body.fuel * 100:.1f}%", (100, 255, 180)))
            lines.append((f"  Thrust: {_fmt_si(body.thrust_force, 'N')}", (100, 255, 180)))
            lines.append((f"  [WASD] rotate/thrust", (140, 140, 180)))

        # Find nearest star for orbital period estimate
        stars = [b for b in self.sim.bodies if isinstance(b, Star)]
        if stars and body not in stars:
            star = min(stars, key=lambda s: body.distance_to(s))
            period = body.orbital_period_around(star)
            dist_au = body.distance_to(star) / 1.496e11
            if period:
                period_days = period / 86400
                lines.append((f"  Dist: {dist_au:.3f} AU", (200, 200, 255)))
                lines.append((f"  Period: {_fmt_time(period)}", (200, 200, 255)))

        y_off = 8
        for (text, col) in lines:
            s = self.font_md.render(text, True, col)
            self.screen.blit(s, (panel_x + 6, panel_y + y_off))
            y_off += 18

    # ------------------------------------------------------------------ #
    #  Bottom bar: controls cheat-sheet                                    #
    # ------------------------------------------------------------------ #

    def _draw_bottom_bar(self) -> None:
        bar = pygame.Surface((self.W, self.BOT_H), pygame.SRCALPHA)
        bar.fill((10, 10, 30, 180))
        self.screen.blit(bar, (0, self.H - self.BOT_H))
        hints = (
            "[SPACE] Pause  [,/.] Warp  [A] Add  [F] Follow  "
            "[R] Reset  [1/2/3] Preset  [S] Save  [L] Load  [ESC] Quit"
        )
        s = self.font_sm.render(hints, True, (140, 140, 170))
        self.screen.blit(s, (8, self.H - self.BOT_H + 5))

    # ------------------------------------------------------------------ #
    #  Notifications                                                       #
    # ------------------------------------------------------------------ #

    def _draw_notifications(self) -> None:
        now = _time.time()
        self._notifications = [n for n in self._notifications if n.expire_at > now]
        y = self.TOP_H + 10
        for notif in self._notifications:
            age = 1.0 - (notif.expire_at - now) / 3.0
            alpha = int(255 * max(0, 1.0 - age))
            r, g, b = notif.color
            s = self.font_md.render(notif.msg, True, (r, g, b, alpha))
            self.screen.blit(s, (10, y))
            y += 20
