"""
renderer.py — Pygame drawing engine for the simulation.

Draws:
  • Starfield background (static random noise)
  • Orbit trails (fading alpha)
  • Habitable zone ring (for stars)
  • Ring systems (Saturn, Uranus)
  • Glow aura for stars
  • Bodies scaled to their world radius (with min pixel size)
  • Velocity vector arrow on selected body
  • Selection ring highlight
  • Asteroid irregular polygons
  • Spacecraft triangle with thrust flame
"""
from __future__ import annotations
import math
import random

import pygame

from renderer.camera import Camera
from bodies.body import CelestialBody
from bodies.star import Star
from bodies.planet import Planet
from bodies.asteroid import Asteroid
from bodies.spacecraft import Spacecraft


# Minimum body size in pixels — so tiny moons are always clickable
_MIN_BODY_PX = 3
_STAR_MIN_PX  = 8


def _clamp_px(val: float, mn: float) -> int:
    return max(int(mn), int(val))


class Renderer:
    """
    Stateless-ish Pygame renderer.

    Usage:
        renderer = Renderer(screen, camera)
        renderer.draw(bodies, sim_time_s)
    """

    # How many trail points to render (never all of them for performance)
    TRAIL_RENDER_LIMIT = 400

    def __init__(self, screen: pygame.Surface, camera: Camera) -> None:
        self.screen = screen
        self.camera = camera
        self.W = screen.get_width()
        self.H = screen.get_height()

        # Generate a static starfield
        rng = random.Random(12345)
        self._starfield = [
            (rng.randint(0, self.W), rng.randint(0, self.H), rng.randint(1, 2),
             rng.randint(150, 255))
            for _ in range(350)
        ]

        # Scratch surfaces for glow effects
        self._glow_surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

    # ------------------------------------------------------------------ #
    #  Main draw call                                                      #
    # ------------------------------------------------------------------ #

    def draw(self, bodies: list[CelestialBody]) -> None:
        self.screen.fill((5, 5, 15))      # deep space background
        self._draw_starfield()
        self._draw_habitable_zones(bodies)
        self._draw_trails(bodies)
        self._draw_ring_systems(bodies)
        self._draw_stars_glow(bodies)
        self._draw_bodies(bodies)
        self._draw_velocity_arrows(bodies)

    # ------------------------------------------------------------------ #
    #  Starfield                                                           #
    # ------------------------------------------------------------------ #

    def _draw_starfield(self) -> None:
        for (x, y, r, brightness) in self._starfield:
            col = (brightness, brightness, brightness)
            pygame.draw.circle(self.screen, col, (x, y), r)

    # ------------------------------------------------------------------ #
    #  Habitable zones                                                     #
    # ------------------------------------------------------------------ #

    def _draw_habitable_zones(self, bodies: list[CelestialBody]) -> None:
        for body in bodies:
            if not isinstance(body, Star):
                continue
            inner, outer = body.habitable_zone()
            cx, cy = self.camera.world_to_screen(body.pos)
            inner_px = int(inner / self.camera.mpp)
            outer_px = int(outer / self.camera.mpp)
            if inner_px > 5000 or outer_px < 2:
                continue
            # Draw a faint green annulus
            surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            pygame.draw.circle(surf, (40, 160, 60, 25), (cx, cy), outer_px)
            pygame.draw.circle(surf, (5, 5, 15,  25), (cx, cy), max(0, inner_px))
            self.screen.blit(surf, (0, 0))
            # Outline rings
            if inner_px >= 2:
                pygame.draw.circle(self.screen, (40, 120, 50), (cx, cy), inner_px, 1)
            if outer_px >= 2:
                pygame.draw.circle(self.screen, (40, 120, 50), (cx, cy), outer_px, 1)

    # ------------------------------------------------------------------ #
    #  Orbit trails                                                        #
    # ------------------------------------------------------------------ #

    def _draw_trails(self, bodies: list[CelestialBody]) -> None:
        for body in bodies:
            if not body.trail or body.fixed:
                continue
            trail = list(body.trail)
            # Decimate if trail is very long
            step = max(1, len(trail) // self.TRAIL_RENDER_LIMIT)
            pts = trail[::step]
            if len(pts) < 2:
                continue
            n = len(pts)
            r, g, b = body.color
            for i in range(1, n):
                alpha = int(255 * (i / n) * 0.6)
                col = (min(255, r), min(255, g), min(255, b), alpha)
                p1 = self.camera.world_to_screen(pts[i - 1])
                p2 = self.camera.world_to_screen(pts[i])
                # Stay on screen check
                if (0 <= p1[0] <= self.W and 0 <= p1[1] <= self.H or
                        0 <= p2[0] <= self.W and 0 <= p2[1] <= self.H):
                    try:
                        pygame.draw.line(self.screen, (col[0], col[1], col[2]), p1, p2, 1)
                    except Exception:
                        pass

    # ------------------------------------------------------------------ #
    #  Ring systems (Saturn, Uranus, …)                                   #
    # ------------------------------------------------------------------ #

    def _draw_ring_systems(self, bodies: list[CelestialBody]) -> None:
        for body in bodies:
            if not isinstance(body, Planet) or not body.has_rings:
                continue
            cx, cy = self.camera.world_to_screen(body.pos)
            inner_px = int(body.ring_inner / self.camera.mpp)
            outer_px = int(body.ring_outer / self.camera.mpp)
            if outer_px < 2 or inner_px > 4000:
                continue
            rc = body.ring_color
            for r_px in range(inner_px, outer_px, max(1, (outer_px - inner_px) // 8)):
                alpha = 120
                pygame.draw.ellipse(
                    self.screen,
                    (rc[0], rc[1], rc[2]),
                    (cx - r_px, cy - r_px // 3, r_px * 2, r_px * 2 // 3),
                    1,
                )

    # ------------------------------------------------------------------ #
    #  Star glow                                                           #
    # ------------------------------------------------------------------ #

    def _draw_stars_glow(self, bodies: list[CelestialBody]) -> None:
        glow = self._glow_surf
        glow.fill((0, 0, 0, 0))
        for body in bodies:
            if not isinstance(body, Star):
                continue
            cx, cy = self.camera.world_to_screen(body.pos)
            body_px = _clamp_px(body.radius / self.camera.mpp, _STAR_MIN_PX)
            r, g, b = body.color
            # Draw 4 concentric glow layers
            for i in range(4, 0, -1):
                glow_r = body_px * (2 + i)
                alpha  = 30 * i
                pygame.draw.circle(
                    glow, (r, g, b, alpha), (cx, cy), glow_r
                )
        self.screen.blit(glow, (0, 0))

    # ------------------------------------------------------------------ #
    #  Bodies                                                              #
    # ------------------------------------------------------------------ #

    def _draw_bodies(self, bodies: list[CelestialBody]) -> None:
        for body in bodies:
            cx, cy = self.camera.world_to_screen(body.pos)
            # Skip if way off screen
            if cx < -2000 or cx > self.W + 2000 or cy < -2000 or cy > self.H + 2000:
                continue

            if isinstance(body, Asteroid):
                self._draw_asteroid(body, cx, cy)
            elif isinstance(body, Spacecraft):
                self._draw_spacecraft(body, cx, cy)
            elif isinstance(body, Star):
                px = _clamp_px(body.radius / self.camera.mpp, _STAR_MIN_PX)
                pygame.draw.circle(self.screen, body.color, (cx, cy), px)
            else:
                px = _clamp_px(body.radius / self.camera.mpp, _MIN_BODY_PX)
                pygame.draw.circle(self.screen, body.color, (cx, cy), px)
                # Atmosphere glow
                if isinstance(body, Planet) and body.atmosphere_thickness > 0:
                    atm_px = int((body.radius + body.atmosphere_thickness) / self.camera.mpp)
                    if atm_px > px:
                        r, g, b = body.color
                        atm_col = (max(0, r - 30), max(0, g - 30), min(255, b + 40))
                        s = pygame.Surface((atm_px * 2, atm_px * 2), pygame.SRCALPHA)
                        pygame.draw.circle(s, (*atm_col, 60), (atm_px, atm_px), atm_px)
                        self.screen.blit(s, (cx - atm_px, cy - atm_px))

            # Name label
            font = pygame.font.SysFont("Consolas", 11)
            name_surf = font.render(body.name, True, (180, 180, 180))
            body_px = max(_MIN_BODY_PX, int(body.radius / self.camera.mpp))
            self.screen.blit(name_surf, (cx + body_px + 3, cy - 6))

            # Selection ring
            if body.selected:
                sel_r = max(body_px + 6, 12)
                pygame.draw.circle(self.screen, (255, 220, 50), (cx, cy), sel_r, 2)

    def _draw_asteroid(self, body: Asteroid, cx: int, cy: int) -> None:
        scale = 1.0 / self.camera.mpp
        pts_world = body.get_polygon(scale=1.0)
        pts_screen = [(cx + int(ox * scale), cy - int(oy * scale)) for (ox, oy) in pts_world]
        if len(pts_screen) >= 3:
            try:
                pygame.draw.polygon(self.screen, body.color, pts_screen)
                pygame.draw.polygon(self.screen, (200, 200, 200), pts_screen, 1)
            except Exception:
                pass
        else:
            pygame.draw.circle(self.screen, body.color, (cx, cy), _MIN_BODY_PX)

    def _draw_spacecraft(self, body: Spacecraft, cx: int, cy: int) -> None:
        size = max(8, int(body.radius / self.camera.mpp))
        angle = body.angle
        # Triangle nose → two base corners
        nose  = (cx + int(math.cos(angle)           * size * 2),
                 cy - int(math.sin(angle)           * size * 2))
        lw    = (cx + int(math.cos(angle + 2.4)    * size),
                 cy - int(math.sin(angle + 2.4)    * size))
        rw    = (cx + int(math.cos(angle - 2.4)    * size),
                 cy - int(math.sin(angle - 2.4)    * size))
        pygame.draw.polygon(self.screen, body.color, [nose, lw, rw])
        pygame.draw.polygon(self.screen, (200, 255, 200), [nose, lw, rw], 1)

        # Thrust flame
        if body.thrusting and body.fuel > 0:
            tail_dir = Vector2D(math.cos(angle + math.pi), math.sin(angle + math.pi))
            flame_len = size * 3
            flame_tip = (cx + int(tail_dir.x * flame_len),
                         cy - int(tail_dir.y * flame_len))
            pygame.draw.line(self.screen, (255, 160, 40), (cx, cy), flame_tip, 3)
            pygame.draw.line(self.screen, (255, 220, 120), (cx, cy), flame_tip, 1)

    # ------------------------------------------------------------------ #
    #  Velocity arrow for selected body                                    #
    # ------------------------------------------------------------------ #

    def _draw_velocity_arrows(self, bodies: list[CelestialBody]) -> None:
        for body in bodies:
            if not body.selected:
                continue
            cx, cy = self.camera.world_to_screen(body.pos)
            speed_px = body.speed / self.camera.mpp   # raw pixel dist per sim-second
            # Scale so arrows are visible but not enormous
            arrow_len = min(max(20.0, speed_px * 1e3), 120.0)
            if body.speed < 1.0:
                continue
            vel_norm = body.vel.normalise_safe()
            ex = cx + int(vel_norm.x * arrow_len)
            ey = cy - int(vel_norm.y * arrow_len)
            pygame.draw.line(self.screen, (80, 200, 255), (cx, cy), (ex, ey), 2)
            # Arrowhead
            head_angle = math.atan2(ey - cy, ex - cx)
            for side in (-0.4, 0.4):
                hx = ex - int(math.cos(head_angle + side) * 8)
                hy = ey - int(math.sin(head_angle + side) * 8)
                pygame.draw.line(self.screen, (80, 200, 255), (ex, ey), (hx, hy), 2)
