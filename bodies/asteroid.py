"""
asteroid.py — Asteroid and Comet classes.
"""
from __future__ import annotations
import random
import math

from physics.vector2d import Vector2D
from bodies.body import CelestialBody

RESOURCE_TYPES = ["Iron", "Nickel", "Silicate", "Carbon", "Ice", "Gold", "Platinum"]


class Asteroid(CelestialBody):
    """
    A small rocky/icy body. Carries a polygon shape for rendering and
    an optional resource type for game-play integrations.

    Parameters
    ----------
    resource_type : str   One of RESOURCE_TYPES.
    spin          : float Angular rotation speed (rad/s) — cosmetic only.
    """

    def __init__(
        self,
        name: str,
        mass: float,
        radius: float,
        pos: Vector2D,
        vel: Vector2D,
        color: tuple[int, int, int] = (130, 120, 110),
        resource_type: str | None = None,
        spin: float = 0.0,
        trail_max: int = 200,
        is_comet: bool = False,
    ) -> None:
        super().__init__(
            name=name,
            mass=mass,
            radius=radius,
            pos=pos,
            vel=vel,
            color=color,
            fixed=False,
            trail_max=trail_max,
            body_type="comet" if is_comet else "asteroid",
        )
        self.resource_type: str = resource_type or random.choice(RESOURCE_TYPES)
        self.spin: float = spin                 # rad/s
        self.angle: float = random.uniform(0, 2 * math.pi)  # current rotation angle
        self.is_comet: bool = is_comet

        # Pre-build polygon vertex offsets for this asteroid instance
        n_verts = random.randint(6, 10)
        self._shape_offsets: list[tuple[float, float]] = []
        for i in range(n_verts):
            theta = 2 * math.pi * i / n_verts
            r = radius * random.uniform(0.7, 1.3)
            self._shape_offsets.append((r * math.cos(theta), r * math.sin(theta)))

    def update_spin(self, dt: float) -> None:
        self.angle += self.spin * dt

    def get_polygon(self, scale: float = 1.0) -> list[tuple[float, float]]:
        """Return rotated polygon vertices in world-space offsets (metres) around pos."""
        cos_a = math.cos(self.angle)
        sin_a = math.sin(self.angle)
        result = []
        for ox, oy in self._shape_offsets:
            rx = ox * cos_a - oy * sin_a
            ry = ox * sin_a + oy * cos_a
            result.append((rx * scale, ry * scale))
        return result

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "resource_type": self.resource_type,
            "spin": self.spin,
            "is_comet": self.is_comet,
        })
        return d
