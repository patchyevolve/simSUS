"""
moon.py — Moon class (semantically a child of a Planet).
"""
from __future__ import annotations

from physics.vector2d import Vector2D
from bodies.body import CelestialBody


class Moon(CelestialBody):
    """
    A natural satellite orbiting a planet.

    Parameters
    ----------
    parent_name : str   Name of the planet this moon orbits (informational).
    """

    def __init__(
        self,
        name: str,
        mass: float,
        radius: float,
        pos: Vector2D,
        vel: Vector2D,
        color: tuple[int, int, int] = (160, 160, 160),
        parent_name: str = "",
        trail_max: int = 400,
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
            body_type="moon",
        )
        self.parent_name: str = parent_name

    def to_dict(self) -> dict:
        d = super().to_dict()
        d["parent_name"] = self.parent_name
        return d
