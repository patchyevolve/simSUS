"""
planet.py — Planet class with atmosphere, rings, surface gravity, and ring system.
"""
from __future__ import annotations

from physics.vector2d import Vector2D
from bodies.body import CelestialBody


# Broad planet type palette
PLANET_COLORS = {
    "rocky":     (160, 130, 100),
    "ocean":     ( 50, 120, 200),
    "gas_giant": (200, 160,  80),
    "ice_giant": ( 80, 180, 220),
    "lava":      (220,  80,  40),
    "desert":    (210, 180, 100),
    "jungle":    ( 60, 160,  70),
    "frozen":    (200, 220, 240),
    "exotic":    (180,  60, 200),
}


class Planet(CelestialBody):
    """
    A planetary body orbiting a star (or another massive body).

    Parameters
    ----------
    planet_type     : str   One of the keys in PLANET_COLORS.
    atmosphere_thickness : float  Render-only, metres (0 = no atmosphere).
    has_rings       : bool  Draw a ring system in renderer.
    ring_inner      : float Ring inner radius (metres from planet centre).
    ring_outer      : float Ring outer radius.
    ring_color      : RGB
    moons           : list  Child Moon objects (populated later).
    """

    def __init__(
        self,
        name: str,
        mass: float,
        radius: float,
        pos: Vector2D,
        vel: Vector2D,
        planet_type: str = "rocky",
        atmosphere_thickness: float = 0.0,
        has_rings: bool = False,
        ring_inner: float = 0.0,
        ring_outer: float = 0.0,
        ring_color: tuple[int, int, int] = (180, 160, 120),
        trail_max: int = 600,
    ) -> None:
        color = PLANET_COLORS.get(planet_type, (180, 180, 180))
        super().__init__(
            name=name,
            mass=mass,
            radius=radius,
            pos=pos,
            vel=vel,
            color=color,
            fixed=False,
            trail_max=trail_max,
            body_type="planet",
        )
        self.planet_type: str = planet_type
        self.atmosphere_thickness: float = float(atmosphere_thickness)
        self.has_rings: bool = has_rings
        self.ring_inner: float = float(ring_inner)
        self.ring_outer: float = float(ring_outer)
        self.ring_color: tuple[int, int, int] = ring_color
        self.moons: list = []   # list[Moon]

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "planet_type": self.planet_type,
            "atmosphere_thickness": self.atmosphere_thickness,
            "has_rings": self.has_rings,
            "ring_inner": self.ring_inner,
            "ring_outer": self.ring_outer,
            "ring_color": list(self.ring_color),
        })
        return d
