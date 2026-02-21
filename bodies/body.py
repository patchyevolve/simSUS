"""
body.py — CelestialBody base class.

Every object in the simulation (star, planet, moon, asteroid, spacecraft)
inherits from CelestialBody. It stores the state vectors that the integrator
updates each step, plus metadata for rendering and UI.
"""
from __future__ import annotations
import math
from collections import deque
from typing import Optional

from physics.vector2d import Vector2D


class CelestialBody:
    """
    Base class for all simulated objects.

    Attributes
    ----------
    name        : str           Unique identifier / display name
    mass        : float         kg
    radius      : float         metres (used for collision detection + render scale)
    pos         : Vector2D      metres, in simulation (world) coordinates
    vel         : Vector2D      m/s
    color       : tuple[int,int,int]   RGB for renderer
    fixed       : bool          If True the integrator skips this body (pinned star)
    trail       : deque         Rolling history of positions for orbit trail rendering
    trail_max   : int           Maximum trail length in steps
    body_type   : str           "star" | "planet" | "moon" | "asteroid" | "spacecraft"
    selected    : bool          True when the user has clicked this body
    """

    # Allow per-instance attribute assignment (no __slots__ so subclasses stay flexible)
    def __init__(
        self,
        name: str,
        mass: float,
        radius: float,
        pos: Vector2D,
        vel: Vector2D,
        color: tuple[int, int, int] = (200, 200, 200),
        fixed: bool = False,
        trail_max: int = 800,
        body_type: str = "body",
    ) -> None:
        if mass <= 0:
            raise ValueError(f"Body '{name}': mass must be positive, got {mass}")
        if radius <= 0:
            raise ValueError(f"Body '{name}': radius must be positive, got {radius}")

        self.name: str = name
        self.mass: float = float(mass)
        self.radius: float = float(radius)
        self.pos: Vector2D = pos
        self.vel: Vector2D = vel
        self.color: tuple[int, int, int] = color
        self.fixed: bool = fixed
        self.trail: deque[Vector2D] = deque(maxlen=trail_max)
        self.trail_max: int = trail_max
        self.body_type: str = body_type
        self.selected: bool = False

        # Optional per-body external force (e.g. spacecraft thrust) added each step
        self._extra_force: Vector2D = Vector2D.zero()

    # ------------------------------------------------------------------ #
    #  State management                                                    #
    # ------------------------------------------------------------------ #

    def record_trail(self) -> None:
        """Append current position to the orbit trail."""
        self.trail.append(self.pos)

    def clear_trail(self) -> None:
        self.trail.clear()

    def add_extra_force(self, force: Vector2D) -> None:
        """Accumulate an extra force (e.g. thrust) to apply next RK stage."""
        self._extra_force = self._extra_force + force

    def consume_extra_force(self) -> Vector2D:
        """Return and reset the accumulated extra force."""
        f = self._extra_force
        self._extra_force = Vector2D.zero()
        return f

    # ------------------------------------------------------------------ #
    #  Derived physical quantities                                         #
    # ------------------------------------------------------------------ #

    @property
    def speed(self) -> float:
        """Scalar speed in m/s."""
        return self.vel.magnitude()

    @property
    def kinetic_energy(self) -> float:
        """½ m v² (Joules)."""
        return 0.5 * self.mass * self.vel.magnitude_sq()

    def surface_gravity(self) -> float:
        """Gravitational acceleration at the body surface: g = G·M / R² (m/s²)."""
        from physics.gravity import G_SI
        return G_SI * self.mass / (self.radius ** 2)

    def gravitational_parameter(self) -> float:
        """Standard gravitational parameter μ = G·M (m³/s²)."""
        from physics.gravity import G_SI
        return G_SI * self.mass

    def distance_to(self, other: "CelestialBody") -> float:
        """Centre-to-centre distance (metres)."""
        return self.pos.distance_to(other.pos)

    def is_colliding_with(self, other: "CelestialBody") -> bool:
        """True when the surfaces overlap (centre distance < sum of radii)."""
        return self.pos.distance_sq_to(other.pos) < (self.radius + other.radius) ** 2

    def orbital_period_around(self, central: "CelestialBody") -> Optional[float]:
        """
        Estimate Keplerian orbital period (seconds) around central body,
        assuming a nearly circular orbit at the current distance.
        T = 2π √(a³ / G·M)
        """
        from physics.gravity import G_SI
        a = self.distance_to(central)
        if a <= 0 or central.mass <= 0:
            return None
        return 2 * math.pi * math.sqrt(a ** 3 / (G_SI * central.mass))

    def merge_with(self, other: "CelestialBody") -> "CelestialBody":
        """
        Perfectly inelastic collision: merge two bodies into one.
        Conserves mass and momentum. The larger body keeps its type / name.
        """
        total_mass = self.mass + other.mass
        # Centre of mass position
        new_pos = (self.pos * self.mass + other.pos * other.mass) / total_mass
        # Momentum conservation
        new_vel = (self.vel * self.mass + other.vel * other.mass) / total_mass
        # Volume-equivalent radius: r_new = (r₁³ + r₂³)^(1/3)
        new_radius = (self.radius ** 3 + other.radius ** 3) ** (1 / 3)

        dominant = self if self.mass >= other.mass else other
        merged = CelestialBody(
            name=dominant.name,
            mass=total_mass,
            radius=new_radius,
            pos=new_pos,
            vel=new_vel,
            color=dominant.color,
            fixed=dominant.fixed,
            trail_max=dominant.trail_max,
            body_type=dominant.body_type,
        )
        return merged

    # ------------------------------------------------------------------ #
    #  Serialisation                                                       #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mass": self.mass,
            "radius": self.radius,
            "pos_x": self.pos.x,
            "pos_y": self.pos.y,
            "vel_x": self.vel.x,
            "vel_y": self.vel.y,
            "color": self.color,
            "fixed": self.fixed,
            "body_type": self.body_type,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CelestialBody":
        return cls(
            name=d["name"],
            mass=d["mass"],
            radius=d["radius"],
            pos=Vector2D(d["pos_x"], d["pos_y"]),
            vel=Vector2D(d["vel_x"], d["vel_y"]),
            color=tuple(d["color"]),
            fixed=d.get("fixed", False),
            body_type=d.get("body_type", "body"),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}('{self.name}', "
            f"mass={self.mass:.3e} kg, r={self.radius:.3e} m, "
            f"pos=({self.pos.x:.3e},{self.pos.y:.3e}), "
            f"vel=({self.vel.x:.3e},{self.vel.y:.3e}))"
        )
