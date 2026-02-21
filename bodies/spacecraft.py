"""
spacecraft.py — User-controlled probe / spacecraft.

The spacecraft can fire its thruster, consuming fuel, which applies an
additional force vector that is injected into the RK4 integrator each step.
"""
from __future__ import annotations
import math

from physics.vector2d import Vector2D
from bodies.body import CelestialBody


class Spacecraft(CelestialBody):
    """
    A thruster-equipped spacecraft.

    Parameters
    ----------
    thrust_force  : float   Maximum thrust in Newtons.
    fuel          : float   Current fuel level (0–1 normalised).
    fuel_rate     : float   Fuel consumed per second at full thrust.
    angle         : float   Nose direction in radians (0 = right, π/2 = up).
    """

    def __init__(
        self,
        name: str,
        mass: float,
        radius: float,
        pos: Vector2D,
        vel: Vector2D,
        color: tuple[int, int, int] = (80, 200, 120),
        thrust_force: float = 1e6,      # N — tiny compared to planetary masses
        fuel: float = 1.0,
        fuel_rate: float = 0.01,        # fraction per second
        angle: float = 0.0,
        trail_max: int = 300,
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
            body_type="spacecraft",
        )
        self.thrust_force: float = float(thrust_force)
        self.fuel: float = max(0.0, min(1.0, float(fuel)))
        self.fuel_rate: float = float(fuel_rate)
        self.angle: float = float(angle)          # radians
        self.thrusting: bool = False
        self.rotating_cw: bool = False
        self.rotating_ccw: bool = False
        self.rotation_speed: float = math.radians(90)  # rad/s

    # ------------------------------------------------------------------ #
    #  Thruster control (called by the event loop each frame)              #
    # ------------------------------------------------------------------ #

    def apply_thrust(self, dt: float) -> None:
        """
        Fire thruster for duration dt (seconds).
        Adds force to the body's extra-force accumulator (consumed by integrator).
        """
        if self.fuel <= 0.0:
            self.thrusting = False
            return
        direction = Vector2D(math.cos(self.angle), math.sin(self.angle))
        force = direction * self.thrust_force
        self.add_extra_force(force)
        self.fuel = max(0.0, self.fuel - self.fuel_rate * dt)
        self.thrusting = True

    def rotate(self, dt: float, clockwise: bool = False) -> None:
        """Rotate nose direction."""
        delta = self.rotation_speed * dt
        self.angle += -delta if clockwise else delta

    def update(self, dt: float) -> None:
        """
        Called each simulation step (before RK4).
        Accumulates thrust force if thrusting flag is set.
        """
        if self.thrusting:
            self.apply_thrust(dt)
        if self.rotating_cw:
            self.rotate(dt, clockwise=True)
        if self.rotating_ccw:
            self.rotate(dt, clockwise=False)

    @property
    def thrust_direction(self) -> Vector2D:
        return Vector2D(math.cos(self.angle), math.sin(self.angle))

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "thrust_force": self.thrust_force,
            "fuel": self.fuel,
            "fuel_rate": self.fuel_rate,
            "angle": self.angle,
        })
        return d
