"""
vector2d.py — Immutable 2-D vector for physics calculations.

Uses __slots__ for minimal memory footprint when millions of vectors are created
per simulation step. All operations return new Vector2D instances.
"""
from __future__ import annotations
import math


class Vector2D:
    """Lightweight immutable 2-D vector."""

    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))

    # ------------------------------------------------------------------ #
    #  Prevent mutation — this is intentionally immutable                  #
    # ------------------------------------------------------------------ #
    def __setattr__(self, name, value):
        raise AttributeError("Vector2D is immutable")

    # ------------------------------------------------------------------ #
    #  Arithmetic                                                          #
    # ------------------------------------------------------------------ #
    def __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector2D:
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vector2D:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector2D:
        if scalar == 0.0:
            raise ZeroDivisionError("Vector2D division by zero")
        return Vector2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> Vector2D:
        return Vector2D(-self.x, -self.y)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vector2D):
            return NotImplemented
        return math.isclose(self.x, other.x) and math.isclose(self.y, other.y)

    def __repr__(self) -> str:
        return f"Vector2D({self.x:.6g}, {self.y:.6g})"

    # ------------------------------------------------------------------ #
    #  Geometric operations                                                #
    # ------------------------------------------------------------------ #
    def dot(self, other: Vector2D) -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y

    def cross(self, other: Vector2D) -> float:
        """2-D 'cross product' (scalar z-component of 3-D cross)."""
        return self.x * other.y - self.y * other.x

    def magnitude_sq(self) -> float:
        """Squared magnitude — cheaper when you only need comparisons."""
        return self.x * self.x + self.y * self.y

    def magnitude(self) -> float:
        """Euclidean length."""
        return math.sqrt(self.magnitude_sq())

    def normalise(self) -> Vector2D:
        """Unit vector in same direction. Raises ZeroDivisionError if zero."""
        mag = self.magnitude()
        if mag == 0.0:
            raise ZeroDivisionError("Cannot normalise zero vector")
        return Vector2D(self.x / mag, self.y / mag)

    def normalise_safe(self, fallback: Vector2D | None = None) -> Vector2D:
        """Normalise, returning fallback (defaults to zero vector) if magnitude is zero."""
        mag = self.magnitude()
        if mag == 0.0:
            return fallback if fallback is not None else Vector2D(0.0, 0.0)
        return Vector2D(self.x / mag, self.y / mag)

    def distance_to(self, other: Vector2D) -> float:
        """Euclidean distance."""
        return (self - other).magnitude()

    def distance_sq_to(self, other: Vector2D) -> float:
        """Squared distance — avoids sqrt when only needed for comparison."""
        return (self - other).magnitude_sq()

    def rotate(self, angle_rad: float) -> Vector2D:
        """Rotate by angle_rad counter-clockwise."""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        return Vector2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a,
        )

    def angle(self) -> float:
        """Angle this vector makes with the positive x-axis (radians, −π to π)."""
        return math.atan2(self.y, self.x)

    def angle_to(self, other: Vector2D) -> float:
        """Signed angle from this vector to other (radians)."""
        return math.atan2(self.cross(other), self.dot(other))

    def perpendicular(self) -> Vector2D:
        """Counter-clockwise perpendicular."""
        return Vector2D(-self.y, self.x)

    def lerp(self, other: Vector2D, t: float) -> Vector2D:
        """Linear interpolation between self and other at parameter t ∈ [0,1]."""
        return Vector2D(self.x + (other.x - self.x) * t, self.y + (other.y - self.y) * t)

    # ------------------------------------------------------------------ #
    #  Conversion helpers                                                  #
    # ------------------------------------------------------------------ #
    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def to_int_tuple(self) -> tuple[int, int]:
        return (int(self.x), int(self.y))

    @classmethod
    def from_polar(cls, r: float, theta: float) -> "Vector2D":
        """Create from polar coordinates (radius, angle in radians)."""
        return cls(r * math.cos(theta), r * math.sin(theta))

    @classmethod
    def zero(cls) -> "Vector2D":
        return cls(0.0, 0.0)
