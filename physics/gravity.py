"""
gravity.py — Gravitational force calculations with softening.

The softening length ε prevents singularities when two bodies occupy the
same (or very close) positions, which occurs near collisions and in the
first frame when objects are placed directly on top of each other.

  F = G * m_a * m_b / (r² + ε²)  *  r̂

SI units throughout: kg, metres, seconds, Newtons.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

from .vector2d import Vector2D

if TYPE_CHECKING:
    from bodies.body import CelestialBody

# Gravitational constant (SI)
G_SI: float = 6.674_30e-11  # m³ kg⁻¹ s⁻²


def gravitational_force(
    body_a: "CelestialBody",
    body_b: "CelestialBody",
    G: float = G_SI,
    softening: float = 1e6,           # metres — adjust per simulation scale
) -> Vector2D:
    """
    Return the gravitational force vector exerted on body_a by body_b.

    Parameters
    ----------
    body_a, body_b : CelestialBody
        Bodies with .mass (kg) and .pos (Vector2D, metres).
    G : float
        Gravitational constant (default SI value).
    softening : float
        Softening length ε in metres. Prevents division by zero.

    Returns
    -------
    Vector2D
        Force in Newtons acting on body_a (towards body_b).
    """
    delta = body_b.pos - body_a.pos          # vector from a → b
    dist_sq = delta.magnitude_sq() + softening * softening
    dist = dist_sq ** 0.5                    # √(r² + ε²)
    magnitude = G * body_a.mass * body_b.mass / dist_sq
    # direction unit vector × magnitude
    return delta * (magnitude / dist)        # == delta.normalise() * magnitude without extra sqrt


def total_acceleration(
    body: "CelestialBody",
    all_bodies: list["CelestialBody"],
    G: float = G_SI,
    softening: float = 1e6,
) -> Vector2D:
    """
    Sum of gravitational accelerations on *body* from all other bodies.

    Returns acceleration (m s⁻²), not force, so mass is divided out.
    """
    acc = Vector2D.zero()
    for other in all_bodies:
        if other is body:
            continue
        if body.mass <= 0:
            continue
        delta = other.pos - body.pos
        dist_sq = delta.magnitude_sq() + softening * softening
        dist = dist_sq ** 0.5
        # a = G * M_other / (r² + ε²)  in direction of delta
        a_mag = G * other.mass / dist_sq
        acc = acc + delta * (a_mag / dist)
    return acc


def orbital_velocity(
    central_mass: float,
    distance: float,
    G: float = G_SI,
) -> float:
    """
    Circular orbital speed v = √(G·M / r).

    Parameters
    ----------
    central_mass : float
        Mass of the central body (kg).
    distance : float
        Orbital radius (m).
    G : float
        Gravitational constant.

    Returns
    -------
    float
        Speed in m/s for a circular orbit at the given distance.
    """
    if distance <= 0 or central_mass <= 0:
        return 0.0
    return (G * central_mass / distance) ** 0.5


def escape_velocity(
    central_mass: float,
    distance: float,
    G: float = G_SI,
) -> float:
    """Escape velocity at distance r from central_mass: v_esc = √(2·G·M / r)."""
    if distance <= 0 or central_mass <= 0:
        return 0.0
    return (2 * G * central_mass / distance) ** 0.5


def total_kinetic_energy(bodies: list["CelestialBody"]) -> float:
    """½ Σ m v² — total kinetic energy of the system (Joules)."""
    return sum(0.5 * b.mass * b.vel.magnitude_sq() for b in bodies)


def total_potential_energy(
    bodies: list["CelestialBody"],
    G: float = G_SI,
    softening: float = 1e6,
) -> float:
    """
    −G Σ_{i<j} m_i m_j / √(r_ij² + ε²) — total gravitational potential energy.
    """
    pe = 0.0
    n = len(bodies)
    for i in range(n):
        for j in range(i + 1, n):
            delta = bodies[j].pos - bodies[i].pos
            dist = (delta.magnitude_sq() + softening * softening) ** 0.5
            pe -= G * bodies[i].mass * bodies[j].mass / dist
    return pe


def total_mechanical_energy(
    bodies: list["CelestialBody"],
    G: float = G_SI,
    softening: float = 1e6,
) -> float:
    """KE + PE — should be (nearly) conserved over simulation time."""
    return total_kinetic_energy(bodies) + total_potential_energy(bodies, G, softening)


def total_linear_momentum(bodies: list["CelestialBody"]) -> Vector2D:
    """Σ m·v — total linear momentum (kg·m/s). Should be conserved."""
    p = Vector2D.zero()
    for b in bodies:
        p = p + b.vel * b.mass
    return p
