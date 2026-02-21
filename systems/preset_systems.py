"""
preset_systems.py — Real Solar System and other built-in presets.

All data in SI units (kg, metres, m/s).
Positions are set at perihelion/aphelion for simplicity; orbital velocity is
calculated from the vis-viva equation or circular approximation.

Sources:
  NASA Planetary Fact Sheet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/
"""
from __future__ import annotations
import math

from physics.vector2d import Vector2D
from physics.gravity import G_SI, orbital_velocity
from bodies.star   import Star
from bodies.planet import Planet
from bodies.moon   import Moon
from bodies.asteroid import Asteroid


AU  = 1.495_978_707e11   # metres per Astronomical Unit
DAY = 86_400.0           # seconds per day


# ──────────────────────────────────────────────────────────────────────────────
#  Helper: place body in circular orbit around central body
# ──────────────────────────────────────────────────────────────────────────────

def _circular_orbit(
    central_mass: float,
    semi_major_axis: float,        # metres
    inclination_angle: float = 0.0,  # angle in xy-plane (offset from +x axis)
    clockwise: bool = False,
) -> tuple[Vector2D, Vector2D]:
    """
    Return (pos, vel) for a circular orbit.
    Body placed at semi_major_axis from origin along the inclination_angle direction.
    Velocity is perpendicular to radius, magnitude = √(G·M / r).
    """
    a = semi_major_axis
    v = orbital_velocity(central_mass, a, G_SI)
    pos = Vector2D.from_polar(a, inclination_angle)
    # Perpendicular direction (counter-clockwise)
    vel_angle = inclination_angle + (math.pi / 2 if not clockwise else -math.pi / 2)
    vel = Vector2D.from_polar(v, vel_angle)
    return pos, vel


# ──────────────────────────────────────────────────────────────────────────────
#  The Solar System
# ──────────────────────────────────────────────────────────────────────────────

def build_solar_system() -> list:
    """
    Assemble the 8-planet Solar System (+ Earth's Moon, Pluto, Halley's comet).
    Returns a flat list of CelestialBody objects.

    All bodies orbit counter-clockwise when viewed from north ecliptic pole.
    """

    # ── Sun ─────────────────────────────────────────────────────────────────
    sun = Star(
        name="Sun",
        mass=1.989e30,
        radius=6.957e8,
        pos=Vector2D.zero(),
        vel=Vector2D.zero(),
        temperature=5778.0,
        fixed=True,             # Pin the Sun at origin
    )

    bodies: list = [sun]

    # ── Planet data ──────────────────────────────────────────────────────────
    #  (name, mass_kg, radius_m, semi_major_axis_m, inclination_offset_deg,
    #   planet_type, atmosphere_thickness, has_rings, ring_inner, ring_outer)
    planet_data = [
        ("Mercury",  3.301e23, 2.439e6,  0.387 * AU,   0,  "rocky",     0,         False, 0, 0),
        ("Venus",    4.867e24, 6.051e6,  0.723 * AU,  45,  "lava",      5e4,       False, 0, 0),
        ("Earth",    5.972e24, 6.371e6,  1.000 * AU,  90,  "ocean",     1e4,       False, 0, 0),
        ("Mars",     6.417e23, 3.389e6,  1.524 * AU, 135,  "desert",    1e3,       False, 0, 0),
        ("Jupiter",  1.898e27, 6.991e7,  5.203 * AU, 180,  "gas_giant", 0,         False, 0, 0),
        ("Saturn",   5.683e26, 5.823e7,  9.537 * AU, 225,  "gas_giant", 0,         True, 7.2e7, 1.4e8),
        ("Uranus",   8.681e25, 2.536e7, 19.191 * AU, 270,  "ice_giant", 0,         True, 4.1e7, 5.1e7),
        ("Neptune",  1.024e26, 2.462e7, 30.069 * AU, 315,  "ice_giant", 0,         False, 0, 0),
    ]

    for (name, mass, radius, sma, offset_deg, ptype, atm, rings, ri, ro) in planet_data:
        offset_rad = math.radians(offset_deg)
        pos, vel = _circular_orbit(sun.mass, sma, offset_rad)
        atm_km = float(atm)
        planet = Planet(
            name=name, mass=mass, radius=radius, pos=pos, vel=vel,
            planet_type=ptype,
            atmosphere_thickness=atm_km,
            has_rings=rings,
            ring_inner=float(ri),
            ring_outer=float(ro),
        )
        bodies.append(planet)

    # ── Earth's Moon ─────────────────────────────────────────────────────────
    earth = next(b for b in bodies if b.name == "Earth")
    moon_sma = 3.844e8  # metres
    moon_v = orbital_velocity(earth.mass, moon_sma, G_SI)
    # Place moon relative to Earth
    moon_pos = earth.pos + Vector2D(moon_sma, 0)
    moon_vel = earth.vel + Vector2D(0, moon_v)
    moon = Moon(
        name="Moon", mass=7.342e22, radius=1.737e6,
        pos=moon_pos, vel=moon_vel,
        color=(200, 200, 195),
        parent_name="Earth",
    )
    bodies.append(moon)

    # ── Pluto ────────────────────────────────────────────────────────────────
    pluto_sma = 39.482 * AU
    pluto_pos, pluto_vel = _circular_orbit(sun.mass, pluto_sma, math.radians(20))
    pluto = Planet(
        name="Pluto", mass=1.303e22, radius=1.188e6,
        pos=pluto_pos, vel=pluto_vel,
        planet_type="frozen",
    )
    bodies.append(pluto)

    # ── Halley's Comet (approximate) ─────────────────────────────────────────
    # Highly eccentric orbit — approximate perihelion position with perihelion vel
    # Perihelion: 0.586 AU, aphelion: 35.08 AU
    halley_peri = 0.586 * AU
    halley_apo  = 35.08 * AU
    halley_a    = (halley_peri + halley_apo) / 2
    halley_v_peri = math.sqrt(G_SI * sun.mass * (2 / halley_peri - 1 / halley_a))
    halley_pos = Vector2D(halley_peri, 0)
    halley_vel = Vector2D(0, -halley_v_peri)   # retrograde orbit
    halley = Asteroid(
        name="Halley's Comet",
        mass=2.2e14,
        radius=5.5e3,   # ~11 km diameter
        pos=halley_pos,
        vel=halley_vel,
        color=(180, 210, 230),
        resource_type="Ice",
        is_comet=True,
    )
    bodies.append(halley)

    # ── Asteroid Belt sample (5 representative bodies) ───────────────────────
    import random
    rng = random.Random(42)  # deterministic seed
    for i in range(5):
        sma = rng.uniform(2.2, 3.2) * AU
        angle = rng.uniform(0, 2 * math.pi)
        pos, vel = _circular_orbit(sun.mass, sma, angle)
        asteroid = Asteroid(
            name=f"Asteroid-{i+1}",
            mass=rng.uniform(1e15, 1e20),
            radius=rng.uniform(1e4, 5e5),
            pos=pos, vel=vel,
            spin=rng.uniform(-1e-4, 1e-4),
        )
        bodies.append(asteroid)

    return bodies


# ──────────────────────────────────────────────────────────────────────────────
#  Binary Star System
# ──────────────────────────────────────────────────────────────────────────────

def build_binary_star() -> list:
    """Two Sun-like stars orbiting their common barycentre, with a planet."""
    m_star = 1.989e30 * 0.9
    separation = 2.0 * AU
    v_star = orbital_velocity(m_star, separation / 2, G_SI)

    star_a = Star("Alpha", m_star, 6.5e8, Vector2D(-separation / 2, 0), Vector2D(0, -v_star),
                  temperature=6000, fixed=False)
    star_b = Star("Beta",  m_star, 6.5e8, Vector2D(+separation / 2, 0), Vector2D(0, +v_star),
                  temperature=5500, fixed=False)

    # A planet in a circumbinary orbit at 5 AU
    planet_sma = 5.0 * AU
    # Total mass for orbital velocity calculation
    total_mass = 2 * m_star
    p_pos, p_vel = _circular_orbit(total_mass, planet_sma, math.radians(45))
    planet = Planet("Tatooine", 5e24, 6e6, p_pos, p_vel, planet_type="desert")

    return [star_a, star_b, planet]


# ──────────────────────────────────────────────────────────────────────────────
#  Figure-8 Three Body
# ──────────────────────────────────────────────────────────────────────────────

def build_figure_eight() -> list:
    """
    Chenciner-Montgomery (2000) figure-8 choreography.
    Three equal masses chasing each other in a stable figure-8 orbit.
    Dimensionless units scaled to AU and years.
    """
    # Reference: Chenciner & Montgomery (2000). Scaled with G=1, m=1, to SI.
    # Scale factor so the figure-8 fits inside the inner solar system
    m = 1.989e30            # solar mass each
    scale_len = 0.5 * AU   # half-AU scale
    # Period for this orbit ≈ 2π√(scale_len³/(G·m)) × factor
    # Initial conditions (dimensionless, from original paper, scaled)
    x1 = Vector2D( 0.97000436 * scale_len, -0.24308753 * scale_len)
    x2 = Vector2D(-0.97000436 * scale_len,  0.24308753 * scale_len)
    x3 = Vector2D.zero()

    # Velocities (dimensionless rescaled)
    v_scale = math.sqrt(G_SI * m / scale_len)
    vx = 0.93240737 / 2 * v_scale
    vy = 0.86473146 / 2 * v_scale
    v3 = Vector2D(vx, vy)
    v1 = v2 = Vector2D(-vx / 2, -vy / 2)

    b1 = Star("Body-1", m, 4e8, x1, v1, temperature=6500, fixed=False)
    b2 = Star("Body-2", m, 4e8, x2, v2, temperature=5000, fixed=False)
    b3 = Star("Body-3", m, 4e8, x3, v3, temperature=4000, fixed=False)
    return [b1, b2, b3]


# ──────────────────────────────────────────────────────────────────────────────
#  Registry
# ──────────────────────────────────────────────────────────────────────────────

PRESETS: dict[str, callable] = {
    "solar_system": build_solar_system,
    "binary_star":  build_binary_star,
    "figure_eight": build_figure_eight,
}
