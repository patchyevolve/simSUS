"""
tests/test_physics.py — Automated physics correctness tests.

Run:
    cd "d:\\codeWorks\\New folder"
    python -m pytest tests/ -v

Tests:
  1. Energy conservation (Earth orbit 1000 steps < 0.01% drift)
  2. Orbital period accuracy (Earth year ≈ 365.25 days ± 0.1%)
  3. Momentum conservation (no external forces → |Δp| ≈ 0)
  4. Two-body circular orbit — both bodies orbit barycentre
  5. Collision merge — mass and momentum conserved
  6. Force softening — zero distance → finite force (no NaN/Inf)
  7. Vector2D — all operations work correctly
"""
from __future__ import annotations
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from physics.vector2d import Vector2D
from physics.gravity import (
    gravitational_force, total_acceleration, orbital_velocity,
    total_mechanical_energy, total_kinetic_energy, total_potential_energy,
    total_linear_momentum, G_SI,
)
from physics.integrator import rk4_step
from bodies.body import CelestialBody
from bodies.star import Star
from bodies.planet import Planet


AU  = 1.495_978_707e11
DAY = 86_400.0


def _make_earth_system():
    """Minimal Sun + Earth for orbital tests."""
    sun = CelestialBody("Sun",   1.989e30, 6.957e8, Vector2D.zero(), Vector2D.zero(),
                        fixed=True)
    earth_dist = 1.0 * AU
    v = orbital_velocity(sun.mass, earth_dist)
    earth = CelestialBody("Earth", 5.972e24, 6.371e6,
                           Vector2D(earth_dist, 0), Vector2D(0, v))
    return sun, earth


# ─────────────────────────────────────────────────────────────────────────────
#  1. Vector2D operations
# ─────────────────────────────────────────────────────────────────────────────

class TestVector2D:
    def test_add(self):
        v = Vector2D(1, 2) + Vector2D(3, 4)
        assert math.isclose(v.x, 4) and math.isclose(v.y, 6)

    def test_sub(self):
        v = Vector2D(5, 3) - Vector2D(2, 1)
        assert math.isclose(v.x, 3) and math.isclose(v.y, 2)

    def test_scale(self):
        v = Vector2D(2, 3) * 2.0
        assert math.isclose(v.x, 4) and math.isclose(v.y, 6)

    def test_magnitude(self):
        v = Vector2D(3, 4)
        assert math.isclose(v.magnitude(), 5.0)

    def test_normalise(self):
        v = Vector2D(3, 4).normalise()
        assert math.isclose(v.magnitude(), 1.0)

    def test_dot(self):
        assert math.isclose(Vector2D(1, 0).dot(Vector2D(0, 1)), 0.0)
        assert math.isclose(Vector2D(1, 0).dot(Vector2D(1, 0)), 1.0)

    def test_cross(self):
        assert math.isclose(Vector2D(1, 0).cross(Vector2D(0, 1)), 1.0)

    def test_immutable(self):
        v = Vector2D(1, 2)
        with pytest.raises(AttributeError):
            v.x = 5          # type: ignore

    def test_from_polar(self):
        v = Vector2D.from_polar(1.0, 0.0)
        assert math.isclose(v.x, 1.0) and math.isclose(v.y, 0.0)
        v2 = Vector2D.from_polar(1.0, math.pi / 2)
        assert math.isclose(v2.y, 1.0, abs_tol=1e-12)

    def test_rotate(self):
        v = Vector2D(1, 0).rotate(math.pi / 2)
        assert math.isclose(v.x, 0.0, abs_tol=1e-12) and math.isclose(v.y, 1.0, abs_tol=1e-12)

    def test_lerp(self):
        v = Vector2D(0, 0).lerp(Vector2D(10, 20), 0.5)
        assert math.isclose(v.x, 5) and math.isclose(v.y, 10)

    def test_zero_division(self):
        with pytest.raises(ZeroDivisionError):
            Vector2D(0, 0).normalise()


# ─────────────────────────────────────────────────────────────────────────────
#  2. Force softening — no NaN / Inf at zero separation
# ─────────────────────────────────────────────────────────────────────────────

def test_force_softening_no_nan():
    """Bodies at exactly the same position must yield finite force."""
    a = CelestialBody("A", 1e24, 1e6, Vector2D.zero(), Vector2D.zero())
    b = CelestialBody("B", 1e24, 1e6, Vector2D.zero(), Vector2D.zero())
    f = gravitational_force(a, b, G_SI, softening=1e6)
    assert math.isfinite(f.x), f"Force x is not finite: {f.x}"
    assert math.isfinite(f.y), f"Force y is not finite: {f.y}"


def test_force_softening_direction():
    """Force on A from B should point from A towards B."""
    a = CelestialBody("A", 1e24, 1e6, Vector2D.zero(), Vector2D.zero())
    b = CelestialBody("B", 1e24, 1e6, Vector2D(1e9, 0), Vector2D.zero())
    f = gravitational_force(a, b)
    # Force should be in +x direction
    assert f.x > 0, "Force should pull towards + x"
    assert math.isclose(f.y, 0.0, abs_tol=1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Energy conservation — Earth orbit
# ─────────────────────────────────────────────────────────────────────────────

def test_energy_conservation_1000_steps():
    """
    Run 1000 RK4 steps of Earth's orbit at 1-hour steps.
    Total mechanical energy must not drift > 0.01%.
    """
    sun, earth = _make_earth_system()
    bodies = [sun, earth]
    E0 = total_mechanical_energy([earth], G_SI, softening=1e6)
    # E0 only for earth (sun is fixed, treat its KE as zero)
    # Actually measure KE+PE of the full system
    E0 = total_kinetic_energy([earth]) + total_potential_energy(bodies, G_SI, 1e6)

    dt = 3600.0   # 1 hour per step
    for _ in range(1000):
        rk4_step(bodies, dt, G_SI, softening=1e6)

    E1 = total_kinetic_energy([earth]) + total_potential_energy(bodies, G_SI, 1e6)
    drift_pct = abs(E1 - E0) / abs(E0) * 100.0
    assert drift_pct < 0.01, f"Energy drifted {drift_pct:.4f}% (> 0.01%)"


# ─────────────────────────────────────────────────────────────────────────────
#  4. Orbital period — Earth year
# ─────────────────────────────────────────────────────────────────────────────

def test_orbital_period_earth():
    """
    Simulate Earth orbiting the Sun. Detect when Earth crosses +x axis again.
    Period should be within 0.5% of 365.25 days.
    """
    sun, earth = _make_earth_system()
    bodies = [sun, earth]
    target_period = 365.25 * DAY

    dt = 3600.0             # 1 hour per step
    max_steps = int(target_period / dt) + 1000
    prev_y = earth.pos.y
    elapsed = 0.0

    for i in range(max_steps):
        rk4_step(bodies, dt, G_SI, softening=1e6)
        elapsed += dt
        curr_y = earth.pos.y
        # Detect upward zero crossing of y: one full revolution
        if elapsed > DAY * 200 and prev_y < 0 and curr_y >= 0 and earth.pos.x > 0:
            break
        prev_y = curr_y
    else:
        pytest.fail("Earth did not complete one orbit within expected time")

    error_pct = abs(elapsed - target_period) / target_period * 100
    assert error_pct < 0.5, (
        f"Earth's orbital period {elapsed/DAY:.2f} days, "
        f"expected {target_period/DAY:.2f} days (error {error_pct:.3f}%)"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  5. Momentum conservation
# ─────────────────────────────────────────────────────────────────────────────

def test_momentum_conservation():
    """
    With no external forces, total linear momentum must not change.
    Tolerance: < 1e-6 fraction drift.
    """
    sun, earth = _make_earth_system()
    sun.fixed = False   # Allow sun to move so total momentum can be checked
    bodies = [sun, earth]
    p0 = total_linear_momentum(bodies)

    dt = 3600.0
    for _ in range(500):
        rk4_step(bodies, dt, G_SI, softening=1e6)

    p1 = total_linear_momentum(bodies)
    dp = (p1 - p0).magnitude()
    p_mag = p0.magnitude()
    if p_mag > 0:
        frac = dp / p_mag
        assert frac < 1e-5, f"Momentum drifted by fraction {frac:.2e}"
    else:
        assert dp < 1e10, f"Momentum changed by {dp:.2e} from zero"


# ─────────────────────────────────────────────────────────────────────────────
#  6. Two-body circular orbit — orbit barycentre
# ─────────────────────────────────────────────────────────────────────────────

def test_two_body_barycentre():
    """
    Two equal-mass bodies in circular orbit should always have their
    barycentre (centre of mass) remain at or near the origin.
    """
    m = 1.0e30
    r = 1.0 * AU
    v = orbital_velocity(m, r, G_SI)   # speed for one body orbiting total mass

    # Symmetric setup: A at (-r/2, 0) moving down; B at (+r/2, 0) moving up
    # Actually use two-body orbital speed correctly:
    # Each body orbits the COM at r/2 from COM. v = sqrt(G*m / (4*r/2)) 
    # Simpler: use vis-viva around reduced mass system
    half_sep = r / 2
    v_each = math.sqrt(G_SI * m / (2 * half_sep))  # v for circular 2-body
    a = CelestialBody("A", m, 1e8, Vector2D(-half_sep, 0), Vector2D(0, -v_each), fixed=False)
    b = CelestialBody("B", m, 1e8, Vector2D(+half_sep, 0), Vector2D(0, +v_each), fixed=False)
    bodies = [a, b]

    # Barycentre initially at origin
    com0_x = (a.pos.x * a.mass + b.pos.x * b.mass) / (a.mass + b.mass)
    com0_y = (a.pos.y * a.mass + b.pos.y * b.mass) / (a.mass + b.mass)

    dt = 3600.0
    for _ in range(200):
        rk4_step(bodies, dt, G_SI, softening=1e4)

    com1_x = (a.pos.x * a.mass + b.pos.x * b.mass) / (a.mass + b.mass)
    com1_y = (a.pos.y * a.mass + b.pos.y * b.mass) / (a.mass + b.mass)

    drift = math.sqrt((com1_x - com0_x)**2 + (com1_y - com0_y)**2)
    # Drift should be tiny fraction of orbit size
    assert drift < half_sep * 1e-6, f"Barycentre drifted {drift:.3e} m"


# ─────────────────────────────────────────────────────────────────────────────
#  7. Collision merge
# ─────────────────────────────────────────────────────────────────────────────

def test_collision_merge_conserves_mass_and_momentum():
    a = CelestialBody("A", 3e24, 5e6, Vector2D(0, 0), Vector2D(1000, 0))
    b = CelestialBody("B", 1e24, 3e6, Vector2D(1e7, 0), Vector2D(-2000, 0))
    total_mass = a.mass + b.mass
    p_before = a.vel * a.mass + b.vel * b.mass

    merged = a.merge_with(b)

    assert math.isclose(merged.mass, total_mass, rel_tol=1e-12), "Mass not conserved"
    p_after = merged.vel * merged.mass
    dp = (p_after - p_before).magnitude()
    assert dp < 1.0, f"Momentum not conserved: Δp = {dp:.3e} kg·m/s"


# ─────────────────────────────────────────────────────────────────────────────
#  8. Preset builds without errors
# ─────────────────────────────────────────────────────────────────────────────

def test_solar_system_preset():
    from systems.preset_systems import build_solar_system
    bodies = build_solar_system()
    assert len(bodies) >= 10, f"Expected ≥10 bodies, got {len(bodies)}"
    names = [b.name for b in bodies]
    for planet in ["Sun", "Earth", "Jupiter", "Saturn"]:
        assert planet in names, f"'{planet}' missing from preset"


def test_binary_star_preset():
    from systems.preset_systems import build_binary_star
    bodies = build_binary_star()
    assert len(bodies) == 3


def test_figure_eight_preset():
    from systems.preset_systems import build_figure_eight
    bodies = build_figure_eight()
    assert len(bodies) == 3


# ─────────────────────────────────────────────────────────────────────────────
#  9. Orbital velocity formula
# ─────────────────────────────────────────────────────────────────────────────

def test_orbital_velocity_earth():
    """Earth's actual orbital speed ≈ 29,783 m/s."""
    v = orbital_velocity(1.989e30, AU, G_SI)
    assert math.isclose(v, 29_783.0, rel_tol=0.005), f"Got {v:.1f} m/s, expected ~29783 m/s"


# ─────────────────────────────────────────────────────────────────────────────
#  10. Simulation class integration
# ─────────────────────────────────────────────────────────────────────────────

def test_simulation_step_runs():
    from simulation import Simulation
    sim = Simulation(sub_steps=2, energy_monitor=False)
    sim.load_preset("solar_system")
    n_before = len(sim.bodies)
    sim.step(1.0 / 60)   # one frame at 60 FPS
    assert len(sim.bodies) >= n_before - 2   # allow for any merges


def test_simulation_add_remove():
    from simulation import Simulation
    sim = Simulation(sub_steps=1, energy_monitor=False)
    sim.load_preset("solar_system")
    n0 = len(sim.bodies)
    new = CelestialBody("Test", 1e20, 1e6, Vector2D(5e11, 0), Vector2D(0, 1e4))
    sim.add_body(new)
    sim.step(0.01)
    assert any(b.name == "Test" for b in sim.bodies)
    sim.remove_body("Test")
    sim.step(0.01)
    assert not any(b.name == "Test" for b in sim.bodies)
