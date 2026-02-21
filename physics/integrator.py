"""
integrator.py — Numerical ODE integrators for orbital mechanics.

Implements:
  • rk4_step        — fixed-step 4th-order Runge-Kutta (workhorse)
  • adaptive_rk4_step — Dormand-Prince RK45 with step-size control
                        (auto-shrinks dt near close approaches)

All integrators operate on a list of CelestialBody objects and update
their .pos and .vel in-place.

Reference: Hairer, Nørsett & Wanner "Solving ODEs I" (Springer, 1993).
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING

from .vector2d import Vector2D
from .gravity import total_acceleration

if TYPE_CHECKING:
    from bodies.body import CelestialBody


# ------------------------------------------------------------------ #
#  State snapshot helpers (avoid mutating bodies during k-stage calc) #
# ------------------------------------------------------------------ #

class _BodyState:
    """Lightweight snapshot of pos + vel for an RK stage."""
    __slots__ = ("pos", "vel")

    def __init__(self, pos: Vector2D, vel: Vector2D) -> None:
        self.pos = pos
        self.vel = vel


def _snapshot(bodies: list["CelestialBody"]) -> list[_BodyState]:
    return [_BodyState(b.pos, b.vel) for b in bodies]


def _apply_snapshot(bodies: list["CelestialBody"], states: list[_BodyState]) -> None:
    for b, s in zip(bodies, states):
        b.pos = s.pos
        b.vel = s.vel


def _derivatives(
    states: list[_BodyState],
    bodies: list["CelestialBody"],
    G: float,
    softening: float,
) -> list[_BodyState]:
    """
    Compute (dpos/dt, dvel/dt) for each body given a set of trial states.
    dpos/dt = vel
    dvel/dt = acceleration from gravity
    """
    # Temporarily apply trial states to compute cross-accelerations
    original = _snapshot(bodies)
    _apply_snapshot(bodies, states)
    derivs = []
    for b in bodies:
        acc = total_acceleration(b, bodies, G, softening)
        derivs.append(_BodyState(b.vel, acc))  # (dpos/dt = vel, dvel/dt = acc)
    _apply_snapshot(bodies, original)
    return derivs


# ------------------------------------------------------------------ #
#  Fixed-step RK4                                                      #
# ------------------------------------------------------------------ #

def rk4_step(
    bodies: list["CelestialBody"],
    dt: float,
    G: float,
    softening: float,
) -> None:
    """
    Advance every body by time step dt using 4th-order Runge-Kutta.

    Classical RK4:
        k1 = f(y_n)
        k2 = f(y_n + ½·dt·k1)
        k3 = f(y_n + ½·dt·k2)
        k4 = f(y_n + dt·k3)
        y_{n+1} = y_n + dt/6 · (k1 + 2k2 + 2k3 + k4)

    This is O(h⁴) accurate per step — far superior to Euler (O(h)) or
    leapfrog (O(h²)) for multi-body problems with varying time-scales.
    """
    if not bodies or dt == 0.0:
        return

    s0 = _snapshot(bodies)           # y_n

    # — k1 —
    k1 = _derivatives(s0, bodies, G, softening)

    # — k2 — trial at t + ½dt
    s1 = [
        _BodyState(
            s0[i].pos + k1[i].pos * (0.5 * dt),
            s0[i].vel + k1[i].vel * (0.5 * dt),
        )
        for i in range(len(bodies))
    ]
    k2 = _derivatives(s1, bodies, G, softening)

    # — k3 — trial at t + ½dt (using k2)
    s2 = [
        _BodyState(
            s0[i].pos + k2[i].pos * (0.5 * dt),
            s0[i].vel + k2[i].vel * (0.5 * dt),
        )
        for i in range(len(bodies))
    ]
    k3 = _derivatives(s2, bodies, G, softening)

    # — k4 — trial at t + dt (using k3)
    s3 = [
        _BodyState(
            s0[i].pos + k3[i].pos * dt,
            s0[i].vel + k3[i].vel * dt,
        )
        for i in range(len(bodies))
    ]
    k4 = _derivatives(s3, bodies, G, softening)

    # — Combine: y_{n+1} = y_n + dt/6 · (k1 + 2k2 + 2k3 + k4) —
    sixth_dt = dt / 6.0
    for i, body in enumerate(bodies):
        if body.fixed:
            continue
        body.pos = s0[i].pos + (
            k1[i].pos + k2[i].pos * 2.0 + k3[i].pos * 2.0 + k4[i].pos
        ) * sixth_dt
        body.vel = s0[i].vel + (
            k1[i].vel + k2[i].vel * 2.0 + k3[i].vel * 2.0 + k4[i].vel
        ) * sixth_dt


# ------------------------------------------------------------------ #
#  Dormand-Prince RK45 adaptive-step integrator                        #
# ------------------------------------------------------------------ #

# Butcher tableau coefficients for Dormand-Prince RK45
_A = [
    [],
    [1 / 5],
    [3 / 40, 9 / 40],
    [44 / 45, -56 / 15, 32 / 9],
    [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
    [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
]
_B5 = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]
_E  = [
    71 / 57600, 0, -71 / 16695, 71 / 1920,
    -17253 / 339200, 22 / 525, -1 / 40,
]   # error coefficients (B5 − B4*)


def adaptive_rk4_step(
    bodies: list["CelestialBody"],
    dt: float,
    G: float,
    softening: float,
    tol: float = 1e-6,
    dt_min: float = 1.0,
    dt_max: float = 86400.0 * 30,   # 30 days max step
) -> float:
    """
    Dormand-Prince RK45 step with error-controlled adaptive step size.

    Attempts to take step dt, measures error vs. embedded 4th-order
    solution, accepts if error < tol, and returns the recommended next dt.

    Parameters
    ----------
    bodies    : list of CelestialBody (updated in-place on acceptance)
    dt        : attempted step size (seconds)
    G, softening : physics params
    tol       : relative + absolute tolerance
    dt_min    : minimum allowed step (prevents infinite shrinkage)
    dt_max    : maximum allowed step

    Returns
    -------
    float
        The recommended dt for the *next* step.
    """
    if not bodies:
        return dt

    while True:
        s0 = _snapshot(bodies)
        n = len(bodies)

        # Compute the 6 Runge-Kutta stages
        stages: list[list[_BodyState]] = [None] * 6  # type: ignore
        stages[0] = _derivatives(s0, bodies, G, softening)

        def stage_state(stage_idx: int) -> list[_BodyState]:
            coeffs = _A[stage_idx]
            result = []
            for i in range(n):
                dp = Vector2D.zero()
                dv = Vector2D.zero()
                for j, c in enumerate(coeffs):
                    dp = dp + stages[j][i].pos * c
                    dv = dv + stages[j][i].vel * c
                result.append(_BodyState(s0[i].pos + dp * dt, s0[i].vel + dv * dt))
            return result

        for si in range(1, 6):
            stages[si] = _derivatives(stage_state(si), bodies, G, softening)
        # 7th stage (FSAL — same as next step's first stage)
        s_final = stage_state(5)   # using B5 coefficients implicitly via stage 5
        stage7 = _derivatives(s_final, bodies, G, softening)
        stages.append(stage7)

        # Estimate 5th-order solution and error
        max_err = 0.0
        new_states = []
        for i in range(n):
            dp5 = Vector2D.zero()
            dv5 = Vector2D.zero()
            dep = Vector2D.zero()
            dev = Vector2D.zero()
            for j, b5 in enumerate(_B5):
                dp5 = dp5 + stages[j][i].pos * b5
                dv5 = dv5 + stages[j][i].vel * b5
            for j, e in enumerate(_E):
                dep = dep + stages[j][i].pos * e
                dev = dev + stages[j][i].vel * e

            pos5 = s0[i].pos + dp5 * dt
            vel5 = s0[i].vel + dv5 * dt
            new_states.append(_BodyState(pos5, vel5))

            # Error norm: max over position and velocity scaled by tolerance
            err_p = (dep * dt).magnitude() / (tol * (1.0 + pos5.magnitude()))
            err_v = (dev * dt).magnitude() / (tol * (1.0 + vel5.magnitude()))
            max_err = max(max_err, err_p, err_v)

        if max_err <= 1.0 or dt <= dt_min:
            # Accept this step
            _apply_snapshot(bodies, new_states)
            # Compute new recommended step size (standard PI controller)
            if max_err > 0:
                factor = min(5.0, max(0.1, 0.9 * max_err ** (-0.2)))
            else:
                factor = 5.0
            return max(dt_min, min(dt_max, dt * factor))
        else:
            # Reject — shrink step and retry
            factor = max(0.1, 0.9 * max_err ** (-0.25))
            dt = max(dt_min, dt * factor)
