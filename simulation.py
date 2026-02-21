"""
simulation.py — Master simulation engine.

Responsibilities:
  • Holds the canonical list of CelestialBody objects.
  • Drives the RK4 integrator each frame (with time-warp sub-stepping).
  • Detects and resolves collisions (perfectly inelastic merge).
  • Emits events: on_collision, on_body_added, on_body_removed.
  • Monitors energy conservation and warns on significant drift.
  • save_state / load_state — full JSON snapshot.

Usage
-----
    sim = Simulation()
    sim.load_preset("solar_system")
    # each frame:
    sim.step(dt_real_seconds)
"""
from __future__ import annotations
import json
import time
from typing import Callable

from physics.gravity import (
    G_SI, total_mechanical_energy, total_linear_momentum
)
from physics.integrator import rk4_step
from physics.vector2d import Vector2D
from bodies.body import CelestialBody
from bodies.spacecraft import Spacecraft
from bodies.asteroid import Asteroid


# ──────────────────────────────────────────────────────────────────────────────
#  Time-warp presets
# ──────────────────────────────────────────────────────────────────────────────
TIME_WARPS = [0.0, 0.1, 0.5, 1.0, 10.0, 100.0, 1_000.0, 10_000.0,
              100_000.0, 1_000_000.0]


class Simulation:
    """Main simulation state machine."""

    def __init__(
        self,
        G: float = G_SI,
        softening: float = 1e6,           # metres
        base_dt: float = 3600.0,          # 1 simulation hour per real second (1× warp)
        sub_steps: int = 8,               # RK4 sub-steps per frame for accuracy
        trail_interval: int = 4,          # record trail every N sub-steps
        energy_monitor: bool = True,
        energy_drift_warn_pct: float = 0.1,
    ) -> None:
        self.G: float = G
        self.softening: float = softening
        self.base_dt: float = base_dt
        self.sub_steps: int = sub_steps
        self.trail_interval: int = trail_interval
        self.energy_monitor: bool = energy_monitor
        self.energy_drift_warn_pct: float = energy_drift_warn_pct

        self.bodies: list[CelestialBody] = []
        self.sim_time: float = 0.0        # seconds of simulated time
        self.real_time: float = 0.0       # seconds of wall-clock time
        self.paused: bool = False
        self.time_warp_index: int = 3     # default index → 1× warp
        self._step_count: int = 0
        self._initial_energy: float | None = None

        # Event callbacks
        self._on_collision: list[Callable] = []
        self._on_body_added: list[Callable] = []
        self._on_body_removed: list[Callable] = []

        self._pending_removals: list[str] = []   # names to remove after integrator
        self._pending_adds: list[CelestialBody] = []

    # ──────────────────────────────────────────────────────────────────────────
    #  Properties
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def time_warp(self) -> float:
        return TIME_WARPS[self.time_warp_index]

    @property
    def effective_dt(self) -> float:
        """Simulation seconds advanced per real second."""
        return self.base_dt * self.time_warp

    def warp_faster(self) -> None:
        self.time_warp_index = min(len(TIME_WARPS) - 1, self.time_warp_index + 1)

    def warp_slower(self) -> None:
        self.time_warp_index = max(1, self.time_warp_index - 1)

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    # ──────────────────────────────────────────────────────────────────────────
    #  Body management
    # ──────────────────────────────────────────────────────────────────────────

    def add_body(self, body: CelestialBody) -> None:
        """Thread-safe-ish deferred add (applied at start of next step)."""
        self._pending_adds.append(body)

    def remove_body(self, name: str) -> None:
        """Schedule named body for removal at end of current step."""
        self._pending_removals.append(name)

    def get_body(self, name: str) -> CelestialBody | None:
        for b in self.bodies:
            if b.name == name:
                return b
        return None

    def _flush_pending(self) -> None:
        """Apply deferred add/remove operations."""
        for name in self._pending_removals:
            before = len(self.bodies)
            self.bodies = [b for b in self.bodies if b.name != name]
            if len(self.bodies) < before:
                for cb in self._on_body_removed:
                    cb(name)
        self._pending_removals.clear()

        for body in self._pending_adds:
            self.bodies.append(body)
            for cb in self._on_body_added:
                cb(body)
        self._pending_adds.clear()

    # ──────────────────────────────────────────────────────────────────────────
    #  Main step
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, real_dt: float) -> None:
        """
        Advance the simulation by real_dt wall-clock seconds.

        Internal flow:
        1. Apply pending adds/removes
        2. If paused or warp=0, skip integration
        3. Pre-step spacecraft thrusters
        4. Sub-step with RK4 (sub_steps iterations)
        5. Record trails every trail_interval sub-steps
        6. Spin asteroids
        7. Detect and resolve collisions
        8. Energy monitoring
        """
        self._flush_pending()
        if self.paused or self.time_warp == 0.0 or not self.bodies:
            return

        sim_dt_total = real_dt * self.effective_dt
        dt_sub = sim_dt_total / self.sub_steps

        # Pre-step spacecraft thrusters (accumulate forces before RK4)
        for body in self.bodies:
            if isinstance(body, Spacecraft):
                body.update(dt_sub)

        for sub in range(self.sub_steps):
            rk4_step(self.bodies, dt_sub, self.G, self.softening)
            self.sim_time += dt_sub
            self._step_count += 1

            # Trail recording
            if self._step_count % self.trail_interval == 0:
                for body in self.bodies:
                    body.record_trail()

            # Asteroid spin
            for body in self.bodies:
                if isinstance(body, Asteroid):
                    body.update_spin(dt_sub)

        self.real_time += real_dt
        self._detect_collisions()

        # Energy drift monitoring
        if self.energy_monitor and self._step_count % (self.sub_steps * 60) == 0:
            self._check_energy()

    # ──────────────────────────────────────────────────────────────────────────
    #  Collision detection & resolution
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_collisions(self) -> None:
        """O(n²) pairwise check — adequate for < 100 bodies."""
        merged_names: set[str] = set()
        new_bodies: list[CelestialBody] = []
        n = len(self.bodies)

        for i in range(n):
            a = self.bodies[i]
            if a.name in merged_names:
                continue
            for j in range(i + 1, n):
                b = self.bodies[j]
                if b.name in merged_names:
                    continue
                if a.is_colliding_with(b):
                    merged = a.merge_with(b)
                    merged_names.add(a.name)
                    merged_names.add(b.name)
                    new_bodies.append(merged)
                    for cb in self._on_collision:
                        cb(a, b, merged)
                    break   # a is consumed; move to next i

        surviving = [b for b in self.bodies if b.name not in merged_names]
        self.bodies = surviving + new_bodies

    # ──────────────────────────────────────────────────────────────────────────
    #  Energy monitor
    # ──────────────────────────────────────────────────────────────────────────

    def _check_energy(self) -> None:
        bodies = [b for b in self.bodies if not b.fixed]
        if len(bodies) < 2:
            return
        E_now = total_mechanical_energy(bodies, self.G, self.softening)
        if self._initial_energy is None:
            self._initial_energy = E_now
            return
        if self._initial_energy == 0.0:
            return
        drift_pct = abs(E_now - self._initial_energy) / abs(self._initial_energy) * 100.0
        if drift_pct > self.energy_drift_warn_pct:
            print(
                f"[SIM WARNING] Energy drift {drift_pct:.3f}% "
                f"(E₀={self._initial_energy:.4e}, E={E_now:.4e})"
            )

    def reset_energy_baseline(self) -> None:
        self._initial_energy = None

    # ──────────────────────────────────────────────────────────────────────────
    #  Event subscriptions
    # ──────────────────────────────────────────────────────────────────────────

    def on_collision(self, callback: Callable) -> None:
        """callback(body_a, body_b, merged_body)"""
        self._on_collision.append(callback)

    def on_body_added(self, callback: Callable) -> None:
        """callback(body)"""
        self._on_body_added.append(callback)

    def on_body_removed(self, callback: Callable) -> None:
        """callback(name: str)"""
        self._on_body_removed.append(callback)

    # ──────────────────────────────────────────────────────────────────────────
    #  Presets
    # ──────────────────────────────────────────────────────────────────────────

    def load_preset(self, preset_name: str) -> None:
        from systems.preset_systems import PRESETS
        if preset_name not in PRESETS:
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {list(PRESETS)}")
        self.bodies = PRESETS[preset_name]()
        self.sim_time = 0.0
        self._step_count = 0
        self._initial_energy = None
        for body in self.bodies:
            body.clear_trail()

    def reset(self) -> None:
        """Reload the default solar system preset."""
        self.load_preset("solar_system")

    def clear_trails(self) -> None:
        """Clear the recorded orbit trail for every body."""
        for body in self.bodies:
            body.clear_trail()

    # ──────────────────────────────────────────────────────────────────────────
    #  Save / Load
    # ──────────────────────────────────────────────────────────────────────────

    def save_state(self, path: str) -> None:
        state = {
            "sim_time": self.sim_time,
            "time_warp_index": self.time_warp_index,
            "bodies": [b.to_dict() for b in self.bodies],
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str) -> None:
        with open(path) as f:
            state = json.load(f)
        self.sim_time = state["sim_time"]
        self.time_warp_index = state.get("time_warp_index", 3)
        self.bodies = [CelestialBody.from_dict(d) for d in state["bodies"]]
        self._initial_energy = None

    # ──────────────────────────────────────────────────────────────────────────
    #  Selection helpers
    # ──────────────────────────────────────────────────────────────────────────

    def select_body_at(self, world_x: float, world_y: float, click_radius: float) -> CelestialBody | None:
        """Return the body closest to the click point, within click_radius metres."""
        best = None
        best_dist = click_radius
        click = Vector2D(world_x, world_y)
        for b in self.bodies:
            d = click.distance_to(b.pos)
            if d < best_dist:
                best_dist = d
                best = b
        # Deselect all, then select winner
        for b in self.bodies:
            b.selected = False
        if best:
            best.selected = True
        return best

    def selected_body(self) -> CelestialBody | None:
        for b in self.bodies:
            if b.selected:
                return b
        return None

    # ──────────────────────────────────────────────────────────────────────────
    #  Diagnostics / repr
    # ──────────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        bodies = [b for b in self.bodies if not b.fixed]
        from physics.gravity import total_kinetic_energy, total_potential_energy
        return {
            "n_bodies": len(self.bodies),
            "sim_time_s": self.sim_time,
            "sim_time_years": self.sim_time / (365.25 * 86400),
            "time_warp": self.time_warp,
            "kinetic_energy": total_kinetic_energy(bodies),
            "potential_energy": total_potential_energy(bodies, self.G, self.softening),
            "paused": self.paused,
        }

    def __repr__(self) -> str:
        return (
            f"<Simulation bodies={len(self.bodies)} "
            f"t={self.sim_time / 86400:.1f}d warp={self.time_warp}×>"
        )
