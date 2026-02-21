"""
main.py — Entry point for the Solar System Simulation Engine.

Run:
    python main.py

Controls:
    Scroll wheel        Zoom in / out
    Middle-mouse drag   Pan camera
    Left-click body     Select (shows stats panel)
    SPACE               Pause / resume
    , / .               Decrease / increase time warp
    A                   Open add-object menu
    F                   Follow selected body
    C                   Clear all orbit trails
    R                   Reset to solar system
    1 / 2 / 3           Load preset (solar system / binary star / figure-8)
    S                   Save state to save.json
    L                   Load state from save.json
    ESC                 Quit
"""
from __future__ import annotations
import sys
import os
import time

# Add project root to python path so relative imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pygame

from simulation import Simulation
from renderer.camera import Camera
from renderer.renderer import Renderer
from ui.ui import HUD
from ui.add_object_menu import AddObjectMenu

# ─────────────────────────────────────────────────────────────
WINDOW_W, WINDOW_H = 1400, 850
FPS_TARGET         = 60
SAVE_PATH          = "save.json"
# ─────────────────────────────────────────────────────────────


def main() -> None:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("☀  Solar System Simulation Engine")
    clock = pygame.time.Clock()

    # ── Build core objects ─────────────────────────────────────
    sim      = Simulation(sub_steps=8, energy_monitor=True)
    camera   = Camera(WINDOW_W, WINDOW_H, meters_per_pixel=3e9)
    renderer = Renderer(screen, camera)
    hud      = HUD(screen, sim)
    add_menu = AddObjectMenu(screen, sim)

    # ── Register event callbacks ───────────────────────────────
    def on_collision(a, b, merged):
        hud.push_notification(
            f"💥 {a.name} + {b.name} → {merged.name}",
            color=(255, 120, 80), duration=4.0
        )

    def on_body_added(body):
        hud.push_notification(f"+ {body.name} added", color=(80, 255, 150))

    def on_body_removed(name):
        hud.push_notification(f"− {name} removed", color=(255, 80, 80))

    sim.on_collision(on_collision)
    sim.on_body_added(on_body_added)
    sim.on_body_removed(on_body_removed)

    # ── Load default preset ────────────────────────────────────
    sim.load_preset("solar_system")

    prev_time = time.perf_counter()

    # ── Main loop ──────────────────────────────────────────────
    running = True
    while running:
        now = time.perf_counter()
        real_dt = min(now - prev_time, 0.05)   # cap at 50 ms to avoid spiral-of-death
        prev_time = now

        # ── Event handling ─────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                _handle_key(event.key, sim, camera, hud, add_menu)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if add_menu.handle_event(event, camera):
                    pass
                elif event.button == 4:          # scroll up → zoom in
                    camera.zoom_in_at(*pygame.mouse.get_pos())
                elif event.button == 5:          # scroll down → zoom out
                    camera.zoom_out_at(*pygame.mouse.get_pos())
                elif event.button == 2:          # middle-mouse → pan
                    camera.begin_pan(*pygame.mouse.get_pos())
                elif event.button == 1:
                    # Select body on left-click (if not in add-menu)
                    if not add_menu.visible or not add_menu._in_panel(*pygame.mouse.get_pos()):
                        mx, my = pygame.mouse.get_pos()
                        world = camera.screen_to_world(mx, my)
                        # click radius = 20 pixels in world coords
                        click_r = 20 * camera.mpp
                        body = sim.select_body_at(world.x, world.y, click_r)
                        if body and camera.follow_body:
                            camera.follow(body)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    camera.end_pan()
                add_menu.handle_event(event, camera)

            elif event.type == pygame.MOUSEMOTION:
                camera.update_pan(*event.pos)
                add_menu.handle_event(event, camera)

            # Spacecraft keyboard thrust (continuous key query handled below)

        # ── Spacecraft controls (held keys) ───────────────────
        _handle_spacecraft_keys(sim)

        # ── Update ─────────────────────────────────────────────
        sim.step(real_dt)
        camera.update()

        # ── Draw ───────────────────────────────────────────────
        renderer.draw(sim.bodies)
        hud.draw()
        add_menu.draw()

        pygame.display.flip()
        clock.tick(FPS_TARGET)

    pygame.quit()
    sys.exit(0)


def _handle_key(key: int, sim: Simulation, camera: Camera,
                hud: HUD, add_menu: AddObjectMenu) -> None:
    if key == pygame.K_ESCAPE:
        pygame.quit(); sys.exit(0)

    elif key == pygame.K_SPACE:
        sim.toggle_pause()
        hud.push_notification("PAUSED" if sim.paused else "RESUMED", duration=1.5)

    elif key == pygame.K_COMMA:
        sim.warp_slower()
        hud.push_notification(f"Time warp ×{sim.time_warp:g}", duration=1.0)

    elif key == pygame.K_PERIOD:
        sim.warp_faster()
        hud.push_notification(f"Time warp ×{sim.time_warp:g}", duration=1.0)

    elif key == pygame.K_a:
        add_menu.toggle()

    elif key == pygame.K_f:
        body = sim.selected_body()
        if body:
            if camera.follow_body is body:
                camera.unfollow()
                hud.push_notification(f"Unfollowing {body.name}", duration=1.5)
            else:
                camera.follow(body)
                hud.push_notification(f"Following {body.name}", duration=1.5)

    elif key == pygame.K_r:
        sim.reset()
        camera.reset()
        hud.push_notification("Solar System reset", duration=2.0)

    elif key == pygame.K_c:
        sim.clear_trails()
        hud.push_notification("Orbit trails cleared", duration=1.5)

    elif key == pygame.K_1:
        sim.load_preset("solar_system")
        camera.reset()
        hud.push_notification("Preset: Solar System", duration=2.0)

    elif key == pygame.K_2:
        sim.load_preset("binary_star")
        camera.zoom_to(2e9)
        hud.push_notification("Preset: Binary Star", duration=2.0)

    elif key == pygame.K_3:
        sim.load_preset("figure_eight")
        camera.zoom_to(5e8)
        hud.push_notification("Preset: Figure-8 Three Body", duration=2.0)

    elif key == pygame.K_s:
        try:
            sim.save_state(SAVE_PATH)
            hud.push_notification(f"Saved → {SAVE_PATH}", duration=2.0)
        except Exception as e:
            hud.push_notification(f"Save failed: {e}", color=(255, 80, 80), duration=3.0)

    elif key == pygame.K_l:
        try:
            sim.load_state(SAVE_PATH)
            hud.push_notification(f"Loaded ← {SAVE_PATH}", duration=2.0)
        except Exception as e:
            hud.push_notification(f"Load failed: {e}", color=(255, 80, 80), duration=3.0)


def _handle_spacecraft_keys(sim: Simulation) -> None:
    from bodies.spacecraft import Spacecraft
    keys = pygame.key.get_pressed()
    for body in sim.bodies:
        if not isinstance(body, Spacecraft) or not body.selected:
            continue
        body.thrusting    = bool(keys[pygame.K_UP]   or keys[pygame.K_w])
        body.rotating_cw  = bool(keys[pygame.K_RIGHT] or keys[pygame.K_d])
        body.rotating_ccw = bool(keys[pygame.K_LEFT]  or keys[pygame.K_a])


if __name__ == "__main__":
    main()
