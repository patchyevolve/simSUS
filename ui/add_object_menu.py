"""
add_object_menu.py — Floating panel to add a new body to the simulation.

Press 'A' to open/close. Select body type, adjust sliders, then click
anywhere on the simulation canvas to place the body at that position.
"""
from __future__ import annotations
import math
import random

import pygame

from physics.vector2d import Vector2D
from physics.gravity import orbital_velocity, G_SI
from bodies.star import Star
from bodies.planet import Planet
from bodies.moon import Moon
from bodies.asteroid import Asteroid
from bodies.spacecraft import Spacecraft


BODY_TYPES = ["Planet", "Star", "Moon", "Asteroid", "Spacecraft"]

# Defaults per type: (mass_kg, radius_m, color, extra kwargs)
_DEFAULTS = {
    "Planet":     (5e24,  6e6,   (100, 160, 220), {}),
    "Star":       (2e30,  6e8,   (255, 200,  80), {"temperature": 5800}),
    "Moon":       (7e22,  1.7e6, (180, 180, 175), {}),
    "Asteroid":   (1e18,  2e5,   (130, 120, 110), {}),
    "Spacecraft": (1e4,   5e4,   ( 80, 200, 120), {"thrust_force": 1e7}),
}


class AddObjectMenu:
    """
    Pygame UI panel — lives at the left side of the screen.
    """

    PANEL_X = 10
    PANEL_Y = 50
    PANEL_W = 220
    PANEL_H = 360

    def __init__(self, screen: pygame.Surface, sim) -> None:
        self.screen = screen
        self.sim = sim
        self.visible: bool = False
        self.placing: bool = False          # waiting for user click to place body
        self.selected_type_idx: int = 0

        pygame.font.init()
        self.font = pygame.font.SysFont("Consolas", 13)
        self.font_sm = pygame.font.SysFont("Consolas", 11)

        # Slider values (normalised 0–1)
        self.mass_log_norm: float = 0.5     # log-scale: 0→1e10 kg, 1→1e32 kg
        self.radius_log_norm: float = 0.4

        self._dragging_slider: str | None = None   # "mass" | "radius"

    # ------------------------------------------------------------------ #
    #  Properties                                                          #
    # ------------------------------------------------------------------ #

    @property
    def body_type(self) -> str:
        return BODY_TYPES[self.selected_type_idx]

    @property
    def mass(self) -> float:
        return 10 ** (10 + self.mass_log_norm * 22)   # 1e10 → 1e32 kg

    @property
    def radius(self) -> float:
        return 10 ** (3 + self.radius_log_norm * 10)  # 1e3 → 1e13 m

    def toggle(self) -> None:
        self.visible = not self.visible
        if not self.visible:
            self.placing = False

    # ------------------------------------------------------------------ #
    #  Event handling                                                      #
    # ------------------------------------------------------------------ #

    def handle_event(self, event: pygame.event.Event, camera) -> bool:
        """
        Returns True if the event was consumed by the menu.
        camera : renderer.Camera — for screen_to_world conversion.
        """
        if not self.visible:
            return False

        mx, my = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._in_panel(mx, my):
                self._handle_panel_click(mx, my)
                return True
            elif self.placing:
                # Place the body at the clicked world position
                world_pos = camera.screen_to_world(mx, my)
                self._place_body(world_pos, camera)
                self.placing = False
                return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._in_mass_slider(my):
                self._dragging_slider = "mass"
            elif self._in_radius_slider(my):
                self._dragging_slider = "radius"

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._dragging_slider = None

        if event.type == pygame.MOUSEMOTION and self._dragging_slider:
            rel_x = (mx - (self.PANEL_X + 10)) / (self.PANEL_W - 20)
            rel_x = max(0.0, min(1.0, rel_x))
            if self._dragging_slider == "mass":
                self.mass_log_norm = rel_x
            elif self._dragging_slider == "radius":
                self.radius_log_norm = rel_x
            return True

        return False

    def _in_panel(self, mx, my) -> bool:
        return (self.PANEL_X <= mx <= self.PANEL_X + self.PANEL_W
                and self.PANEL_Y <= my <= self.PANEL_Y + self.PANEL_H)

    def _in_mass_slider(self, my) -> bool:
        sy = self.PANEL_Y + 175
        return abs(my - sy) < 10

    def _in_radius_slider(self, my) -> bool:
        sy = self.PANEL_Y + 225
        return abs(my - sy) < 10

    def _handle_panel_click(self, mx, my) -> None:
        # Type selector buttons  (row of buttons at top of panel)
        btn_y = self.PANEL_Y + 42
        btn_w = (self.PANEL_W - 20) // len(BODY_TYPES)
        for i, btype in enumerate(BODY_TYPES):
            bx = self.PANEL_X + 10 + i * btn_w
            if bx <= mx <= bx + btn_w and btn_y <= my <= btn_y + 20:
                self.selected_type_idx = i
                # Reset sliders to sensible defaults for this type
                defaults = _DEFAULTS[btype]
                self.mass_log_norm  = (math.log10(defaults[0]) - 10) / 22
                self.radius_log_norm = (math.log10(defaults[1]) - 3) / 10
                return

        # "Place" button
        place_btn_y = self.PANEL_Y + self.PANEL_H - 35
        if self.PANEL_X + 10 <= mx <= self.PANEL_X + self.PANEL_W - 10 and place_btn_y <= my:
            self.placing = True

    # ------------------------------------------------------------------ #
    #  Place body in simulation                                             #
    # ------------------------------------------------------------------ #

    def _place_body(self, world_pos: Vector2D, camera) -> None:
        btype = self.body_type
        mass = self.mass
        radius = self.radius

        # Auto orbital velocity towards nearest star
        stars = [b for b in self.sim.bodies if isinstance(b, Star)]
        if stars:
            nearest_star = min(stars, key=lambda s: world_pos.distance_to(s.pos))
            dist = world_pos.distance_to(nearest_star.pos)
            v_orb = orbital_velocity(nearest_star.mass, dist, G_SI) if dist > 0 else 0.0
            radial = world_pos - nearest_star.pos
            perp = radial.perpendicular().normalise_safe()
            auto_vel = nearest_star.vel + perp * v_orb
        else:
            auto_vel = Vector2D.zero()

        if btype == "Planet":
            body = Planet("NewPlanet", mass, radius, world_pos, auto_vel)
        elif btype == "Star":
            body = Star("NewStar", mass, radius, world_pos, auto_vel,
                        temperature=5000, fixed=False)
        elif btype == "Moon":
            body = Moon("NewMoon", mass, radius, world_pos, auto_vel)
        elif btype == "Asteroid":
            body = Asteroid("NewAsteroid", mass, radius, world_pos, auto_vel,
                            spin=random.uniform(-1e-4, 1e-4))
        elif btype == "Spacecraft":
            body = Spacecraft("Probe", mass, radius, world_pos, auto_vel)
        else:
            return

        self.sim.add_body(body)

    # ------------------------------------------------------------------ #
    #  Draw                                                                #
    # ------------------------------------------------------------------ #

    def draw(self) -> None:
        if not self.visible:
            return

        # Panel background
        panel = pygame.Surface((self.PANEL_W, self.PANEL_H), pygame.SRCALPHA)
        panel.fill((8, 12, 35, 220))
        self.screen.blit(panel, (self.PANEL_X, self.PANEL_Y))

        x0 = self.PANEL_X + 10
        y = self.PANEL_Y + 10

        # Title
        s = self.font.render("➕ Add Body", True, (255, 220, 80))
        self.screen.blit(s, (x0, y))
        y += 26

        # Type selector buttons
        btn_w = (self.PANEL_W - 20) // len(BODY_TYPES)
        for i, btype in enumerate(BODY_TYPES):
            bx = x0 + i * btn_w
            col = (60, 100, 180) if i == self.selected_type_idx else (30, 35, 60)
            pygame.draw.rect(self.screen, col, (bx, y, btn_w - 2, 20), border_radius=3)
            label = self.font_sm.render(btype[:5], True, (220, 220, 255))
            self.screen.blit(label, (bx + 2, y + 3))
        y += 30

        def _draw_slider(label: str, norm_val: float, display_val: str, sy: int) -> None:
            s = self.font_sm.render(f"{label}: {display_val}", True, (180, 200, 255))
            self.screen.blit(s, (x0, sy - 14))
            track_x = x0
            track_w = self.PANEL_W - 20
            pygame.draw.rect(self.screen, (40, 40, 70), (track_x, sy - 4, track_w, 8), border_radius=4)
            thumb_x = int(track_x + norm_val * track_w)
            pygame.draw.circle(self.screen, (100, 160, 255), (thumb_x, sy), 7)

        # Mass slider
        _draw_slider("Mass", self.mass_log_norm, f"{self.mass:.2e} kg", self.PANEL_Y + 175)
        # Radius slider
        _draw_slider("Radius", self.radius_log_norm, f"{self.radius:.2e} m", self.PANEL_Y + 225)

        # Info
        y2 = self.PANEL_Y + 250
        s = self.font_sm.render(f"Type: {self.body_type}", True, (200, 220, 200))
        self.screen.blit(s, (x0, y2))
        y2 += 16
        if self.placing:
            s = self.font.render("Click to place →", True, (80, 255, 150))
            self.screen.blit(s, (x0, y2))

        # Place button
        btn_y = self.PANEL_Y + self.PANEL_H - 35
        btn_col = (30, 160, 80) if not self.placing else (20, 90, 50)
        pygame.draw.rect(self.screen, btn_col,
                         (x0, btn_y, self.PANEL_W - 20, 28), border_radius=5)
        label = "Click on map to place" if self.placing else "▶  Place Body"
        s = self.font.render(label, True, (220, 255, 220))
        self.screen.blit(s, (x0 + 6, btn_y + 6))
