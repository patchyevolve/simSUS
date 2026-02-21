"""
camera.py — World ↔ screen coordinate transformations.

Supports:
  • Zoom (scroll wheel) with smooth lerp
  • Pan (middle-mouse drag)
  • "Follow body" mode — camera tracks a selected body
  • world_to_screen / screen_to_world conversions
"""
from __future__ import annotations

from physics.vector2d import Vector2D


class Camera:
    """
    Manages the viewport onto the simulation world.

    Parameters
    ----------
    screen_width, screen_height : int   Window dimensions in pixels.
    meters_per_pixel : float            Initial scale: world metres → 1 screen pixel.
    """

    ZOOM_FACTOR_STEP = 1.15     # multiply/divide per scroll tick
    ZOOM_MIN = 1e3              # pixels represent this many metres minimum
    ZOOM_MAX = 1e14             # maximum metres per pixel (fully zoomed out)
    FOLLOW_LERP = 0.07          # camera follows body at this fraction per frame

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        meters_per_pixel: float = 3e9,
    ) -> None:
        self.screen_width  = screen_width
        self.screen_height = screen_height
        self.mpp: float = meters_per_pixel     # world metres per screen pixel

        # Camera centre in world space (metres)
        self.center: Vector2D = Vector2D.zero()
        self._target_center: Vector2D = Vector2D.zero()

        self.follow_body = None                # CelestialBody | None

        # Pan state
        self._panning: bool = False
        self._pan_start_mouse: tuple[int, int] = (0, 0)
        self._pan_start_center: Vector2D = Vector2D.zero()

    # ------------------------------------------------------------------ #
    #  Core transforms                                                     #
    # ------------------------------------------------------------------ #

    def world_to_screen(self, world_pos: Vector2D) -> tuple[int, int]:
        """Convert world-space position (metres) → screen pixel (x, y)."""
        dx = (world_pos.x - self.center.x) / self.mpp
        dy = (world_pos.y - self.center.y) / self.mpp
        sx = int(self.screen_width  / 2 + dx)
        sy = int(self.screen_height / 2 - dy)   # y-axis is flipped
        return (sx, sy)

    def screen_to_world(self, sx: int, sy: int) -> Vector2D:
        """Convert screen pixel → world-space position (metres)."""
        dx = (sx - self.screen_width  / 2) * self.mpp
        dy = (sy - self.screen_height / 2) * self.mpp
        return Vector2D(self.center.x + dx, self.center.y - dy)

    def world_radius_to_pixels(self, radius_m: float) -> float:
        """Convert a world-space radius in metres to pixels."""
        return radius_m / self.mpp

    def pixels_per_metre(self) -> float:
        return 1.0 / self.mpp

    # ------------------------------------------------------------------ #
    #  Zoom                                                                #
    # ------------------------------------------------------------------ #

    def zoom_in(self) -> None:
        self.mpp = max(self.ZOOM_MIN, self.mpp / self.ZOOM_FACTOR_STEP)

    def zoom_out(self) -> None:
        self.mpp = min(self.ZOOM_MAX, self.mpp * self.ZOOM_FACTOR_STEP)

    def zoom_to(self, metres_per_pixel: float) -> None:
        self.mpp = max(self.ZOOM_MIN, min(self.ZOOM_MAX, metres_per_pixel))

    def zoom_in_at(self, sx: int, sy: int) -> None:
        """Zoom in centred on screen point (sx, sy) — keeps that point fixed."""
        world_before = self.screen_to_world(sx, sy)
        self.zoom_in()
        world_after = self.screen_to_world(sx, sy)
        delta = world_after - world_before
        self.center = self.center - delta

    def zoom_out_at(self, sx: int, sy: int) -> None:
        world_before = self.screen_to_world(sx, sy)
        self.zoom_out()
        world_after = self.screen_to_world(sx, sy)
        delta = world_after - world_before
        self.center = self.center - delta

    # ------------------------------------------------------------------ #
    #  Pan                                                                 #
    # ------------------------------------------------------------------ #

    def begin_pan(self, mx: int, my: int) -> None:
        self._panning = True
        self._pan_start_mouse = (mx, my)
        self._pan_start_center = self.center
        self.follow_body = None   # break follow when user pans

    def update_pan(self, mx: int, my: int) -> None:
        if not self._panning:
            return
        dx = (mx - self._pan_start_mouse[0]) * self.mpp
        dy = (my - self._pan_start_mouse[1]) * self.mpp
        self.center = Vector2D(
            self._pan_start_center.x - dx,
            self._pan_start_center.y + dy,  # y flipped
        )

    def end_pan(self) -> None:
        self._panning = False

    # ------------------------------------------------------------------ #
    #  Follow mode                                                         #
    # ------------------------------------------------------------------ #

    def follow(self, body) -> None:
        self.follow_body = body

    def unfollow(self) -> None:
        self.follow_body = None

    def update(self) -> None:
        """Call once per frame to smoothly nudge camera towards followed body."""
        if self.follow_body is not None:
            target = self.follow_body.pos
            self.center = self.center.lerp(target, self.FOLLOW_LERP)

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self.center = Vector2D.zero()
        self.follow_body = None
        self.mpp = 3e9

    def __repr__(self) -> str:
        return (
            f"<Camera center=({self.center.x:.3e},{self.center.y:.3e}) "
            f"mpp={self.mpp:.3e}>"
        )
