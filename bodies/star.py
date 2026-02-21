"""
star.py — Star class with blackbody colour, luminosity, and habitable zone.
"""
from __future__ import annotations
import math

from physics.vector2d import Vector2D
from bodies.body import CelestialBody


# Stefan-Boltzmann constant (W m⁻² K⁻⁴)
_SIGMA = 5.670_374_419e-8
# Solar reference values
_SOLAR_LUMINOSITY = 3.828e26   # Watts
_SOLAR_RADIUS     = 6.957e8    # metres
_SOLAR_MASS       = 1.989e30   # kg
_SOLAR_TEMP       = 5778.0     # K


def _blackbody_color(temp_k: float) -> tuple[int, int, int]:
    """
    Approximate blackbody RGB from temperature in Kelvin.
    Algorithm by Tanner Helland (2012), valid 1000 K – 40000 K.
    """
    t = temp_k / 100.0
    # Red channel
    if t <= 66:
        r = 255
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        r = max(0, min(255, r))
    # Green channel
    if t <= 66:
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
    g = max(0, min(255, g))
    # Blue channel
    if t >= 66:
        b = 255
    elif t <= 19:
        b = 0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    b = max(0, min(255, b))
    return (int(r), int(g), int(b))


class Star(CelestialBody):
    """
    A stellar body.

    Parameters
    ----------
    temperature : float   Surface temperature in Kelvin.  Determines colour
                          and luminosity (via Stefan-Boltzmann law).
    luminosity  : float   Override luminosity (Watts). If None, calculated
                          from temperature and radius.
    """

    def __init__(
        self,
        name: str,
        mass: float,
        radius: float,
        pos: Vector2D,
        vel: Vector2D,
        temperature: float = _SOLAR_TEMP,
        luminosity: float | None = None,
        fixed: bool = True,
        trail_max: int = 0,
    ) -> None:
        color = _blackbody_color(temperature)
        super().__init__(
            name=name,
            mass=mass,
            radius=radius,
            pos=pos,
            vel=vel,
            color=color,
            fixed=fixed,
            trail_max=trail_max,
            body_type="star",
        )
        self.temperature: float = float(temperature)

        if luminosity is not None:
            self.luminosity: float = float(luminosity)
        else:
            # L = 4π R² σ T⁴
            self.luminosity = 4.0 * math.pi * radius ** 2 * _SIGMA * temperature ** 4

    # ------------------------------------------------------------------ #
    #  Habitable zone (simple flux-based estimate)                         #
    # ------------------------------------------------------------------ #

    def habitable_zone(self) -> tuple[float, float]:
        """
        Conservative habitable zone boundaries (metres).

        Uses the Kasting / Kopparapu (2013) flux limits:
            inner edge: S_eff ≈ 1.1   (runaway greenhouse)
            outer edge: S_eff ≈ 0.356 (maximum greenhouse)

        Returns
        -------
        (inner_radius, outer_radius) in metres.
        """
        # L_star / L_sun gives solar-equivalent luminosity
        l_ratio = self.luminosity / _SOLAR_LUMINOSITY
        # Distance for S_eff = S_eff_sun from Earth (1 AU = 1.496e11 m)
        AU = 1.496e11
        inner = AU * math.sqrt(l_ratio / 1.1)
        outer = AU * math.sqrt(l_ratio / 0.356)
        return (inner, outer)

    @property
    def habitable_zone_inner(self) -> float:
        return self.habitable_zone()[0]

    @property
    def habitable_zone_outer(self) -> float:
        return self.habitable_zone()[1]

    # ------------------------------------------------------------------ #
    #  Class type helpers                                                   #
    # ------------------------------------------------------------------ #

    @property
    def spectral_class(self) -> str:
        """Morgan-Keenan spectral class letter (O B A F G K M)."""
        t = self.temperature
        if t >= 30000: return "O"
        if t >= 10000: return "B"
        if t >= 7500:  return "A"
        if t >= 6000:  return "F"
        if t >= 5200:  return "G"
        if t >= 3700:  return "K"
        return "M"

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"temperature": self.temperature, "luminosity": self.luminosity})
        return d
