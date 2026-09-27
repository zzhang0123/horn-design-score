"""One-port theta/phi power gain on a complete regular angular grid."""

from __future__ import annotations

from dataclasses import dataclass

import healpy as hp
import numpy as np
from scipy.interpolate import RegularGridInterpolator


@dataclass(frozen=True, eq=False)
class BeamPattern:
    freqs_mhz: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    gain_theta_dbi: np.ndarray
    gain_phi_dbi: np.ndarray
    convention: str = "accepted_power"

    def __post_init__(self) -> None:
        f, t, p = (np.asarray(x, dtype=float) for x in
                   (self.freqs_mhz, self.theta_deg, self.phi_deg))
        if f.ndim != 1 or t.ndim != 1 or p.ndim != 1 or min(map(len, (f, t, p))) < 2:
            raise ValueError("frequency, theta and phi must be one-dimensional grids")
        if not all(np.all(np.isfinite(x)) and np.all(np.diff(x) > 0) for x in (f, t, p)):
            raise ValueError("frequency, theta and phi must be finite and strictly increasing")
        if np.any(f <= 0) or not np.isclose(t[0], 0, atol=1e-8) or not np.isclose(t[-1], 180, atol=1e-8):
            raise ValueError("frequency must be positive and theta must cover 0..180 degrees")
        if p.size < 4 or p[0] < 0 or p[-1] >= 360 or np.max(np.diff(np.r_[p, p[0] + 360])) > 2 * np.median(np.diff(p)):
            raise ValueError("phi must cover a periodic full 360 degree grid without a large gap")
        if self.convention not in ("accepted_power", "shape_only"):
            raise ValueError("convention must be accepted_power or shape_only")
        for name in ("gain_theta_dbi", "gain_phi_dbi"):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != (len(f), len(t), len(p)) or np.any(np.isnan(a)) or np.any(np.isposinf(a)):
                raise ValueError(f"{name} must have shape (frequency, theta, phi), finite or -inf")
            object.__setattr__(self, name, a)
        for name, value in (("freqs_mhz", f), ("theta_deg", t), ("phi_deg", p)):
            object.__setattr__(self, name, value)

    def sample_healpix(self, nside: int) -> np.ndarray:
        """Interpolate *linear power* onto local RING pixels; phi=0 is south for zenith scans."""
        if not hp.isnsideok(nside):
            raise ValueError("invalid HEALPix nside")
        theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        points = np.column_stack((np.rad2deg(theta),
                                  (np.rad2deg(phi) - self.phi_deg[0]) % 360 + self.phi_deg[0]))
        p_ext = np.r_[self.phi_deg, self.phi_deg[0] + 360]
        out = np.empty((len(self.freqs_mhz), len(theta)), dtype=float)
        for i in range(len(self.freqs_mhz)):
            total = np.power(10.0, self.gain_theta_dbi[i] / 10.0) + np.power(10.0, self.gain_phi_dbi[i] / 10.0)
            interpolator = RegularGridInterpolator(
                (self.theta_deg, p_ext), np.column_stack((total, total[:, :1])),
                bounds_error=True,
            )
            out[i] = interpolator(points)
        if not np.all(np.isfinite(out)) or np.any(out < 0):
            raise ValueError("interpolated power gain is invalid")
        eta = out.mean(axis=1)
        if np.any(eta <= 0):
            raise ValueError("beam has zero total response")
        if self.convention == "shape_only":
            out /= eta[:, None]
        elif np.any(eta > 1.001):
            raise ValueError("integrated accepted-power gain exceeds physical efficiency 1")
        return out

    @classmethod
    def load_npz(cls, path: str) -> "BeamPattern":
        with np.load(path, allow_pickle=False) as data:
            return cls(*(data[k] for k in ("freqs_mhz", "theta_deg", "phi_deg", "gain_theta_dbi", "gain_phi_dbi")),
                       convention=str(data["convention"]) if "convention" in data else "accepted_power")
