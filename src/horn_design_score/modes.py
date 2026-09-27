"""Primary packed-alm beam input for limtod_jax drift-scan scoring."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import numpy as np

from .protocol import Protocol


@dataclass(frozen=True, eq=False)
class BeamModes:
    """One-port, Stokes-I *power gain* in beam-local healpy packed alm layout.

    ``full_alm`` is the all-direction accepted-power beam. Supplying
    ``sky_alm`` plus ``eta_rad`` and ``h_partition`` skips the expensive
    horizon preparation during design search. ``sky_ref_alm`` additionally
    skips the fixed beam-local to celestial rotation. Prepared arrays must
    derive from the same full beam, using half weight on horizon-centred
    pixels for the physical sky/ground split.
    """

    freqs_mhz: np.ndarray
    full_alm: np.ndarray  # (frequency, nalm), beam-local, real-field symmetry
    nside: int
    lmax: int
    sky_alm: np.ndarray | None = None  # horizon-masked beam-local alms
    eta_rad: np.ndarray | None = None
    h_partition: np.ndarray | None = None
    convention: str = "accepted_power"
    sky_ref_alm: np.ndarray | None = None  # masked beam at LST=0.5 degrees
    ref_latitude_deg: float = 53.23625

    def __post_init__(self) -> None:
        f = np.array(self.freqs_mhz, dtype=float, copy=True)
        a = np.array(self.full_alm, dtype=np.complex128, copy=True)
        if f.ndim != 1 or f.size < 2 or np.any(~np.isfinite(f)) or np.any(f <= 0) or np.any(np.diff(f) <= 0):
            raise ValueError("freqs_mhz must be positive and strictly increasing")
        if not hp.isnsideok(self.nside) or self.lmax < 0 or self.lmax > 3 * self.nside - 1:
            raise ValueError("invalid nside/lmax for HEALPix beam")
        nalm = hp.Alm.getsize(self.lmax)
        if a.shape != (f.size, nalm) or not np.all(np.isfinite(a)):
            raise ValueError("full_alm must have shape (frequency, packed nalm) and be finite")
        if np.any(np.abs(a[:, :self.lmax + 1].imag) > 1e-10 * np.maximum(1, np.abs(a[:, :self.lmax + 1].real))):
            raise ValueError("m=0 alm coefficients of a real power beam must be real")
        if self.convention not in ("accepted_power", "shape_only"):
            raise ValueError("unknown gain convention")
        prepared = (self.sky_alm is not None, self.eta_rad is not None, self.h_partition is not None)
        if any(prepared) and not all(prepared):
            raise ValueError("sky_alm, eta_rad and h_partition must be supplied together")
        if self.sky_ref_alm is not None and not all(prepared):
            raise ValueError("sky_ref_alm requires the prepared local sky beam")
        if not np.isfinite(self.ref_latitude_deg) or not -90 <= self.ref_latitude_deg <= 90:
            raise ValueError("invalid reference latitude")
        if all(prepared):
            sky = np.array(self.sky_alm, dtype=np.complex128, copy=True)
            eta = np.array(self.eta_rad, dtype=float, copy=True)
            h = np.array(self.h_partition, dtype=float, copy=True)
            if sky.shape != a.shape or not np.all(np.isfinite(sky)):
                raise ValueError("sky_alm must match full_alm shape and be finite")
            if eta.shape != f.shape or h.shape != f.shape or np.any(~np.isfinite(eta)) or np.any(~np.isfinite(h)):
                raise ValueError("eta_rad and h_partition must have one finite value per channel")
            if np.any(eta <= 0) or np.any(h < 0) or np.any(h > eta + 1e-6):
                raise ValueError("invalid beam efficiency or sky partition")
            if self.convention == "accepted_power" and np.any(eta > 1.001):
                raise ValueError("accepted-power efficiency exceeds unity")
            if self.convention == "shape_only" and not np.allclose(eta, 1., atol=1e-6):
                raise ValueError("shape_only prepared modes require unit efficiency")
            object.__setattr__(self, "sky_alm", sky)
            object.__setattr__(self, "eta_rad", eta)
            object.__setattr__(self, "h_partition", h)
            if self.sky_ref_alm is not None:
                ref = np.array(self.sky_ref_alm, dtype=np.complex128, copy=True)
                if ref.shape != a.shape or not np.all(np.isfinite(ref)):
                    raise ValueError("sky_ref_alm must match full_alm shape and be finite")
                object.__setattr__(self, "sky_ref_alm", ref)
                ref.flags.writeable = False
            for arr in (sky, eta, h):
                arr.flags.writeable = False
        object.__setattr__(self, "freqs_mhz", f)
        object.__setattr__(self, "full_alm", a)
        f.flags.writeable = False
        a.flags.writeable = False

    def prepare(self, latitude_deg: float = 53.23625) -> "BeamModes":
        """Derive a fixed zenith horizon mask and the physical power split.

        The zenith sky window assigns half of each horizon-centred pixel to
        the sky. This matches the physical sky/ground partition and avoids
        an O(1/nside) energy mismatch from a strict horizon cut. The map is
        re-analysed once and the prepared result can be saved to NPZ.
        """
        if not np.isfinite(latitude_deg) or not -90 <= latitude_deg <= 90:
            raise ValueError("invalid reference latitude")
        if self.sky_ref_alm is not None and latitude_deg == self.ref_latitude_deg:
            return self
        if self.sky_alm is not None:
            masked = self.sky_alm
            eta = self.eta_rad
            h = self.h_partition
            self_full = self.full_alm
        else:
            theta, _ = hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)))
            partition = np.where(theta < np.pi / 2 - 1e-14, 1.,
                                 np.where(theta > np.pi / 2 + 1e-14, 0., 0.5))
            mask = partition
            eta = np.empty(len(self.freqs_mhz))
            h = np.empty_like(eta)
            masked = np.empty_like(self.full_alm)
            for i, alm in enumerate(self.full_alm):
                physical_map = hp.alm2map(alm, self.nside, lmax=self.lmax)
                eta[i] = np.mean(physical_map)
                h[i] = np.mean(physical_map * partition)
                masked[i] = hp.map2alm(physical_map * mask, lmax=self.lmax, iter=3)
            if self.convention == "shape_only":
                if np.any(eta <= 0):
                    raise ValueError("shape-only beam has zero integral")
                masked /= eta[:, None]
                self_full = self.full_alm / eta[:, None]
                h /= eta
                eta[:] = 1.0
            else:
                self_full = self.full_alm
                if np.any(eta <= 0) or np.any(eta > 1.001):
                    raise ValueError("accepted-power beam efficiency outside (0, 1.001]")
        from limTOD.simulator import zyz_of_pointing
        psi, beta, phi = zyz_of_pointing(0.5, latitude_deg, 0., 90., 0.)
        ref = masked.copy()
        for row in ref:
            hp.rotate_alm(row, phi, beta, psi, lmax=self.lmax)
        return BeamModes(self.freqs_mhz, self_full, self.nside, self.lmax,
                         sky_alm=masked, eta_rad=eta, h_partition=h,
                         convention=self.convention, sky_ref_alm=ref,
                         ref_latitude_deg=latitude_deg)

    def validate_for(self, protocol: Protocol) -> None:
        if not np.array_equal(self.freqs_mhz, protocol.freqs_mhz):
            raise ValueError("beam and protocol frequency grids must match")
        if self.nside != protocol.sky_nside:
            raise ValueError("beam and sky HEALPix nside must match")

    def save_npz(self, path: str | Path) -> None:
        prepared = self if self.sky_ref_alm is not None else self.prepare()
        np.savez(path, freqs_mhz=prepared.freqs_mhz, full_alm=prepared.full_alm,
                 nside=prepared.nside, lmax=prepared.lmax, sky_alm=prepared.sky_alm,
                 eta_rad=prepared.eta_rad, h_partition=prepared.h_partition,
                 sky_ref_alm=prepared.sky_ref_alm, convention=prepared.convention,
                 ref_latitude_deg=prepared.ref_latitude_deg)

    @classmethod
    def load_npz(cls, path: str | Path) -> "BeamModes":
        with np.load(path, allow_pickle=False) as data:
            kwargs = {key: data[key] for key in ("sky_alm", "eta_rad", "h_partition", "sky_ref_alm") if key in data}
            return cls(data["freqs_mhz"], data["full_alm"], int(data["nside"]),
                       int(data["lmax"]), convention=str(data["convention"]) if "convention" in data else "accepted_power",
                       ref_latitude_deg=float(data["ref_latitude_deg"]) if "ref_latitude_deg" in data else 53.23625,
                       **kwargs)


def beam_modes_from_grid(beam, nside: int, lmax: int | None = None) -> BeamModes:
    """Optional converter; the primary scoring input is already packed alm."""
    lmax = 3 * nside - 1 if lmax is None else lmax
    maps = beam.sample_healpix(nside)
    alm = np.stack([hp.map2alm(row, lmax=lmax) for row in maps])
    return BeamModes(beam.freqs_mhz, alm, nside, lmax, convention=beam.convention)
