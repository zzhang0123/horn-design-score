"""Precompute the RA-averaged sky seen by each local HEALPix direction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import healpy as hp
import numpy as np
from scipy.special import eval_legendre

from .protocol import Protocol


@dataclass(frozen=True, eq=False)
class ZonalKernel:
    nside: int
    latitude_deg: float
    freqs_mhz: np.ndarray
    mean_sky_k: dict[str, np.ndarray]  # (frequency, local pixel)
    sky_fraction: np.ndarray  # equator-centred pixels contribute half to each hemisphere
    method: str = "harmonic"
    source_fingerprint: str = ""

    @classmethod
    def build(cls, protocol: Protocol, nside: int | None = None,
              *, method: str = "harmonic") -> "ZonalKernel":
        """Build an m=0 sky kernel once; candidate evaluations use only dot products.

        ``harmonic`` matches limTOD's band-limited spherical-harmonic convention.
        ``ring`` is a cheaper approximation useful for diagnosing pixel effects.
        """
        nside = protocol.sky_nside if nside is None else nside
        if not hp.isnsideok(nside):
            raise ValueError("invalid kernel nside")
        if method not in ("harmonic", "ring"):
            raise ValueError("method must be harmonic or ring")
        x, _, z = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
        lat = np.deg2rad(protocol.latitude_deg)
        sin_dec = np.sin(lat) * z - np.cos(lat) * x  # limTOD zenith rotation: phi=0 points south
        out: dict[str, np.ndarray] = {}
        if method == "harmonic":
            lmax = 3 * protocol.sky_nside - 1
            # Y_l0 is real. The m=0 sky is invariant to LST and is evaluated
            # directly in the local frame at the declination of each pixel.
            basis = np.stack([np.sqrt((2 * ell + 1) / (4 * np.pi)) * eval_legendre(ell, sin_dec)
                              for ell in range(lmax + 1)])
            for name, sky in protocol.sky_maps_k.items():
                # Centering makes a spatially uniform sky exact even when
                # finite-HEALPix harmonic quadrature leaks a tiny monopole.
                monopole = sky.mean(axis=1)
                alm0 = np.stack([hp.map2alm(row - mean, lmax=lmax)[
                    hp.Alm.getidx(lmax, np.arange(lmax + 1), 0)].real
                    for row, mean in zip(sky, monopole)])
                out[name] = monopole[:, None] + alm0 @ basis
        else:
            starts, counts, ring_z, _, _ = hp.ringinfo(protocol.sky_nside,
                                                      np.arange(1, 4 * protocol.sky_nside))
            for name, sky in protocol.sky_maps_k.items():
                ring_mean = np.add.reduceat(sky, starts.astype(int), axis=1) / counts[None, :]
                out[name] = np.stack([np.interp(sin_dec, ring_z[::-1], row[::-1])
                                      for row in ring_mean])
        sky_fraction = np.where(z > 1e-14, 1.0, np.where(z < -1e-14, 0.0, 0.5))
        return cls(nside, protocol.latitude_deg, protocol.freqs_mhz.copy(), out,
                   sky_fraction, method, protocol.fingerprint)

    def validate_for(self, protocol: Protocol) -> None:
        if self.latitude_deg != protocol.latitude_deg or not np.array_equal(self.freqs_mhz, protocol.freqs_mhz):
            raise ValueError("kernel and protocol site/frequency mismatch")
        if set(self.mean_sky_k) != set(protocol.sky_maps_k):
            raise ValueError("kernel and protocol sky scenarios differ")
        if self.source_fingerprint != protocol.fingerprint:
            raise ValueError("kernel was built from different sky or protocol inputs")

    def save_npz(self, path: str | Path) -> None:
        """Save the expensive one-time sky projection for reuse across processes."""
        np.savez(path, nside=self.nside, latitude_deg=self.latitude_deg,
                 freqs_mhz=self.freqs_mhz, sky_fraction=self.sky_fraction,
                 method=self.method, source_fingerprint=self.source_fingerprint,
                 **{f"sky_{name}": value for name, value in self.mean_sky_k.items()})

    @classmethod
    def load_npz(cls, path: str | Path, protocol: Protocol) -> "ZonalKernel":
        with np.load(path, allow_pickle=False) as data:
            skies = {key[4:]: data[key] for key in data.files if key.startswith("sky_") and key != "sky_fraction"}
            out = cls(int(data["nside"]), float(data["latitude_deg"]), data["freqs_mhz"],
                      skies, data["sky_fraction"], str(data["method"]), str(data["source_fingerprint"]))
        out.validate_for(protocol)
        return out


def foreground_fast(gain: np.ndarray, kernel: ZonalKernel, protocol: Protocol) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Return foreground d0, celestial signal throughput h and radiation efficiency."""
    kernel.validate_for(protocol)
    if gain.shape != (protocol.freqs_mhz.size, hp.nside2npix(kernel.nside)):
        raise ValueError("gain map has wrong shape")
    eta = gain.mean(axis=1)
    sky_fraction = kernel.sky_fraction
    h = np.mean(gain * sky_fraction[None, :], axis=1)
    ground_loss = ground_loss_fast(gain, kernel, protocol)
    foreground = {name: np.mean(gain * sky * sky_fraction[None, :], axis=1) + ground_loss
                  for name, sky in kernel.mean_sky_k.items()}
    return foreground, h, eta


def ground_loss_fast(gain: np.ndarray, kernel: ZonalKernel, protocol: Protocol) -> np.ndarray:
    """Return the ground-pickup plus loss temperature per channel."""
    ground = np.mean(gain * (1 - kernel.sky_fraction)[None, :], axis=1) * protocol.ground_k
    return ground + (1.0 - gain.mean(axis=1)) * protocol.loss_k


def reference_d0_fast(gain: np.ndarray, kernel: ZonalKernel, protocol: Protocol) -> dict[str, np.ndarray]:
    """Return each channel's beam weighting the reference-channel sky map."""
    j = protocol.bcf_reference_index
    return {name: np.mean(gain * (sky[j] * kernel.sky_fraction)[None, :], axis=1)
            for name, sky in kernel.mean_sky_k.items()}
