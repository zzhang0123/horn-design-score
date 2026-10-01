"""Fast drift-scan m-mode score for packed beam alms.

This implements limtod_jax.driftscan's documented packed-alm contraction in
NumPy/healpy. Mixing JAX's and healpy's OpenMP runtimes aborts on the current
macOS environment, so both transform sides stay in one runtime here.
"""

from __future__ import annotations

from dataclasses import dataclass

import healpy as hp
import numpy as np

from .modes import BeamModes
from .protocol import Protocol
from .score import ScoreResult, _score_spectra, edges_beam_factor


@dataclass(eq=False)
class HarmonicScorer:
    """Prepare sky quadrature alms and phase synthesis once for many beams."""

    protocol: Protocol
    lmax: int

    def __post_init__(self) -> None:
        if self.lmax < 0 or self.lmax > min(3 * self.protocol.sky_nside - 1, 359):
            raise ValueError("invalid sky/beam band-limit")
        nside = self.protocol.sky_nside
        npix = hp.nside2npix(nside)
        # Unlike beam hp.map2alm(iter=3), the sky uses an iter=0 pixel
        # quadrature analysis scaled to a plain pixel sum. This is the exact
        # limtod_jax.driftscan input contract.
        self.names = tuple(sorted(self.protocol.sky_maps_k))
        self.sky_quad = np.stack([
            np.stack([hp.map2alm(row, lmax=self.lmax, iter=0) * (npix / (4 * np.pi))
                      for row in self.protocol.sky_maps_k[name]])
            for name in self.names
        ])  # (scenario, frequency, packed alm)
        self.ones_quad = hp.map2alm(np.ones(npix), lmax=self.lmax, iter=0) * (npix / (4 * np.pi))
        # Zonal (m=0) harmonics of each sky at the beam-factor reference channel.
        self.reference_quad = self.sky_quad[:, self.protocol.bcf_reference_index, :self.lmax + 1]
        lst = np.arange(360, dtype=float) + 0.5
        self.phase = np.exp(1j * np.deg2rad(lst[:, None] - lst[0])
                            * np.arange(1, self.lmax + 1)[None, :])

    def _prepared(self, beam: BeamModes) -> BeamModes:
        beam.validate_for(self.protocol)
        if beam.lmax != self.lmax:
            raise ValueError("beam lmax differs from prepared sky lmax")
        return beam.prepare(self.protocol.latitude_deg)

    def _ground_loss(self, b: BeamModes) -> np.ndarray:
        return ((b.eta_rad - b.h_partition) * self.protocol.ground_k
                + (1 - b.eta_rad) * self.protocol.loss_k)

    def _reference_d0(self, b: BeamModes) -> dict[str, np.ndarray]:
        """Full-LST mean of each channel's beam on the reference-channel sky."""
        zonal = np.conj(b.sky_ref_alm[:, :self.lmax + 1])
        d0 = (zonal @ self.reference_quad.T).real / hp.nside2npix(b.nside)
        return {name: d0[:, j] for j, name in enumerate(self.names)}

    def beam_factor(self, beam: BeamModes) -> dict[str, np.ndarray]:
        """Return the EDGES beam chromaticity factor C(nu) for each sky."""
        reference = self._reference_d0(self._prepared(beam))
        return {name: edges_beam_factor(d0, self.protocol) for name, d0 in reference.items()}

    def forward(self, beam: BeamModes) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Return d0 spectra, per-bin thermal errors, sky throughput and efficiency."""
        return self._spectra(self._prepared(beam))

    def _spectra(self, b: BeamModes) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
        beam_ref = b.sky_ref_alm
        starts = hp.Alm.getidx(self.lmax, np.arange(self.lmax + 1), np.arange(self.lmax + 1))
        product = np.conj(beam_ref)[None, :, :] * self.sky_quad
        modes = np.add.reduceat(product, starts, axis=-1) / hp.nside2npix(b.nside)
        ones_mode = np.sum(np.conj(beam_ref[:, :self.lmax + 1]) * self.ones_quad[:self.lmax + 1], axis=1).real / hp.nside2npix(b.nside)
        ground_loss = self._ground_loss(b)
        foreground: dict[str, np.ndarray] = {}
        noise: dict[str, np.ndarray] = {}
        for j, name in enumerate(self.names):
            c = modes[j]
            # Exact Fourier synthesis on the fixed full-turn LST bin centres;
            # no high-m truncation is introduced by the scorer.
            tod = c[:, 0].real[:, None] + 2 * np.real(c[:, 1:] @ self.phase.T)
            tod += ground_loss[:, None]
            foreground[name] = c[:, 0].real + ground_loss
            t_sys = tod + self.protocol.receiver_k
            noise[name] = np.sqrt(np.sum(t_sys**2 / self.protocol.visits[None, :], axis=1)) / (
                360 * np.sqrt(self.protocol.bandwidth_hz * self.protocol.integration_s))
        return foreground, noise, ones_mode, b.eta_rad

    def score(self, beam: BeamModes, *, fit_spectrum: bool = False) -> ScoreResult:
        """Return one score; optionally fit the injected global-signal template."""
        b = self._prepared(beam)
        foreground, noise, throughput, eta = self._spectra(b)
        if np.any(throughput <= 0):
            return ScoreResult(0., "invalid_sky_throughput", "masked harmonic beam has nonpositive response", {}, "per_bin_tsys_mmodes")
        return _score_spectra(foreground, throughput, eta, noise,
                              self.protocol, "per_bin_tsys_mmodes",
                              reference_d0=self._reference_d0(b),
                              additive_k=self._ground_loss(b),
                              fit_spectrum=fit_spectrum)
