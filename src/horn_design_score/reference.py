"""Slow limTOD-based forward and m-mode reference for shortlisted beams."""

from __future__ import annotations

import healpy as hp
import numpy as np

from .beam import BeamPattern
from .modes import BeamModes
from .kernel import ZonalKernel, foreground_fast, ground_loss_fast
from .protocol import Protocol
from .score import ScoreResult, _score_spectra


def reference_forward(beam: BeamPattern, protocol: Protocol,
                      channel_indices: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return per-scenario (d0, exact thermal sigma) on selected channels.

    This path is intentionally slow and uses the sky-map nside. A small set of
    channels is sufficient to check the zonal fast path while developing.
    """
    return _grid_spectra(beam, protocol, channel_indices, with_reference=False)[:2]


def _channel_indices(protocol: Protocol, channel_indices: np.ndarray | None) -> np.ndarray:
    indices = (np.arange(protocol.freqs_mhz.size) if channel_indices is None
               else np.asarray(channel_indices, dtype=int))
    if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= protocol.freqs_mhz.size):
        raise ValueError("invalid channel indices")
    return indices


def _grid_spectra(beam: BeamPattern, protocol: Protocol,
                  channel_indices: np.ndarray | None, *, with_reference: bool
                  ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Return (d0, sigma, reference-sky d0, ground and loss) on selected channels.

    ``with_reference`` repeats the TOD synthesis on the beam-factor reference
    sky map, doubling the cost; without it the third item is empty.
    """
    from limTOD import generate_TOD_sky, mmode

    if not np.array_equal(beam.freqs_mhz, protocol.freqs_mhz):
        raise ValueError("beam and protocol frequency grids must match")
    nside = protocol.sky_nside
    indices = _channel_indices(protocol, channel_indices)
    gain = beam.sample_healpix(nside)
    kernel = ZonalKernel.build(protocol, nside)
    _, h, _ = foreground_fast(gain, kernel, protocol)
    npix = gain.shape[1]
    sky_fraction = kernel.sky_fraction
    ground_loss = ground_loss_fast(gain, kernel, protocol)
    lst = np.arange(360, dtype=float) + 0.5
    zero = np.zeros(360)
    zenith = np.full(360, 90.0)
    window = mmode.LSTWindow(lst)

    def sky_tod(i: int, sky_row: np.ndarray) -> np.ndarray:
        if h[i] <= 0:
            return np.zeros_like(lst)
        # limTOD rotates a finite-lmax beam and may change its integral,
        # especially with a hard horizon. Renormalize after rotation, then
        # restore the independently integrated sky throughput h.
        return h[i] * generate_TOD_sky(gain[i] * sky_fraction / npix, sky_row, lst,
                                       protocol.latitude_deg, zero, zenith, zero,
                                       normalize_beam=True)

    foreground: dict[str, np.ndarray] = {}
    noise: dict[str, np.ndarray] = {}
    reference: dict[str, np.ndarray] = {}
    for name, sky in protocol.sky_maps_k.items():
        tod = np.stack([sky_tod(i, sky[i]) + ground_loss[i] for i in indices], axis=1)
        # The closed, uniformly spaced m=0 estimate is exactly the LST mean.
        solved = mmode.solve(tod, window, m_trunc=0)
        fg = solved.d0
        if not np.allclose(fg, tod.mean(axis=0), rtol=1e-10, atol=1e-10):
            raise ArithmeticError("limTOD m-mode d0 differs from closed LST mean")
        foreground[name] = fg
        t_sys = tod + protocol.receiver_k
        noise[name] = np.sqrt(np.sum(t_sys**2 / protocol.visits[:, None], axis=0)) / (
            360 * np.sqrt(protocol.bandwidth_hz * protocol.integration_s))
        if with_reference:
            template = sky[protocol.bcf_reference_index]
            reference[name] = np.array([sky_tod(i, template).mean() for i in indices])
    return foreground, noise, reference, ground_loss[indices]


def reference_score(beam: BeamPattern, protocol: Protocol,
                    *, fit_spectrum: bool = False) -> ScoreResult:
    """Use limTOD full-TOD synthesis and m-mode solve on all channels."""
    gain = beam.sample_healpix(protocol.sky_nside)
    kernel = ZonalKernel.build(protocol)
    _, h, eta = foreground_fast(gain, kernel, protocol)
    foreground, noise, reference, ground_loss = _grid_spectra(
        beam, protocol, None, with_reference=protocol.foreground_model == "edges_beam_factor")
    return _score_spectra(foreground, h, eta, noise, protocol, "per_bin_tsys_limtod",
                          reference_d0=reference, additive_k=ground_loss,
                          fit_spectrum=fit_spectrum)


def reference_modes_forward(beam: BeamModes, protocol: Protocol,
                            channel_indices: np.ndarray | None = None
                            ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """Independent limTOD map rotation and m=0 solve for a packed-alm beam.

    This intentionally performs 360 map syntheses per selected channel. It
    checks the harmonic contraction, phase and noise without using the fast
    scorer's Fourier synthesis. The same already horizon-masked local alm is
    used, so the comparison isolates the m-mode calculation.
    """
    beam.validate_for(protocol)
    return _modes_spectra(beam.prepare(protocol.latitude_deg), protocol, channel_indices)[:3]


def _modes_spectra(b: BeamModes, protocol: Protocol, channel_indices: np.ndarray | None
                   ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray,
                              dict[str, np.ndarray], np.ndarray]:
    """Return (d0, sigma, throughput, reference-sky d0, ground and loss).

    ``b`` is a prepared beam. The reference-sky d0 reuses each pointed beam
    map on the beam-factor reference channel of the same sky.
    """
    from limTOD import mmode
    from limTOD.simulator import pointing_beam_in_eq_sys

    indices = _channel_indices(protocol, channel_indices)
    lst = np.arange(360, dtype=float) + 0.5
    window = mmode.LSTWindow(lst)
    ground_loss = ((b.eta_rad - b.h_partition) * protocol.ground_k
                   + (1 - b.eta_rad) * protocol.loss_k)
    foreground: dict[str, np.ndarray] = {}
    noise: dict[str, np.ndarray] = {}
    reference: dict[str, np.ndarray] = {}
    throughput = np.zeros(indices.size)
    for name, sky in protocol.sky_maps_k.items():
        template = sky[protocol.bcf_reference_index]
        tod = np.empty((360, indices.size))
        template_tod = np.empty_like(tod)
        for j, i in enumerate(indices):
            for k, angle in enumerate(lst):
                pointed = pointing_beam_in_eq_sys(
                    b.sky_alm[i], angle, protocol.latitude_deg, 0., 90., 0.,
                    b.nside, normalize=False, truncate_frac_thres=0.)
                if name == next(iter(protocol.sky_maps_k)):
                    throughput[j] += pointed.mean() / len(lst)
                tod[k, j] = np.mean(pointed * sky[i]) + ground_loss[i]
                template_tod[k, j] = np.mean(pointed * template)
        d0 = mmode.solve(tod, window, m_trunc=0).d0
        foreground[name] = d0
        reference[name] = template_tod.mean(axis=0)
        t_sys = tod + protocol.receiver_k
        noise[name] = np.sqrt(np.sum(t_sys**2 / protocol.visits[:, None], axis=0)) / (
            360 * np.sqrt(protocol.bandwidth_hz * protocol.integration_s))
    return foreground, noise, throughput, reference, ground_loss[indices]


def reference_modes_score(beam: BeamModes, protocol: Protocol,
                          *, fit_spectrum: bool = False) -> ScoreResult:
    """Full-LST limTOD oracle for a shortlisted packed-alm beam."""
    beam.validate_for(protocol)
    b = beam.prepare(protocol.latitude_deg)
    foreground, noise, throughput, reference, ground_loss = _modes_spectra(b, protocol, None)
    return _score_spectra(foreground, throughput, b.eta_rad,
                          noise, protocol, "per_bin_tsys_limtod",
                          reference_d0=reference, additive_k=ground_loss,
                          fit_spectrum=fit_spectrum)
