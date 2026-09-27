"""Deterministic analytic inputs for a quick beam-design benchmark.

These are generated models, not the CNN-PL/GLEAM or GSM2008/GLEAM maps and
not a 21cmVAE prediction. No external data files or downloads are needed.
"""

from __future__ import annotations

import healpy as hp
import numpy as np
from scipy.special import erf


_REFERENCE_MHZ = 70.0
_CMB_K = 2.725
_SIGNAL_CENTRE_MHZ = 72.0
_SIGNAL_SIGMA_MHZ = 10.0
_SIGNAL_DEPTH_K = -0.14
_QUAD_NODES = (-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0))
_QUAD_WEIGHTS = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)


def _channels(freqs_mhz: np.ndarray, bandwidth_hz: float) -> tuple[np.ndarray, float]:
    freqs = np.asarray(freqs_mhz, dtype=float)
    width = float(bandwidth_hz) / 1e6
    if (freqs.ndim != 1 or freqs.size == 0 or not np.all(np.isfinite(freqs))
            or not np.isfinite(width) or width <= 0
            or np.any(freqs <= width / 2)):
        raise ValueError("channel centres must be finite and exceed half the positive bandwidth")
    return freqs, width


def default_signal_k(freqs_mhz: np.ndarray, bandwidth_hz: float = 1e6) -> np.ndarray:
    """Channel-averaged, illustrative Gaussian absorption in Kelvin.

    The centre is 72 MHz, depth is -0.14 K, and Gaussian sigma is 10 MHz.
    It is a fixed design template, not a physical 21 cm prediction.
    """
    freqs, width = _channels(freqs_mhz, bandwidth_hz)
    scale = np.sqrt(2.0) * _SIGNAL_SIGMA_MHZ
    upper = erf((freqs + width / 2 - _SIGNAL_CENTRE_MHZ) / scale)
    lower = erf((freqs - width / 2 - _SIGNAL_CENTRE_MHZ) / scale)
    return (_SIGNAL_DEPTH_K * _SIGNAL_SIGMA_MHZ
            * np.sqrt(np.pi / 2.0) * (upper - lower) / width)


def _foreground_cube(freqs: np.ndarray, width: float, amplitude: np.ndarray,
                     index: np.ndarray, curvature: np.ndarray) -> np.ndarray:
    cube = np.zeros((freqs.size, amplitude.size), dtype=float)
    for node, weight in zip(_QUAD_NODES, _QUAD_WEIGHTS):
        log_ratio = np.log((freqs + width * node / 2) / _REFERENCE_MHZ)[:, None]
        cube += (weight / 2) * amplitude[None, :] * np.exp(
            index[None, :] * log_ratio + curvature[None, :] * log_ratio**2
        )
    return cube + _CMB_K


def default_foreground_scenarios(
    freqs_mhz: np.ndarray, nside: int, bandwidth_hz: float = 1e6,
) -> dict[str, np.ndarray]:
    """Two smooth analytic equatorial HEALPix RING foreground cubes in Kelvin.

    Each has a broad bright strip, a localized bright region, and spatially
    varying spectral index and curvature. They are channel averaged and
    intended only as reproducible design-search examples.
    """
    freqs, width = _channels(freqs_mhz, bandwidth_hz)
    if not hp.isnsideok(nside):
        raise ValueError("nside must be a valid HEALPix resolution")
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    strip_a = np.exp(-0.5 * ((0.6 * y + 0.8 * z) / 0.20)**2)
    bright_a = np.exp(-(1 - (0.8 * x + 0.6 * y)) / 0.035)
    amplitude_a = 650 + 1000 * strip_a + 1400 * bright_a
    index_a = -2.50 + 0.08 * z - 0.06 * x
    curvature_a = -0.025 * (1 + y)

    strip_b = np.exp(-0.5 * ((-0.3 * x + np.sqrt(0.91) * z) / 0.26)**2)
    bright_b = np.exp(-(1 - (-0.4 * x + 0.6 * y + np.sqrt(0.48) * z)) / 0.045)
    amplitude_b = 800 + 1300 * strip_b + 1800 * bright_b
    index_b = -2.57 + 0.09 * x + 0.05 * y
    curvature_b = -0.04 * (1 - z)

    return {
        "analytic_a": _foreground_cube(freqs, width, amplitude_a, index_a, curvature_a),
        "analytic_b": _foreground_cube(freqs, width, amplitude_b, index_b, curvature_b),
    }
