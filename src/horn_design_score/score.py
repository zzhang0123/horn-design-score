"""A single deterministic, foreground-residual-penalized design score."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .beam import BeamPattern
from .kernel import ZonalKernel, foreground_fast
from .protocol import Protocol


@dataclass(frozen=True)
class ScoreResult:
    value: float
    status: str
    detail: str
    scenarios: dict[str, dict[str, float]]
    noise_method: str
    fitted_spectra: dict[str, FittedSpectrum] | None = None


@dataclass(frozen=True)
class FittedSpectrum:
    """Fixed-template fit to a noiseless injected sky scenario.

    ``intrinsic_k`` is the inferred global 21 cm spectrum before beam
    attenuation; ``antenna_k`` includes the beam sky throughput. Neither is
    a free-form reconstruction or a fit to measured data.
    """

    status: str
    frequencies_mhz: list[float]
    amplitude: float | None = None
    intrinsic_k: list[float] | None = None
    antenna_k: list[float] | None = None
    residual_chi2: float | None = None
    detail: str = ""
    model_kind: str = "fixed_template_amplitude"
    data_kind: str = "noiseless_A1_injection"


def _score_spectra(foreground: dict[str, np.ndarray], h: np.ndarray, eta: np.ndarray,
                   noise: dict[str, np.ndarray], protocol: Protocol, noise_method: str,
                   *, fit_spectrum: bool = False) -> ScoreResult:
    f = protocol.freqs_mhz
    x = np.log(f / 70.0)
    v = np.vander(x, protocol.foreground_order + 1, increasing=True)
    template = h * protocol.signal_k
    scenarios: dict[str, dict[str, float]] = {}
    spectra: dict[str, FittedSpectrum] | None = {} if fit_spectrum else None
    for name, fg in foreground.items():
        sigma = noise[name]
        if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)) or np.any(fg <= 0):
            return ScoreResult(0., "invalid_forward_model", name, {}, noise_method)
        # Fit foreground only, before adding the 21 cm signal. A nonlinear fit
        # is needed: a log-space polynomial fit is not a Kelvin-space fit.
        try:
            a0 = np.polynomial.polynomial.polyfit(x, np.log(fg), protocol.foreground_order)
            def model(a: np.ndarray) -> np.ndarray:
                with np.errstate(over="raise", invalid="raise"):
                    return np.exp(v @ a)
            opt = least_squares(lambda a: (fg - model(a)) / sigma, a0,
                                jac=lambda a: -(model(a)[:, None] * v) / sigma[:, None],
                                max_nfev=500, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        except (FloatingPointError, ValueError, OverflowError) as exc:
            return ScoreResult(0., "foreground_fit_failed", f"{name}: {exc}", {}, noise_method)
        if not opt.success or not np.all(np.isfinite(opt.x)):
            return ScoreResult(0., "foreground_fit_failed", name, {}, noise_method)
        fitted = model(opt.x)
        jac = fitted[:, None] * v / sigma[:, None]
        q, r = np.linalg.qr(jac, mode="reduced")
        if np.min(np.abs(np.diag(r))) < 1e-12 * np.max(np.abs(np.diag(r))):
            return ScoreResult(0., "foreground_rank_deficient", name, {}, noise_method)
        u = template / sigma
        u -= q @ (q.T @ u)
        residual = (fg - fitted) / sigma
        residual -= q @ (q.T @ residual)
        information = float(u @ u)
        if information <= 1e-24 or not np.isfinite(information):
            return ScoreResult(0., "signal_unidentifiable", name, {}, noise_method)
        residual_power = float(residual @ residual)
        bias = float((u @ residual) / information)
        value = float(np.sqrt(information / (1.0 + residual_power)))
        scenarios[name] = {"score": value, "information": information,
                           "amplitude_bias": bias, "residual_chi2": residual_power,
                           "sigma_amplitude": float(1 / np.sqrt(information)),
                           "eta_min": float(np.min(eta)), "eta_max": float(np.max(eta)),
                           "sky_throughput_mean": float(np.mean(h))}
        if spectra is not None:
            # Fit a noiseless mock with the protocol's A=1 signal injected.
            # This is intentionally separate from the foreground-only fit
            # that defines the FOM, so requesting spectra cannot change it.
            data = fg + template
            def joint_residual(params: np.ndarray) -> np.ndarray:
                return (data - model(params[:-1]) - params[-1] * template) / sigma
            def joint_jac(params: np.ndarray) -> np.ndarray:
                fg_model = model(params[:-1])
                return np.column_stack((-fg_model[:, None] * v / sigma[:, None],
                                        -template / sigma))
            try:
                joint = least_squares(joint_residual, np.r_[opt.x, 1.],
                                      jac=joint_jac, max_nfev=500,
                                      xtol=1e-12, ftol=1e-12, gtol=1e-12)
                if not joint.success or not np.all(np.isfinite(joint.x)):
                    raise ValueError("joint foreground/template fit did not converge")
                amplitude = float(joint.x[-1])
                spectra[name] = FittedSpectrum(
                    "ok", f.tolist(), amplitude,
                    (amplitude * protocol.signal_k).tolist(),
                    (amplitude * template).tolist(),
                    float(joint.fun @ joint.fun))
            except (FloatingPointError, ValueError, OverflowError) as exc:
                spectra[name] = FittedSpectrum("fit_failed", f.tolist(), detail=str(exc))
    return ScoreResult(min(s["score"] for s in scenarios.values()), "ok", "",
                       scenarios, noise_method, spectra)


def score_beam(beam, protocol: Protocol, kernel=None,
               *, nside: int | None = None, fit_spectrum: bool = False) -> ScoreResult:
    """Score a packed-alm beam, or a legacy angular-grid beam.

    ``BeamModes`` uses limtod_jax m-modes and per-bin thermal noise; reuse a
    ``HarmonicScorer`` as ``kernel`` across many candidates. ``BeamPattern``
    retains the earlier angular-grid approximation and takes ``ZonalKernel``.
    """
    from .modes import BeamModes
    if isinstance(beam, BeamModes):
        from .harmonic import HarmonicScorer
        scorer = HarmonicScorer(protocol, beam.lmax) if kernel is None else kernel
        if not isinstance(scorer, HarmonicScorer) or scorer.protocol.fingerprint != protocol.fingerprint:
            raise ValueError("prepared harmonic scorer does not match protocol")
        return scorer.score(beam, fit_spectrum=fit_spectrum)
    if not isinstance(beam, BeamPattern):
        raise TypeError("beam must be BeamModes or BeamPattern")
    if not np.array_equal(beam.freqs_mhz, protocol.freqs_mhz):
        raise ValueError("beam and protocol frequency grids must match exactly")
    kernel = ZonalKernel.build(protocol, nside) if kernel is None else kernel
    kernel.validate_for(protocol)
    try:
        gain = beam.sample_healpix(kernel.nside)
    except ValueError as exc:
        return ScoreResult(0., "invalid_beam", str(exc), {}, "lst_mean_tsys")
    fg, h, eta = foreground_fast(gain, kernel, protocol)
    if np.any(eta > 1.001) or np.any(eta <= 0):
        return ScoreResult(0., "invalid_efficiency", "radiation efficiency outside allowed range", {}, "lst_mean_tsys")
    noise_scale = np.sqrt(np.sum(1.0 / protocol.visits)) / (
        360.0 * np.sqrt(protocol.bandwidth_hz * protocol.integration_s))
    noise = {name: (spectrum + protocol.receiver_k) * noise_scale for name, spectrum in fg.items()}
    return _score_spectra(fg, h, eta, noise, protocol, "lst_mean_tsys",
                          fit_spectrum=fit_spectrum)
