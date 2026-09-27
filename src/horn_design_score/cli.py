"""Command line entry point for reproducible scores and a synthetic demo."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from dataclasses import asdict
from pathlib import Path

import healpy as hp
import numpy as np

from .beam import BeamPattern
from .modes import BeamModes, beam_modes_from_grid
from .harmonic import HarmonicScorer
from .kernel import ZonalKernel
from .protocol import Protocol
from .reference import reference_score, reference_modes_score
from .score import score_beam
from . import __version__


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _demo(directory: Path) -> None:
    """Write explicitly synthetic inputs; these are never a science reference."""
    directory.mkdir(parents=True, exist_ok=True)
    f = np.arange(55., 121.)
    theta = np.linspace(0., 180., 91)
    phi = np.arange(0., 360., 5.)
    # A broad zenith beam with accepted-power efficiency below unity.
    power = 1.8 * np.exp(-0.5 * (theta / 45.)**2)
    gt = np.broadcast_to(10 * np.log10(power)[None, :, None],
                         (f.size, theta.size, phi.size)).copy()
    gp = np.full_like(gt, -np.inf)
    np.savez(directory / "demo_beam.npz", freqs_mhz=f, theta_deg=theta, phi_deg=phi,
             gain_theta_dbi=gt, gain_phi_dbi=gp, convention="accepted_power")
    nside = 8
    ang, ra = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    dec_shape = 1 + 0.2 * np.cos(ang) + 0.1 * np.cos(ra)
    index_shape = 2.5 + 0.1 * np.sin(ra) * np.sin(ang)
    sky_a = 1500 * dec_shape[None, :] * (f[:, None] / 70.)**(-index_shape[None, :])
    sky_b = 1600 * dec_shape[None, :] * (f[:, None] / 70.)**(-2.55 - 0.08 * np.cos(ang)[None, :])
    signal = -0.14 * np.exp(-0.5 * ((f - 72.) / 10.)**2)
    visits = np.full(360, 181.)
    np.savez(directory / "demo_protocol.npz", freqs_mhz=f, signal_k=signal,
             visits=visits, sky_analytic_a=sky_a, sky_analytic_b=sky_b)
    grid = BeamPattern.load_npz(str(directory / "demo_beam.npz"))
    beam_modes_from_grid(grid, nside=8).save_npz(directory / "demo_beam_modes.npz")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="horn-score")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("make-demo", help="write small synthetic inputs, not a science protocol")
    demo.add_argument("directory", type=Path)
    score = sub.add_parser("score", help="score one beam against a frozen protocol NPZ")
    score.add_argument("--beam", required=True, type=Path)
    score.add_argument("--protocol", required=True, type=Path)
    score.add_argument("--nside", type=int)
    score.add_argument("--kernel-cache", type=Path, help="reuse the one-time m=0 sky projection")
    score.add_argument("--reference", action="store_true", help="slow limTOD TOD and m-mode check")
    score.add_argument("--fitted-spectrum", action="store_true",
                       help="include joint-fit 21 cm template spectrum for each sky scenario")
    score.add_argument("--require-reference-band", action="store_true", help="require 55..120 MHz at 1 MHz spacing; sky and signal are not verified")
    score.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "make-demo":
        _demo(args.directory)
        print(args.directory.resolve())
        return 0
    with np.load(args.beam, allow_pickle=False) as data:
        harmonic = "full_alm" in data
    beam = BeamModes.load_npz(args.beam) if harmonic else BeamPattern.load_npz(str(args.beam))
    if harmonic and args.nside is not None and args.nside != beam.nside:
        parser.error("--nside must match the packed-alm beam nside")
    protocol = Protocol.load_npz(args.protocol, require_reference_band=args.require_reference_band)
    kernel = None
    if harmonic and not args.reference:
        if args.kernel_cache:
            parser.error("--kernel-cache currently applies only to angular-grid input; reuse HarmonicScorer in Python")
        kernel = HarmonicScorer(protocol, beam.lmax)
    elif not args.reference:
        if args.kernel_cache and args.kernel_cache.exists():
            kernel = ZonalKernel.load_npz(args.kernel_cache, protocol)
        else:
            kernel = ZonalKernel.build(protocol, args.nside)
            if args.kernel_cache:
                kernel.save_npz(args.kernel_cache)
    if args.reference and args.nside is not None and args.nside != protocol.sky_nside:
        parser.error("reference requires the protocol sky nside")
    result = ((reference_modes_score(beam, protocol, fit_spectrum=args.fitted_spectrum)
               if harmonic else reference_score(beam, protocol, fit_spectrum=args.fitted_spectrum))
              if args.reference else score_beam(beam, protocol, kernel,
                                                fit_spectrum=args.fitted_spectrum))
    record = asdict(result)
    if result.fitted_spectra is None:
        record.pop("fitted_spectra")
    record["provenance"] = {"beam_sha256": _sha256(args.beam),
                            "protocol_sha256": _sha256(args.protocol),
                            "package_version": __version__,
                            "limtod_version": version("limTOD"),
                            "protocol_id": protocol.protocol_id,
                            "protocol_fingerprint": protocol.fingerprint,
                            "nside": protocol.sky_nside if args.nside is None else args.nside,
                            "lmax": beam.lmax if harmonic else None,
                            "horizon_rule": "half_horizon_pixel" if harmonic else "zonal_grid",
                            "beam_convention": beam.convention,
                            "input_kind": "beam_alm" if harmonic else "angular_grid"}
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
    if args.output:
        args.output.write_text(payload + "\n")
    else:
        print(payload)
    return 0 if result.status == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
