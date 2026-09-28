"""Replay the matched-beam-factor RHINO survey setup from limTOD T-006 caches.

This is a deterministic fixed-template diagnostic, not the survey's Bayesian
21cmVAE fit. It reads existing limTOD study products without modifying them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares


SKIES = {"cnnpl+ps": ("cnnpl", "gleam"), "gsm+ps": ("gsm", "gleam")}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fit(foreground: np.ndarray, sigma: np.ndarray, factor: np.ndarray,
         signal: np.ndarray, freqs: np.ndarray, order: int) -> dict:
    v = np.vander(np.log(freqs / 70.0), order + 1, increasing=True)
    model = lambda a: factor * np.exp(v @ a)
    a0 = np.linalg.lstsq(v, np.log(foreground / factor), rcond=None)[0]
    fit = least_squares(lambda a: (foreground - model(a)) / sigma, a0,
                        method="lm", max_nfev=20000, xtol=1e-14,
                        ftol=1e-14, gtol=1e-14)
    if not fit.success:
        raise RuntimeError("foreground fit failed")
    q, _ = np.linalg.qr(model(fit.x)[:, None] * v / sigma[:, None])
    u = signal / sigma
    u -= q @ (q.T @ u)
    residual = (foreground - model(fit.x)) / sigma
    residual -= q @ (q.T @ residual)
    information = float(u @ u)
    residual_chi2 = float(residual @ residual)
    injected = foreground + signal
    joint = least_squares(
        lambda z: (injected - model(z[:-1]) - z[-1] * signal) / sigma,
        np.r_[fit.x, 1.0], method="lm", max_nfev=20000,
        xtol=1e-14, ftol=1e-14, gtol=1e-14,
    )
    if not joint.success:
        raise RuntimeError("joint foreground and signal fit failed")
    amplitude = float(joint.x[-1])
    return {
        "score": float(np.sqrt(information / (1.0 + residual_chi2))),
        "expected_delta_chi2": information,
        "foreground_residual_chi2": residual_chi2,
        "fitted_amplitude": amplitude,
        "fitted_intrinsic_spectrum_k": (amplitude * signal).tolist(),
        "joint_residual_chi2": float(joint.fun @ joint.fun),
        "sigma_d0_70_mK": float(1000 * sigma[15]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True,
                        help="limTOD/studies directory containing both mmode_d0_21cm studies")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--order", type=int, default=5, choices=(3, 4, 5, 6))
    args = parser.parse_args()

    root = args.study_root.expanduser()
    first = root / "mmode_d0_21cm_8h"
    year = root / "mmode_d0_21cm_lstint"
    sky_cache = first / "cache"
    site_cache = year / "cache"
    tod_path = site_cache / "tod_jbo.npz"
    weights_path = site_cache / "weights_jbo_0.npy"
    sky_paths = {
        component: sky_cache / f"sky_{component}_eq_n64_g256_55-85x31.npy"
        for component in {c for parts in SKIES.values() for c in parts}
    }
    required = [first / "config.py", first / "signal21.py", tod_path,
                weights_path, *sky_paths.values()]
    missing = [path for path in required if not path.is_file()]
    if missing:
        parser.error(
            "limTOD's RHINO study source and generated caches are not included "
            "in its Git checkout; this local replay requires them. Missing: "
            + ", ".join(str(path) for path in missing)
        )
    freqs = np.arange(55.0, 86.0)
    input_paths = [tod_path, weights_path]
    sys.path.insert(0, str(first))
    import signal21  # limTOD study's original fiducial 21cmVAE spectrum

    signal = signal21.t21_fid(freqs)
    weights = np.load(weights_path)
    if weights.shape != (360,) or np.any(weights <= 0):
        raise ValueError("expected 360 positive Jodrell Bank LST-bin visit counts")
    sky_means = {}
    for component, path in sky_paths.items():
        cube = np.load(path, mmap_mode="r")
        if cube.shape != (31, 49152):
            raise ValueError(f"unexpected sky cube shape: {path}")
        sky_means[component] = np.mean(cube, axis=1)
        input_paths.append(path)

    results = {}
    with np.load(tod_path, allow_pickle=False) as data:
        if data["lst"].shape != (360,):
            raise ValueError("expected a closed 360-bin LST scan")
        for beam in ("dry", "wet"):
            scenarios = {}
            for name, parts in SKIES.items():
                tod = sum(data[f"tod_{beam}_{component}"] for component in parts)
                foreground = tod.mean(axis=0)
                mono = sum(sky_means[component] for component in parts)
                matched_factor = foreground / mono
                sigma = np.sqrt(np.sum(
                    (tod + 100.0)**2 / (1e6 * 240.0 * weights[:, None]), axis=0,
                )) / 360.0
                scenarios[name] = _fit(foreground, sigma, matched_factor,
                                       signal, freqs, args.order)
            results[beam] = {
                "score": min(s["score"] for s in scenarios.values()),
                "scenarios": scenarios,
            }

    record = {
        "setup": {
            "source": "limTOD mmode_d0_21cm_lstint, Jodrell Bank, Sun below 0 deg",
            "frequencies_mhz": freqs.tolist(),
            "foreground_order": args.order,
            "beam_factor": "matched C(nu) = d0_model / all_sky_mean_model",
            "signal": "study fiducial 21cmVAE, noiseless A=1 injection",
            "signal_sha256": hashlib.sha256(np.ascontiguousarray(signal).tobytes()).hexdigest(),
            "score": "sqrt(expected_delta_chi2 / (1 + foreground_residual_chi2))",
            "input_sha256": {str(path.relative_to(root)): _sha256(path)
                             for path in input_paths},
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    for beam, result in results.items():
        print(f"{beam}: FOM={result['score']:.6g}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
