"""Compare RHINO ground-plane CST wet/dry horns under one analytic protocol.

CST exports contain directivity, not accepted-power efficiency. This example
scores their normalized angular shapes and keeps raw CST data outside the repo.
The default phi mapping is an unverified horn-mount assumption.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import perf_counter

import healpy as hp
import numpy as np

from limTOD.cstbeam import cst_beam_maps, cst_frequency_table
from horn_design_score import BeamModes, HarmonicScorer, Protocol, __version__


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cst-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nside", type=int, default=32)
    parser.add_argument("--lmax", type=int)
    parser.add_argument("--phi0-deg", type=float, default=0.0)
    parser.add_argument("--phi-sense", choices=("ccw", "cw"), default="ccw")
    args = parser.parse_args()

    lmax = 3 * args.nside - 1 if args.lmax is None else args.lmax
    if not hp.isnsideok(args.nside):
        parser.error("--nside must be a valid HEALPix resolution")
    if lmax < 0 or lmax > min(3 * args.nside - 1, 359):
        parser.error("--lmax must be between 0 and min(3*nside-1, 359)")
    paths = {
        "dry": args.cst_root.expanduser() / "HornDryGround",
        "wet": args.cst_root.expanduser() / "HornWetGround",
    }
    tables = {name: cst_frequency_table(path) for name, path in paths.items()}
    # Use the exact common integer-MHz exports: 1 MHz noise channels cannot
    # be treated as independent on an overlapping half-MHz centre grid.
    common = sorted(set(tables["dry"]) & set(tables["wet"]))
    freqs = np.array([freq for freq in common if freq.is_integer()], dtype=float)
    if freqs.size < 8:
        raise ValueError("wet/dry CST directories need at least eight common integer-MHz channels")

    start = perf_counter()
    beams: dict[str, BeamModes] = {}
    for name, path in paths.items():
        maps = cst_beam_maps(path, freqs, nside=args.nside,
                             phi0_deg=args.phi0_deg, phi_sense=args.phi_sense)
        alms = np.stack([hp.map2alm(row, lmax=lmax, iter=3) for row in maps])
        beams[name] = BeamModes(freqs, alms, args.nside, lmax, convention="shape_only")

    protocol = Protocol.default(beams["dry"])
    scorer = HarmonicScorer(protocol, lmax)
    results = {
        name: asdict(scorer.score(beam, fit_spectrum=True))
        for name, beam in beams.items()
    }
    record = {
        "setup": {
            "frequencies_mhz": freqs.tolist(),
            "nside": args.nside,
            "lmax": lmax,
            "beam_convention": "shape_only",
            "phi0_deg": args.phi0_deg,
            "phi_sense": args.phi_sense,
            "phi_mapping_verified": False,
            "protocol_id": protocol.protocol_id,
            "protocol_fingerprint": protocol.fingerprint,
            "package_version": __version__,
            "cst_sha256": {
                name: {str(freq): _sha256(table[freq]) for freq in freqs}
                for name, table in tables.items()
            },
        },
        "results": results,
        "elapsed_s": perf_counter() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    for name, result in results.items():
        print(f"{name}: {result['status']}, FOM={result['value']:.6g}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
