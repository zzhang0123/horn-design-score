import json

import healpy as hp
import numpy as np

from horn_design_score.cli import main


def test_cli_scores_bundled_default_without_protocol_file(tmp_path):
    freqs = np.arange(60., 90., 2.)
    lmax = 3
    alm = np.zeros((freqs.size, hp.Alm.getsize(lmax)), dtype=complex)
    alm[:, 0] = np.sqrt(4 * np.pi) / 2
    beam_path = tmp_path / "beam.npz"
    output_path = tmp_path / "score.json"
    np.savez(beam_path, freqs_mhz=freqs, full_alm=alm, nside=4, lmax=lmax)

    assert main(["score", "--beam", str(beam_path), "--default-protocol",
                 "--output", str(output_path)]) == 0
    result = json.loads(output_path.read_text())
    assert result["status"] == "ok"
    assert np.isfinite(result["value"])
    assert set(result["scenarios"]) == {"analytic_a", "analytic_b"}
    assert result["provenance"]["protocol_id"] == "analytic-default-v3"
    assert result["provenance"]["foreground_model"] == "edges_beam_factor"
    assert result["provenance"]["beam_factor_reference_mhz"] == 74.
    assert result["provenance"]["protocol_source"] == "analytic_default"
    assert result["provenance"]["protocol_sha256"] is None
