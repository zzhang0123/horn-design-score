import json

import healpy as hp
import numpy as np
import pytest

from horn_design_score import (BeamEfficiencyError, BeamModes, HarmonicScorer, Protocol,
                               reference_modes_forward, score_beam)
from horn_design_score.cli import main

NSIDE, LMAX = 8, 23
FREQS = np.arange(55., 67.)
ROOT_4PI = np.sqrt(4 * np.pi)


def _beam(scale: float = 0.9, *, convention: str = "accepted_power") -> BeamModes:
    """Chromatic, azimuthally asymmetric beam with a visible back lobe."""
    theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    width = np.deg2rad(60. * 60. / FREQS) / 2.355
    maps = np.exp(-0.5 * (theta[None, :] / width[:, None])**2) + 0.02
    maps = maps * (1 + 0.2 * np.sin(theta) * np.cos(phi))[None, :]
    maps *= scale / maps.mean(axis=1, keepdims=True)
    alm = np.stack([hp.map2alm(row, lmax=LMAX, iter=3) for row in maps])
    return BeamModes(FREQS, alm, NSIDE, LMAX, convention=convention)


def _isotropic(efficiency: float) -> BeamModes:
    alm = np.zeros((FREQS.size, hp.Alm.getsize(LMAX)), dtype=complex)
    alm[:, 0] = efficiency * ROOT_4PI
    return BeamModes(FREQS, alm, NSIDE, LMAX)


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    return Protocol.default(_beam())


def test_prepared_efficiency_and_throughput_are_harmonic(protocol):
    beam = _beam().prepare(protocol.latitude_deg)
    np.testing.assert_allclose(beam.eta_rad, beam.full_alm[:, 0].real / ROOT_4PI, rtol=1e-14)
    np.testing.assert_allclose(beam.h_partition, beam.sky_alm[:, 0].real / ROOT_4PI, rtol=1e-14)
    _, _, throughput, eta = HarmonicScorer(protocol, LMAX).forward(beam)
    np.testing.assert_array_equal(eta, beam.eta_rad)
    # The scorer's response to a uniform sky adds pixel-quadrature leakage.
    np.testing.assert_allclose(throughput, beam.h_partition, atol=2e-3)
    assert np.all(throughput < eta)


def test_isothermal_sky_ground_and_loss_give_isothermal_d0():
    visits = 180 * (1 + 0.4 * np.sin(np.deg2rad(np.arange(360) + 0.5)))
    sky = np.full((FREQS.size, hp.nside2npix(NSIDE)), 300.)
    protocol = Protocol(FREQS, {"sky": sky}, np.zeros(FREQS.size), visits,
                        ground_k=300., loss_k=300.)
    beam = _beam().prepare(protocol.latitude_deg)
    fast = HarmonicScorer(protocol, LMAX).forward(beam)[0]["sky"]
    np.testing.assert_allclose(fast, 300., rtol=1e-12)
    slow = reference_modes_forward(beam, protocol, [7])[0]["sky"]
    np.testing.assert_allclose(slow, 300., rtol=1e-10)


def test_shape_only_beam_is_normalized_to_unit_harmonic_efficiency(protocol):
    beam = _beam(3.7, convention="shape_only").prepare(protocol.latitude_deg)
    np.testing.assert_array_equal(beam.eta_rad, 1.)
    np.testing.assert_allclose(beam.full_alm[:, 0].real / ROOT_4PI, 1., rtol=1e-12)
    assert HarmonicScorer(protocol, LMAX).score(beam).status == "ok"


@pytest.mark.parametrize("efficiency, status", [
    (0.9, "ok"),
    (1.0009, "ok"),                  # inside the 1.001 quadrature tolerance
    (1.0011, "invalid_efficiency"),  # just outside it
    (1.17, "invalid_efficiency"),
    (0.0, "invalid_efficiency"),
    (-0.5, "invalid_efficiency"),
])
def test_efficiency_boundary_sets_status(protocol, efficiency, status):
    beam = _isotropic(efficiency)
    scorer = HarmonicScorer(protocol, LMAX)
    result = scorer.score(beam)
    assert result.status == status
    assert score_beam(beam, protocol, scorer).status == status
    if status == "ok":
        assert result.value > 0
    else:
        assert result.value == 0.
        assert result.scenarios == {}
        assert result.detail
        with pytest.raises(BeamEfficiencyError):
            beam.prepare(protocol.latitude_deg)
        with pytest.raises(BeamEfficiencyError):
            scorer.forward(beam)


def test_chromatic_beam_with_excess_gain_scores_zero(protocol):
    result = HarmonicScorer(protocol, LMAX).score(_beam(1.17))
    assert (result.value, result.status) == (0., "invalid_efficiency")


def test_input_mismatch_still_raises(protocol):
    other = BeamModes(FREQS + 1., _beam().full_alm, NSIDE, LMAX)
    with pytest.raises(ValueError, match="frequency grids") as excinfo:
        HarmonicScorer(protocol, LMAX).score(other)
    assert not isinstance(excinfo.value, BeamEfficiencyError)


@pytest.mark.parametrize("excess, accepted", [(5e-4, True), (2e-3, False)])
def test_prepared_partition_tolerance(protocol, excess, accepted):
    beam = _beam().prepare(protocol.latitude_deg)
    args = (FREQS, beam.full_alm, NSIDE, LMAX)
    kwargs = dict(sky_alm=beam.sky_alm, eta_rad=beam.eta_rad,
                  h_partition=beam.eta_rad + excess)
    if accepted:
        BeamModes(*args, **kwargs)
    else:
        with pytest.raises(BeamEfficiencyError, match="sky partition"):
            BeamModes(*args, **kwargs)


def test_cli_writes_zero_score_for_invalid_efficiency(tmp_path):
    beam_path, output_path = tmp_path / "beam.npz", tmp_path / "score.json"
    np.savez(beam_path, freqs_mhz=FREQS, full_alm=_isotropic(1.2).full_alm,
             nside=NSIDE, lmax=LMAX)
    assert main(["score", "--beam", str(beam_path), "--default-protocol",
                 "--output", str(output_path)]) == 2
    record = json.loads(output_path.read_text())
    assert record["status"] == "invalid_efficiency"
    assert record["value"] == 0.
    assert record["provenance"]["protocol_id"] == "analytic-default-v3"
