import json

import healpy as hp
import numpy as np
import pytest

from horn_design_score import (BeamEfficiencyError, BeamModes, HarmonicScorer, Protocol,
                               reference_modes_forward, reference_modes_score, score_beam)
from horn_design_score.cli import main

NSIDE, LMAX = 8, 23
FREQS = np.arange(55., 67.)
ROOT_4PI = np.sqrt(4 * np.pi)


def _beam(scale: float | np.ndarray = 0.9, *, convention: str = "accepted_power") -> BeamModes:
    """Chromatic, azimuthally asymmetric beam with a visible back lobe.

    ``scale`` is the pixel-mean gain, one value or one per channel.
    """
    theta, phi = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    width = np.deg2rad(60. * 60. / FREQS) / 2.355
    maps = np.exp(-0.5 * (theta[None, :] / width[:, None])**2) + 0.02
    maps = maps * (1 + 0.2 * np.sin(theta) * np.cos(phi))[None, :]
    maps *= np.reshape(scale, (-1, 1)) / maps.mean(axis=1, keepdims=True)
    alm = np.stack([hp.map2alm(row, lmax=LMAX, iter=3) for row in maps])
    return BeamModes(FREQS, alm, NSIDE, LMAX, convention=convention)


def _isotropic(efficiency: float, *, convention: str = "accepted_power") -> BeamModes:
    alm = np.zeros((FREQS.size, hp.Alm.getsize(LMAX)), dtype=complex)
    alm[:, 0] = efficiency * ROOT_4PI
    return BeamModes(FREQS, alm, NSIDE, LMAX, convention=convention)


def _offset_gaussian(centre_deg: float, fwhm_deg: float, *, upward_only: bool,
                     efficiency: float) -> BeamModes:
    """Nside-4 Gaussian lobe centred ``centre_deg`` from zenith, a00 set to ``efficiency``."""
    nside, lmax = 4, 11
    vec = np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))
    centre = np.array([np.sin(np.deg2rad(centre_deg)), 0., np.cos(np.deg2rad(centre_deg))])
    angle = np.arccos(np.clip(centre @ vec, -1, 1))
    gain = np.exp(-0.5 * (angle / (np.deg2rad(fwhm_deg) / 2.355))**2)
    if upward_only:
        gain = gain * (vec[2] > 0)
    alm = hp.map2alm(gain, lmax=lmax, iter=3)
    alm = alm * (efficiency * ROOT_4PI / alm[0].real)
    return BeamModes(FREQS, np.tile(alm, (FREQS.size, 1)), nside, lmax)


def _prepared_kwargs(beam: BeamModes, *, scale: float = 1., partition: np.ndarray | None = None) -> dict:
    """Prepared arrays of ``beam``, rescaled, with the sky monopole optionally replaced."""
    sky = beam.sky_alm * scale
    h = beam.h_partition * scale
    if partition is not None:
        sky[:, 0] = partition * ROOT_4PI
        h = partition
    return dict(full_alm=beam.full_alm * scale, sky_alm=sky, eta_rad=beam.eta_rad * scale,
                h_partition=h)


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


@pytest.mark.parametrize("excess, accepted", [(1e-3 - 1e-9, True), (1e-3 + 1e-9, False)])
def test_prepared_partition_tolerance(protocol, excess, accepted):
    beam = _beam().prepare(protocol.latitude_deg)
    kwargs = _prepared_kwargs(beam, partition=beam.eta_rad + excess)
    if accepted:
        BeamModes(FREQS, nside=NSIDE, lmax=LMAX, **kwargs)
    else:
        with pytest.raises(BeamEfficiencyError, match="sky partition"):
            BeamModes(FREQS, nside=NSIDE, lmax=LMAX, **kwargs)


def test_prepared_arrays_outside_physical_range_raise_efficiency_error(protocol):
    beam = _beam().prepare(protocol.latitude_deg)
    with pytest.raises(BeamEfficiencyError, match="exceeds unity"):
        BeamModes(FREQS, nside=NSIDE, lmax=LMAX, **_prepared_kwargs(beam, scale=1.2))
    with pytest.raises(BeamEfficiencyError, match="sky partition"):
        BeamModes(FREQS, nside=NSIDE, lmax=LMAX,
                  **_prepared_kwargs(beam, partition=np.full(FREQS.size, -1e-6)))


@pytest.mark.parametrize("field", ["eta_rad", "h_partition"])
def test_prepared_arrays_must_equal_alm_monopoles(protocol, field, tmp_path):
    """Files prepared by 0.6.0 or earlier stored pixel means, off by 1e-4 or more."""
    beam = _beam().prepare(protocol.latitude_deg)
    kwargs = _prepared_kwargs(beam)
    kwargs[field] = kwargs[field] + 1e-4
    with pytest.raises(ValueError, match="prepare") as excinfo:
        BeamModes(FREQS, nside=NSIDE, lmax=LMAX, **kwargs)
    assert not isinstance(excinfo.value, BeamEfficiencyError)

    path = tmp_path / "stale.npz"
    np.savez(path, freqs_mhz=FREQS, nside=NSIDE, lmax=LMAX, sky_ref_alm=beam.sky_ref_alm,
             convention="accepted_power", ref_latitude_deg=beam.ref_latitude_deg, **kwargs)
    with pytest.raises(ValueError, match="prepare"):
        BeamModes.load_npz(path)


def test_prepare_reuses_prepared_arrays(protocol):
    scorer = HarmonicScorer(protocol, LMAX)
    fresh = _beam().prepare(protocol.latitude_deg)
    local = BeamModes(FREQS, nside=NSIDE, lmax=LMAX, **_prepared_kwargs(fresh))
    assert local.sky_ref_alm is None
    again = local.prepare(protocol.latitude_deg)
    np.testing.assert_array_equal(again.eta_rad, fresh.eta_rad)
    np.testing.assert_array_equal(again.h_partition, fresh.h_partition)
    np.testing.assert_allclose(again.sky_ref_alm, fresh.sky_ref_alm, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(scorer.score(local).value, scorer.score(fresh).value, rtol=1e-12)

    moved = fresh.prepare(10.)
    assert moved.ref_latitude_deg == 10.
    np.testing.assert_array_equal(moved.eta_rad, fresh.eta_rad)
    np.testing.assert_array_equal(moved.sky_alm, fresh.sky_alm)
    assert not np.allclose(moved.sky_ref_alm, fresh.sky_ref_alm)


def test_three_temperatures_fix_each_term_of_d0():
    """Distinct sky, ground and loss temperatures, and an efficiency that varies by channel."""
    t_sky, t_ground, t_loss = 250., 290., 310.
    sky = np.full((FREQS.size, hp.nside2npix(NSIDE)), t_sky)
    protocol = Protocol(FREQS, {"sky": sky}, np.zeros(FREQS.size), np.full(360, 180.),
                        ground_k=t_ground, loss_k=t_loss)
    beam = _beam(np.linspace(0.6, 0.9, FREQS.size)).prepare(protocol.latitude_deg)
    d0, _, throughput, eta = HarmonicScorer(protocol, LMAX).forward(beam)
    assert np.ptp(eta) > 0.25

    def expected(h, efficiency):
        return h * t_sky + (efficiency - h) * t_ground + (1 - efficiency) * t_loss

    np.testing.assert_allclose(d0["sky"], expected(throughput, eta), rtol=1e-12)
    subset = [7, 2]
    slow_d0, _, slow_throughput = reference_modes_forward(beam, protocol, subset)
    np.testing.assert_allclose(slow_throughput, throughput[subset], rtol=1e-10)
    np.testing.assert_allclose(slow_d0["sky"], expected(slow_throughput, eta[subset]), rtol=1e-10)


def test_throughput_above_efficiency_scores_zero_on_both_paths():
    beam = _offset_gaussian(20., 25., upward_only=True, efficiency=0.995)
    protocol = Protocol.default(beam)
    prepared = beam.prepare(protocol.latitude_deg)  # the stored monopoles pass
    scorer = HarmonicScorer(protocol, beam.lmax)
    _, _, throughput, eta = scorer.forward(prepared)
    assert np.all(prepared.h_partition < eta + 1e-3)
    assert np.all(throughput > eta + 1e-3)
    for result in (scorer.score(prepared), reference_modes_score(prepared, protocol)):
        assert (result.value, result.status) == (0., "invalid_efficiency")
        assert "throughput" in result.detail
        assert result.scenarios == {}


def test_nonpositive_throughput_has_one_status_on_both_paths():
    beam = _offset_gaussian(165., 30., upward_only=False, efficiency=0.9)
    protocol = Protocol.default(beam)
    prepared = beam.prepare(protocol.latitude_deg)
    scorer = HarmonicScorer(protocol, beam.lmax)
    assert np.all(prepared.h_partition > 0)
    assert np.all(scorer.forward(prepared)[2] <= 0)
    for result in (scorer.score(prepared), reference_modes_score(prepared, protocol)):
        assert (result.value, result.status) == (0., "invalid_sky_throughput")


def test_reference_modes_score_reports_invalid_efficiency(protocol):
    result = reference_modes_score(_isotropic(1.17), protocol)
    assert (result.value, result.status) == (0., "invalid_efficiency")
    assert result.noise_method == "per_bin_tsys_limtod"


def test_shape_only_beam_with_zero_integral_scores_zero(protocol):
    result = HarmonicScorer(protocol, LMAX).score(_isotropic(0., convention="shape_only"))
    assert (result.value, result.status) == (0., "invalid_efficiency")
    assert "zero integral" in result.detail


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
