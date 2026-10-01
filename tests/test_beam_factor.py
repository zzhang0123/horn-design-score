import healpy as hp
import numpy as np
import pytest

from horn_design_score import (BeamModes, BeamPattern, HarmonicScorer, Protocol,
                               reference_modes_score, score_beam)

NSIDE, LMAX = 8, 23
FREQS = np.arange(55., 86.)


def _gaussian_beam(fwhm_deg: np.ndarray) -> BeamModes:
    theta, _ = hp.pix2ang(NSIDE, np.arange(hp.nside2npix(NSIDE)))
    sigma = np.deg2rad(fwhm_deg) / 2.355
    maps = np.exp(-0.5 * (theta[None, :] / sigma[:, None])**2) + 1e-3
    maps *= 0.9 / maps.mean(axis=1, keepdims=True)
    alm = np.stack([hp.map2alm(row, lmax=LMAX, iter=3) for row in maps])
    return BeamModes(FREQS, alm, NSIDE, LMAX)


@pytest.fixture(scope="module")
def achromatic() -> BeamModes:
    return _gaussian_beam(np.full(FREQS.size, 60.))


@pytest.fixture(scope="module")
def chromatic() -> BeamModes:
    return _gaussian_beam(60. * 70. / FREQS)


@pytest.mark.parametrize("freqs, expected_mhz", [
    (np.arange(55., 86.), 75.),
    (np.arange(55., 67.), 66.),       # band below 75 MHz: upper edge
    (np.arange(100., 201., 10.), 100.),  # band above 75 MHz: lower edge
    (np.arange(60., 90., 2.), 74.),   # 74 and 76 tie: lower channel
])
def test_bcf_reference_is_channel_nearest_75_mhz(freqs, expected_mhz):
    protocol = Protocol(freqs, {"sky": np.ones((freqs.size, 12))},
                        np.zeros(freqs.size), np.ones(360))
    assert protocol.foreground_model == "edges_beam_factor"
    assert protocol.bcf_reference_mhz == expected_mhz
    assert freqs[protocol.bcf_reference_index] == expected_mhz


def test_edges_factor_is_beam_weighted_reference_sky_ratio(chromatic):
    protocol = Protocol.default(chromatic)
    j = protocol.bcf_reference_index
    # Independent route: the reference-frequency map at every channel, with
    # no ground or loss term, gives the numerator and denominator directly.
    template = {name: np.repeat(sky[j][None, :], FREQS.size, axis=0)
                for name, sky in protocol.sky_maps_k.items()}
    template_protocol = Protocol(FREQS, template, protocol.signal_k, protocol.visits,
                                 ground_k=0., loss_k=0.)
    expected = HarmonicScorer(template_protocol, LMAX).forward(chromatic)[0]
    factor = HarmonicScorer(protocol, LMAX).beam_factor(chromatic)
    assert set(factor) == set(protocol.sky_maps_k)
    for name, values in factor.items():
        np.testing.assert_allclose(values, expected[name] / expected[name][j], rtol=1e-12)
        assert values[j] == 1.
        assert np.ptp(values) > 1e-3


def test_edges_factor_is_exact_for_single_spectrum_sky(chromatic):
    base = Protocol.default(chromatic)
    j = base.bcf_reference_index
    spectrum = (FREQS / 75.)**-2.5
    skies = {name: spectrum[:, None] * sky[j][None, :]
             for name, sky in base.sky_maps_k.items()}
    common = (FREQS, skies, base.signal_k, base.visits)
    edges = score_beam(chromatic, Protocol(*common))
    plain = score_beam(chromatic, Protocol(*common, foreground_model="plain_log_polynomial"))
    assert edges.status == plain.status == "ok"
    for name in skies:
        # Ground and loss terms are 300 K here; the fit must remove them.
        assert edges.scenarios[name]["residual_chi2"] < 1e-8
        assert plain.scenarios[name]["residual_chi2"] > 1.


def test_edges_factor_keeps_chromatic_residual_that_matched_factor_cancels(achromatic, chromatic):
    protocol = Protocol.default(achromatic)
    scorer = HarmonicScorer(protocol, LMAX)
    flat, varying = scorer.score(achromatic), scorer.score(chromatic)
    assert flat.status == varying.status == "ok"
    assert varying.value < flat.value
    for name in protocol.sky_maps_k:
        assert (varying.scenarios[name]["residual_chi2"]
                > 10 * flat.scenarios[name]["residual_chi2"])

    matched = Protocol(FREQS, protocol.sky_maps_k, protocol.signal_k, protocol.visits,
                       foreground_model="matched_beam_factor")
    matched_scorer = HarmonicScorer(matched, LMAX)
    flat_m, varying_m = matched_scorer.score(achromatic), matched_scorer.score(chromatic)
    for name in protocol.sky_maps_k:
        np.testing.assert_allclose(varying_m.scenarios[name]["residual_chi2"],
                                   flat_m.scenarios[name]["residual_chi2"], rtol=1e-2)


def test_edges_fit_recovers_injected_template(chromatic):
    protocol = Protocol.default(chromatic)
    result = HarmonicScorer(protocol, LMAX).score(chromatic, fit_spectrum=True)
    plain = HarmonicScorer(protocol, LMAX).score(chromatic)
    assert result.value == plain.value
    for spectrum in result.fitted_spectra.values():
        assert spectrum.status == "ok"
        assert np.isfinite(spectrum.amplitude)


def test_grid_path_edges_factor_absorbs_throughput_ripple(analytic_inputs):
    beam, base = analytic_inputs
    efficiency = 0.8 + 0.05 * np.sin(np.arange(beam.freqs_mhz.size))
    gain = beam.gain_theta_dbi + 10 * np.log10(efficiency)[:, None, None]
    rippled = BeamPattern(beam.freqs_mhz, beam.theta_deg, beam.phi_deg, gain, gain)
    # Default ground_k = loss_k = 300 K, so the additive term ripples too.
    common = (base.freqs_mhz, base.sky_maps_k, base.signal_k, base.visits)
    edges = score_beam(rippled, Protocol(*common, foreground_order=2))
    plain = score_beam(rippled, Protocol(*common, foreground_order=2,
                                         foreground_model="plain_log_polynomial"))
    assert edges.status == plain.status == "ok"
    assert edges.scenarios["analytic"]["residual_chi2"] < 1e-8
    assert plain.scenarios["analytic"]["residual_chi2"] > 1.


def test_limtod_modes_reference_agrees_with_fast_edges_score(analytic_inputs):
    _, base = analytic_inputs
    nside, lmax = 4, 7
    freqs = base.freqs_mhz
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    # Beam whose tilt changes with frequency, on a sky with varying index.
    tilt = 0.25 * (freqs / 60.)[:, None]
    gain = 0.55 * (1 + tilt * (np.sin(theta) * np.cos(phi))[None, :])
    alm = np.stack([hp.map2alm(row, lmax=lmax) for row in gain])
    beam = BeamModes(freqs, alm, nside, lmax).prepare(base.latitude_deg)
    index = -2.5 + 0.1 * np.cos(theta) + 0.05 * np.sin(theta) * np.cos(phi)
    sky = 1000 * (1 + 0.25 * np.cos(phi)) * (freqs[:, None] / 70.)**index[None, :]
    protocol = Protocol(freqs, {"sky": sky}, base.signal_k, base.visits, foreground_order=2)
    fast = HarmonicScorer(protocol, lmax).score(beam)
    slow = reference_modes_score(beam, protocol)
    assert fast.status == slow.status == "ok"
    np.testing.assert_allclose(slow.value, fast.value, rtol=1e-6)
    np.testing.assert_allclose(slow.scenarios["sky"]["residual_chi2"],
                               fast.scenarios["sky"]["residual_chi2"], rtol=1e-4)
    assert fast.scenarios["sky"]["residual_chi2"] > 0
