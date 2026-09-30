import numpy as np

from horn_design_score import BeamPattern, Protocol, ZonalKernel, score_beam
from horn_design_score.score import _score_spectra


def test_exact_smooth_foreground_has_no_residual(analytic_inputs):
    beam, protocol = analytic_inputs
    result = score_beam(beam, protocol, ZonalKernel.build(protocol))
    assert result.status == "ok"
    scenario = result.scenarios["analytic"]
    assert result.value > 0
    assert scenario["residual_chi2"] < 1e-12
    assert abs(scenario["amplitude_bias"]) < 1e-8
    np.testing.assert_allclose(result.value, np.sqrt(scenario["information"]), rtol=1e-12)


def test_lower_efficiency_lowers_information(analytic_inputs):
    beam, protocol = analytic_inputs
    reference = score_beam(beam, protocol)
    low = BeamPattern(beam.freqs_mhz, beam.theta_deg, beam.phi_deg,
                      beam.gain_theta_dbi - 10 * np.log10(2),
                      beam.gain_phi_dbi - 10 * np.log10(2))
    result = score_beam(low, protocol)
    assert result.status == "ok"
    assert result.scenarios["analytic"]["information"] < reference.scenarios["analytic"]["information"]
    assert result.scenarios["analytic"]["eta_max"] == 0.5


def test_optional_joint_fit_recovers_template_without_changing_score(analytic_inputs):
    beam, protocol = analytic_inputs
    plain = score_beam(beam, protocol)
    fitted = score_beam(beam, protocol, fit_spectrum=True)
    assert plain.fitted_spectra is None
    assert fitted.value == plain.value
    spectrum = fitted.fitted_spectra["analytic"]
    assert spectrum.status == "ok"
    np.testing.assert_allclose(spectrum.frequencies_mhz, protocol.freqs_mhz)
    np.testing.assert_allclose(spectrum.amplitude, 1., atol=1e-8)
    np.testing.assert_allclose(spectrum.intrinsic_k, protocol.signal_k, atol=1e-9)
    np.testing.assert_allclose(spectrum.antenna_k,
                               fitted.scenarios["analytic"]["sky_throughput_mean"] * protocol.signal_k,
                               atol=1e-9)


def test_matched_beam_factor_is_used_in_foreground_and_joint_fits():
    freqs = np.arange(55., 75.)
    monopole = 1000. * (freqs / 70.)**-2.5
    factor = 1. + 0.04 * np.sin((freqs - 55.) * 0.9)
    foreground = {"sky": factor * monopole}
    skies = {"sky": np.repeat(monopole[:, None], 12, axis=1)}
    signal = -0.1 * np.exp(-0.5 * ((freqs - 65.) / 3.)**2)
    common = dict(freqs_mhz=freqs, sky_maps_k=skies, signal_k=signal,
                  visits=np.ones(360), foreground_order=2)
    matched = Protocol(**common, foreground_model="matched_beam_factor")
    plain = Protocol(**common, foreground_model="plain_log_polynomial")
    noise = {"sky": np.ones_like(freqs)}
    args = (foreground, np.ones_like(freqs), np.ones_like(freqs), noise)
    corrected = _score_spectra(*args, matched, "test", fit_spectrum=True)
    uncorrected = _score_spectra(*args, plain, "test")
    assert corrected.status == "ok"
    assert corrected.scenarios["sky"]["residual_chi2"] < 1e-16
    assert uncorrected.scenarios["sky"]["residual_chi2"] > 1.
    assert abs(corrected.fitted_spectra["sky"].amplitude - 1.) < 1e-8
