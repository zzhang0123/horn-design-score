import numpy as np

from horn_design_score import BeamPattern, ZonalKernel, score_beam


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
