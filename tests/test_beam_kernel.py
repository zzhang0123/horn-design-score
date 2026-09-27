import healpy as hp
import numpy as np
import pytest

from horn_design_score import BeamPattern, ZonalKernel
from horn_design_score.kernel import foreground_fast


def test_isotropic_power_gain_and_horizon(analytic_inputs):
    beam, protocol = analytic_inputs
    gain = beam.sample_healpix(4)
    np.testing.assert_allclose(gain, 1, atol=1e-14)
    kernel = ZonalKernel.build(protocol)
    foreground, h, eta = foreground_fast(gain, kernel, protocol)
    np.testing.assert_allclose(eta, 1, atol=1e-14)
    np.testing.assert_allclose(h, 0.5, atol=1e-14)
    np.testing.assert_allclose(foreground["analytic"], 500 * (protocol.freqs_mhz / 70.)**-2.5,
                               rtol=1e-13)


def test_zonal_average_removes_ra_harmonic(analytic_inputs):
    _, protocol = analytic_inputs
    theta, phi = hp.pix2ang(4, np.arange(hp.nside2npix(4)))
    sky = 1000 + 100 * np.cos(theta) + 70 * np.cos(phi)
    from horn_design_score import Protocol
    p = Protocol(protocol.freqs_mhz, {"sky": np.tile(sky, (12, 1))},
                 protocol.signal_k, protocol.visits)
    kernel = ZonalKernel.build(p, method="ring")
    x, _, z = hp.pix2vec(4, np.arange(hp.nside2npix(4)))
    sin_dec = np.sin(np.deg2rad(p.latitude_deg)) * z - np.cos(np.deg2rad(p.latitude_deg)) * x
    # At the two polar caps interpolation uses the nearest HEALPix ring.
    expected = 1000 + 100 * np.clip(sin_dec, -0.9791666667, 0.9791666667)
    np.testing.assert_allclose(kernel.mean_sky_k["sky"][0], expected, atol=1e-7)


def test_reject_nonphysical_and_incomplete_beam(analytic_inputs):
    beam, _ = analytic_inputs
    too_large = BeamPattern(beam.freqs_mhz, beam.theta_deg, beam.phi_deg,
                            beam.gain_theta_dbi + 10, beam.gain_phi_dbi + 10)
    with pytest.raises(ValueError, match="efficiency"):
        too_large.sample_healpix(4)
    with pytest.raises(ValueError, match="theta"):
        BeamPattern(beam.freqs_mhz, beam.theta_deg[:-1], beam.phi_deg,
                    beam.gain_theta_dbi[:, :-1], beam.gain_phi_dbi[:, :-1])
