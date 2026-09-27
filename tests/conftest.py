import numpy as np
import healpy as hp
import pytest

from horn_design_score import BeamPattern, Protocol


@pytest.fixture
def analytic_inputs():
    f = np.arange(55., 67.)
    theta = np.linspace(0., 180., 37)
    phi = np.arange(0., 360., 10.)
    # G_theta = G_phi = 0.5: total accepted-power gain is exactly one.
    half_dbi = 10 * np.log10(0.5)
    gain = np.full((len(f), len(theta), len(phi)), half_dbi)
    beam = BeamPattern(f, theta, phi, gain, gain)
    nside = 4
    npix = hp.nside2npix(nside)
    sky = (1000 * (f / 70.)**-2.5)[:, None] * np.ones((1, npix))
    signal = -0.14 * np.exp(-0.5 * ((f - 60.) / 2.)**2)
    protocol = Protocol(f, {"analytic": sky}, signal, np.full(360, 180.),
                        ground_k=0., loss_k=0., foreground_order=2)
    return beam, protocol
