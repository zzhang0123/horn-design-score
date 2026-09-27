import healpy as hp
import numpy as np

from horn_design_score import (
    BeamModes,
    HarmonicScorer,
    beam_modes_from_grid,
    reference_modes_forward,
    score_beam,
    Protocol,
)


def test_prepared_modes_roundtrip_and_score(analytic_inputs, tmp_path):
    grid, protocol = analytic_inputs
    raw = beam_modes_from_grid(grid, nside=4, lmax=7)
    assert isinstance(raw, BeamModes)
    prepared = raw.prepare(protocol.latitude_deg)
    assert prepared.sky_ref_alm.shape == (len(grid.freqs_mhz), hp.Alm.getsize(7))
    path = tmp_path / "modes.npz"
    prepared.save_npz(path)
    restored = BeamModes.load_npz(path)
    scorer = HarmonicScorer(protocol, 7)
    direct = score_beam(raw, protocol, scorer)
    cached = score_beam(restored, protocol, scorer)
    assert direct.status == cached.status == "ok"
    np.testing.assert_allclose(direct.value, cached.value, rtol=1e-12)
    assert not restored.full_alm.flags.writeable
    assert not restored.sky_ref_alm.flags.writeable
    fitted = scorer.score(restored, fit_spectrum=True)
    assert fitted.value == cached.value
    spectrum = fitted.fitted_spectra["analytic"]
    assert spectrum.status == "ok"
    np.testing.assert_allclose(spectrum.intrinsic_k, protocol.signal_k, atol=1e-8)
    np.testing.assert_allclose(spectrum.antenna_k,
                               scorer.forward(restored)[2] * protocol.signal_k,
                               atol=1e-8)


def test_harmonic_d0_and_noise_against_limtod(analytic_inputs):
    grid, protocol = analytic_inputs
    beam = beam_modes_from_grid(grid, nside=4, lmax=7).prepare(protocol.latitude_deg)
    fast_fg, fast_noise, fast_h, _ = HarmonicScorer(protocol, 7).forward(beam)
    slow_fg, slow_noise, slow_h = reference_modes_forward(beam, protocol, [0])
    np.testing.assert_allclose(fast_fg["analytic"][0], slow_fg["analytic"][0], rtol=1e-10)
    np.testing.assert_allclose(fast_noise["analytic"][0], slow_noise["analytic"][0], rtol=1e-10)
    np.testing.assert_allclose(fast_h[0], slow_h[0], rtol=1e-10)
    np.testing.assert_allclose(fast_h[0], 0.5, atol=1e-4)
    np.testing.assert_allclose(fast_fg["analytic"][0],
                               protocol.sky_maps_k["analytic"][0, 0] * 0.5,
                               rtol=1e-4)


def test_harmonic_phase_with_anisotropic_sky_and_visits(analytic_inputs):
    _, base = analytic_inputs
    nside = 4
    theta, phi = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    local_gain = 0.55 * (1 + 0.25 * np.sin(theta) * np.cos(phi))
    local_alm = hp.map2alm(local_gain, lmax=7)
    beam = BeamModes(base.freqs_mhz, np.tile(local_alm, (len(base.freqs_mhz), 1)),
                     nside, 7).prepare(base.latitude_deg)
    sky = base.sky_maps_k["analytic"] * (1 + 0.25 * np.cos(phi) + 0.1 * np.sin(phi))[None, :]
    visits = 180 * (1 + 0.4 * np.sin(np.deg2rad(np.arange(360) + 0.5)))
    protocol = Protocol(base.freqs_mhz, {"anisotropic": sky}, base.signal_k,
                        visits, ground_k=0., loss_k=0., foreground_order=2)
    fast_fg, fast_noise, _, _ = HarmonicScorer(protocol, 7).forward(beam)
    slow_fg, slow_noise, _ = reference_modes_forward(beam, protocol, [0])
    np.testing.assert_allclose(fast_fg["anisotropic"][0], slow_fg["anisotropic"][0], rtol=1e-10)
    np.testing.assert_allclose(fast_noise["anisotropic"][0], slow_noise["anisotropic"][0], rtol=1e-10)
