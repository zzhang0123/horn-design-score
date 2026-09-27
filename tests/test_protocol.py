import numpy as np
import pytest

from horn_design_score import BeamModes, Protocol


def test_default_protocol_fixes_design_setup_and_roundtrips(tmp_path):
    freqs = np.arange(60., 90., 2.)
    beam_path = tmp_path / "beam.npz"
    np.savez(beam_path, freqs_mhz=freqs,
             full_alm=np.full((freqs.size, 1), np.sqrt(4 * np.pi) / 2),
             nside=1, lmax=0)
    beam = BeamModes.load_npz(beam_path)
    skies = {
        "cnn_pl_gleam": np.full((freqs.size, 12), 1500.),
        "gsm2008_gleam": np.full((freqs.size, 12), 1600.),
    }
    signal = -0.1 * np.exp(-0.5 * ((freqs - 75.) / 10.)**2)
    visits = np.full(360, 180.)

    protocol = Protocol.default(beam, skies, signal, visits)
    assert protocol.protocol_id == "jbo-default-v2"
    np.testing.assert_array_equal(protocol.freqs_mhz, beam.freqs_mhz)
    assert set(protocol.sky_maps_k) == set(skies)
    assert protocol.latitude_deg == 53.23625
    assert protocol.receiver_k == 100.
    assert protocol.ground_k == protocol.loss_k == 300.
    assert protocol.integration_s == 240.
    assert protocol.bandwidth_hz == 1e6
    assert protocol.foreground_order == 5

    path = tmp_path / "default.npz"
    protocol.save_npz(path)
    assert Protocol.load_npz(path).fingerprint == protocol.fingerprint

    with pytest.raises(ValueError, match="jbo-default-v2"):
        Protocol.default(beam, {"other": skies["cnn_pl_gleam"]}, signal, visits)
    with pytest.raises(ValueError, match="jbo-default-v2"):
        Protocol(freqs, skies, signal, visits, protocol_id="jbo-default-v2",
                 receiver_k=75.)
    with pytest.raises(ValueError, match="one finite value per channel"):
        Protocol.default(beam, skies, signal[:-1], visits)


def test_old_default_protocol_remains_loadable(tmp_path):
    freqs = np.arange(55., 121.)
    skies = {
        "cnn_pl_gleam": np.full((freqs.size, 12), 1500.),
        "gsm2008_gleam": np.full((freqs.size, 12), 1600.),
    }
    old = Protocol(freqs, skies, np.zeros(freqs.size), np.full(360, 180.),
                   protocol_id="jbo-default-v1")
    path = tmp_path / "old-default.npz"
    old.save_npz(path)
    assert Protocol.load_npz(path).fingerprint == old.fingerprint


def test_protocol_npz_roundtrip_preserves_observing_settings(analytic_inputs, tmp_path):
    _, base = analytic_inputs
    protocol = Protocol(
        base.freqs_mhz, base.sky_maps_k, base.signal_k, base.visits,
        latitude_deg=42.5, receiver_k=75., ground_k=285., loss_k=295.,
        integration_s=180., bandwidth_hz=500_000., foreground_order=3,
        protocol_id="roundtrip-test",
    )
    path = tmp_path / "protocol.npz"
    protocol.save_npz(path)
    restored = Protocol.load_npz(path)
    assert restored.fingerprint == protocol.fingerprint
    assert restored.protocol_id == "roundtrip-test"
    assert restored.latitude_deg == 42.5
    assert restored.receiver_k == 75.
    assert restored.ground_k == 285.
    assert restored.loss_k == 295.
    assert restored.integration_s == 180.
    assert restored.bandwidth_hz == 500_000.
    assert restored.foreground_order == 3
    np.testing.assert_array_equal(restored.sky_maps_k["analytic"], base.sky_maps_k["analytic"])
    with pytest.raises(ValueError, match="reference band"):
        Protocol.load_npz(path, require_reference_band=True)


def test_protocol_rejects_nonfinite_observing_settings(analytic_inputs):
    _, base = analytic_inputs
    with pytest.raises(ValueError, match="site or temperature"):
        Protocol(base.freqs_mhz, base.sky_maps_k, base.signal_k, base.visits,
                 receiver_k=np.nan)
