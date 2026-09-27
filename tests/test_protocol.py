import numpy as np
import pytest

from horn_design_score import Protocol


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
