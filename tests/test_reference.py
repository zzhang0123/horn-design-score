import numpy as np
import pytest

from horn_design_score import ZonalKernel, reference_forward
from horn_design_score.kernel import foreground_fast


@pytest.mark.filterwarnings("ignore:.*monopole.*")
def test_limtod_closed_scan_matches_fast_isotropic(analytic_inputs):
    beam, protocol = analytic_inputs
    gain = beam.sample_healpix(4)
    fast, _, _ = foreground_fast(gain, ZonalKernel.build(protocol), protocol)
    slow, sigma = reference_forward(beam, protocol, np.array([0]))
    np.testing.assert_allclose(slow["analytic"][0], fast["analytic"][0], rtol=1e-5)
    assert sigma["analytic"][0] > 0
