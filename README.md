# horn-design-score

Score a horn beam for a fixed global 21 cm drift-scan experiment. Input: beam
spherical-harmonic modes. Output: one dimensionless figure of merit (larger is
better), with an optional fitted 21 cm template spectrum.

## Try it

Install the local limTOD `mmode-solver` checkout first, then this package:

```bash
python3 -m pip install -e /path/to/limTOD
python3 -m pip install -e .
horn-score make-demo demo
horn-score score --beam demo/demo_beam_modes.npz --protocol demo/demo_protocol.npz
horn-score score --beam demo/demo_beam_modes.npz --protocol demo/demo_protocol.npz \
  --fitted-spectrum --output demo/score.json
```

The demo is synthetic; its score is **not** a scientific beam ranking.

## Use your beam

The beam NPZ needs `freqs_mhz`, `nside`, `lmax`, and `full_alm` with shape
`(frequency, (lmax+1)(lmax+2)/2)` in healpy packed order. `full_alm` is the
all-direction **Stokes-I power gain of one port**, in limTOD's beam-local frame
(pole at zenith, φ=0 south, φ=90° east). Its all-sky mean must give a radiation
efficiency between 0 and 1. If starting from θ/φ power gains in dBi, convert
their *linear* gains to Stokes I before calculating alms; the optional
`beam_modes_from_grid` helper does this for a regular angular grid.

The protocol NPZ stores a copy of the beam's `freqs_mhz` for matching, plus
`signal_k`, `visits` (360 positive LST-bin counts), and at least one
`sky_<name>` equatorial HEALPix RING map cube in K with shape
`(frequency, pixel)`. Frequencies and nside must match the beam.
The beam file sets the frequency grid. `Protocol.default(beam, ...)` copies
that grid and fixes the Jodrell Bank zenith setup: full 360-bin LST coverage,
240 s per visit, 1 MHz channel bandwidth, a 100 K receiver, 300 K ground/loss,
and a fifth-order foreground fit. Supply both foreground sky cubes, a smooth
21 cm template, and the 2025 night-time visit counts on the beam's channels:

```python
from horn_design_score import BeamModes, Protocol

beam = BeamModes.load_npz("beam.npz")
protocol = Protocol.default(
    beam, {"cnn_pl_gleam": cnn_pl_cube_k, "gsm2008_gleam": gsm2008_cube_k},
    signal_k, visits_2025,
)
protocol.save_npz("protocol.npz")
```

The package does not include or verify these reference arrays. For each sky,
the foreground model fits the exponential of a fifth-order polynomial in
log frequency, giving a smooth positive spectrum. The 21 cm model is the
fixed `signal_k` spectrum, injected with unit amplitude; the optional spectrum
output fits only that amplitude in a noiseless mock.

Reuse one protocol and scorer for many beams:

```python
from horn_design_score import BeamModes, HarmonicScorer, Protocol

protocol = Protocol.load_npz("protocol.npz")
beam = BeamModes.load_npz("beam.npz").prepare(protocol.latitude_deg)
scorer = HarmonicScorer(protocol, beam.lmax)
result = scorer.score(beam, fit_spectrum=True)
print(result.value, result.status)
for name, fit in result.fitted_spectra.items():
    print(name, fit.amplitude, fit.intrinsic_k, fit.antenna_k)
```

`prepare()` computes the horizon-weighted and reference-frame beam modes once;
save them with `beam.save_npz()` to skip that work in later runs. Without
`fit_spectrum=True` (CLI: `--fitted-spectrum`), no fitted spectra are returned.
`intrinsic_k` is the fitted global template before beam attenuation;
`antenna_k` includes the beam throughput. The fit uses a **noiseless mock with
template amplitude 1** and estimates only the template's amplitude. It is not
a free-form spectrum recovery or a fit to observed data. Requesting it does
not change the figure of merit.

## Interpretation and limits

The score is the minimum across sky scenarios of
`sqrt(information / (1 + foreground_residual_chi2))`, after fitting and
projecting out a fixed smooth foreground family. It is a design ranking
metric, not a detection significance. The fast scorer follows limTOD's m-mode
algebra using NumPy/healpy; `horn-score score ... --reference` uses limTOD's
slower map rotation and `m=0` solve for a cross-check. See [DESIGN.md](DESIGN.md)
for the model and numerical conventions.

The intended Jodrell Bank 2025 sky, visits and smoothed 21 cm template are
**not bundled**. Freeze and archive those inputs before comparing designs;
all candidate beams must share the same frequency grid. The optional
`--require-reference-band` check asks specifically for 55–120 MHz and is not
part of the default. The current model omits polarized leakage, ionosphere,
calibration drift and manufacturing tolerances.
