# T-001: Scoring a global 21 cm telescope beam with m modes

## Purpose and current status

`horn-design-score` evaluates a candidate drift-scan beam with one dimensionless figure of merit (FOM). It measures how well a fixed global 21 cm template survives foreground fitting in the full-LST, $m=0$ spectrum, while accounting for beam throughput, thermal noise, and foreground residuals. The score is intended for repeated candidate evaluation in a design search.

The installable package, synthetic demonstration, harmonic fast path, and independent limTOD checks are implemented. A scientifically interpretable ranking of physical horns still requires frozen, verified 2025 Jodrell Bank sky maps, observing visits, and signal inputs. The synthetic demonstration is an engineering check, not a result about real horn designs.

The upstream dependency is limTOD's `mmode-solver` branch. Its RHINO task T-006 motivates the use of the full-LST $d_0$ observable. An eight-hour LST arc can leak higher $m$ modes into a nominal monopole estimate; this package currently scores only complete LST coverage and makes no partial-arc leakage claim.

## Fixed reference experiment

Every candidate in a ranking must use exactly the same protocol. Proposed reference settings are:

| Item | Fixed choice |
| --- | --- |
| Site and pointing | Jodrell Bank, latitude 53.23625°, longitude −2.30744°; zenith drift scan |
| Visits | During 2025, 240 s samples while the Sun is below the horizon, accumulated into 360 one-degree LST bins; every bin must have positive coverage |
| Frequency | 55–120 MHz, channel-averaged at 1 MHz |
| Foregrounds | Two equatorial sky cubes: CNN-PL + GLEAM and GSM2008 + GLEAM; freeze provenance, hashes, pixelization, and frequency interpolation |
| Signal | One fixed, spectrally smooth, channel-averaged global 21 cm template $T_{21}(\nu)$, injected with amplitude $A=1$ |
| Receiver and environment | One receiving port, unpolarized Stokes I, $T_{\rm rx}=100$ K, $T_{\rm ground}=T_{\rm loss}=300$ K, independent thermal noise in each 1 MHz channel |
| Foreground fit | Positive log-polynomial $\exp[\sum_{k=0}^{5}a_k\ln^k(\nu/70\,\mathrm{MHz})]$, with one fixed order and fitting rule for all candidates |

The 55–85 MHz sub-band alone gives weak signal–foreground separation in the motivating study. Extending a measured beam to 120 MHz needs validated chromatic beam data; the benchmark must not silently extrapolate it. A signal derived from 21cmVAE is suitable only after removing interpolation kinks. Sky interpolation artifacts and the adequacy of the foreground order must be checked with reference and achromatic beams before ranking candidates. Any correction or uncertainty model must be frozen for the whole benchmark, never tuned to an individual beam.

This idealized protocol excludes ionospheric variation, RFI, daily calibration drift, and polarized leakage. Changing site, band, visits, fitting model, or physical assumptions creates a new protocol version and a new ranking. An eight-hour arc or another site is a separate diagnostic experiment.

## Beam input and physical convention

The primary input is `BeamModes`: healpy-packed `full_alm[frequency, alm]` coefficients of the **full-sky, beam-local, Stokes-I power gain**. Users can convert electromagnetic simulation output to these spherical harmonics themselves. `BeamModes.prepare(latitude_deg)` computes and caches the horizon-weighted local sky coefficients, the once-rotated reference celestial coefficients, the radiative efficiency $\eta_{\rm rad}$, and the sky/ground partition $h$. Prepared arrays can be supplied directly to avoid repeated HEALPix transforms and Wigner rotations in a design loop.

An optional `BeamPattern` adapter accepts sampled direction grids and two polarized gains in dBi. The two gains must describe the **same receiving port** for orthogonal incident polarization states. Convert each dBi value to linear power gain before adding:

\[
G(\nu,\Omega)=10^{G_{\theta,\mathrm{dBi}}(\nu,\Omega)/10}
                 +10^{G_{\phi,\mathrm{dBi}}(\nu,\Omega)/10}.
\]

Two separate ports, field-amplitude dB, directivity, and realized-gain conventions require an explicit conversion. These two power gains cannot reconstruct a complex Jones matrix or predict polarized leakage. The reference model treats $G$ as accepted-power gain with a perfect match and requires physically meaningful $0<\eta_{\rm rad}\leq1$, allowing only small numerical quadrature tolerance.

With local horizon directions $H_+$ and ground directions $H_-$, and with $T_{\rm sky}$ denoting the foreground-only sky, the forward model is

\[
\begin{aligned}
T_{\rm ant}(t,\nu)={}&\frac{1}{4\pi}\int_{H_+}G(\nu,\Omega)
             T_{\rm sky}(R_t\Omega,\nu)\,d\Omega\\
&+\frac{1}{4\pi}\int_{H_-}G(\nu,\Omega)T_{\rm ground}\,d\Omega
 +(1-\eta_{\rm rad})T_{\rm loss}+h(\nu)A T_{21}(\nu),\\
h(\nu)={}&\frac{1}{4\pi}\int_{H_+}G(\nu,\Omega)\,d\Omega,
\qquad
\eta_{\rm rad}(\nu)=\frac{1}{4\pi}\int_{4\pi}G(\nu,\Omega)\,d\Omega.
\end{aligned}
\]

The discrete implementation assigns horizon-centred HEALPix pixels consistently between sky and ground. Its harmonic $m=0$ sky throughput may differ slightly from the geometric partition at finite `nside`; angular convergence is therefore an acceptance check. limTOD's upstream `normalize_beam=True` result without a horizon mask does not establish this package's absolute-gain convention. A shape-only beam normalized to $\eta_{\rm rad}=1$ belongs to a separately labelled ranking.

## Observable, noise, and score

Complete LST coverage makes $d_0$ the mean of the 360 LST-bin temperatures:

\[
d_0(\nu)=\frac{1}{360}\sum_{b=1}^{360}T_b(\nu),\qquad
\sigma_{d_0}^2(\nu)=\frac{1}{360^2}\sum_{b=1}^{360}
\frac{T_{\rm sys,b}^2(\nu)}{\Delta\nu\,(240\,\mathrm{s})\,n_b},
\quad T_{\rm sys,b}=T_{\rm ant,fg,b}+T_{\rm rx}.
\]

The visit count $n_b$ may vary by bin, but every bin must have positive coverage. Partial coverage would require an explicit estimator, covariance, conditioning analysis, and higher-$m$ leakage treatment before it could be scored.

For each frozen foreground sky $j$, let $f_j$ be its signal-free $d_0$ spectrum and $\hat f_j$ its fitted smooth foreground. Let $J_j$ be the foreground-model Jacobian, $W_j^{\mathsf T}W_j=C_j^{-1}$ the thermal-noise whitening, and $Q_j=\operatorname{orth}(W_jJ_j)$. After projecting out the locally fitted foreground tangent space, define

\[
\begin{aligned}
P_j&=I-Q_jQ_j^{\mathsf T},\\
u_j&=P_jW_j\,[h\odot T_{21}],\\
r_j&=P_jW_j(f_j-\hat f_j),\\
I_j&=u_j^{\mathsf T}u_j,\\
b_j&=u_j^{\mathsf T}r_j/I_j,\\
S_j&=\sqrt{I_j/(1+r_j^{\mathsf T}r_j)},\\
\mathrm{FOM}&=\min_j S_j.
\end{aligned}
\]

The minimum makes the design score conservative across the two predeclared skies. Higher is better. The score penalizes the *total* whitened foreground residual, while $b_j$ is a diagnostic for template-aligned bias. Invalid or unidentifiable candidates and failed foreground fits receive FOM 0 with an explicit status. The FOM is a design metric, not a calibrated detection significance, posterior probability, or guarantee of full signal-shape recovery.

With `fit_spectrum=True`, the scorer also fits a positive smooth foreground plus one amplitude of the **fixed** $T_{21}$ template to noiseless mock data with injected $A=1$. It can return the intrinsic fitted spectrum $A_{\rm fit}T_{21}$, the antenna-convolved spectrum $h A_{\rm fit}T_{21}$, and fit residuals for each sky. This is neither a free-form spectrum reconstruction nor an analysis of observed data. An optional spectrum-fit failure is reported separately and does not alter the already computed design FOM.

## Numerical implementation and use

`HarmonicScorer` caches sky quadrature harmonics and protocol state. Once per reference protocol, sky maps are transformed with the selected HEALPix quadrature and all input arrays/settings receive a reproducibility fingerprint. For each candidate, the prepared beam is contracted with the sky in packed harmonic order, $\sum_\ell B^*_{\ell m}S_{\ell m}$; phase factors synthesize the 360-bin drift-scan TOD. The $m=0$ term yields the full-LST mean, while the bin temperatures give the thermal-noise covariance. Foreground fitting and the QR projection then produce the score. The NumPy/healpy hot path avoids the JAX/healpy OpenMP conflict observed on the current macOS host.

The independent, slower check uses limTOD map rotation via `pointing_beam_in_eq_sys` and `mmode.solve(m_trunc=0)`. Existing synthetic tests exercise isotropic and anisotropic skies and nonuniform visits. A legacy zonal grid path remains diagnostic only: its LST-mean system-temperature approximation is unsuitable for candidate ranking. A prepared synthetic candidate was measured near 0.9 ms in a small `nside=8` setup; this is not a throughput claim for the proposed `nside=64` reference benchmark.

The package import is `horn_design_score`. The Python entry point is `score_beam(beam, protocol, kernel=scorer)` or `HarmonicScorer.score`; construct the scorer once and reuse it across candidates. The CLI scores one candidate NPZ and writes JSON including input hashes. See [README.md](README.md) for the short installation and usage example.

## Acceptance boundary

Engineering acceptance already covers package installation, API and CLI use, the synthetic example, and targeted harmonic-versus-limTOD checks. Scientific acceptance still requires:

1. Freeze and inspect the real 2025 visit histogram, both foreground cubes, the smooth channel-averaged signal, interpolation rules, units, and input hashes.
2. Run physically plausible baseline and achromatic/chromatic control beams; validate foreground-model residuals and signal/null injections without candidate-specific retuning.
3. Check angular (`nside`, `lmax`) and frequency convergence, ranking stability, false detections, and the slow limTOD cross-check on representative beams.
4. Benchmark full reference-resolution runtime and memory on the intended search hardware.
5. Keep results from partial LST arcs outside this FOM until an explicit leakage-aware estimator and covariance have been validated.

Manufacturing feasibility, impedance mismatch, polarized sky leakage, calibration error, and actual observing conditions remain outside the present score. A numerical winner under this protocol is a candidate for those further checks, not an experimentally validated optimum.
