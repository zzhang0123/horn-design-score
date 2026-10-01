# T-001: Scoring a global 21 cm telescope beam with m modes

## Purpose and current status

`horn-design-score` evaluates a candidate drift-scan beam with one dimensionless figure of merit (FOM). It measures how well a fixed global 21 cm template survives foreground fitting in the full-LST, $m=0$ spectrum, while accounting for beam throughput, thermal noise, and foreground residuals. The score is intended for repeated candidate evaluation in a design search.

The installable package, synthetic demonstration, harmonic fast path, and independent limTOD checks are implemented. A scientifically interpretable ranking of physical horns still requires frozen, verified 2025 Jodrell Bank sky maps, observing visits, and signal inputs. The synthetic demonstration is an engineering check, not a result about real horn designs.

The upstream dependency is limTOD's `mmode-solver` branch. Its RHINO task T-006 motivates the use of the full-LST $d_0$ observable. An eight-hour LST arc can leak higher $m$ modes into a nominal monopole estimate; this package currently scores only complete LST coverage and makes no partial-arc leakage claim.

## Built-in analytic benchmark

`Protocol.default(beam)` now supplies all non-beam inputs needed for a reproducible design-search benchmark. It uses two deterministic equatorial HEALPix skies, `analytic_a` and `analytic_b`, with smooth bright structures and spatially varying power-law index and curvature. Their 70 MHz amplitudes and angular patterns are analytic; they are not CNN-PL, GSM2008, or GLEAM products. Each sky adds a 2.725 K constant and is averaged over each channel with three-point Gauss–Legendre quadrature. The signal is a channel-averaged Gaussian absorption with centre 72 MHz, depth −0.14 K, and standard deviation 10 MHz. The observing schedule uses 180 uniform 240 s visits per LST bin. The beam provides the channel centres and `nside`; the assumed channel bandwidth is 1 MHz.

These analytic inputs allow immediate, repeatable scoring and software optimization. Their score does not establish a ranking under the real sky or actual 2025 visits. Passing frozen sky cubes, signal, and visits to `Protocol.default` replaces the generated inputs. A preset id (`analytic-default-v3`, `jbo-default-v4`) fixes the sky labels, the observing settings and the foreground model. It does not fix the sky, signal or visit arrays; those are identified by the protocol fingerprint.

## Proposed scientific reference setup

Every candidate in a ranking must use exactly the same protocol. Proposed reference settings are:

| Item | Fixed choice |
| --- | --- |
| Site and pointing | Jodrell Bank, latitude 53.23625°, longitude −2.30744°; zenith drift scan. The score uses only the latitude. |
| Visits | During 2025, 240 s samples while the Sun is below the horizon, accumulated into 360 one-degree LST bins; every bin must have positive coverage |
| Frequency | Channel centres come from the input beam file; use the same grid for every candidate and sample the sky and signal on that grid. The assumed channel bandwidth is 1 MHz. |
| Foregrounds | Two equatorial sky cubes: CNN-PL + GLEAM and GSM2008 + GLEAM; freeze provenance, hashes, pixelization, and frequency interpolation |
| Signal | One fixed, spectrally smooth, channel-averaged global 21 cm template $T_{21}(\nu)$, injected with amplitude $A=1$ |
| Receiver and environment | One receiving port, unpolarized Stokes I, $T_{\rm rx}=100$ K, $T_{\rm ground}=T_{\rm loss}=300$ K, independent thermal noise in each 1 MHz channel |
| Foreground fit | Ground and loss terms subtracted, then the EDGES beam chromaticity factor $C(\nu)=D(\nu)/D(\nu_{\rm ref})$ times a positive fifth-order log-polynomial; $\nu_{\rm ref}$ is the channel nearest 75 MHz; one fixed fitting rule for all candidates |

The protocol records the beam's frequencies for alignment and reproducibility; it does not choose their range. The 55–85 MHz sub-band alone gives weak signal–foreground separation in the motivating study. A 55–120 MHz benchmark is appropriate only when the input beams, skies, and signal cover it; the benchmark must not silently extrapolate a measured beam. A signal derived from 21cmVAE is suitable only after removing interpolation kinks. Sky interpolation artifacts and the adequacy of the foreground order must be checked with reference and achromatic beams before ranking candidates. Any correction or uncertainty model must be frozen for the whole benchmark, never tuned to an individual beam.

This idealized protocol excludes ionospheric variation, RFI, daily calibration drift, and polarized leakage. A different beam frequency grid or input dataset has a different protocol fingerprint and must be ranked separately. Changing the observing or fitting assumptions requires a new preset version. An eight-hour arc or another site is a separate diagnostic experiment.

## Beam input and physical convention

The primary input is `BeamModes`: healpy-packed `full_alm[frequency, alm]` coefficients of the **full-sky, beam-local, Stokes-I power gain**. Users can convert electromagnetic simulation output to these spherical harmonics themselves. `BeamModes.prepare(latitude_deg)` computes and caches the horizon-weighted local sky coefficients, the once-rotated reference celestial coefficients, the radiative efficiency $\eta_{\rm rad}$, and the horizon-weighted monopole, both as $a_{00}/\sqrt{4\pi}$ of the corresponding coefficients. Prepared arrays can be supplied directly to avoid repeated HEALPix transforms and Wigner rotations in a design loop.

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

The discrete implementation assigns half of each horizon-centred HEALPix pixel to the sky and analyses the horizon-weighted beam with `map2alm(iter=3)`. $\eta_{\rm rad}$ is $a_{00}/\sqrt{4\pi}$ of the full beam, the integral of the band-limited gain. One $h$ is used throughout the harmonic scorer: its full-LST mean response to a uniform unit sky. That $h$ multiplies $T_{21}$, weights the sky term, and sets the ground fraction $\eta_{\rm rad}-h$, so an isothermal sky, ground and loss give an isothermal $d_0$. It differs from the horizon-weighted monopole stored by `prepare()` by pixel-quadrature leakage (below $2\times10^{-3}$ at `nside=8` in the package tests); angular convergence is therefore an acceptance check. `prepare()` rejects $\eta_{\rm rad}$ outside $(0,1.001]$ and a horizon-weighted monopole above $\eta_{\rm rad}+10^{-3}$; the scorer applies the same upper bound to $h$. The legacy grid path uses pixel means for $\eta_{\rm rad}$ and $h$. limTOD's upstream `normalize_beam=True` result without a horizon mask does not establish this package's absolute-gain convention. A shape-only beam normalized to $\eta_{\rm rad}=1$ belongs to a separately labelled ranking.

## Observable, noise, and score

Complete LST coverage makes $d_0$ the mean of the 360 LST-bin temperatures:

\[
d_0(\nu)=\frac{1}{360}\sum_{b=1}^{360}T_b(\nu),\qquad
\sigma_{d_0}^2(\nu)=\frac{1}{360^2}\sum_{b=1}^{360}
\frac{T_{\rm sys,b}^2(\nu)}{\Delta\nu\,(240\,\mathrm{s})\,n_b},
\quad T_{\rm sys,b}=T_{\rm ant,fg,b}+T_{\rm rx}.
\]

The visit count $n_b$ may vary by bin, but every bin must have positive coverage. Partial coverage would require an explicit estimator, covariance, conditioning analysis, and higher-$m$ leakage treatment before it could be scored.

For each frozen foreground sky $j$, let $f_j$ be its signal-free $d_0$ spectrum and $g=(\eta_{\rm rad}-h)T_{\rm ground}+(1-\eta_{\rm rad})T_{\rm loss}$ the ground and loss term, which follows from the beam alone. With the default `edges_beam_factor` model the fitted foreground is $\hat f_j=g+C_j\exp(Va_j)$, with the EDGES beam chromaticity factor

\[
C_j(\nu)=\frac{D_j(\nu)}{D_j(\nu_{\rm ref})},\qquad
D_j(\nu)=\frac{1}{360}\sum_{b=1}^{360}\frac{1}{4\pi}\int_{H_+}G(\nu,\Omega)\,
T_j(R_b\Omega,\nu_{\rm ref})\,d\Omega.
\]

$D_j$ weights the sky map of one reference channel by the beam at $\nu$; $\nu_{\rm ref}$ is the channel nearest 75 MHz, the lower one on a tie. $G$ is the accepted-power gain, so $C_j$ includes the throughput ratio $h(\nu)/h(\nu_{\rm ref})$ and equals it for an isotropic sky. $C_j$ is exact when the sky is one angular pattern times one spectrum. Spatial variation of the sky spectrum coupled to beam chromaticity is not removed and stays in the residual $r_j$ below. Let $J_j=\operatorname{diag}(\hat f_j-g)V$ be the foreground-model Jacobian, $W_j^{\mathsf T}W_j=N_j^{-1}$ the thermal-noise whitening, and $Q_j=\operatorname{orth}(W_jJ_j)$. After projecting out the locally fitted foreground tangent space, define

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

The minimum makes the design score conservative across the two predeclared skies. Higher is better. The score penalizes the *total* whitened foreground residual, while $b_j$ is a diagnostic for template-aligned bias. $C_j$ and $g$ are computed from the candidate beam and from the mock's own sky map at $\nu_{\rm ref}$, so the score does not include errors in the beam model or in the reference sky map; a separate mismatched-model benchmark is needed for those. Two other foreground models remain selectable through `Protocol(foreground_model=...)`. Neither subtracts $g$: they fit $\hat f_j=C_j\exp(Va_j)$ to $f_j$ and use $J_j=\operatorname{diag}(\hat f_j)V$. `matched_beam_factor` uses $C_j=f_j/\langle T_j\rangle$ with the mock's own sky at every frequency. The fit then reduces to a log-polynomial fit of the sky monopole, and the unwhitened residual is $C_j$ times the monopole fit residual, so $r_j^{\mathsf T}r_j$ does not respond to beam chromaticity; it still varies with the beam through the noise whitening. Its score is neither an upper nor a lower bound on the default score, and it does not rank beams by chromaticity. `plain_log_polynomial` uses $C_j=1$. A candidate receives FOM 0 with an explicit status when its efficiency or sky partition is outside the allowed range or its sky throughput exceeds $\eta_{\rm rad}+10^{-3}$ (`invalid_efficiency`), its sky throughput is not positive (`invalid_sky_throughput`, packed-alm paths only), a forward spectrum or noise level is not positive (`invalid_forward_model`), the foreground model is not positive (`invalid_foreground_model`), the foreground fit does not converge (`foreground_fit_failed`), the foreground tangent space is rank deficient (`foreground_rank_deficient`), or the projected template vanishes (`signal_unidentifiable`, $I_j\le10^{-24}$). `HarmonicScorer.score`, `score_beam` and `reference_modes_score` return these statuses. On the legacy grid path `score_beam` returns `invalid_beam` or `invalid_efficiency` for a non-physical gain, while `reference_score` raises. A beam and protocol on different frequency grids or `nside` raise an error instead. So does a prepared beam whose stored efficiency or partition is not physical, or does not equal the monopole of its coefficients. Files prepared by version 0.6.0 or earlier stored pixel means and are rejected for the second reason; build the beam from `full_alm` and call `prepare()` again. A candidate with small but positive throughput or information is scored and reported as `ok`. The FOM is a design metric, not a calibrated detection significance, posterior probability, or guarantee of full signal-shape recovery.

With `fit_spectrum=True`, the scorer also fits a positive smooth foreground plus one amplitude of the **fixed** $T_{21}$ template to noiseless mock data with injected $A=1$. It returns, for each sky, the fitted amplitude, the intrinsic fitted spectrum $A_{\rm fit}T_{21}$, the antenna-convolved spectrum $h A_{\rm fit}T_{21}$, and the $\chi^2$ of the fit. This is neither a free-form spectrum reconstruction nor an analysis of observed data. An optional spectrum-fit failure is reported separately and does not alter the already computed design FOM.

## Numerical implementation and use

`HarmonicScorer` caches sky quadrature harmonics and protocol state. Once per reference protocol, sky maps are transformed with the selected HEALPix quadrature and all input arrays/settings receive a reproducibility fingerprint. For each candidate, the prepared beam is contracted with the sky in packed harmonic order, $\sum_\ell B^*_{\ell m}S_{\ell m}$; phase factors synthesize the 360-bin drift-scan TOD. The $m=0$ term yields the full-LST mean, while the bin temperatures give the thermal-noise covariance. Foreground fitting and the QR projection then produce the score. LST bins are sampled at their centres, 0.5° to 359.5°, as instantaneous values. $V$ has columns $[\ln(\nu/70\,\mathrm{MHz})]^k$ for $k=0,\dots,N$. The coefficients $a_j$ are fitted by noise-weighted nonlinear least squares in Kelvin, started from an unweighted log-space fit. The tangent space is declared rank deficient when the smallest QR diagonal is below $10^{-12}$ of the largest. `HarmonicScorer` requires `lmax` $\le\min(3\,\mathrm{nside}-1,\,359)$; a protocol needs at least eight channels and a foreground order of at most the channel count minus three. The NumPy/healpy hot path avoids the JAX/healpy OpenMP conflict observed on the current macOS host.

The independent, slower check uses limTOD map rotation via `pointing_beam_in_eq_sys` and `mmode.solve(m_trunc=0)`. Existing synthetic tests exercise isotropic and anisotropic skies and nonuniform visits. A legacy zonal grid path remains diagnostic only: its LST-mean system-temperature approximation is unsuitable for candidate ranking. The CLI still scores an angular-grid NPZ with status `ok`; the output marks it with `input_kind: angular_grid`. A prepared synthetic candidate was measured near 0.9 ms in a small `nside=8` setup; this is not a throughput claim for the proposed `nside=64` reference benchmark.

The package import is `horn_design_score`. The Python entry point is `score_beam(beam, protocol, kernel=scorer)` or `HarmonicScorer.score`; construct the scorer once and reuse it across candidates. The CLI scores one candidate NPZ and writes JSON including input hashes. See [README.md](README.md) for the short installation and usage example.

## Acceptance boundary

Engineering acceptance already covers package installation, API and CLI use, the synthetic example, and targeted harmonic-versus-limTOD checks. Scientific acceptance still requires:

1. Freeze and inspect the real 2025 visit histogram, both foreground cubes, the smooth channel-averaged signal, interpolation rules, units, and input hashes.
2. Run physically plausible baseline and achromatic/chromatic control beams; validate foreground-model residuals and signal/null injections without candidate-specific retuning.
3. Check angular (`nside`, `lmax`) and frequency convergence, ranking stability, false detections, and the slow limTOD cross-check on representative beams.
4. Benchmark full reference-resolution runtime and memory on the intended search hardware.
5. Keep results from partial LST arcs outside this FOM until an explicit leakage-aware estimator and covariance have been validated.

Manufacturing feasibility, impedance mismatch, polarized sky leakage, calibration error, and actual observing conditions remain outside the present score. A numerical winner under this protocol is a candidate for those further checks, not an experimentally validated optimum.
