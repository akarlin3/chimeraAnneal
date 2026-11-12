# chimeraAnneal (`chimera-anneal`)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20564867.svg)](https://doi.org/10.5281/zenodo.20564867)

**Coupled phase-oscillator chimera survival, graze-versus-absorption collapse detection, Weibull aging hazard, and topology-dependent finite-size scaling.**

`chimera-anneal` is the standalone scientific Python library accompanying the *Anneal* study:

> **Avery Karlin** (2026). *"Chimera Collapse Ages: Topology-Dependent Finite-Size Scaling in Mean-Field and Ring Oscillator Systems."* Submitted to *Physical Review E* (Zenodo archive: [10.5281/zenodo.20564867](https://doi.org/10.5281/zenodo.20564867)).

All stochastic initializations and bootstrap estimators use strictly deterministic cross-platform `numpy.random.PCG64` streams, with a lightweight core requiring only `numpy>=1.24` and `scipy>=1.10` (plus optional `numba` acceleration via `pip install chimera-anneal[fast]`).

---

## AveryCorp Scientific Suites Architecture

`chimeraAnneal` is a core pillar of **Suite II** in the **AveryCorp Open Scientific Software Ecosystem**, sharing a standardized `AuditReport` diagnostic schema (`chimera-anneal audit --json` / `run_audit()`) across two non-overlapping suites:

- **Suite I: Clinical Quantitative-MRI Trust Spine (`Diagnose -> Calibrate -> Benchmark`)**
  - [`sextantIVIM`](https://github.com/akarlin3/sextantIVIM) (Stage 1: Identifiability & Railing Gate) -> [`conformalIVIM`](https://github.com/akarlin3/conformalIVIM) (Stage 2: Conformal UQ & Drift Gate) -> [`IVIMBench`](https://github.com/akarlin3/IVIMBench) (Stage 3: DRO Phantoms & Closed-Loop RT Benchmark).
- **Suite II: Scientific Identifiability & Null-Audit Suite**
  - [`weakSINDY`](https://github.com/akarlin3/weakSINDY): Weak-form fractional PDE/ODE identifiability & ensemble inclusion gate ·
  - **[`chimeraAnneal`](https://github.com/akarlin3/chimeraAnneal) (*This Package*)**: Absorbing collapse vs. transient graze separation & finite-size hazard gate ·
  - [`proteusMine`](https://github.com/akarlin3/proteusMine): Structure-first catalytic triad screening, `pLDDT` stratification & TOST non-superiority gate.

---

## Mathematical Overview

### 1. Coupled-Oscillator Chimera Substrates (`chimera_anneal.substrates`)

1. **Two-Population Sakaguchi–Kuramoto Mean Field (`TwoPopMeanField`)**  
   Evolves two identical subpopulations $\sigma \in \{1, 2\}$ of $N$ oscillators each ($2N$ total) with intra-population coupling $p = (1 + A)/2$, inter-population coupling $q = (1 - A)/2$, and phase lag $\alpha = \pi/2 - \beta$:
   $$\frac{d\theta_i^{(\sigma)}}{dt} = \omega + p R_\sigma \sin(\Phi_\sigma - \theta_i^{(\sigma)} - \alpha) + q R_{\sigma'} \sin(\Phi_{\sigma'} - \theta_i^{(\sigma)} - \alpha), \qquad R_\sigma e^{i\Phi_\sigma} = \frac{1}{N}\sum_{j=1}^N e^{i\theta_j^{(\sigma)}}.$$
   Supports exact $O(N)$ fixed-step RK4 and adaptive 8th-order Dormand–Prince (`DOP853`) integration.

2. **Nonlocally Coupled Ring (`NonlocalRingSubstrate`)**  
   Evolves $N$ oscillators on a periodic ring with top-hat window ($r = P/N$) or cosine coupling kernel $G(x - y) \propto 1 + A_{\cos}\cos(x - y)$ near the chimera existence boundary ($\beta_c^\infty \approx 0.137$ at $r = 0.15$):
   $$\frac{d\theta_k}{dt} = \omega - \sum_{j=0}^{N-1} G_{k-j}\sin(\theta_k - \theta_j + \alpha).$$
   Evaluated in $O(N)$ (circular moving-average or `numba` sliding sum) or $O(N\log N)$ real FFT convolution.

3. **Parameter-Free Reduced Macroscopic Flow (`ReducedTwoPopFlow`)**  
   Integrates the exact 2D ($\rho_1 \equiv 1$ Ott–Antonsen manifold) and 3D $(\rho_1, \rho_2, \psi)$ collective equations of Abrams, Mirollo, Strogatz & Wiley (*Phys. Rev. Lett.* **101**, 084103, 2008), locating the saddle-node ($A_{\mathrm{SN}}$), supercritical Hopf ($A_H$), and homoclinic boundary-crisis ($A_{\mathrm{hc}}$) bifurcations without fitted parameters.

### 2. Graze-vs-Absorption Collapse Detection (`chimera_anneal.detection`)

Conventional first-passage criteria ($\min(R_1, R_2) \ge \theta_{\mathrm{sync}}$ held for $W$) misfire on breathing mean-field chimeras, where up to 98% of sustained near-synchrony excursions self-recover into a dispersed incoherent halo (*grazes*). `detect_chimera_collapse` pairs first-passage graze timing $t_{\mathrm{graze}}$ with a hysteresis-gated absorption time $t_{\mathrm{abs}}$ verified against post-crossing recovery dips, alongside the Kemeth–Haugland–Krischer discrete-Laplacian curvature metric $g_0(t)$.

### 3. Weibull Aging & Topology-Dependent Finite-Size Scaling (`chimera_anneal.hazard`)

- **Right-censored Weibull MLE (`fit_weibull_survival`)**: Fits $S(t) = \exp[-(t/\lambda)^k]$ with profile-likelihood and `PCG64` bootstrap 95% CIs plus likelihood-ratio tests against memoryless Poisson escape ($k = 1$). Across both topologies, chimera collapse ages ($k > 1$).
- **Non-parametric curves (`kaplan_meier_estimator`, `instantaneous_hazard_rate`)**: Right-censored Kaplan–Meier survival with Greenwood CIs, Nelson–Aalen cumulative hazard, and Epanechnikov kernel-smoothed hazard rates $h(t)$.
- **Finite-size scaling (`fit_finite_size_scaling`)**: Evaluates power-law $\tau(N) = c N^\gamma$ and exponential $\tau(N) = a e^{\kappa N}$ lifetime laws with stratified `PCG64` bootstrap 95% CIs.

---

## Installation

```bash
pip install chimera-anneal
# Optional Numba JIT acceleration for large ring ensembles:
pip install "chimera-anneal[fast]"
```

For local development:

```bash
git clone https://github.com/akarlin3/chimeraAnneal.git
cd chimeraAnneal
pip install -e ".[dev]"
pytest -q
```

---

## Quickstart

### Python API

```python
from chimera_anneal import (
    TwoPopMeanField,
    NonlocalRingSubstrate,
    ReducedTwoPopFlow,
    detect_chimera_collapse,
    fit_weibull_survival,
)

# 1. Parameter-free Ott-Antonsen bifurcation check at beta = 0.05
A_hc = ReducedTwoPopFlow.locate_homoclinic(beta=0.05)
print(f"Homoclinic boundary crisis A_hc(0.05) = {A_hc:.4f}")

# 2. Integrate a deterministic 2-population breathing chimera (A = 0.5 > A_hc)
engine = TwoPopMeanField(n_per_pop=16, A=0.5, beta=0.05)
traj = engine.integrate(seed=100015, t_max=300.0, dt=0.05, method="rk4")
det = detect_chimera_collapse(traj)
print(f"First graze: {det.t_graze:.1f}s | True absorption: {det.t_abs:.1f}s (grazes={det.n_grazes})")

# 3. Right-censored Weibull aging fit across a seeded ring ensemble
ring = NonlocalRingSubstrate(n_oscillators=48, kernel="tophat", coupling_radius=0.15, beta=0.135)
taus, events = [], []
for k in range(20):
    r_traj = ring.integrate(seed=20260609 + k, t_max=250.0)
    r_det = detect_chimera_collapse(r_traj, ring_eps=0.04, hold_time=10.0)
    taus.append(r_det.t_abs)
    events.append(r_det.event_abs)

fit = fit_weibull_survival(taus, events, seed=2026)
print(f"Weibull shape k = {fit.shape_k:.2f} (95% CI {fit.shape_ci95}), scale = {fit.scale_lambda:.1f}s")
```

### Command-Line Interface

```bash
# Seeded mean-field vs. nonlocal-ring survival & finite-size scaling summary table
chimera-anneal demo

# Custom finite-size scaling benchmark
chimera-anneal benchmark --n-sizes 32,64,128 --trials 50
```

---

## License & Citation

Released under the [MIT License](LICENSE). Please see [`CITATION.cff`](CITATION.cff) to cite this software and the accompanying *Physical Review E* study.
