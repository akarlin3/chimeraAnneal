"""Deterministic test suite for chimeraAnneal (chimera-anneal v1.0.0)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from chimera_anneal import (
    NonlocalRingSubstrate,
    ReducedTwoPopFlow,
    TwoPopMeanField,
    __version__,
    detect_chimera_collapse,
    fit_finite_size_scaling,
    fit_weibull_survival,
    hold_time_hysteresis_gate,
    instantaneous_hazard_rate,
    kaplan_meier_estimator,
    local_curvature_threshold_gate,
    make_pcg64_rng,
)
from chimera_anneal.cli import main as cli_main


def test_version_and_pcg64_determinism() -> None:
    assert __version__ == "1.0.0"
    rng_a = make_pcg64_rng(100015)
    rng_b = make_pcg64_rng(100015)
    assert isinstance(rng_a.bit_generator, np.random.PCG64)
    np.testing.assert_array_equal(rng_a.normal(size=32), rng_b.normal(size=32))


def test_twopop_meanfield_determinism_and_rk4_dop853_agreement() -> None:
    eng = TwoPopMeanField(n_per_pop=16, p=0.75, q=0.25, beta=0.05)
    assert math.isclose(eng.A, 0.5, rel_tol=1e-12)
    assert math.isclose(eng.alpha, 0.5 * math.pi - 0.05, rel_tol=1e-12)

    traj1 = eng.integrate(seed=100015, t_max=25.0, dt=0.02, decimate=5, method="rk4")
    traj2 = eng.integrate(seed=100015, t_max=25.0, dt=0.02, decimate=5, method="rk4")
    np.testing.assert_array_equal(traj1.R1, traj2.R1)
    np.testing.assert_array_equal(traj1.R2, traj2.R2)
    np.testing.assert_array_equal(traj1.theta_final, traj2.theta_final)

    traj_dop = eng.integrate(seed=100015, t_max=25.0, dt=0.02, decimate=5, method="dop853")
    np.testing.assert_allclose(traj1.R1, traj_dop.R1, atol=2e-4)
    np.testing.assert_allclose(traj1.R2, traj_dop.R2, atol=2e-4)


def test_nonlocal_ring_tophat_and_cosine_determinism() -> None:
    ring_th = NonlocalRingSubstrate(
        n_oscillators=48, kernel="tophat", coupling_radius=0.15, beta=0.12, use_numba=False
    )
    assert math.isclose(float(np.sum(ring_th.kernel_weights)), 1.0, rel_tol=1e-12)
    r1 = ring_th.integrate(seed=20260609, t_max=20.0, dt=0.05, decimate=5, stop_eps=None)
    r2 = ring_th.integrate(seed=20260609, t_max=20.0, dt=0.05, decimate=5, stop_eps=None)
    np.testing.assert_array_equal(r1.rho_std, r2.rho_std)
    np.testing.assert_array_equal(r1.theta_final, r2.theta_final)

    r_dop = ring_th.integrate(seed=20260609, t_max=20.0, dt=0.05, decimate=5, method="dop853")
    np.testing.assert_allclose(r1.rho_mean, r_dop.rho_mean, atol=2e-3)

    # Cosine kernel check + chimera curvature gate
    ring_cos = NonlocalRingSubstrate(
        n_oscillators=64, kernel="cosine", cosine_amplitude=0.995, beta=0.18
    )
    assert math.isclose(float(np.sum(ring_cos.kernel_weights)), 1.0, rel_tol=1e-12)
    rc = ring_cos.integrate(seed=20260609, t_max=15.0, dt=0.05, decimate=5, stop_eps=None)
    gate = local_curvature_threshold_gate(rc.theta_final)
    assert 0.0 <= float(gate["g0"]) <= 1.0


def test_reduced_twopop_flow_manifold_restriction_and_bifurcations() -> None:
    flow = ReducedTwoPopFlow(A=0.35, beta=0.05)
    r_test, psi_test = 0.62, -0.18
    rhs2 = flow.rhs_2d((r_test, psi_test))
    rhs3 = flow.rhs_3d((1.0, r_test, psi_test))
    # On rho1 == 1, drho1/dt == 0 and (drho2/dt, dpsi/dt) == rhs_2d identically
    assert abs(rhs3[0]) < 1e-15
    np.testing.assert_allclose(rhs3[1:], rhs2, atol=1e-13)
    assert flow.transverse_rate(r_test, psi_test) < 0.0

    sn = ReducedTwoPopFlow.locate_saddle_node(0.05)["A_SN"]
    hopf = ReducedTwoPopFlow.locate_hopf(0.05)
    ahc = ReducedTwoPopFlow.locate_homoclinic(0.05, t_max=3000.0, tol=2e-3)
    # Abrams et al. (2008) ordering at beta=0.05: A_SN (~0.0948) < A_Hopf (~0.270) < A_hc (~0.410) < 0.5
    assert 0.08 < sn < hopf < ahc < 0.45

    post_hc = ReducedTwoPopFlow(A=0.5, beta=0.05).integrate_3d((0.998, 0.45, 0.1), t_max=150.0)
    assert post_hc["captured"] is True
    assert post_hc["t_capture"] is not None and float(post_hc["t_capture"]) < 150.0


def test_graze_vs_absorption_separation() -> None:
    # Construct a breathing order-parameter trace with two self-healing grazes (>= 5s above 0.85
    # followed by recovery dips below 0.80) before irreversible absorption at t = 150s.
    t = np.linspace(0.0, 220.0, 2201)
    r1 = np.full_like(t, 0.995)
    r2 = np.full_like(t, 0.55)
    # Graze 1: [30, 37]s -> recovers to 0.45
    r2[(t >= 30.0) & (t <= 37.0)] = 0.90
    # Graze 2: [85, 92]s -> recovers to 0.50
    r2[(t >= 85.0) & (t <= 92.0)] = 0.91
    # True absorption: [150, 220]s -> stays at 0.99
    r2[t >= 150.0] = 0.99

    det = detect_chimera_collapse(
        t=t,
        R1=r1,
        R2=r2,
        sync_threshold=0.85,
        hold_time=5.0,
        recovery_threshold=0.80,
        recovery_hold=5.0,
        verification_horizon=100.0,
    )
    assert det.event_graze == 1
    assert math.isclose(det.t_graze, 30.0, abs_tol=0.2)
    assert det.n_grazes == 2
    assert det.event_abs == 1
    assert math.isclose(det.t_abs, 150.0, abs_tol=0.2)
    assert math.isclose(det.overtrigger_fraction, 2.0 / 3.0, rel_tol=1e-6)

    # Also verify real dynamical graze-vs-absorption separation on seed=100015 (Anneal Fig. 1a)
    eng = TwoPopMeanField(n_per_pop=16, A=0.5, beta=0.05)
    real_traj = eng.integrate(seed=100015, t_max=260.0, dt=0.05, decimate=4)
    real_det = detect_chimera_collapse(real_traj)
    assert real_det.t_abs >= real_det.t_graze


def test_weibull_aging_recovery_km_and_scaling() -> None:
    rng = make_pcg64_rng(424242)
    k_true = 2.4
    lam_true = 80.0
    samples = lam_true * rng.weibull(k_true, size=400)
    t_cap = 140.0
    events = (samples <= t_cap).astype(int)
    observed = np.minimum(samples, t_cap)

    wf = fit_weibull_survival(observed, events, n_boot=150, seed=424242)
    assert wf.aging_detected is True
    assert wf.shape_ci95[0] > 1.8
    assert abs(wf.shape_k - k_true) < 0.25
    assert abs(wf.scale_lambda - lam_true) < 8.0
    assert wf.lrt_pvalue < 1e-12

    # Memoryless exponential (k = 1) control should NOT falsely claim k > 1 aging
    exp_samples = rng.exponential(scale=50.0, size=400)
    exp_ev = (exp_samples <= 120.0).astype(int)
    exp_obs = np.minimum(exp_samples, 120.0)
    wf_exp = fit_weibull_survival(exp_obs, exp_ev, n_boot=100, seed=424242)
    assert wf_exp.shape_ci95[0] <= 1.15 <= wf_exp.shape_ci95[1] or abs(wf_exp.shape_k - 1.0) < 0.15

    km = kaplan_meier_estimator(observed, events)
    assert len(km.times) > 50
    assert np.all(np.diff(km.survival) <= 1e-12)
    assert 40.0 < km.median_lifetime < 95.0

    haz = instantaneous_hazard_rate(observed, events)
    assert float(haz["spearman_rho_time"]) > 0.7

    # Finite-size power-law scaling recovery: tau(N) = 5.0 * N^1.5
    n_sizes = [16, 32, 64, 128]
    by_n = {
        n: (5.0 * (n ** 1.5)) * rng.weibull(2.2, size=80)
        for n in n_sizes
    }
    sc = fit_finite_size_scaling(n_sizes, by_n, n_boot=200, seed=99)
    assert abs(sc.power_law_gamma - 1.5) < 0.12
    assert sc.power_law_gamma_ci95[0] < 1.5 < sc.power_law_gamma_ci95[1]


def test_cli_demo_and_benchmark(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rc_demo = cli_main(["demo", "--trials", "4", "--seed", "20260609"])
    assert rc_demo == 0
    out_demo = capsys.readouterr().out
    assert "chimera-anneal v1.0.0" in out_demo
    assert "TwoPopMeanField" in out_demo
    assert "NonlocalRing" in out_demo

    json_path = tmp_path / "bench.json"
    rc_bench = cli_main(
        [
            "benchmark",
            "--n-sizes",
            "16,32",
            "--trials",
            "5",
            "--t-max",
            "80.0",
            "--json-out",
            str(json_path),
        ]
    )
    assert rc_bench == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["n_sizes"] == [16, 32]
    assert "mean_field" in payload and "nonlocal_ring" in payload
