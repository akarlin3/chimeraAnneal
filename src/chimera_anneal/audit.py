"""Standardized AveryCorp Suite II chimera survival & collapse-scaling audit."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .detection import detect_chimera_collapse
from .hazard import fit_finite_size_scaling, fit_weibull_survival
from .substrates import (
    MeanFieldTrajectory,
    NonlocalRingSubstrate,
    ReducedTwoPopFlow,
    TwoPopMeanField,
    make_pcg64_rng,
)

SUITE_I_NAME: Literal["Suite I: Clinical Quantitative-MRI Trust Spine"] = (
    "Suite I: Clinical Quantitative-MRI Trust Spine"
)
SUITE_II_NAME: Literal["Suite II: Scientific Identifiability & Null-Audit Suite"] = (
    "Suite II: Scientific Identifiability & Null-Audit Suite"
)
SuiteLiteral = Literal[
    "Suite I: Clinical Quantitative-MRI Trust Spine",
    "Suite II: Scientific Identifiability & Null-Audit Suite",
]
StatusLiteral = Literal["PASS", "WARN", "HALT"]


def _to_builtin(val: Any) -> float | int | str | bool:
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    return str(val)


@dataclass(frozen=True)
class AuditReport:
    """Standardized AveryCorp scientific identifiability and diagnostic audit report."""

    package: str
    version: str
    suite: SuiteLiteral
    stage_role: str
    status: StatusLiteral
    verdict: str
    metrics: dict[str, float | int | str | bool]
    thresholds: dict[str, float | int | str]
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the audit report to a JSON-compatible Python dictionary."""
        return {
            "package": str(self.package),
            "version": str(self.version),
            "suite": str(self.suite),
            "stage_role": str(self.stage_role),
            "status": str(self.status),
            "verdict": str(self.verdict),
            "metrics": {str(k): _to_builtin(v) for k, v in self.metrics.items()},
            "thresholds": {str(k): _to_builtin(v) for k, v in self.thresholds.items()},
            "seed": int(self.seed),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize the audit report to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary_line(self) -> str:
        """Return a concise one-line human-readable status summary."""
        return (
            f"[{self.status}] {self.package} v{self.version} "
            f"({self.suite} | {self.stage_role}) -> {self.verdict}"
        )


def _verify_graze_hysteresis_gate(seed: int) -> dict[str, float | int | bool]:
    """Verify hysteresis collapse detector separates breathing grazes from absorbing collapse."""
    rng = make_pcg64_rng(seed)
    t = np.linspace(0.0, 120.0, 1201)
    r1 = np.ones_like(t)
    # Two transient breathing grazes near synchrony followed by true absorption at t >= 82
    r2 = 0.55 + 0.01 * rng.normal(size=t.size)
    r2[(t >= 20.0) & (t <= 26.0)] = 0.91
    r2[(t >= 50.0) & (t <= 56.0)] = 0.93
    r2[t >= 82.0] = 0.992
    r2 = np.clip(r2, 0.05, 1.0)
    psi = np.zeros_like(t)

    synth_traj = MeanFieldTrajectory(
        t=t,
        R1=r1,
        R2=r2,
        Phi1=psi,
        Phi2=psi,
        g0=0.5 * (r1 + r2),
        theta_final=np.zeros(32, dtype=np.float64),
        n_per_pop=16,
        p=0.75,
        q=0.25,
        beta=0.05,
        alpha=float(0.5 * np.pi - 0.05),
        seed=int(seed),
        method="synthetic",
        stopped_early=False,
    )
    det = detect_chimera_collapse(
        synth_traj,
        sync_threshold=0.85,
        hold_time=4.0,
        recovery_threshold=0.80,
        recovery_hold=4.0,
        verification_horizon=30.0,
    )
    separated = (
        det.n_grazes >= 2
        and det.event_abs == 1
        and (det.t_abs - det.t_graze) >= 50.0
    )
    return {
        "graze_gate_separated": bool(separated),
        "n_transient_grazes": int(det.n_grazes),
        "graze_to_absorption_gap": float(det.t_abs - det.t_graze),
    }


def run_audit(
    seed: int = 42,
    *,
    mf_trials: int = 10,
    ring_trials: int = 8,
    min_weibull_k: float = 1.0,
) -> AuditReport:
    """Run the Suite II chimera collapse graze-separation & finite-size hazard audit."""
    from . import __version__

    # 1. Parameter-free Ott-Antonsen bifurcation ordering (A_SN ~ 0.095 < A_Hopf ~ 0.270 < 0.50)
    a_sn = float(ReducedTwoPopFlow.locate_saddle_node(0.05)["A_SN"])
    a_hopf = float(ReducedTwoPopFlow.locate_hopf(0.05))
    bif_ordered = 0.08 < a_sn < a_hopf < 0.50

    # 2. Hysteresis graze-vs-absorption separation check
    graze_check = _verify_graze_hysteresis_gate(int(seed))

    # 3. Seeded Mean-Field survival & finite-size scaling ensemble
    mf_sizes = (8, 16)
    mf_taus: dict[int, list[float]] = {}
    mf_evs: dict[int, list[int]] = {}
    mf_overtrig: list[float] = []
    for n_val in mf_sizes:
        eng = TwoPopMeanField(n_per_pop=n_val, A=0.50, beta=0.05)
        t_list: list[float] = []
        e_list: list[int] = []
        for k in range(mf_trials):
            traj = eng.integrate(
                seed=int(seed) + 1000 * n_val + k,
                t_max=220.0,
                dt=0.05,
                decimate=4,
                stop_sync_eps=0.03,
                stop_hold=8.0,
            )
            det = detect_chimera_collapse(
                traj,
                sync_threshold=0.85,
                hold_time=4.0,
                recovery_threshold=0.80,
                recovery_hold=4.0,
                verification_horizon=60.0,
            )
            t_list.append(det.t_abs)
            e_list.append(det.event_abs)
            mf_overtrig.append(det.overtrigger_fraction)
        mf_taus[n_val] = t_list
        mf_evs[n_val] = e_list

    wf_mf = fit_weibull_survival(mf_taus[8], mf_evs[8], n_boot=80, seed=int(seed))
    sc_mf = fit_finite_size_scaling(mf_sizes, mf_taus, mf_evs, n_boot=80, seed=int(seed))

    # 4. Seeded Nonlocal-Ring post-boundary collapse ensemble (beta=0.145 > beta_c ~ 0.137)
    ring_n = 32
    ring = NonlocalRingSubstrate(
        n_oscillators=ring_n,
        kernel="tophat",
        coupling_radius=0.15,
        beta=0.145,
    )
    rg_taus: list[float] = []
    rg_evs: list[int] = []
    for k in range(ring_trials):
        rtraj = ring.integrate(
            seed=int(seed) + 5000 + k,
            t_max=220.0,
            dt=0.05,
            decimate=10,
            stop_eps=0.04,
            dt_hold=15.0,
        )
        rdet = detect_chimera_collapse(
            rtraj,
            ring_eps=0.04,
            hold_time=10.0,
            ring_recovery_eps=0.08,
            recovery_hold=5.0,
            verification_horizon=50.0,
        )
        rg_taus.append(rdet.t_abs)
        rg_evs.append(rdet.event_abs)
    wf_rg = fit_weibull_survival(rg_taus, rg_evs, n_boot=80, seed=int(seed) + 1)
    ring_k = float(wf_rg.shape_k) if np.isfinite(wf_rg.shape_k) else 1.0

    aging_ok = (wf_mf.shape_k >= min_weibull_k) and np.isfinite(sc_mf.power_law_gamma)
    graze_ok = bool(graze_check["graze_gate_separated"]) and bif_ordered

    if aging_ok and graze_ok:
        status: StatusLiteral = "PASS"
        verdict = (
            f"ABSORBING_HAZARD_VERIFIED: Hysteresis gate separates transient grazes "
            f"(gap = {graze_check['graze_to_absorption_gap']:.1f}), Weibull k(MF) = "
            f"{wf_mf.shape_k:.2f}, k(Ring) = {ring_k:.2f}, scaling gamma = "
            f"{sc_mf.power_law_gamma:+.3f}."
        )
    elif graze_ok:
        status = "WARN"
        verdict = (
            f"HAZARD_BORDERLINE: Graze separation passed, but Weibull shape k = "
            f"{wf_mf.shape_k:.2f} is below aging threshold {min_weibull_k:.2f}."
        )
    else:
        status = "HALT"
        verdict = (
            f"DETECTOR_FAILURE: Graze separation = {graze_ok}, "
            f"bifurcation order = {bif_ordered}."
        )

    return AuditReport(
        package="chimera-anneal",
        version=__version__,
        suite=SUITE_II_NAME,
        stage_role="Suite II: Absorbing Collapse vs. Transient Graze Separation & Finite-Size Hazard Gate",
        status=status,
        verdict=verdict,
        metrics={
            "oa_bifurcation_ordered": bif_ordered,
            "graze_gate_separated": bool(graze_check["graze_gate_separated"]),
            "n_transient_grazes_filtered": int(graze_check["n_transient_grazes"]),
            "mean_field_overtrigger_fraction": round(float(np.mean(mf_overtrig)), 6),
            "weibull_shape_k_mean_field": round(float(wf_mf.shape_k), 6),
            "weibull_shape_k_ring": round(float(ring_k), 6),
            "mean_field_power_law_gamma": round(float(sc_mf.power_law_gamma), 6),
        },
        thresholds={
            "min_weibull_shape_k": float(min_weibull_k),
            "min_transient_grazes_filtered": 2,
            "sync_threshold": 0.85,
        },
        seed=int(seed),
    )
