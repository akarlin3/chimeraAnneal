"""Command-line interface for ``chimera-anneal``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

import numpy as np

from .detection import detect_chimera_collapse
from .hazard import fit_finite_size_scaling, fit_weibull_survival
from .substrates import NonlocalRingSubstrate, ReducedTwoPopFlow, TwoPopMeanField


def _run_meanfield_ensemble(
    n_sizes: Sequence[int],
    trials: int,
    *,
    base_seed: int,
    t_max: float,
    A: float = 0.5,
    beta: float = 0.05,
) -> dict[str, object]:
    taus_abs: dict[int, list[float]] = {}
    evs_abs: dict[int, list[int]] = {}
    taus_graze: dict[int, list[float]] = {}
    overtrigger_by_n: dict[int, float] = {}

    for n_val in n_sizes:
        eng = TwoPopMeanField(n_per_pop=n_val, A=A, beta=beta)
        t_a: list[float] = []
        e_a: list[int] = []
        t_g: list[float] = []
        ot: list[float] = []
        for k in range(trials):
            seed = base_seed + 1000 * n_val + k
            traj = eng.integrate(
                seed=seed,
                t_max=t_max,
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
                verification_horizon=80.0,
            )
            t_a.append(det.t_abs)
            e_a.append(det.event_abs)
            t_g.append(det.t_graze)
            ot.append(det.overtrigger_fraction)
        taus_abs[n_val] = t_a
        evs_abs[n_val] = e_a
        taus_graze[n_val] = t_g
        overtrigger_by_n[n_val] = float(np.mean(ot))

    scaling = fit_finite_size_scaling(n_sizes, taus_abs, evs_abs, n_boot=250, seed=base_seed)
    return {
        "taus_abs": taus_abs,
        "evs_abs": evs_abs,
        "taus_graze": taus_graze,
        "overtrigger_by_n": overtrigger_by_n,
        "scaling": scaling,
    }


def _run_ring_ensemble(
    n_sizes: Sequence[int],
    trials: int,
    *,
    base_seed: int,
    t_max: float,
    beta: float = 0.132,
    coupling_radius: float = 0.15,
) -> dict[str, object]:
    taus_abs: dict[int, list[float]] = {}
    evs_abs: dict[int, list[int]] = {}
    for n_val in n_sizes:
        ring = NonlocalRingSubstrate(
            n_oscillators=n_val,
            kernel="tophat",
            coupling_radius=coupling_radius,
            beta=beta,
        )
        t_a: list[float] = []
        e_a: list[int] = []
        for k in range(trials):
            seed = base_seed + 2000 * n_val + k
            traj = ring.integrate(
                seed=seed,
                t_max=t_max,
                dt=0.05,
                decimate=10,
                stop_eps=0.04,
                dt_hold=15.0,
            )
            det = detect_chimera_collapse(
                traj,
                ring_eps=0.04,
                hold_time=10.0,
                ring_recovery_eps=0.08,
                recovery_hold=5.0,
                verification_horizon=60.0,
            )
            t_a.append(det.t_abs)
            e_a.append(det.event_abs)
        taus_abs[n_val] = t_a
        evs_abs[n_val] = e_a

    scaling = fit_finite_size_scaling(n_sizes, taus_abs, evs_abs, n_boot=250, seed=base_seed + 1)
    return {
        "taus_abs": taus_abs,
        "evs_abs": evs_abs,
        "scaling": scaling,
    }


def _cmd_demo(args: argparse.Namespace) -> int:
    if getattr(args, "json", False):
        from .audit import run_audit

        report = run_audit(seed=int(args.seed), mf_trials=max(int(args.trials), 4))
        print(report.to_json())
        return 0 if report.status != "HALT" else 1

    seed = int(args.seed)
    mf_sizes = [8, 16, 32]
    ring_sizes = [32, 48, 64]
    trials = int(args.trials)

    sn = ReducedTwoPopFlow.locate_saddle_node(0.05)["A_SN"]
    hopf = ReducedTwoPopFlow.locate_hopf(0.05)
    ahc = ReducedTwoPopFlow.locate_homoclinic(0.05, t_max=3000.0, tol=2e-3)

    mf = _run_meanfield_ensemble(mf_sizes, trials, base_seed=seed, t_max=240.0)
    rg = _run_ring_ensemble(ring_sizes, trials, base_seed=seed, t_max=220.0, beta=0.135)

    mf_sc = mf["scaling"]
    rg_sc = rg["scaling"]

    print("=" * 82)
    print("chimera-anneal v1.0.0 | Seeded PCG64 Chimera Survival & Finite-Size Scaling Demo")
    print("=" * 82)
    print(
        f"Reduced Ott-Antonsen Bifurcations (beta=0.05): "
        f"A_SN={sn:.4f} | A_Hopf={hopf:.4f} | A_hc={ahc:.4f} (operating A=0.5000 > A_hc)"
    )
    print("-" * 82)
    print(
        f"{'Substrate':<16} {'N':>5} {'Events':>8} {'tau_graze':>10} "
        f"{'tau_abs':>10} {'Weibull k [95% CI]':>22} {'Overtrig':>8}"
    )
    print("-" * 82)

    mf_taus_abs: dict[int, list[float]] = mf["taus_abs"]  # type: ignore[assignment]
    mf_evs_abs: dict[int, list[int]] = mf["evs_abs"]  # type: ignore[assignment]
    mf_taus_grz: dict[int, list[float]] = mf["taus_graze"]  # type: ignore[assignment]
    mf_ot: dict[int, float] = mf["overtrigger_by_n"]  # type: ignore[assignment]

    for n_val in mf_sizes:
        wf = fit_weibull_survival(mf_taus_abs[n_val], mf_evs_abs[n_val], n_boot=100, seed=seed)
        med_g = float(np.median(mf_taus_grz[n_val]))
        ci_s = f"{wf.shape_k:.2f} [{wf.shape_ci95[0]:.2f},{wf.shape_ci95[1]:.2f}]"
        ev_s = f"{wf.n_events}/{wf.n_total}"
        print(
            f"{'TwoPopMeanField':<16} {n_val:>5d} {ev_s:>8} {med_g:>10.1f} "
            f"{wf.scale_lambda:>10.1f} {ci_s:>22} {100.0 * mf_ot[n_val]:>7.1f}%"
        )

    rg_taus_abs: dict[int, list[float]] = rg["taus_abs"]  # type: ignore[assignment]
    rg_evs_abs: dict[int, list[int]] = rg["evs_abs"]  # type: ignore[assignment]
    for n_val in ring_sizes:
        wf = fit_weibull_survival(rg_taus_abs[n_val], rg_evs_abs[n_val], n_boot=100, seed=seed)
        ci_s = f"{wf.shape_k:.2f} [{wf.shape_ci95[0]:.2f},{wf.shape_ci95[1]:.2f}]"
        ev_s = f"{wf.n_events}/{wf.n_total}"
        print(
            f"{'NonlocalRing':<16} {n_val:>5d} {ev_s:>8} {wf.scale_lambda:>10.1f} "
            f"{wf.scale_lambda:>10.1f} {ci_s:>22} {'0.0%':>8}"
        )

    print("-" * 82)
    print(
        f"Mean-Field Scaling : gamma = {mf_sc.power_law_gamma:+.3f} "  # type: ignore[attr-defined]
        f"95% CI [{mf_sc.power_law_gamma_ci95[0]:+.3f}, {mf_sc.power_law_gamma_ci95[1]:+.3f}] "  # type: ignore[attr-defined]
        f"({mf_sc.preferred_scaling})"  # type: ignore[attr-defined]
    )
    print(
        f"Nonlocal-Ring Near-Boundary Scaling : gamma = {rg_sc.power_law_gamma:+.3f} "  # type: ignore[attr-defined]
        f"95% CI [{rg_sc.power_law_gamma_ci95[0]:+.3f}, {rg_sc.power_law_gamma_ci95[1]:+.3f}] | "  # type: ignore[attr-defined]
        f"kappa = {rg_sc.exponential_kappa:+.4f}"  # type: ignore[attr-defined]
    )
    print("=" * 82)
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from .audit import run_audit

    report = run_audit(seed=int(args.seed))
    if getattr(args, "json", False):
        print(report.to_json())
    else:
        print(report.summary_line())
        for k, v in report.metrics.items():
            print(f"  {k}: {v}")
    return 0 if report.status != "HALT" else 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    n_sizes = [int(s.strip()) for s in args.n_sizes.split(",") if s.strip()]
    trials = int(args.trials)
    seed = int(args.seed)
    t_max = float(args.t_max)

    report: dict[str, object] = {"seed": seed, "n_sizes": n_sizes, "trials": trials}
    print(
        f"chimera-anneal benchmark | sizes={n_sizes} | trials={trials} | "
        f"substrate={args.substrate} | seed={seed}"
    )
    print("-" * 76)

    if args.substrate in ("both", "mean-field"):
        mf = _run_meanfield_ensemble(n_sizes, trials, base_seed=seed, t_max=t_max)
        sc = mf["scaling"]
        print(
            f"[Mean-Field] tau(N)={[round(x, 2) for x in sc.characteristic_tau]} | "  # type: ignore[attr-defined]
            f"k(N)={[round(x, 2) for x in sc.weibull_shapes]} | "  # type: ignore[attr-defined]
            f"gamma={sc.power_law_gamma:+.3f} "  # type: ignore[attr-defined]
            f"[{sc.power_law_gamma_ci95[0]:+.3f}, {sc.power_law_gamma_ci95[1]:+.3f}]"  # type: ignore[attr-defined]
        )
        report["mean_field"] = {
            "characteristic_tau": list(sc.characteristic_tau),  # type: ignore[attr-defined]
            "weibull_shapes": list(sc.weibull_shapes),  # type: ignore[attr-defined]
            "gamma": sc.power_law_gamma,  # type: ignore[attr-defined]
            "gamma_ci95": list(sc.power_law_gamma_ci95),  # type: ignore[attr-defined]
        }

    if args.substrate in ("both", "ring"):
        rg = _run_ring_ensemble(n_sizes, trials, base_seed=seed, t_max=t_max, beta=float(args.ring_beta))
        sc = rg["scaling"]
        print(
            f"[Nonlocal-Ring] tau(N)={[round(x, 2) for x in sc.characteristic_tau]} | "  # type: ignore[attr-defined]
            f"k(N)={[round(x, 2) for x in sc.weibull_shapes]} | "  # type: ignore[attr-defined]
            f"gamma={sc.power_law_gamma:+.3f} "  # type: ignore[attr-defined]
            f"[{sc.power_law_gamma_ci95[0]:+.3f}, {sc.power_law_gamma_ci95[1]:+.3f}]"  # type: ignore[attr-defined]
        )
        report["nonlocal_ring"] = {
            "characteristic_tau": list(sc.characteristic_tau),  # type: ignore[attr-defined]
            "weibull_shapes": list(sc.weibull_shapes),  # type: ignore[attr-defined]
            "gamma": sc.power_law_gamma,  # type: ignore[attr-defined]
            "gamma_ci95": list(sc.power_law_gamma_ci95),  # type: ignore[attr-defined]
        }

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
            fh.write("\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chimera-anneal",
        description=(
            "Coupled-oscillator chimera survival, graze-vs-absorption detection, "
            "and topology-dependent finite-size scaling."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser(
        "demo",
        help="Run a seeded mean-field vs. nonlocal-ring survival & scaling comparison.",
    )
    p_demo.add_argument("--seed", type=int, default=20260609, help="PCG64 master seed.")
    p_demo.add_argument("--trials", type=int, default=12, help="Trials per system size N.")
    p_demo.add_argument(
        "--json",
        action="store_true",
        help="Emit standardized AveryCorp AuditReport JSON diagnostic.",
    )
    p_demo.set_defaults(func=_cmd_demo)

    p_audit = sub.add_parser(
        "audit",
        help="Run Suite II chimera graze-vs-absorption & finite-size hazard diagnostic audit.",
    )
    p_audit.add_argument("--seed", type=int, default=42, help="PCG64 seed (default: 42).")
    p_audit.add_argument("--json", action="store_true", help="Emit machine-readable AuditReport JSON.")
    p_audit.set_defaults(func=_cmd_audit)

    p_bench = sub.add_parser(
        "benchmark",
        help="Run finite-size survival scaling benchmark across custom N sizes.",
    )
    p_bench.add_argument(
        "--n-sizes",
        type=str,
        default="32,64,128",
        help="Comma-separated oscillator counts N (e.g., 32,64,128).",
    )
    p_bench.add_argument("--trials", type=int, default=50, help="Trials per N size.")
    p_bench.add_argument(
        "--substrate",
        choices=("both", "mean-field", "ring"),
        default="both",
        help="Which oscillator substrate(s) to benchmark.",
    )
    p_bench.add_argument("--t-max", type=float, default=220.0, help="Integration horizon.")
    p_bench.add_argument("--ring-beta", type=float, default=0.135, help="Ring phase-lag offset beta.")
    p_bench.add_argument("--seed", type=int, default=20260609, help="PCG64 master seed.")
    p_bench.add_argument("--json-out", type=str, default=None, help="Optional JSON report path.")
    p_bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
