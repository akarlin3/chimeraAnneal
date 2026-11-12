"""Operational chimera-death detectors separating transient grazes from absorption.

In breathing two-population chimeras, up to 98% of conventional first-passage
threshold crossings (``min(R1, R2) >= sync_threshold`` sustained for
``hold_time``) self-recover into a dispersed incoherent halo—a phenomenon termed
a near-synchrony *graze*. True absorbing collapse requires a hysteresis-gated
absorption detector confirming no recovery excursion below ``recovery_threshold``
within a post-crossing verification horizon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .substrates import MeanFieldTrajectory, RingTrajectory, kemeth_curvature_g0


@dataclass(frozen=True)
class CollapseDetectionResult:
    """Operational detection summary distinguishing grazes from true absorption."""

    substrate: str
    t_graze: float
    event_graze: int
    t_abs: float
    event_abs: int
    n_grazes: int
    graze_times: tuple[float, ...]
    overtrigger_fraction: float
    mean_chimera_coherence: float
    mean_curvature_g0: float
    t_max: float


def local_curvature_threshold_gate(
    theta: np.ndarray,
    *,
    threshold_frac: float = 0.015,
    chimera_band: tuple[float, float] = (0.10, 0.92),
) -> dict[str, float | bool]:
    """Evaluate Kemeth local discrete-Laplacian curvature coherence ``g0`` on ``theta``.

    Returns ``g0`` and booleans classifying ``chimera``, ``synchronized``, or
    ``incoherent`` spatial structure.
    """
    g0 = kemeth_curvature_g0(np.asarray(theta, dtype=np.float64), threshold_frac=threshold_frac)
    lo, hi = chimera_band
    return {
        "g0": g0,
        "is_chimera": bool(lo <= g0 <= hi),
        "is_synchronized": bool(g0 > hi),
        "is_incoherent": bool(g0 < lo),
    }


def hold_time_hysteresis_gate(
    t: np.ndarray,
    signal: np.ndarray,
    *,
    enter_threshold: float = 0.85,
    enter_hold: float = 5.0,
    recovery_threshold: float = 0.80,
    recovery_hold: float = 5.0,
    verification_horizon: float = 120.0,
    direction: Literal["above", "below"] = "above",
) -> dict[str, float | int | tuple[float, ...]]:
    """Apply a two-level hold-time hysteresis gate to an order-parameter trace.

    Parameters
    ----------
    t : np.ndarray
        Monotonically increasing sample times.
    signal : np.ndarray
        Order-parameter time series. When ``direction="above"`` (e.g.
        ``min(R1, R2)`` in the two-population system), synchrony crossings enter
        above ``enter_threshold`` and recover when dipping below
        ``recovery_threshold``. When ``direction="below"`` (e.g. spatial spread
        ``rho_std`` on the ring), synchrony crossings enter below
        ``enter_threshold`` and recover when rising above ``recovery_threshold``.
    """
    tt = np.asarray(t, dtype=np.float64)
    sig = np.asarray(signal, dtype=np.float64)
    if tt.ndim != 1 or sig.shape != tt.shape or len(tt) < 2:
        raise ValueError("t and signal must be 1D arrays of identical length >= 2")

    t_max = float(tt[-1])
    if direction == "above":
        in_sync = sig >= enter_threshold
        in_rec = sig < recovery_threshold
    elif direction == "below":
        in_sync = sig <= enter_threshold
        in_rec = sig > recovery_threshold
    else:
        raise ValueError("direction must be 'above' or 'below'")

    # Locate all maximal sustained excursions into the synchronized band
    candidate_crossings: list[tuple[int, int, float]] = []
    n = len(tt)
    i = 0
    while i < n:
        if not in_sync[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and in_sync[j + 1]:
            j += 1
        if tt[j] - tt[i] >= enter_hold:
            candidate_crossings.append((i, j, float(tt[i])))
        i = j + 1

    if not candidate_crossings:
        return {
            "t_graze": t_max,
            "event_graze": 0,
            "t_abs": t_max,
            "event_abs": 0,
            "n_grazes": 0,
            "graze_times": (),
            "overtrigger_fraction": 0.0,
        }

    t_graze = candidate_crossings[0][2]
    event_graze = 1

    graze_times: list[float] = []
    t_abs = t_max
    event_abs = 0

    for start_idx, _end_idx, t_cross in candidate_crossings:
        # Inspect verification window [t_cross, min(t_max, t_cross + verification_horizon)]
        t_verify_end = min(t_max, t_cross + verification_horizon)
        end_verify_idx = int(np.searchsorted(tt, t_verify_end, side="right"))

        recovered = False
        k = start_idx + 1
        while k < end_verify_idx:
            if not in_rec[k]:
                k += 1
                continue
            m = k
            while m + 1 < n and in_rec[m + 1]:
                m += 1
            if tt[m] - tt[k] >= recovery_hold:
                recovered = True
                break
            k = m + 1

        if recovered:
            graze_times.append(t_cross)
        else:
            # Verify that the trace remained observed long enough (either >= enter_hold
            # at end of integration after stopping early, or spans a clean post-crossing window)
            if t_max - t_cross >= enter_hold:
                t_abs = t_cross
                event_abs = 1
                break

    total_firings = len(graze_times) + event_abs
    overtrigger = (len(graze_times) / total_firings) if total_firings > 0 else 0.0

    return {
        "t_graze": float(t_graze),
        "event_graze": int(event_graze),
        "t_abs": float(t_abs),
        "event_abs": int(event_abs),
        "n_grazes": len(graze_times),
        "graze_times": tuple(graze_times),
        "overtrigger_fraction": float(overtrigger),
    }


def detect_chimera_collapse(
    trajectory: MeanFieldTrajectory | RingTrajectory | None = None,
    *,
    t: np.ndarray | None = None,
    R1: np.ndarray | None = None,
    R2: np.ndarray | None = None,
    rho_std: np.ndarray | None = None,
    g0: np.ndarray | None = None,
    sync_threshold: float = 0.85,
    ring_eps: float = 0.04,
    hold_time: float = 5.0,
    recovery_threshold: float = 0.80,
    ring_recovery_eps: float = 0.08,
    recovery_hold: float = 5.0,
    verification_horizon: float = 120.0,
    burn_in: float = 10.0,
) -> CollapseDetectionResult:
    """Detect chimera collapse while separating self-healing grazes from absorption.

    Accepts either a ``MeanFieldTrajectory`` / ``RingTrajectory`` dataclass directly
    or raw order-parameter arrays ``(t, R1, R2)`` or ``(t, rho_std)``.
    """
    if isinstance(trajectory, MeanFieldTrajectory):
        tt = trajectory.t
        r1 = trajectory.R1
        r2 = trajectory.R2
        g0_arr = trajectory.g0
        substrate = "mean-field"
        m_sig = np.minimum(r1, r2)
        gate = hold_time_hysteresis_gate(
            tt,
            m_sig,
            enter_threshold=sync_threshold,
            enter_hold=hold_time,
            recovery_threshold=recovery_threshold,
            recovery_hold=recovery_hold,
            verification_horizon=verification_horizon,
            direction="above",
        )
        t_end = float(gate["t_abs"]) if int(gate["event_abs"]) == 1 else float(tt[-1])
        mask = (tt >= burn_in) & (tt < t_end)
        if not np.any(mask):
            mask = np.ones_like(tt, dtype=bool)
        coherence = float(np.mean(np.abs(r1[mask] - r2[mask])))
        mean_g0 = float(np.mean(g0_arr[mask]))
    elif isinstance(trajectory, RingTrajectory):
        tt = trajectory.t
        rs = trajectory.rho_std
        g0_arr = trajectory.g0
        substrate = "nonlocal-ring"
        gate = hold_time_hysteresis_gate(
            tt,
            rs,
            enter_threshold=ring_eps,
            enter_hold=hold_time,
            recovery_threshold=ring_recovery_eps,
            recovery_hold=recovery_hold,
            verification_horizon=verification_horizon,
            direction="below",
        )
        t_end = float(gate["t_abs"]) if int(gate["event_abs"]) == 1 else float(tt[-1])
        mask = (tt >= burn_in) & (tt < t_end)
        if not np.any(mask):
            mask = np.ones_like(tt, dtype=bool)
        coherence = float(np.mean(rs[mask]))
        mean_g0 = float(np.mean(g0_arr[mask]))
    else:
        if t is None:
            raise ValueError("Provide either `trajectory` or `t` with `(R1, R2)` or `rho_std`.")
        tt = np.asarray(t, dtype=np.float64)
        if R1 is not None and R2 is not None:
            r1 = np.asarray(R1, dtype=np.float64)
            r2 = np.asarray(R2, dtype=np.float64)
            substrate = "mean-field"
            m_sig = np.minimum(r1, r2)
            gate = hold_time_hysteresis_gate(
                tt,
                m_sig,
                enter_threshold=sync_threshold,
                enter_hold=hold_time,
                recovery_threshold=recovery_threshold,
                recovery_hold=recovery_hold,
                verification_horizon=verification_horizon,
                direction="above",
            )
            t_end = float(gate["t_abs"]) if int(gate["event_abs"]) == 1 else float(tt[-1])
            mask = (tt >= burn_in) & (tt < t_end)
            if not np.any(mask):
                mask = np.ones_like(tt, dtype=bool)
            coherence = float(np.mean(np.abs(r1[mask] - r2[mask])))
            mean_g0 = float(np.mean(g0[mask])) if g0 is not None else float("nan")
        elif rho_std is not None:
            rs = np.asarray(rho_std, dtype=np.float64)
            substrate = "nonlocal-ring"
            gate = hold_time_hysteresis_gate(
                tt,
                rs,
                enter_threshold=ring_eps,
                enter_hold=hold_time,
                recovery_threshold=ring_recovery_eps,
                recovery_hold=recovery_hold,
                verification_horizon=verification_horizon,
                direction="below",
            )
            t_end = float(gate["t_abs"]) if int(gate["event_abs"]) == 1 else float(tt[-1])
            mask = (tt >= burn_in) & (tt < t_end)
            if not np.any(mask):
                mask = np.ones_like(tt, dtype=bool)
            coherence = float(np.mean(rs[mask]))
            mean_g0 = float(np.mean(g0[mask])) if g0 is not None else float("nan")
        else:
            raise ValueError("Specify `(R1, R2)` for mean-field or `rho_std` for ring detection.")

    return CollapseDetectionResult(
        substrate=substrate,
        t_graze=float(gate["t_graze"]),
        event_graze=int(gate["event_graze"]),
        t_abs=float(gate["t_abs"]),
        event_abs=int(gate["event_abs"]),
        n_grazes=int(gate["n_grazes"]),
        graze_times=gate["graze_times"],  # type: ignore[arg-type]
        overtrigger_fraction=float(gate["overtrigger_fraction"]),
        mean_chimera_coherence=coherence,
        mean_curvature_g0=mean_g0,
        t_max=float(tt[-1]),
    )
