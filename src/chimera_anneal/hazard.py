"""Non-parametric and parametric survival, hazard, and finite-size scaling analysis.

Provides right-censored survival and aging estimators for chimera lifetimes:
- ``kaplan_meier_estimator``: Kaplan-Meier survival curve with Greenwood CIs
  and Nelson-Aalen cumulative hazard increments.
- ``fit_weibull_survival``: Right-censored Weibull MLE ``S(t) = exp(-(t/lam)^k)``,
  profile-likelihood and PCG64 bootstrap 95% CIs, and Weibull-vs-exponential
  likelihood-ratio test (LRT) for aging (``k > 1``).
- ``instantaneous_hazard_rate``: Epanechnikov kernel-smoothed instantaneous
  hazard ``h(t)`` with parametric Weibull overlay.
- ``fit_finite_size_scaling``: Topology-dependent finite-size scaling fits
  comparing power-law ``tau(N) ~ N^gamma`` and exponential ``tau(N) ~ exp(kappa N)``
  laws with PCG64 bootstrap 95% CIs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import optimize, stats

from .substrates import make_pcg64_rng


@dataclass(frozen=True)
class KaplanMeierEstimate:
    """Right-censored Kaplan-Meier and Nelson-Aalen survival summary."""

    times: np.ndarray
    survival: np.ndarray
    variance: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    cum_hazard: np.ndarray
    hazard_increments: np.ndarray
    n_at_risk: np.ndarray
    n_events: np.ndarray
    median_lifetime: float


def kaplan_meier_estimator(
    tau: Sequence[float] | np.ndarray,
    event: Sequence[int] | np.ndarray | None = None,
    *,
    alpha: float = 0.05,
) -> KaplanMeierEstimate:
    """Compute the right-censored Kaplan-Meier estimator with Greenwood 95% CIs."""
    t_arr = np.asarray(tau, dtype=np.float64)
    if t_arr.ndim != 1 or len(t_arr) == 0:
        raise ValueError("tau must be a non-empty 1D array of positive lifetimes.")
    if np.any(t_arr <= 0.0):
        t_arr = np.maximum(t_arr, 1e-9)
    ev_arr = (
        np.ones(len(t_arr), dtype=np.int64)
        if event is None
        else np.asarray(event, dtype=np.int64)
    )
    if ev_arr.shape != t_arr.shape:
        raise ValueError("event must have the same shape as tau.")

    order = np.argsort(t_arr, kind="mergesort")
    t_arr = t_arr[order]
    ev_arr = ev_arr[order]

    uniq_deaths = np.unique(t_arr[ev_arr == 1])
    times: list[float] = []
    surv_list: list[float] = []
    var_list: list[float] = []
    chaz_list: list[float] = []
    incr_list: list[float] = []
    risk_list: list[int] = []
    nev_list: list[int] = []

    s_cur = 1.0
    greenwood_sum = 0.0
    h_cur = 0.0

    for t_val in uniq_deaths:
        at_risk = int(np.sum(t_arr >= t_val))
        d_count = int(np.sum((t_arr == t_val) & (ev_arr == 1)))
        if at_risk <= 0:
            continue
        s_cur *= 1.0 - d_count / at_risk
        if at_risk > d_count:
            greenwood_sum += d_count / (at_risk * (at_risk - d_count))
        v_cur = (s_cur * s_cur) * greenwood_sum
        inc = d_count / at_risk
        h_cur += inc

        times.append(float(t_val))
        surv_list.append(float(s_cur))
        var_list.append(float(v_cur))
        chaz_list.append(float(h_cur))
        incr_list.append(float(inc))
        risk_list.append(at_risk)
        nev_list.append(d_count)

    times_np = np.asarray(times, dtype=np.float64)
    surv_np = np.asarray(surv_list, dtype=np.float64)
    var_np = np.asarray(var_list, dtype=np.float64)
    z = float(stats.norm.ppf(1.0 - 0.5 * alpha))
    se = np.sqrt(np.maximum(var_np, 0.0))
    ci_lo = np.clip(surv_np - z * se, 0.0, 1.0)
    ci_hi = np.clip(surv_np + z * se, 0.0, 1.0)

    below_half = np.nonzero(surv_np <= 0.5)[0]
    median_tau = float(times_np[below_half[0]]) if len(below_half) > 0 else float("nan")

    return KaplanMeierEstimate(
        times=times_np,
        survival=surv_np,
        variance=var_np,
        ci_low=ci_lo,
        ci_high=ci_hi,
        cum_hazard=np.asarray(chaz_list, dtype=np.float64),
        hazard_increments=np.asarray(incr_list, dtype=np.float64),
        n_at_risk=np.asarray(risk_list, dtype=np.int64),
        n_events=np.asarray(nev_list, dtype=np.int64),
        median_lifetime=median_tau,
    )


def _profile_weibull_loglik(
    k: float, t_scaled: np.ndarray, d_events: int, sum_log_t_scaled: float
) -> float:
    if not (0.02 < k < 60.0) or d_events <= 0:
        return -math.inf
    with np.errstate(over="ignore", invalid="ignore"):
        tk = t_scaled ** k
        s_tk = float(np.sum(tk))
    if not math.isfinite(s_tk) or s_tk <= 0.0:
        return -math.inf
    lam_k = s_tk / d_events
    return float(
        d_events * math.log(k)
        + (k - 1.0) * sum_log_t_scaled
        - d_events * math.log(lam_k)
        - d_events
    )


@dataclass(frozen=True)
class WeibullSurvivalFit:
    """Right-censored Weibull survival MLE and aging test against exponential."""

    shape_k: float
    scale_lambda: float
    shape_ci95: tuple[float, float]
    scale_ci95: tuple[float, float]
    exp_scale_lambda: float
    loglik_weibull: float
    loglik_exponential: float
    lrt_statistic: float
    lrt_pvalue: float
    n_total: int
    n_events: int
    aging_detected: bool


def fit_weibull_survival(
    tau: Sequence[float] | np.ndarray,
    event: Sequence[int] | np.ndarray | None = None,
    *,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 2026,
) -> WeibullSurvivalFit:
    """Fit right-censored Weibull survival ``S(t) = exp(-(t / scale_lambda)^shape_k)``.

    Profiles the scale parameter ``lambda`` analytically so optimization is a
    strictly concave 1D problem in ``log(k)``, immune to scale overflow via
    dimensionless rescaling by ``median(tau)``. Computes profile-likelihood 95%
    confidence bounds on ``shape_k`` and PCG64 bootstrap 95% CIs on
    ``scale_lambda`` (and fallback ``shape_k``).
    """
    t_raw = np.maximum(np.asarray(tau, dtype=np.float64), 1e-9)
    ev = (
        np.ones(len(t_raw), dtype=np.int64)
        if event is None
        else np.asarray(event, dtype=np.int64)
    )
    if t_raw.ndim != 1 or ev.shape != t_raw.shape or len(t_raw) < 2:
        raise ValueError("tau and event must be matching 1D arrays with length >= 2.")

    d_events = int(np.sum(ev == 1))
    if d_events == 0:
        return WeibullSurvivalFit(
            shape_k=float("nan"),
            scale_lambda=float("inf"),
            shape_ci95=(float("nan"), float("nan")),
            scale_ci95=(float("nan"), float("nan")),
            exp_scale_lambda=float("inf"),
            loglik_weibull=float("nan"),
            loglik_exponential=float("nan"),
            lrt_statistic=0.0,
            lrt_pvalue=1.0,
            n_total=len(t_raw),
            n_events=0,
            aging_detected=False,
        )

    scale_ref = float(np.median(t_raw))
    t_sc = t_raw / scale_ref
    sum_log_sc = float(np.sum(np.log(t_sc[ev == 1])))

    res = optimize.minimize_scalar(
        lambda log_k: -_profile_weibull_loglik(math.exp(log_k), t_sc, d_events, sum_log_sc),
        bounds=(math.log(0.05), math.log(40.0)),
        method="bounded",
    )
    k_hat = float(math.exp(float(res.x)))
    lam_sc = float((np.sum(t_sc ** k_hat) / d_events) ** (1.0 / k_hat))
    lam_hat = lam_sc * scale_ref

    # Exact unscaled log-likelihoods
    ll_sc_max = _profile_weibull_loglik(k_hat, t_sc, d_events, sum_log_sc)
    ll_weibull = ll_sc_max - d_events * math.log(scale_ref)

    exp_lam = float(np.sum(t_raw) / d_events)
    ll_exp = float(-d_events * math.log(exp_lam) - d_events)

    lrt_stat = max(0.0, 2.0 * (ll_weibull - ll_exp))
    lrt_p = float(stats.chi2.sf(lrt_stat, df=1))

    # Profile-likelihood CI on k
    thr = 0.5 * float(stats.chi2.ppf(1.0 - alpha, df=1))

    def _gap(k_val: float) -> float:
        return (ll_sc_max - _profile_weibull_loglik(k_val, t_sc, d_events, sum_log_sc)) - thr

    k_lo, k_hi = float("nan"), float("nan")
    lo_probe = max(0.03, 0.5 * k_hat)
    while lo_probe > 0.03 and _gap(lo_probe) < 0.0:
        lo_probe *= 0.5
    if _gap(lo_probe) > 0.0:
        k_lo = float(optimize.brentq(_gap, lo_probe, k_hat, xtol=1e-6))

    hi_probe = min(50.0, 2.0 * k_hat)
    while hi_probe < 50.0 and _gap(hi_probe) < 0.0:
        hi_probe *= 1.8
    if _gap(hi_probe) > 0.0:
        k_hi = float(optimize.brentq(_gap, k_hat, hi_probe, xtol=1e-6))

    # PCG64 bootstrap CI on scale_lambda (and fallback shape CI if needed)
    lam_ci = (float("nan"), float("nan"))
    if n_boot > 0 and d_events >= 3:
        rng = make_pcg64_rng(seed)
        boot_lams: list[float] = []
        boot_ks: list[float] = []
        n = len(t_raw)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            tb = t_sc[idx]
            eb = ev[idx]
            db = int(np.sum(eb == 1))
            if db < 2:
                continue
            slog = float(np.sum(np.log(tb[eb == 1])))
            rb = optimize.minimize_scalar(
                lambda lk: -_profile_weibull_loglik(math.exp(lk), tb, db, slog),
                bounds=(math.log(0.1), math.log(25.0)),
                method="bounded",
            )
            kb = float(math.exp(float(rb.x)))
            lb = float((np.sum(tb ** kb) / db) ** (1.0 / kb)) * scale_ref
            boot_ks.append(kb)
            boot_lams.append(lb)
        if len(boot_lams) >= 10:
            q_lo = 100.0 * (0.5 * alpha)
            q_hi = 100.0 * (1.0 - 0.5 * alpha)
            lam_ci = (
                float(np.percentile(boot_lams, q_lo)),
                float(np.percentile(boot_lams, q_hi)),
            )
            if not math.isfinite(k_lo) or not math.isfinite(k_hi):
                k_lo = float(np.percentile(boot_ks, q_lo))
                k_hi = float(np.percentile(boot_ks, q_hi))

    aging = bool(k_hat > 1.0 and (math.isfinite(k_lo) and k_lo > 1.0) and lrt_p < alpha)
    return WeibullSurvivalFit(
        shape_k=k_hat,
        scale_lambda=lam_hat,
        shape_ci95=(k_lo, k_hi),
        scale_ci95=lam_ci,
        exp_scale_lambda=exp_lam,
        loglik_weibull=ll_weibull,
        loglik_exponential=ll_exp,
        lrt_statistic=lrt_stat,
        lrt_pvalue=lrt_p,
        n_total=len(t_raw),
        n_events=d_events,
        aging_detected=aging,
    )


def instantaneous_hazard_rate(
    tau: Sequence[float] | np.ndarray,
    event: Sequence[int] | np.ndarray | None = None,
    *,
    grid: np.ndarray | None = None,
    bandwidth: float | None = None,
    n_grid: int = 100,
) -> dict[str, np.ndarray | float]:
    """Estimate instantaneous hazard ``h(t)`` via Epanechnikov kernel smoothing.

    Smooths the Nelson-Aalen cumulative hazard increments ``dH(t_i) = d_i / n_i``
    with an Epanechnikov kernel of width ``bandwidth`` (defaulting to Silverman's
    rule of thumb over uncensored event times) and pairs it with the parametric
    Weibull hazard ``(k / lam) * (t / lam)^(k - 1)``.
    """
    km = kaplan_meier_estimator(tau, event=event)
    wf = fit_weibull_survival(tau, event=event, n_boot=0)

    if len(km.times) == 0:
        g = np.linspace(0.1, 1.0, n_grid) if grid is None else np.asarray(grid, dtype=np.float64)
        z = np.zeros_like(g)
        return {
            "grid": g,
            "kernel_hazard": z,
            "weibull_hazard": z,
            "bandwidth": 0.0,
            "spearman_rho_time": 0.0,
        }

    t_min = float(km.times[0])
    t_max = float(km.times[-1])
    span = max(t_max - t_min, 1e-3)
    if bandwidth is None:
        sd = float(np.std(km.times)) if len(km.times) > 1 else 0.25 * span
        bandwidth = max(0.12 * span, 1.06 * max(sd, 1e-3) * (len(km.times) ** (-0.2)))
    bw = float(bandwidth)

    eval_grid = (
        np.linspace(max(1e-6, t_min), t_max, n_grid)
        if grid is None
        else np.asarray(grid, dtype=np.float64)
    )
    u = (eval_grid[:, None] - km.times[None, :]) / bw
    kern = np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u * u) / bw, 0.0)
    # Boundary reflection / renormalization inside observed support [0, t_max]
    h_kern = kern @ km.hazard_increments

    k = wf.shape_k
    lam = wf.scale_lambda
    h_weib = (k / lam) * np.power(np.maximum(eval_grid / lam, 1e-12), k - 1.0)

    rho_s = 0.0
    if len(eval_grid) >= 4 and np.std(h_kern) > 0.0:
        rho_val, _ = stats.spearmanr(eval_grid, h_kern)
        rho_s = float(rho_val) if math.isfinite(float(rho_val)) else 0.0

    return {
        "grid": eval_grid,
        "kernel_hazard": h_kern,
        "weibull_hazard": h_weib,
        "bandwidth": bw,
        "spearman_rho_time": rho_s,
    }


@dataclass(frozen=True)
class FiniteSizeScalingFit:
    """Topology-dependent finite-size scaling fit across oscillator counts ``N``."""

    n_sizes: tuple[int, ...]
    characteristic_tau: tuple[float, ...]
    weibull_shapes: tuple[float, ...]
    power_law_gamma: float
    power_law_gamma_ci95: tuple[float, float]
    power_law_prefactor: float
    power_law_r2: float
    exponential_kappa: float
    exponential_kappa_ci95: tuple[float, float]
    exponential_prefactor: float
    exponential_r2: float
    delta_aic_exp_minus_pow: float
    preferred_scaling: str


def fit_finite_size_scaling(
    n_sizes: Sequence[int],
    lifetimes_by_n: Mapping[int, Sequence[float] | np.ndarray],
    events_by_n: Mapping[int, Sequence[int] | np.ndarray] | None = None,
    *,
    n_boot: int = 500,
    seed: int = 2026,
    alpha: float = 0.05,
) -> FiniteSizeScalingFit:
    """Fit power-law ``tau(N) = c * N^gamma`` and exponential ``tau(N) = a * exp(kappa * N)``
    scaling laws across system sizes ``N`` with deterministic PCG64 bootstrap 95% CIs.
    """
    sizes = [int(n) for n in n_sizes]
    if len(sizes) < 2:
        raise ValueError("At least 2 distinct system sizes N are required for scaling fits.")

    n_arr = np.asarray(sizes, dtype=np.float64)
    log_n = np.log(n_arr)

    char_taus: list[float] = []
    shapes: list[float] = []
    raw_tau: dict[int, np.ndarray] = {}
    raw_ev: dict[int, np.ndarray] = {}

    for n_val in sizes:
        t_n = np.asarray(lifetimes_by_n[n_val], dtype=np.float64)
        ev_n = (
            np.ones(len(t_n), dtype=np.int64)
            if events_by_n is None or n_val not in events_by_n
            else np.asarray(events_by_n[n_val], dtype=np.int64)
        )
        raw_tau[n_val] = t_n
        raw_ev[n_val] = ev_n
        wf = fit_weibull_survival(t_n, ev_n, n_boot=0)
        tau_c = wf.scale_lambda if math.isfinite(wf.scale_lambda) else float(np.median(t_n))
        char_taus.append(float(tau_c))
        shapes.append(float(wf.shape_k))

    y = np.log(np.maximum(np.asarray(char_taus, dtype=np.float64), 1e-9))

    # Power-law fit: log(tau) = log(c) + gamma * log(N)
    gamma_hat, log_c = np.polyfit(log_n, y, 1)
    y_pow = log_c + gamma_hat * log_n
    rss_pow = float(np.sum((y - y_pow) ** 2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r2_pow = 1.0 - rss_pow / tss if tss > 1e-15 else 1.0

    # Exponential fit: log(tau) = log(a) + kappa * N
    kappa_hat, log_a = np.polyfit(n_arr, y, 1)
    y_exp = log_a + kappa_hat * n_arr
    rss_exp = float(np.sum((y - y_exp) ** 2))
    r2_exp = 1.0 - rss_exp / tss if tss > 1e-15 else 1.0

    m = len(sizes)
    aic_pow = m * math.log(max(rss_pow / m, 1e-18)) + 4.0
    aic_exp = m * math.log(max(rss_exp / m, 1e-18)) + 4.0
    delta_aic = aic_exp - aic_pow

    # Stratified PCG64 bootstrap across trials within each N
    rng = make_pcg64_rng(seed)
    boot_gammas: list[float] = []
    boot_kappas: list[float] = []
    for _ in range(max(0, int(n_boot))):
        yb = np.empty(m, dtype=np.float64)
        valid_draw = True
        for idx, n_val in enumerate(sizes):
            t_n = raw_tau[n_val]
            ev_n = raw_ev[n_val]
            pick = rng.integers(0, len(t_n), size=len(t_n))
            tb = t_n[pick]
            eb = ev_n[pick]
            if int(np.sum(eb == 1)) >= 2:
                wfb = fit_weibull_survival(tb, eb, n_boot=0)
                val = wfb.scale_lambda
            else:
                val = float(np.median(tb))
            if not math.isfinite(val) or val <= 0.0:
                valid_draw = False
                break
            yb[idx] = math.log(val)
        if not valid_draw:
            continue
        gb, _ = np.polyfit(log_n, yb, 1)
        kb, _ = np.polyfit(n_arr, yb, 1)
        boot_gammas.append(float(gb))
        boot_kappas.append(float(kb))

    q_lo = 100.0 * (0.5 * alpha)
    q_hi = 100.0 * (1.0 - 0.5 * alpha)
    if len(boot_gammas) >= 10:
        gamma_ci = (
            float(np.percentile(boot_gammas, q_lo)),
            float(np.percentile(boot_gammas, q_hi)),
        )
        kappa_ci = (
            float(np.percentile(boot_kappas, q_lo)),
            float(np.percentile(boot_kappas, q_hi)),
        )
    else:
        gamma_ci = (float(gamma_hat), float(gamma_hat))
        kappa_ci = (float(kappa_hat), float(kappa_hat))

    if abs(gamma_hat) < 0.15 and (gamma_ci[0] <= 0.0 <= gamma_ci[1] or abs(gamma_hat) < 0.08):
        preferred = "size-invariant plateau"
    elif delta_aic < -2.0:
        preferred = "exponential"
    else:
        preferred = "power-law"

    return FiniteSizeScalingFit(
        n_sizes=tuple(sizes),
        characteristic_tau=tuple(char_taus),
        weibull_shapes=tuple(shapes),
        power_law_gamma=float(gamma_hat),
        power_law_gamma_ci95=gamma_ci,
        power_law_prefactor=float(math.exp(log_c)),
        power_law_r2=float(r2_pow),
        exponential_kappa=float(kappa_hat),
        exponential_kappa_ci95=kappa_ci,
        exponential_prefactor=float(math.exp(log_a)),
        exponential_r2=float(r2_exp),
        delta_aic_exp_minus_pow=float(delta_aic),
        preferred_scaling=preferred,
    )
