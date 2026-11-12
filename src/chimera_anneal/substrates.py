"""Coupled phase-oscillator chimera substrates and reduced Ott-Antonsen flow.

Implements the two canonical chimera substrates studied in the companion
*Anneal* manuscript:
1. ``TwoPopMeanField``: Two-population Sakaguchi-Kuramoto coupled oscillator
   engine with intra-/inter-population couplings ``(p, q)`` (or coupling
   disparity ``A = p - q`` with ``p + q = 1``), phase lag ``alpha = pi/2 - beta``,
   and deterministic ``PCG64`` initialization + RK4 / DOP853 integration.
2. ``NonlocalRingSubstrate``: Nonlocally coupled ``N``-oscillator ring with
   top-hat or cosine coupling kernel ``G(x - y)`` near the chimera existence
   boundary, supporting optional ``numba`` acceleration (``chimera-anneal[fast]``)
   with a fast vectorized NumPy/FFT fallback.
3. ``ReducedTwoPopFlow``: Parameter-free 2D and 3D Ott-Antonsen /
   Watanabe-Strogatz macroscopic collective flow validator (Abrams et al., 2008).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import uniform_filter1d
from scipy.optimize import brentq

try:
    from numba import njit  # type: ignore[import-not-found]

    HAS_NUMBA = True
except ImportError:  # pragma: no cover - standard NumPy/SciPy-only install
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _decorator(fn):
            return fn

        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _decorator


def make_pcg64_rng(seed: int) -> np.random.Generator:
    """Create a strictly deterministic cross-platform PCG64 random generator."""
    return np.random.Generator(np.random.PCG64(int(seed)))


def kemeth_curvature_g0(theta: np.ndarray, threshold_frac: float = 0.01) -> float:
    """Compute the Kemeth-Haugland-Krischer discrete Laplacian curvature metric g0.

    For phase configuration ``theta`` on a ring (or concatenated subpopulations),
    computes wrapped second differences ``D_m = wrap(theta_{m+1} - 2 theta_m + theta_{m-1})``
    with maximum possible magnitude ``D_max = pi`` (or ``4`` on unrolled phases;
    here wrapped differences lie in ``[-pi, pi]``, so ``D_max = 2*pi`` for second
    differences of wrapped first differences) and returns the fraction of sites
    with ``|D_m| < threshold_frac * D_max``. Genuine chimeras satisfy ``0 < g0 < 1``,
    whereas full synchrony yields ``g0 = 1`` and pure incoherence yields ``g0 ~ 0``.
    """
    d1 = np.angle(np.exp(1j * (np.roll(theta, -1) - theta)))
    d2 = d1 - np.roll(d1, 1)
    d_max = 2.0 * math.pi
    return float(np.mean(np.abs(d2) < threshold_frac * d_max))


# ---------------------------------------------------------------------------
# 1. Two-Population Sakaguchi-Kuramoto Mean-Field Substrate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MeanFieldTrajectory:
    """Recorded order-parameter trajectory from ``TwoPopMeanField``."""

    t: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    Phi1: np.ndarray
    Phi2: np.ndarray
    g0: np.ndarray
    theta_final: np.ndarray
    n_per_pop: int
    p: float
    q: float
    beta: float
    alpha: float
    seed: int
    method: str
    stopped_early: bool


class TwoPopMeanField:
    """Two-population Sakaguchi-Kuramoto coupled phase-oscillator engine.

    Governs two identical populations ``sigma in {1, 2}`` of ``n_per_pop`` phase
    oscillators each (``2 * n_per_pop`` total):

        d(theta_i^sigma)/dt = omega
            + (p / N) sum_j sin(theta_j^sigma - theta_i^sigma - alpha)
            + (q / N) sum_j sin(theta_j^{sigma'} - theta_i^sigma - alpha)

    where ``p`` is intra-population coupling, ``q`` is inter-population coupling
    (related to Abrams-Strogatz disparity ``A`` via ``p = (1 + A)/2``,
    ``q = (1 - A)/2``), and ``alpha = pi/2 - beta``. Using complex population
    order parameters ``Z_sigma = R_sigma exp(i Phi_sigma)``, the all-to-all sum
    evaluates identically in ``O(N)`` operations per step.
    """

    def __init__(
        self,
        n_per_pop: int = 16,
        *,
        p: float | None = None,
        q: float | None = None,
        A: float = 0.5,
        beta: float = 0.05,
        omega: float = 0.0,
    ) -> None:
        if n_per_pop < 2:
            raise ValueError("n_per_pop must be >= 2")
        if (p is None) != (q is None):
            raise ValueError("Provide both (p, q) or neither (using disparity A).")
        if p is None or q is None:
            p_val = 0.5 * (1.0 + float(A))
            q_val = 0.5 * (1.0 - float(A))
        else:
            p_val = float(p)
            q_val = float(q)
        if p_val <= 0.0 or q_val < 0.0:
            raise ValueError("Couplings must satisfy p > 0 and q >= 0.")

        self.n_per_pop = int(n_per_pop)
        self.p = p_val
        self.q = q_val
        self.A = (p_val - q_val) / (p_val + q_val)
        self.beta = float(beta)
        self.alpha = 0.5 * math.pi - self.beta
        self.omega = float(omega)

    def initial_conditions(
        self,
        seed: int = 100015,
        *,
        coherent_scale: float = 0.05,
        incoherent_span: float = math.pi,
    ) -> np.ndarray:
        """Generate deterministic PCG64 chimera initial conditions (length ``2N``).

        Population 1 starts tightly clustered (Gaussian with standard deviation
        ``coherent_scale``), and Population 2 starts dispersed uniformly over
        ``[-incoherent_span, +incoherent_span]``.
        """
        rng = make_pcg64_rng(seed)
        th1 = rng.normal(loc=0.0, scale=coherent_scale, size=self.n_per_pop)
        th2 = rng.uniform(low=-incoherent_span, high=incoherent_span, size=self.n_per_pop)
        return np.concatenate([th1, th2]).astype(np.float64)

    def order_parameters(self, theta: np.ndarray) -> tuple[float, float, float, float]:
        """Return ``(R1, R2, Phi1, Phi2)`` for concatenated state ``theta``."""
        n = self.n_per_pop
        z1 = np.exp(1j * theta[:n]).mean()
        z2 = np.exp(1j * theta[n:]).mean()
        return (
            float(np.abs(z1)),
            float(np.abs(z2)),
            float(np.angle(z1)),
            float(np.angle(z2)),
        )

    def _deriv_and_obs(self, x: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
        n = self.n_per_pop
        th1 = x[:n]
        th2 = x[n:]
        z1 = np.exp(1j * th1).mean()
        z2 = np.exp(1j * th2).mean()
        r1 = float(np.abs(z1))
        phi1 = float(np.angle(z1))
        r2 = float(np.abs(z2))
        phi2 = float(np.angle(z2))

        out = np.empty_like(x)
        out[:n] = (
            self.omega
            + self.p * r1 * np.sin(phi1 - th1 - self.alpha)
            + self.q * r2 * np.sin(phi2 - th1 - self.alpha)
        )
        out[n:] = (
            self.omega
            + self.p * r2 * np.sin(phi2 - th2 - self.alpha)
            + self.q * r1 * np.sin(phi1 - th2 - self.alpha)
        )
        return out, r1, r2, phi1, phi2

    def rhs(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate ``d(theta)/dt`` at phase vector ``theta``."""
        d, _, _, _, _ = self._deriv_and_obs(np.asarray(theta, dtype=np.float64))
        return d

    def integrate(
        self,
        *,
        seed: int = 100015,
        theta0: np.ndarray | None = None,
        t_max: float = 300.0,
        dt: float = 0.05,
        decimate: int = 4,
        method: Literal["rk4", "dop853"] = "rk4",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        stop_sync_eps: float | None = None,
        stop_hold: float = 10.0,
    ) -> MeanFieldTrajectory:
        """Integrate the ``2N``-oscillator system deterministically.

        Parameters
        ----------
        seed : int
            Seed for ``PCG64`` initial-condition generation when ``theta0`` is None.
        theta0 : np.ndarray, optional
            Explicit initial phases of length ``2 * n_per_pop``.
        t_max : float
            Integration horizon.
        dt : float
            Fixed step size for ``method="rk4"`` (or base sample grid spacing for
            ``method="dop853"``).
        decimate : int
            Record macroscopic order parameters every ``decimate`` steps.
        method : {"rk4", "dop853"}
            Deterministic fixed-step 4th-order Runge-Kutta or adaptive 8th-order
            Dormand-Prince (``scipy.integrate.solve_ivp``).
        stop_sync_eps : float, optional
            If set during ``rk4`` integration, halt once ``|R1 - R2| < stop_sync_eps``
            AND ``min(R1, R2) > 1 - stop_sync_eps`` continuously for ``stop_hold``
            time units (strict post-absorption early stop).
        """
        x0 = (
            self.initial_conditions(seed=seed)
            if theta0 is None
            else np.asarray(theta0, dtype=np.float64).copy()
        )
        if x0.shape != (2 * self.n_per_pop,):
            raise ValueError(f"Expected theta0 of shape ({2 * self.n_per_pop},), got {x0.shape}")

        decimate = max(1, int(decimate))
        method_norm = method.lower()

        if method_norm == "dop853":
            n_steps = max(1, int(round(t_max / (dt * decimate))))
            t_eval = np.linspace(0.0, float(t_max), n_steps + 1)
            sol = solve_ivp(
                lambda _t, y: self.rhs(y),
                (0.0, float(t_max)),
                x0,
                method="DOP853",
                t_eval=t_eval,
                rtol=rtol,
                atol=atol,
            )
            m = sol.y.shape[1]
            r1_arr = np.empty(m, dtype=np.float64)
            r2_arr = np.empty(m, dtype=np.float64)
            p1_arr = np.empty(m, dtype=np.float64)
            p2_arr = np.empty(m, dtype=np.float64)
            g0_arr = np.empty(m, dtype=np.float64)
            for idx in range(m):
                xk = sol.y[:, idx]
                r1, r2, p1, p2 = self.order_parameters(xk)
                r1_arr[idx] = r1
                r2_arr[idx] = r2
                p1_arr[idx] = p1
                p2_arr[idx] = p2
                g0_arr[idx] = 0.5 * (
                    kemeth_curvature_g0(xk[: self.n_per_pop], threshold_frac=0.02)
                    + kemeth_curvature_g0(xk[self.n_per_pop :], threshold_frac=0.02)
                )
            return MeanFieldTrajectory(
                t=sol.t.copy(),
                R1=r1_arr,
                R2=r2_arr,
                Phi1=p1_arr,
                Phi2=p2_arr,
                g0=g0_arr,
                theta_final=sol.y[:, -1].copy(),
                n_per_pop=self.n_per_pop,
                p=self.p,
                q=self.q,
                beta=self.beta,
                alpha=self.alpha,
                seed=int(seed),
                method="dop853",
                stopped_early=False,
            )

        if method_norm != "rk4":
            raise ValueError(f"Unsupported integration method {method!r}; expected 'rk4' or 'dop853'.")

        n_steps = max(1, int(round(t_max / dt)))
        cap = n_steps // decimate + 3
        t_out = np.empty(cap, dtype=np.float64)
        r1_out = np.empty(cap, dtype=np.float64)
        r2_out = np.empty(cap, dtype=np.float64)
        p1_out = np.empty(cap, dtype=np.float64)
        p2_out = np.empty(cap, dtype=np.float64)
        g0_out = np.empty(cap, dtype=np.float64)

        x = x0
        idx = 0
        below_start: float | None = None
        stopped_early = False

        for step in range(n_steps):
            k1, r1, r2, p1, p2 = self._deriv_and_obs(x)
            t_now = step * dt
            if step % decimate == 0:
                t_out[idx] = t_now
                r1_out[idx] = r1
                r2_out[idx] = r2
                p1_out[idx] = p1
                p2_out[idx] = p2
                g0_out[idx] = 0.5 * (
                    kemeth_curvature_g0(x[: self.n_per_pop], threshold_frac=0.02)
                    + kemeth_curvature_g0(x[self.n_per_pop :], threshold_frac=0.02)
                )
                idx += 1

            if stop_sync_eps is not None:
                if abs(r1 - r2) < stop_sync_eps and min(r1, r2) > (1.0 - stop_sync_eps):
                    if below_start is None:
                        below_start = t_now
                    elif t_now - below_start >= stop_hold:
                        if step % decimate != 0:
                            t_out[idx] = t_now
                            r1_out[idx] = r1
                            r2_out[idx] = r2
                            p1_out[idx] = p1
                            p2_out[idx] = p2
                            g0_out[idx] = 0.5 * (
                                kemeth_curvature_g0(x[: self.n_per_pop], threshold_frac=0.02)
                                + kemeth_curvature_g0(x[self.n_per_pop :], threshold_frac=0.02)
                            )
                            idx += 1
                        stopped_early = True
                        break
                else:
                    below_start = None

            k2, _, _, _, _ = self._deriv_and_obs(x + 0.5 * dt * k1)
            k3, _, _, _, _ = self._deriv_and_obs(x + 0.5 * dt * k2)
            k4, _, _, _, _ = self._deriv_and_obs(x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not stopped_early:
            t_final = n_steps * dt
            if idx == 0 or abs(t_out[idx - 1] - t_final) > 1e-12:
                _, r1, r2, p1, p2 = self._deriv_and_obs(x)
                t_out[idx] = t_final
                r1_out[idx] = r1
                r2_out[idx] = r2
                p1_out[idx] = p1
                p2_out[idx] = p2
                g0_out[idx] = 0.5 * (
                    kemeth_curvature_g0(x[: self.n_per_pop], threshold_frac=0.02)
                    + kemeth_curvature_g0(x[self.n_per_pop :], threshold_frac=0.02)
                )
                idx += 1

        return MeanFieldTrajectory(
            t=t_out[:idx],
            R1=r1_out[:idx],
            R2=r2_out[:idx],
            Phi1=p1_out[:idx],
            Phi2=p2_out[:idx],
            g0=g0_out[:idx],
            theta_final=x,
            n_per_pop=self.n_per_pop,
            p=self.p,
            q=self.q,
            beta=self.beta,
            alpha=self.alpha,
            seed=int(seed),
            method="rk4",
            stopped_early=stopped_early,
        )


# ---------------------------------------------------------------------------
# 2. Nonlocally Coupled N-Oscillator Ring Substrate (Numba + NumPy/FFT)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _numba_tophat_deriv(theta: np.ndarray, radius_p: int, alpha: float, omega: float) -> np.ndarray:
    n = theta.shape[0]
    cr = np.cos(theta)
    ci = np.sin(theta)
    w = 2 * radius_p + 1
    inv = 1.0 / w
    sr = 0.0
    si = 0.0
    for d in range(-radius_p, radius_p + 1):
        idx = d % n
        sr += cr[idx]
        si += ci[idx]
    out = np.empty(n, dtype=np.float64)
    for k in range(n):
        mr = sr * inv
        mi = si * inv
        ca = math.cos(theta[k] + alpha)
        sa = math.sin(theta[k] + alpha)
        out[k] = omega - (sa * mr - ca * mi)
        rem = (k - radius_p) % n
        add = (k + 1 + radius_p) % n
        sr += cr[add] - cr[rem]
        si += ci[add] - ci[rem]
    return out


@njit(cache=True)
def _numba_tophat_local_rho(theta: np.ndarray, radius_p: int) -> np.ndarray:
    n = theta.shape[0]
    cr = np.cos(theta)
    ci = np.sin(theta)
    w = 2 * radius_p + 1
    inv = 1.0 / w
    sr = 0.0
    si = 0.0
    for d in range(-radius_p, radius_p + 1):
        idx = d % n
        sr += cr[idx]
        si += ci[idx]
    rho = np.empty(n, dtype=np.float64)
    for k in range(n):
        mr = sr * inv
        mi = si * inv
        rho[k] = math.sqrt(mr * mr + mi * mi)
        rem = (k - radius_p) % n
        add = (k + 1 + radius_p) % n
        sr += cr[add] - cr[rem]
        si += ci[add] - ci[rem]
    return rho


@dataclass(frozen=True)
class RingTrajectory:
    """Recorded spatial-coherence trajectory from ``NonlocalRingSubstrate``."""

    t: np.ndarray
    rho_mean: np.ndarray
    rho_std: np.ndarray
    rho_min: np.ndarray
    rho_max: np.ndarray
    R_global: np.ndarray
    g0: np.ndarray
    theta_final: np.ndarray
    n_oscillators: int
    kernel: str
    coupling_radius: float
    cosine_amplitude: float
    beta: float
    alpha: float
    seed: int
    backend: str
    stopped_early: bool


class NonlocalRingSubstrate:
    """Nonlocally coupled ``N``-oscillator Kuramoto-Sakaguchi ring.

    Evolves ``N`` identical oscillators at spatial sites ``x_k = 2*pi*k / N``:

        d(theta_k)/dt = omega - sum_{j=0}^{N-1} G_{k-j} sin(theta_k - theta_j + alpha)

    with ``alpha = pi/2 - beta`` and normalized circular coupling kernel ``G``:
    - ``kernel="tophat"``: Wolfrum-Omel'chenko top-hat window of index radius
      ``P = max(1, round(coupling_radius * N))`` and weight ``1 / (2P + 1)``.
    - ``kernel="cosine"``: Abrams-Strogatz cosine kernel
      ``G(x) proportional to 1 + cosine_amplitude * cos(x)``, normalized to sum
      to unity on the discrete ring and evaluated via real FFT convolution.

    Automatically uses ``numba`` JIT compilation when ``chimera-anneal[fast]`` is
    installed and ``use_numba=True``, with an exact vectorized NumPy/FFT fallback.
    """

    def __init__(
        self,
        n_oscillators: int = 64,
        *,
        kernel: Literal["tophat", "cosine"] = "tophat",
        coupling_radius: float = 0.15,
        cosine_amplitude: float = 0.995,
        beta: float = 0.12,
        omega: float = 0.0,
        use_numba: bool = True,
    ) -> None:
        if n_oscillators < 8:
            raise ValueError("n_oscillators must be >= 8")
        if kernel not in ("tophat", "cosine"):
            raise ValueError(f"Unsupported kernel {kernel!r}; expected 'tophat' or 'cosine'.")
        self.n_oscillators = int(n_oscillators)
        self.kernel = kernel
        self.coupling_radius = float(coupling_radius)
        self.radius_p = max(1, min(self.n_oscillators // 2 - 1, int(round(self.coupling_radius * self.n_oscillators))))
        self.cosine_amplitude = float(cosine_amplitude)
        self.beta = float(beta)
        self.alpha = 0.5 * math.pi - self.beta
        self.omega = float(omega)
        self.use_numba = bool(use_numba and HAS_NUMBA and kernel == "tophat")

        # Build discrete normalized circular kernel and its rFFT spectrum
        grid = 2.0 * math.pi * np.arange(self.n_oscillators, dtype=np.float64) / self.n_oscillators
        if kernel == "tophat":
            k_vec = np.zeros(self.n_oscillators, dtype=np.float64)
            idx = np.arange(-self.radius_p, self.radius_p + 1) % self.n_oscillators
            k_vec[idx] = 1.0 / (2 * self.radius_p + 1)
        else:
            raw = 1.0 + self.cosine_amplitude * np.cos(grid)
            k_vec = raw / float(np.sum(raw))
        self.kernel_weights = k_vec
        self._kernel_rfft = np.fft.rfft(k_vec)

    def initial_conditions(
        self,
        seed: int = 20260609,
        *,
        incoherent_frac: float = 0.5,
        coherent_scale: float = 0.05,
    ) -> np.ndarray:
        """Seed a canonical chimera state (coherent arc + centered incoherent arc) via PCG64."""
        rng = make_pcg64_rng(seed)
        n = self.n_oscillators
        if self.kernel == "cosine":
            # Abrams-Strogatz Gaussian-envelope spatial chimera seed
            x = 2.0 * math.pi * np.arange(n, dtype=np.float64) / n - math.pi
            env = 6.0 * np.exp(-0.76 * x * x)
            u = rng.uniform(-0.5, 0.5, size=n)
            return (u * env).astype(np.float64)
        theta = rng.normal(loc=0.0, scale=coherent_scale, size=n)
        width = max(2, int(round(incoherent_frac * n)))
        start = (n - width) // 2
        idx = np.arange(start, start + width) % n
        theta[idx] = rng.uniform(-math.pi, math.pi, size=width)
        return theta.astype(np.float64)

    def local_order_parameter(self, theta: np.ndarray) -> np.ndarray:
        """Compute the complex local mean field ``W_k = sum_j G_{k-j} exp(i theta_j)``."""
        c = np.exp(1j * theta)
        if self.kernel == "tophat":
            size = 2 * self.radius_p + 1
            re = uniform_filter1d(c.real, size=size, mode="wrap")
            im = uniform_filter1d(c.imag, size=size, mode="wrap")
            return re + 1j * im
        re = np.fft.irfft(np.fft.rfft(c.real) * self._kernel_rfft, n=self.n_oscillators)
        im = np.fft.irfft(np.fft.rfft(c.imag) * self._kernel_rfft, n=self.n_oscillators)
        return re + 1j * im

    def rhs(self, theta: np.ndarray) -> np.ndarray:
        """Evaluate ``d(theta)/dt`` on the ring."""
        x = np.asarray(theta, dtype=np.float64)
        if self.use_numba:
            return _numba_tophat_deriv(x, self.radius_p, self.alpha, self.omega)
        wbar = self.local_order_parameter(x)
        return self.omega - np.imag(np.exp(1j * (x + self.alpha)) * np.conj(wbar))

    def _spatial_summary(self, x: np.ndarray) -> tuple[float, float, float, float, float, float]:
        if self.use_numba:
            rho = _numba_tophat_local_rho(x, self.radius_p)
        else:
            # Local spatial coherence uses the canonical top-hat neighborhood of radius_p
            # so cosine-kernel rings (whose coupling is 2-harmonic smooth) also resolve
            # coherent vs incoherent spatial domains cleanly.
            c = np.exp(1j * x)
            size = 2 * self.radius_p + 1
            re = uniform_filter1d(c.real, size=size, mode="wrap")
            im = uniform_filter1d(c.imag, size=size, mode="wrap")
            rho = np.hypot(re, im)
        r_glob = float(np.abs(np.exp(1j * x).mean()))
        g0 = kemeth_curvature_g0(x, threshold_frac=0.015)
        return (
            float(rho.mean()),
            float(rho.std()),
            float(rho.min()),
            float(rho.max()),
            r_glob,
            g0,
        )

    def integrate(
        self,
        *,
        seed: int = 20260609,
        theta0: np.ndarray | None = None,
        t_max: float = 500.0,
        dt: float = 0.05,
        decimate: int = 10,
        method: Literal["rk4", "dop853"] = "rk4",
        stop_eps: float | None = 0.04,
        dt_hold: float = 30.0,
    ) -> RingTrajectory:
        """Integrate the ``N``-oscillator ring deterministically."""
        x0 = (
            self.initial_conditions(seed=seed)
            if theta0 is None
            else np.asarray(theta0, dtype=np.float64).copy()
        )
        if x0.shape != (self.n_oscillators,):
            raise ValueError(f"Expected theta0 of shape ({self.n_oscillators},), got {x0.shape}")

        decimate = max(1, int(decimate))
        backend = "numba" if self.use_numba else "numpy-fft"
        method_norm = method.lower()

        if method_norm == "dop853":
            n_eval = max(1, int(round(t_max / (dt * decimate))))
            t_eval = np.linspace(0.0, float(t_max), n_eval + 1)
            sol = solve_ivp(
                lambda _t, y: self.rhs(y),
                (0.0, float(t_max)),
                x0,
                method="DOP853",
                t_eval=t_eval,
                rtol=1e-7,
                atol=1e-9,
            )
            m = sol.y.shape[1]
            mean_a = np.empty(m, dtype=np.float64)
            std_a = np.empty(m, dtype=np.float64)
            min_a = np.empty(m, dtype=np.float64)
            max_a = np.empty(m, dtype=np.float64)
            rg_a = np.empty(m, dtype=np.float64)
            g0_a = np.empty(m, dtype=np.float64)
            for idx in range(m):
                mn, sd, rmin, rmax, rg, g0 = self._spatial_summary(sol.y[:, idx])
                mean_a[idx] = mn
                std_a[idx] = sd
                min_a[idx] = rmin
                max_a[idx] = rmax
                rg_a[idx] = rg
                g0_a[idx] = g0
            return RingTrajectory(
                t=sol.t.copy(),
                rho_mean=mean_a,
                rho_std=std_a,
                rho_min=min_a,
                rho_max=max_a,
                R_global=rg_a,
                g0=g0_a,
                theta_final=sol.y[:, -1].copy(),
                n_oscillators=self.n_oscillators,
                kernel=self.kernel,
                coupling_radius=self.coupling_radius,
                cosine_amplitude=self.cosine_amplitude,
                beta=self.beta,
                alpha=self.alpha,
                seed=int(seed),
                backend=f"{backend}-dop853",
                stopped_early=False,
            )

        if method_norm != "rk4":
            raise ValueError(f"Unsupported method {method!r}; expected 'rk4' or 'dop853'.")

        n_steps = max(1, int(round(t_max / dt)))
        cap = n_steps // decimate + 3
        t_a = np.empty(cap, dtype=np.float64)
        mean_a = np.empty(cap, dtype=np.float64)
        std_a = np.empty(cap, dtype=np.float64)
        min_a = np.empty(cap, dtype=np.float64)
        max_a = np.empty(cap, dtype=np.float64)
        rg_a = np.empty(cap, dtype=np.float64)
        g0_a = np.empty(cap, dtype=np.float64)

        x = x0
        idx = 0
        below_samples = 0
        hold_samples = max(1, int(round(dt_hold / (decimate * dt)))) + 2
        stopped_early = False

        for step in range(n_steps):
            if step % decimate == 0:
                mn, sd, rmin, rmax, rg, g0 = self._spatial_summary(x)
                t_a[idx] = step * dt
                mean_a[idx] = mn
                std_a[idx] = sd
                min_a[idx] = rmin
                max_a[idx] = rmax
                rg_a[idx] = rg
                g0_a[idx] = g0
                idx += 1

                if stop_eps is not None and stop_eps > 0.0:
                    if sd < stop_eps:
                        below_samples += 1
                        if below_samples >= hold_samples:
                            stopped_early = True
                            break
                    else:
                        below_samples = 0

            k1 = self.rhs(x)
            k2 = self.rhs(x + 0.5 * dt * k1)
            k3 = self.rhs(x + 0.5 * dt * k2)
            k4 = self.rhs(x + dt * k3)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        if not stopped_early:
            t_final = n_steps * dt
            if idx == 0 or abs(t_a[idx - 1] - t_final) > 1e-12:
                mn, sd, rmin, rmax, rg, g0 = self._spatial_summary(x)
                t_a[idx] = t_final
                mean_a[idx] = mn
                std_a[idx] = sd
                min_a[idx] = rmin
                max_a[idx] = rmax
                rg_a[idx] = rg
                g0_a[idx] = g0
                idx += 1

        return RingTrajectory(
            t=t_a[:idx],
            rho_mean=mean_a[:idx],
            rho_std=std_a[:idx],
            rho_min=min_a[:idx],
            rho_max=max_a[:idx],
            R_global=rg_a[:idx],
            g0=g0_a[:idx],
            theta_final=x,
            n_oscillators=self.n_oscillators,
            kernel=self.kernel,
            coupling_radius=self.coupling_radius,
            cosine_amplitude=self.cosine_amplitude,
            beta=self.beta,
            alpha=self.alpha,
            seed=int(seed),
            backend=backend,
            stopped_early=stopped_early,
        )


# ---------------------------------------------------------------------------
# 3. Parameter-Free Reduced 2D/3D Ott-Antonsen / Watanabe-Strogatz Flow
# ---------------------------------------------------------------------------
class ReducedTwoPopFlow:
    """Parameter-free reduced 2D/3D Ott-Antonsen / Watanabe-Strogatz flow validator.

    Implements the exact collective macroscopic equations of Abrams, Mirollo,
    Strogatz & Wiley (*Phys. Rev. Lett.* 101, 084103, 2008) for couplings
    ``mu = (1 + A)/2``, ``nu = (1 - A)/2``, and ``alpha = pi/2 - beta``:
    - 2D invariant-manifold system ``(r, psi)`` at ``rho_1 == 1`` (Eq. 12).
    - 3D rotational-symmetry reduction ``(rho_1, rho_2, psi)`` (Eq. 10), which
      restricts identically to the 2D flow on the attracting ``rho_1 = 1``
      Ott-Antonsen manifold.
    """

    def __init__(self, A: float = 0.5, beta: float = 0.05) -> None:
        self.A = float(A)
        self.beta = float(beta)
        self.mu = 0.5 * (1.0 + self.A)
        self.nu = 0.5 * (1.0 - self.A)
        self.alpha = 0.5 * math.pi - self.beta

    def rhs_2d(self, state: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
        """Evaluate 2D Ott-Antonsen manifold flow ``(dr/dt, dpsi/dt)`` at ``rho_1 = 1``."""
        r, psi = float(state[0]), float(state[1])
        mu, nu, a = self.mu, self.nu, self.alpha
        r_safe = max(r, 1e-12)
        rdot = 0.5 * (1.0 - r * r) * (mu * r * math.cos(a) + nu * math.cos(psi - a))
        psidot = (
            ((1.0 + r * r) / (2.0 * r_safe)) * (mu * r * math.sin(a) - nu * math.sin(psi - a))
            - mu * math.sin(a)
            - nu * r * math.sin(psi + a)
        )
        return np.array([rdot, psidot], dtype=np.float64)

    def rhs_3d(
        self, state: np.ndarray | list[float] | tuple[float, float, float]
    ) -> np.ndarray:
        """Evaluate 3D collective flow ``(drho1/dt, drho2/dt, dpsi/dt)``."""
        r1, r2, psi = float(state[0]), float(state[1]), float(state[2])
        mu, nu, a = self.mu, self.nu, self.alpha
        ca, sa = math.cos(a), math.sin(a)
        r1_safe = max(r1, 1e-12)
        r2_safe = max(r2, 1e-12)
        r1dot = 0.5 * (1.0 - r1 * r1) * (mu * r1 * ca + nu * r2 * math.cos(psi + a))
        r2dot = 0.5 * (1.0 - r2 * r2) * (mu * r2 * ca + nu * r1 * math.cos(psi - a))
        phi1dot = -((1.0 + r1 * r1) / (2.0 * r1_safe)) * (
            mu * r1 * sa + nu * r2 * math.sin(psi + a)
        )
        phi2dot = -((1.0 + r2 * r2) / (2.0 * r2_safe)) * (
            mu * r2 * sa + nu * r1 * math.sin(a - psi)
        )
        return np.array([r1dot, r2dot, phi1dot - phi2dot], dtype=np.float64)

    def transverse_rate(self, r2: float, psi: float) -> float:
        """Transverse stability rate ``d(drho1/dt)/drho1`` at ``rho1 = 1`` (< 0 attracting)."""
        return -(
            self.mu * math.cos(self.alpha)
            + self.nu * float(r2) * math.cos(float(psi) + self.alpha)
        )

    @staticmethod
    def fixed_point_family(
        beta: float, n_grid: int = 40001
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Parametric stationary-chimera curve ``(psi, r, A)`` (Eqs. 13-14 of Abrams 2008)."""
        psi_grid = np.linspace(-1.45, 1.45, n_grid)
        b = float(beta)
        num = np.sin(2.0 * b + psi_grid)
        den = np.sin(2.0 * b - psi_grid) + 2.0 * np.sin(psi_grid)
        valid = np.abs(den) > 1e-12
        rad = np.zeros_like(psi_grid)
        rad[valid] = num[valid] / den[valid]
        valid &= (rad > 0.0) & (rad < 1.0)
        r = np.sqrt(np.clip(rad, 0.0, None))
        d_a = np.sin(b + psi_grid) - r * math.sin(b)
        valid &= np.abs(d_a) > 1e-12
        psi_v = psi_grid[valid]
        r_v = r[valid]
        a_v = (np.sin(b + psi_v) + r_v * math.sin(b)) / (np.sin(b + psi_v) - r_v * math.sin(b))
        keep = (a_v > 0.0) & (a_v < 1.0)
        return psi_v[keep], r_v[keep], a_v[keep]

    @classmethod
    def locate_saddle_node(cls, beta: float) -> dict[str, float]:
        """Locate the saddle-node bifurcation ``A_SN(beta)`` where stationary chimeras appear."""
        psi_v, r_v, a_v = cls.fixed_point_family(beta, n_grid=120001)
        if len(a_v) == 0:
            raise RuntimeError(f"No stationary chimera branch found for beta={beta}")
        idx = int(np.argmin(a_v))
        return {"A_SN": float(a_v[idx]), "r_SN": float(r_v[idx]), "psi_SN": float(psi_v[idx])}

    @classmethod
    def get_stationary_chimera(
        cls, A: float, beta: float, branch: Literal["stable", "saddle"] = "stable"
    ) -> dict[str, float | bool] | None:
        """Solve for the stationary chimera fixed point and its 2D Jacobian stability."""
        psi_v, r_v, a_v = cls.fixed_point_family(beta, n_grid=80001)
        crossings: list[float] = []
        for i in range(len(a_v) - 1):
            if (a_v[i] - A) * (a_v[i + 1] - A) <= 0.0 and a_v[i] != a_v[i + 1]:
                frac = (A - a_v[i]) / (a_v[i + 1] - a_v[i])
                crossings.append(float(r_v[i] + frac * (r_v[i + 1] - r_v[i])))
        if not crossings:
            return None
        rs = sorted(set(round(x, 9) for x in crossings))
        r_star = rs[-1] if branch == "saddle" else rs[0]

        flow = cls(A=A, beta=beta)
        b = float(beta)
        s = max(-1.0, min(1.0, r_star * math.sin(b) * (A + 1.0) / (A - 1.0)))
        psi_candidates = (math.asin(s) - b, math.pi - math.asin(s) - b, -math.pi - math.asin(s) - b)
        psi_star = min(psi_candidates, key=lambda ps: float(np.linalg.norm(flow.rhs_2d((r_star, ps)))))

        h = 1e-6
        j_mat = np.zeros((2, 2), dtype=np.float64)
        for col in range(2):
            sp = np.array([r_star, psi_star], dtype=np.float64)
            sm = np.array([r_star, psi_star], dtype=np.float64)
            sp[col] += h
            sm[col] -= h
            j_mat[:, col] = (flow.rhs_2d(sp) - flow.rhs_2d(sm)) / (2.0 * h)
        det = float(np.linalg.det(j_mat))
        tr = float(np.trace(j_mat))
        return {
            "r": r_star,
            "psi": float(psi_star),
            "det": det,
            "trace": tr,
            "linearly_stable": bool(det > 0.0 and tr < 0.0),
        }

    @classmethod
    def locate_hopf(cls, beta: float) -> float:
        """Locate the supercritical Hopf bifurcation ``A_H(beta)`` (``trace(J) = 0, det(J) > 0``)."""
        sn = cls.locate_saddle_node(beta)["A_SN"]
        a_grid = np.linspace(sn + 1e-3, min(0.75, sn + 0.45), 120)

        def _trace_spiral(a_val: float) -> float:
            fp = cls.get_stationary_chimera(a_val, beta, branch="stable")
            if fp is None or float(fp["det"]) <= 0.0:
                return float("nan")
            return float(fp["trace"])

        prev_a: float | None = None
        prev_tr: float | None = None
        for a_val in a_grid:
            tr = _trace_spiral(float(a_val))
            if (
                prev_tr is not None
                and prev_a is not None
                and math.isfinite(tr)
                and math.isfinite(prev_tr)
                and prev_tr * tr < 0.0
            ):
                return float(brentq(_trace_spiral, prev_a, float(a_val), xtol=1e-7))
            prev_a, prev_tr = float(a_val), tr
        raise RuntimeError(f"Could not bracket Hopf bifurcation for beta={beta}")

    @classmethod
    def locate_homoclinic(
        cls, beta: float, *, t_max: float = 4000.0, sync_r: float = 0.995, tol: float = 5e-4
    ) -> float:
        """Locate the homoclinic (saddle-loop) boundary crisis ``A_hc(beta)`` by bisection."""
        a_lo = cls.locate_hopf(beta) + 1e-3
        a_hi = min(0.75, a_lo + 0.18)

        def _escapes(a_val: float) -> bool:
            fp = cls.get_stationary_chimera(a_val, beta, branch="stable")
            if fp is None:
                return True
            flow = cls(A=a_val, beta=beta)
            state0 = np.array([min(0.99, float(fp["r"]) + 1e-2), float(fp["psi"])])

            def _hit_sync(_t: float, y: np.ndarray) -> float:
                return float(y[0] - sync_r)

            _hit_sync.terminal = True  # type: ignore[attr-defined]
            _hit_sync.direction = 1  # type: ignore[attr-defined]
            sol = solve_ivp(
                lambda _t, y: flow.rhs_2d(y),
                (0.0, t_max),
                state0,
                method="DOP853",
                rtol=1e-9,
                atol=1e-11,
                events=_hit_sync,
            )
            return bool(sol.t_events and len(sol.t_events[0]) > 0)

        while not _escapes(a_hi) and a_hi < 0.95:
            a_hi += 0.08
        while a_hi - a_lo > tol:
            mid = 0.5 * (a_lo + a_hi)
            if _escapes(mid):
                a_hi = mid
            else:
                a_lo = mid
        return 0.5 * (a_lo + a_hi)

    def integrate_2d(
        self,
        state0: tuple[float, float] | np.ndarray,
        *,
        t_max: float = 200.0,
        n_points: int = 2001,
    ) -> dict[str, np.ndarray]:
        """Integrate the 2D Ott-Antonsen reduced flow via DOP853."""
        t_eval = np.linspace(0.0, float(t_max), n_points)
        sol = solve_ivp(
            lambda _t, y: self.rhs_2d(y),
            (0.0, float(t_max)),
            np.asarray(state0, dtype=np.float64),
            method="DOP853",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-11,
        )
        return {"t": sol.t, "r": sol.y[0], "psi": sol.y[1]}

    def integrate_3d(
        self,
        state0: tuple[float, float, float] | np.ndarray,
        *,
        t_max: float = 200.0,
        n_points: int = 2001,
        capture_theta: float = 0.85,
    ) -> dict[str, np.ndarray | float | bool | None]:
        """Integrate the 3D reduced collective flow and report capture & transverse stability."""
        t_eval = np.linspace(0.0, float(t_max), n_points)
        sol = solve_ivp(
            lambda _t, y: self.rhs_3d(y),
            (0.0, float(t_max)),
            np.asarray(state0, dtype=np.float64),
            method="DOP853",
            t_eval=t_eval,
            rtol=1e-9,
            atol=1e-11,
        )
        r1, r2, psi = sol.y[0], sol.y[1], sol.y[2]
        m = np.minimum(r1, r2)
        above = np.nonzero(m >= capture_theta)[0]
        t_cap: float | None = None
        for idx in above:
            if np.all(m[idx:] >= capture_theta - 0.05):
                t_cap = float(sol.t[idx])
                break
        tr = np.array([self.transverse_rate(r2[k], psi[k]) for k in range(len(sol.t))])
        return {
            "t": sol.t,
            "rho1": r1,
            "rho2": r2,
            "psi": psi,
            "captured": t_cap is not None,
            "t_capture": t_cap,
            "transverse_rate_mean": float(np.mean(tr)),
            "transverse_attracting_frac": float(np.mean(tr < 0.0)),
        }
