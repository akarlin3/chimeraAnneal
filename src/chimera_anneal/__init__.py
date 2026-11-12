"""chimeraAnneal: Coupled-oscillator chimera survival & finite-size scaling."""
from __future__ import annotations

from .detection import (
    CollapseDetectionResult,
    detect_chimera_collapse,
    hold_time_hysteresis_gate,
    local_curvature_threshold_gate,
)
from .hazard import (
    FiniteSizeScalingFit,
    KaplanMeierEstimate,
    WeibullSurvivalFit,
    fit_finite_size_scaling,
    fit_weibull_survival,
    instantaneous_hazard_rate,
    kaplan_meier_estimator,
)
from .substrates import (
    HAS_NUMBA,
    MeanFieldTrajectory,
    NonlocalRingSubstrate,
    ReducedTwoPopFlow,
    RingTrajectory,
    TwoPopMeanField,
    kemeth_curvature_g0,
    make_pcg64_rng,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "HAS_NUMBA",
    "make_pcg64_rng",
    "kemeth_curvature_g0",
    "MeanFieldTrajectory",
    "TwoPopMeanField",
    "RingTrajectory",
    "NonlocalRingSubstrate",
    "ReducedTwoPopFlow",
    "CollapseDetectionResult",
    "local_curvature_threshold_gate",
    "hold_time_hysteresis_gate",
    "detect_chimera_collapse",
    "KaplanMeierEstimate",
    "kaplan_meier_estimator",
    "WeibullSurvivalFit",
    "fit_weibull_survival",
    "instantaneous_hazard_rate",
    "FiniteSizeScalingFit",
    "fit_finite_size_scaling",
]
