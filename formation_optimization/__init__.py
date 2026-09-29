"""Offline formation-configuration optimization and validation tools."""

from .optimizer import (
    FormationOptimizationResult,
    build_follower_planar_iteration_matrix,
    follower_planar_spectral_radius,
    optimize_nominal_configuration,
)
from .spectral_norm_optimizer import (
    MinimumSpectralNormResult,
    SpectralNormOptimizationResult,
    follower_planar_matrix_metrics,
    minimize_follower_planar_spectral_norm,
    optimize_nominal_configuration_by_spectral_norm,
)

__all__ = [
    "FormationOptimizationResult",
    "build_follower_planar_iteration_matrix",
    "follower_planar_spectral_radius",
    "optimize_nominal_configuration",
    "SpectralNormOptimizationResult",
    "MinimumSpectralNormResult",
    "follower_planar_matrix_metrics",
    "minimize_follower_planar_spectral_norm",
    "optimize_nominal_configuration_by_spectral_norm",
]
