import numpy as np


def clip_vector_norm(vectors, max_norm):
    """Project vectors onto an L2 ball while preserving direction."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scales = np.minimum(1.0, max_norm / np.maximum(norms, 1e-8))
    return vectors * scales


class ResidualVelocityMixer:
    """Scale normalized residual actions and combine them with a nominal input."""

    def __init__(self, residual_speed_limit=0.6, command_speed_limit=1.0):
        if residual_speed_limit <= 0 or command_speed_limit <= 0:
            raise ValueError("Velocity limits must be positive.")
        if residual_speed_limit > command_speed_limit:
            raise ValueError(
                "residual_speed_limit must not exceed command_speed_limit."
            )

        self.residual_speed_limit = float(residual_speed_limit)
        self.command_speed_limit = float(command_speed_limit)

    def scale_residual(self, normalized_residual):
        normalized_residual = clip_vector_norm(normalized_residual, 1.0)
        return normalized_residual * self.residual_speed_limit

    def combine(self, nominal_velocity, residual_velocity):
        nominal_velocity = np.asarray(nominal_velocity, dtype=np.float32)
        residual_velocity = np.asarray(residual_velocity, dtype=np.float32)
        if nominal_velocity.shape != residual_velocity.shape:
            raise ValueError(
                f"Nominal shape {nominal_velocity.shape} does not match "
                f"residual shape {residual_velocity.shape}."
            )
        return clip_vector_norm(
            nominal_velocity + residual_velocity,
            self.command_speed_limit,
        )
