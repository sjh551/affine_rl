import numpy as np


def clip_vector_norm(vectors, max_norm):
    """Project vectors onto an L2 ball while preserving their directions."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if max_norm <= 0:
        raise ValueError("max_norm must be positive.")

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    scales = np.minimum(1.0, max_norm / np.maximum(norms, 1e-8))
    return vectors * scales


class NominalPController:
    """Position P controller with reserved authority for an RL residual."""

    def __init__(
        self,
        kp=0.5,
        nominal_speed_limit=0.4,
        residual_speed_limit=0.6,
        command_speed_limit=1.0,
    ):
        if kp <= 0:
            raise ValueError("kp must be positive.")
        if nominal_speed_limit <= 0 or residual_speed_limit <= 0:
            raise ValueError("Nominal and residual speed limits must be positive.")
        if nominal_speed_limit + residual_speed_limit > command_speed_limit + 1e-8:
            raise ValueError(
                "Nominal and residual limits must not exceed the final command "
                "limit when added together."
            )

        self.kp = float(kp)
        self.nominal_speed_limit = float(nominal_speed_limit)
        self.residual_speed_limit = float(residual_speed_limit)
        self.command_speed_limit = float(command_speed_limit)

    def desired_positions(self, target_position, leader_offsets):
        target_position = np.asarray(target_position, dtype=np.float32)
        leader_offsets = np.asarray(leader_offsets, dtype=np.float32)
        return target_position[:2] + leader_offsets

    def compute_nominal(self, current_positions, target_position, leader_offsets):
        current_positions = np.asarray(current_positions, dtype=np.float32)
        desired = self.desired_positions(target_position, leader_offsets)
        if current_positions.shape != desired.shape:
            raise ValueError(
                f"Position shape {current_positions.shape} does not match "
                f"desired shape {desired.shape}."
            )
        return clip_vector_norm(
            self.kp * (desired - current_positions),
            self.nominal_speed_limit,
        )

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
