import numpy as np

from .residual_velocity_controller import ResidualVelocityMixer, clip_vector_norm


class NominalPController(ResidualVelocityMixer):
    """Position P controller with reserved authority for an RL residual."""

    def __init__(
        self,
        kp=0.5,
        nominal_speed_limit=0.4,
        residual_speed_limit=0.6,
        command_speed_limit=1.4,
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

        super().__init__(residual_speed_limit, command_speed_limit)
        self.kp = float(kp)
        self.nominal_speed_limit = float(nominal_speed_limit)

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
