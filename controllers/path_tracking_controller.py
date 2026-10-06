import numpy as np

from .residual_velocity_controller import clip_vector_norm


class PathTrackingController:
    """Track independent planar polylines with a lookahead P controller."""

    def __init__(self, kp=0.8, speed_limit=0.35, lookahead_distance=1.0):
        if kp <= 0 or speed_limit <= 0 or lookahead_distance <= 0:
            raise ValueError("Controller parameters must be positive.")
        self.kp = float(kp)
        self.speed_limit = float(speed_limit)
        self.lookahead_distance = float(lookahead_distance)
        self.paths = []
        self.segment_indices = []

    def set_paths(self, paths):
        checked_paths = []
        for path in paths:
            path = np.asarray(path, dtype=np.float32)
            if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
                raise ValueError("Every path must have shape (N, 2), N >= 2.")
            checked_paths.append(path)
        self.paths = checked_paths
        self.segment_indices = [0] * len(checked_paths)

    @staticmethod
    def _project_to_segment(position, start, end):
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        if length_sq <= 1e-12:
            return start.copy(), 0.0
        ratio = float(np.dot(position - start, segment) / length_sq)
        ratio = float(np.clip(ratio, 0.0, 1.0))
        return start + ratio * segment, ratio

    def _closest_progress(self, path_index, position):
        path = self.paths[path_index]
        first_segment = self.segment_indices[path_index]
        best = None
        for segment_index in range(first_segment, len(path) - 1):
            projected, ratio = self._project_to_segment(
                position,
                path[segment_index],
                path[segment_index + 1],
            )
            distance_sq = float(np.dot(position - projected, position - projected))
            candidate = (distance_sq, segment_index, ratio, projected)
            if best is None or candidate[0] < best[0]:
                best = candidate

        _, segment_index, ratio, projected = best
        self.segment_indices[path_index] = max(
            self.segment_indices[path_index],
            segment_index,
        )
        return segment_index, ratio, projected

    def _lookahead_point(self, path_index, position):
        path = self.paths[path_index]
        segment_index, _, point = self._closest_progress(path_index, position)
        remaining = self.lookahead_distance

        while segment_index < len(path) - 1:
            segment_end = path[segment_index + 1]
            available = float(np.linalg.norm(segment_end - point))
            if available >= remaining and available > 1e-8:
                return point + (segment_end - point) * (remaining / available)
            remaining -= available
            segment_index += 1
            point = path[segment_index]

        return path[-1].copy()

    def compute_nominal(self, current_positions):
        current_positions = np.asarray(current_positions, dtype=np.float32)
        if current_positions.shape != (len(self.paths), 2):
            raise ValueError(
                f"Expected current_positions shape ({len(self.paths)}, 2), "
                f"got {current_positions.shape}."
            )

        lookahead_points = np.stack([
            self._lookahead_point(index, position)
            for index, position in enumerate(current_positions)
        ])
        return clip_vector_norm(
            self.kp * (lookahead_points - current_positions),
            self.speed_limit,
        )

    def get_paths(self):
        return [path.copy() for path in self.paths]
