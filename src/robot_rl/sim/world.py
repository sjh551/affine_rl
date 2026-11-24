"""PyBullet connection, scene building, sensor queries and physical stepping."""

import numpy as np
import pybullet as p
import pybullet_data

from robot_rl.maps.layouts import Box
from robot_rl.robots.ranger import Ranger


class BulletWorld:
    def __init__(self, render_mode, time_step):
        self.render = render_mode == "human"
        self.time_step = time_step
        self.client_id = p.connect(p.GUI if self.render else p.DIRECT)
        if self.client_id < 0:
            raise RuntimeError("Unable to connect to PyBullet.")

    def reset_scene(self, course, goal, *, walls, obstacle_seed):
        client = self.client_id
        if client is None:
            raise RuntimeError("The PyBullet environment is closed.")
        p.resetSimulation(physicsClientId=client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.setTimeStep(self.time_step, physicsClientId=client)
        p.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1,
                                    physicsClientId=client)
        p.loadURDF("plane.urdf", physicsClientId=client)
        self.wall_ids = self._add_walls(course.half_extents) if walls else set()
        self.obstacle_ids = set()
        if course.obstacles is not None:
            self.obstacle_ids = self._add_obstacles(course.obstacles(obstacle_seed), course.scale)
        if self.render:
            marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.2, rgbaColor=(0, 1, 0, 1),
                                         physicsClientId=client)
            p.createMultiBody(baseVisualShapeIndex=marker, basePosition=(*goal, 0.2), physicsClientId=client)
        self._debug_lines = {}

    def spawn_robots(self, positions, heading, leader_count):
        self.robots = [Ranger(self.client_id, xy, heading, i < leader_count)
                       for i, xy in enumerate(positions)]
        self.robot_ids = {robot.body for robot in self.robots}
        for _ in range(round(0.5 / self.time_step)):
            p.stepSimulation(physicsClientId=self.client_id)

    def robot_states(self):
        return tuple(np.asarray(values) for values in zip(*(robot.state() for robot in self.robots)))

    def command_robots(self, start, velocities, yaws, heading):
        for i, velocity in enumerate(velocities, start=start):
            self.robots[i].command(velocity, yaws[i], heading)

    def advance(self, substeps, control_stride, update_control):
        for substep in range(substeps):
            if substep % control_stride == 0:
                update_control(substep)
            p.stepSimulation(physicsClientId=self.client_id)
            collision = self.contact_reason()
            if collision:
                return collision
        return None

    def contact_reason(self):
        for contact in p.getContactPoints(physicsClientId=self.client_id):
            a, b = contact[1:3]
            if a in self.robot_ids:
                other = b
            elif b in self.robot_ids:
                other = a
            else:
                continue
            if other in self.robot_ids and a != b:
                return "Crash_Team"
            if other in self.wall_ids:
                return "Wall_Out"
            if other in self.obstacle_ids:
                return "Crash_Obstacle"
        return None

    def ray_fractions(self, origins, ends):
        hits = p.rayTestBatch(origins.tolist(), ends.tolist(), collisionFilterMask=2, physicsClientId=self.client_id)
        return np.asarray([hit[2] for hit in hits], dtype=np.float32)

    def focus(self, center):
        if self.render:
            p.resetDebugVisualizerCamera(cameraDistance=15, cameraYaw=0, cameraPitch=-60,
                                         cameraTargetPosition=(*center, 0.2), physicsClientId=self.client_id)

    def draw_formation(self, positions, edges):
        if self.render:
            for i, j in edges:
                self._debug_lines[i, j] = p.addUserDebugLine(
                    (*positions[i], 0.35), (*positions[j], 0.35), (0.2, 0.6, 1.0),
                    replaceItemUniqueId=self._debug_lines.get((i, j), -1), physicsClientId=self.client_id,
                )

    def _add_walls(self, half_extents):
        x_limit, y_limit = half_extents
        ids = set()
        for sign in (-1, 1):
            for position, extents in (
                ((sign * (x_limit + 0.15), 0, 0.5), (0.15, y_limit + 0.3, 0.5)),
                ((0, sign * (y_limit + 0.15), 0.5), (x_limit, 0.15, 0.5)),
            ):
                dimensions = {"halfExtents": extents}
                shapes = self._make_shapes(p.GEOM_BOX, dimensions, dimensions, (0.6, 0.6, 0.6, 1))
                ids.add(self._place_barrier(position, shapes))
        return ids

    def _add_obstacles(self, obstacles, scale):
        ids, shapes = set(), {}
        for obstacle in obstacles:
            if isinstance(obstacle, Box):
                extents = tuple(scale * value / 2 for value in obstacle.size)
                key = (p.GEOM_BOX, *extents)
                collision = visual = {"halfExtents": extents}
            else:
                radius, height = scale * obstacle.radius, scale * obstacle.height
                key = (p.GEOM_CYLINDER, radius, height)
                collision = {"radius": radius, "height": height}
                visual = {"radius": radius, "length": height}
            if key not in shapes:
                shapes[key] = self._make_shapes(key[0], collision, visual, (0.25, 0.32, 0.29, 1.0))
            center = tuple(scale * value for value in obstacle.center)
            ids.add(self._place_barrier(center, shapes[key]))
        return ids

    def _make_shapes(self, geometry, collision, visual, color):
        return (
            p.createCollisionShape(geometry, **collision, physicsClientId=self.client_id),
            p.createVisualShape(geometry, **visual, rgbaColor=color, physicsClientId=self.client_id),
        )

    def _place_barrier(self, position, shapes):
        collision, visual = shapes
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
                                 basePosition=position, physicsClientId=self.client_id)
        p.setCollisionFilterGroupMask(body, -1, 2, 3, physicsClientId=self.client_id)
        return body

    def close(self):
        if self.client_id is not None and p.isConnected(physicsClientId=self.client_id):
            p.disconnect(physicsClientId=self.client_id)
        self.client_id = None
