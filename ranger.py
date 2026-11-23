"""Ranger wheel and steering control for a planar world-frame velocity command."""

from pathlib import Path

import numpy as np
import pybullet as p


URDF_PATH = Path(__file__).resolve().parent / "assets/ranger_mini_v3/urdf/ranger_mini.urdf"
WHEEL_RADIUS = 0.09
WHEEL_OFFSETS = ((0.25, 0.19), (0.25, -0.19), (-0.25, 0.19), (-0.25, -0.19))


class Ranger:
    def __init__(self, client, xy, heading, leader):
        self.client = client
        orientation = p.getQuaternionFromEuler((0, 0, heading))
        self.body = p.loadURDF(str(URDF_PATH), [*xy, 0.315], orientation, physicsClientId=client)
        color = (0.4, 0.85, 0.45, 1) if leader else (0.3, 0.5, 0.95, 1)
        p.changeVisualShape(self.body, -1, rgbaColor=color, physicsClientId=client)
        p.changeDynamics(self.body, -1, lateralFriction=0.5, linearDamping=0.04,
                         angularDamping=0.1, physicsClientId=client)
        dynamics = p.getDynamicsInfo(self.body, -1, physicsClientId=client)
        self.base_from_com = p.invertTransform(dynamics[3], dynamics[4])

        joints = {}
        joint_count = p.getNumJoints(self.body, physicsClientId=client)
        for index in range(joint_count):
            info = p.getJointInfo(self.body, index, physicsClientId=client)
            joints[info[1].decode()] = info
        # Group 2 is reserved for obstacles and walls. LiDAR ignores group 1.
        for link in range(-1, joint_count):
            p.setCollisionFilterGroupMask(self.body, link, 1, 3, physicsClientId=client)

        corners = ("fl", "fr", "rl", "rr")
        steering_info = [joints[f"{corner}_steering_joint"] for corner in corners]
        self.steering = [info[0] for info in steering_info]
        self.steering_limits = [(info[8], info[9]) for info in steering_info]
        self.wheels = [joints[f"{corner}_wheel"][0] for corner in corners]
        for joint in self.wheels:
            p.changeDynamics(self.body, joint, lateralFriction=0.8, rollingFriction=0.002,
                             spinningFriction=0.002, physicsClientId=client)
        self._drive(0.0, 0.0, 0.0)

    def state(self):
        com, orientation = p.getBasePositionAndOrientation(self.body, physicsClientId=self.client)
        linear, angular = p.getBaseVelocity(self.body, physicsClientId=self.client)
        # Bullet reports the center of mass. Formation coordinates use the base link origin.
        origin, rotation = p.multiplyTransforms(com, orientation, *self.base_from_com)
        offset = np.asarray(origin) - com
        velocity = np.asarray(linear) + np.cross(angular, offset)
        yaw = p.getEulerFromQuaternion(rotation)[2]
        return origin[:2], velocity[:2], yaw, angular[2]

    def command(self, velocity, yaw, heading):
        c, s = np.cos(yaw), np.sin(yaw)
        vx, vy = velocity
        yaw_error = (heading - yaw + np.pi) % (2 * np.pi) - np.pi
        omega = np.clip(2 * yaw_error, -1.0, 1.0)
        self._drive(c * vx + s * vy, -s * vx + c * vy, omega)

    def _drive(self, vx, vy, omega):
        joint_states = p.getJointStates(self.body, self.steering, physicsClientId=self.client)
        angles, speeds = [], []
        for (wx, wy), joint, (lower, upper) in zip(WHEEL_OFFSETS, joint_states, self.steering_limits):
            wheel_vx, wheel_vy = vx - omega * wy, vy + omega * wx
            speed = np.hypot(wheel_vx, wheel_vy)
            current = joint[0]
            if speed < 1e-6:
                angles.append(current)
                speeds.append(0.0)
                continue

            # The URDF steering axis points along -z. Reversing a wheel gives an
            # equivalent direction and can require less movement of the steering joint.
            angle = -np.arctan2(wheel_vy, wheel_vx)
            reverse = angle - np.pi if angle > 0 else angle + np.pi
            candidates = [(a, direction) for a, direction in ((angle, 1), (reverse, -1))
                          if lower <= a <= upper]
            chosen, direction = min(candidates, key=lambda pair: abs(pair[0] - current))
            angles.append(chosen)
            speeds.append(direction * speed / WHEEL_RADIUS * max(0.0, np.cos(chosen - current)))

        p.setJointMotorControlArray(self.body, self.steering, p.POSITION_CONTROL,
                                   targetPositions=angles, forces=[80.0] * 4,
                                   physicsClientId=self.client)
        p.setJointMotorControlArray(self.body, self.wheels, p.VELOCITY_CONTROL,
                                   targetVelocities=speeds, forces=[20.0] * 4,
                                   physicsClientId=self.client)
