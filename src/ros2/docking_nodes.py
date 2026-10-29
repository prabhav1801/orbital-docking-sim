"""ROS 2-style node wrappers. There is no rclpy dependency: `publish` just prints.

These only illustrate how the pose estimator and controller would be split into nodes.
"""
import numpy as np

from src.vision.vision_pose_node import ArUcoVisionPoseEstimator


class MockNode:
    def __init__(self, name):
        self.name = name

    def publish(self, topic, message):
        print(f"[{self.name}] {topic}: {message}")


class VisionPoseNode(MockNode):
    """Runs ArUco + solvePnP on a camera frame and publishes the marker translation."""

    def __init__(self):
        super().__init__("vision_pose_node")
        self.estimator = ArUcoVisionPoseEstimator()

    def process_camera_frame(self, frame):
        found, tvec, _ = self.estimator.estimate_relative_pose(frame)
        if not found:
            return None
        self.publish("/camera/relative_pose", tvec)
        return tvec


class GNCControllerNode(MockNode):
    """Applies a controller (anything with compute_control) and clips to the thrust limit."""

    def __init__(self, controller, max_thrust=5.0):
        super().__init__("gnc_controller_node")
        self.controller = controller
        self.max_thrust = max_thrust

    def evaluate_control(self, estimated_state):
        force = np.clip(self.controller.compute_control(estimated_state), -self.max_thrust, self.max_thrust)
        self.publish("/thruster/cmd_force", force)
        return force
