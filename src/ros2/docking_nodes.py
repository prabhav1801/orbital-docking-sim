import numpy as np
from src.vision.vision_pose_node import ArUcoVisionPoseEstimator

class MockROS2Node:
    def __init__(self, node_name):
        self.node_name = node_name

    def publish(self, topic, message):
        print(f"[{self.node_name}] Publishing to {topic}: {message}")

class VisionPoseNode(MockROS2Node):
    """
    ROS 2 Node for optical relative navigation, subscribing to raw camera frames,
    running ArUco PnP pose estimation, and publishing relative states.
    """
    def __init__(self):
        super().__init__("vision_pose_node")
        self.estimator = ArUcoVisionPoseEstimator()

    def process_camera_frame(self, frame):
        found, tvec, rvec = self.estimator.estimate_relative_pose(frame)
        if found:
            self.publish("/camera/relative_pose", tvec)
            return tvec
        return None

class GNCControllerNode(MockROS2Node):
    """
    ROS 2 Node for Guidance, Navigation, and Control (GNC), subscribing to estimated states
    and evaluating saturated LQR / RL control laws.
    """
    def __init__(self):
        super().__init__("gnc_controller_node")

    def evaluate_control(self, estimated_state):
        force_cmd = -np.eye(3) @ estimated_state[0:3]
        clamped_force = np.clip(force_cmd, -5.0, 5.0)
        self.publish("/thruster/cmd_force", clamped_force)
        return clamped_force

