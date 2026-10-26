import cv2
import numpy as np

class ArUcoVisionPoseEstimator:
    """
    ArUco / AprilTag fiducial marker relative pose estimation module.
    Extracts 6-DOF translation and rotation vectors via Perspective-n-Point (PnP) solver
    to feed optical relative state measurements into the Clohessy-Wiltshire state estimator.
    """
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        self.camera_matrix = camera_matrix if camera_matrix is not None else np.array([
            [800.0, 0, 320.0],
            [0, 800.0, 240.0],
            [0, 0, 1.0]
        ], dtype=np.float32)
        
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((4, 1), dtype=np.float32)
        
        marker_size = 0.5
        self.marker_points_3d = np.array([
            [-marker_size/2,  marker_size/2, 0],
            [ marker_size/2,  marker_size/2, 0],
            [ marker_size/2, -marker_size/2, 0],
            [-marker_size/2, -marker_size/2, 0]
        ], dtype=np.float32)
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

    def estimate_relative_pose(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None and len(ids) > 0:
            success, rvec, tvec = cv2.solvePnP(
                self.marker_points_3d, 
                corners[0][0], 
                self.camera_matrix, 
                self.dist_coeffs, 
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if success:
                return True, tvec.flatten(), rvec.flatten()
                
        return False, None, None

