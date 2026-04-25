import pyrealsense2 as rs
import numpy as np
import cv2
import time
from collections import deque
import sys

# --- Fixed Calibration Parameters ---
origin_cam_cm = np.array([0.0, 21.0, 25.0]) # Base is 0 X, 21 Y (Down), 25 Z (Forward)
origin_cam_m = origin_cam_cm / 100.0        # RealSense projection requires meters

phi = np.radians(30.0)

# Matrix: Camera Frame -> Robot Frame
R_cam_to_robot = np.array([
    [ 0,  np.cos(phi), -np.sin(phi)],
    [ 1,            0,            0],
    [ 0, -np.sin(phi), -np.cos(phi)]
])

# Inverse Matrix for drawing 3D robot axes in the camera frame
R_robot_to_cam = R_cam_to_robot.T

# 1. Setup Camera (1280x720 HD)
pipeline = rs.pipeline()
config = rs.config()

try:
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)
except RuntimeError as e:
    print(f"\n[ERROR] Pipeline Start Failed: {e}")
    sys.exit(1)

align = rs.align(rs.stream.color)

# 2. Setup OpenCV & ArUco
cv2.namedWindow('RealSense AR Tracker', cv2.WINDOW_AUTOSIZE)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Tracking State Variables
trajectory_px = deque(maxlen=50) # Store 2D pixels for the trail
prev_time = time.time()
prev_pos_robot = None
velocity = 0.0

try:
    while True:
        try:
            frames = pipeline.wait_for_frames(5000) 
        except RuntimeError:
            continue 

        aligned_frames = align.process(frames)
        c_frame = aligned_frames.get_color_frame()
        d_frame = aligned_frames.get_depth_frame()
        
        if not c_frame or not d_frame:
            continue

        intrinsics = c_frame.profile.as_video_stream_profile().intrinsics
        frame = np.asanyarray(c_frame.get_data())
        current_time = time.time()

        # --- 1. Draw Static Camera Axes Legend (Bottom Left) ---
        h, w, _ = frame.shape
        cam_leg_orig = (50, h - 50)
        cv2.line(frame, cam_leg_orig, (100, h - 50), (0, 0, 255), 3) # X (Red, Right)
        cv2.line(frame, cam_leg_orig, (50, h - 10), (0, 255, 0), 3)  # Y (Green, Down)
        cv2.circle(frame, cam_leg_orig, 5, (255, 0, 0), -1)          # Z (Blue, Forward/Into Screen)
        cv2.putText(frame, "Camera Frame", (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- 2. Project & Draw Tilted Robot Axes on Table ---
        if origin_cam_m[2] > 0: # Ensure the base is in front of the camera lens
            # Project Origin to Pixel
            base_px = rs.rs2_project_point_to_pixel(intrinsics, origin_cam_m)
            base_px = (int(base_px[0]), int(base_px[1]))
            
            # Calculate tips of X, Y, Z axes (10cm length) in 3D
            axis_len = 0.1 
            tip_x = origin_cam_m + R_robot_to_cam @ np.array([axis_len, 0, 0])
            tip_y = origin_cam_m + R_robot_to_cam @ np.array([0, axis_len, 0])
            tip_z = origin_cam_m + R_robot_to_cam @ np.array([0, 0, axis_len])
            
            # Project tips to 2D Pixels
            px_x = tuple(map(int, rs.rs2_project_point_to_pixel(intrinsics, tip_x)))
            px_y = tuple(map(int, rs.rs2_project_point_to_pixel(intrinsics, tip_y)))
            px_z = tuple(map(int, rs.rs2_project_point_to_pixel(intrinsics, tip_z)))
            
            # Draw AR lines
            cv2.line(frame, base_px, px_x, (0, 0, 255), 3) # X Axis
            cv2.line(frame, base_px, px_y, (0, 255, 0), 3) # Y Axis
            cv2.line(frame, base_px, px_z, (255, 0, 0), 3) # Z Axis
            cv2.putText(frame, "Robot Base", (base_px[0] - 40, base_px[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- 3. Detect ArUco Marker ---
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            c = corners[0][0]
            center_u, center_v = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
            depth_m = d_frame.get_distance(center_u, center_v)
            
            if depth_m > 0:
                # Store 2D pixel for visual trail
                trajectory_px.append((center_u, center_v))

                # Calculate 3D Position
                current_pt_cam = np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [center_u, center_v], depth_m)) * 100
                pos_robot = R_cam_to_robot @ (current_pt_cam - origin_cam_cm)
                
                # Calculate Velocity
                dt = current_time - prev_time
                if dt > 0 and prev_pos_robot is not None:
                    velocity = np.linalg.norm(pos_robot - prev_pos_robot) / dt
                prev_pos_robot = pos_robot
                
                # Draw Info
                cv2.putText(frame, f"Pos: X:{pos_robot[0]:.1f} Y:{pos_robot[1]:.1f} Z:{pos_robot[2]:.1f} cm", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Vel: {velocity:.1f} cm/s", 
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # Optional: Pin a floating text bubble right next to the marker
                cv2.putText(frame, f"{velocity:.1f} cm/s", (center_u + 15, center_v - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- 4. Draw Trajectory Trail ---
        for i in range(1, len(trajectory_px)):
            cv2.line(frame, trajectory_px[i-1], trajectory_px[i], (255, 0, 255), 2)

        prev_time = current_time

        # Render Stream
        cv2.imshow('RealSense AR Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()