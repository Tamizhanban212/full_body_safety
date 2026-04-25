import pyrealsense2 as rs
import numpy as np
import cv2
import time
from collections import deque

# 1. Configure RealSense Pipeline (Color + Depth)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) # Enabled Depth

# Start streaming and create alignment object
pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

# 2. Setup ArUco Detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# 3. State Variables for Tracking
trajectory = deque(maxlen=50)  # Stores the last 50 pixel coordinates for the trail
prev_time = time.time()
prev_pos = None  # Stores the previous 3D position
velocity = 0.0

try:
    while True:
        # Wait for coherent frames and align depth to color
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue

        # Get camera intrinsics for 3D deprojection
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        frame = np.asanyarray(color_frame.get_data())

        # Detect Markers
        corners, ids, rejected = detector.detectMarkers(frame)
        current_time = time.time()

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Calculate the center pixel (u, v) of the first detected marker
            c = corners[0][0]
            center_u = int(np.mean(c[:, 0]))
            center_v = int(np.mean(c[:, 1]))
            
            # Extract depth (Z) in meters
            depth_m = depth_frame.get_distance(center_u, center_v)
            
            if depth_m > 0:
                # Deproject 2D pixel to 3D point (X, Y, Z) in meters
                point_3d_m = rs.rs2_deproject_pixel_to_point(intrinsics, [center_u, center_v], depth_m)
                
                # Convert meters to cm
                pos_cm = np.array(point_3d_m) * 100 
                
                # Calculate Time Delta and Velocity
                dt = current_time - prev_time
                if dt > 0 and prev_pos is not None:
                    # Euclidean distance formula in 3D: sqrt(dx^2 + dy^2 + dz^2)
                    dist_cm = np.linalg.norm(pos_cm - prev_pos)
                    velocity = dist_cm / dt
                
                # Update previous variables and trajectory history
                prev_pos = pos_cm
                trajectory.append((center_u, center_v))
                
                # Draw Position and Velocity on the frame
                pos_text = f"Pos: X:{pos_cm[0]:.1f} Y:{pos_cm[1]:.1f} Z:{pos_cm[2]:.1f} cm"
                vel_text = f"Vel: {velocity:.1f} cm/s"
                cv2.putText(frame, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, vel_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 4. Draw Trajectory Trail
        for i in range(1, len(trajectory)):
            # Draw lines connecting the historical center points
            cv2.line(frame, trajectory[i-1], trajectory[i], (255, 0, 0), 2)

        prev_time = current_time

        # Show stream
        cv2.imshow('RealSense ArUco Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()