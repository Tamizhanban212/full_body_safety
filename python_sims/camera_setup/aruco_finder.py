import pyrealsense2 as rs
import numpy as np
import cv2

# 1. Configure RealSense Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# 2. Setup ArUco Detector (OpenCV 4.7.0+ syntax)
# Use DICT_4X4_50 or DICT_6X6_250 depending on your printed marker
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

try:
    while True:
        # Wait for a coherent frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        frame = np.asanyarray(color_frame.get_data())

        # 3. Detect Markers
        corners, ids, rejected = detector.detectMarkers(frame)

        # 4. Draw results
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            print(f"Detected IDs: {ids.flatten()}")

        # Show stream
        cv2.imshow('RealSense ArUco Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()