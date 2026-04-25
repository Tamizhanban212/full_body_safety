import pyrealsense2 as rs
import numpy as np
import cv2
import time
from collections import deque
import sys
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets, QtGui

# --- 1. Physical Board Configuration ---
MARKER_SIZE = 0.05
SPACING = 0.10
TRACKING_TIMEOUT = 0.5 # Seconds to remember an obstacle after it vanishes

s = MARKER_SIZE / 2
corners_0 = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]]) 
corners_1 = np.array([[-s+SPACING, s, 0], [s+SPACING, s, 0], [s+SPACING, -s, 0], [-s+SPACING, -s, 0]]) 
corners_2 = np.array([[-s, s+SPACING, 0], [s, s+SPACING, 0], [s, -s+SPACING, 0], [-s, -s+SPACING, 0]]) 

board_obj_points = np.array([corners_0, corners_1, corners_2], dtype=np.float32)
board_ids = np.array([[0], [1], [2]], dtype=np.int32)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
board = cv2.aruco.Board(board_obj_points, aruco_dict, board_ids)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# --- 2. Camera Setup ---
pipeline = rs.pipeline()
config = rs.config()

try:
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    pipeline.start(config)
except RuntimeError as e:
    print(f"[ERROR] Pipeline Failed: {e}")
    sys.exit(1)

align = rs.align(rs.stream.color)
cv2.namedWindow('RealSense Tracking', cv2.WINDOW_AUTOSIZE)

# --- 3. UI and State Setup ---
app = QtWidgets.QApplication.instance()
if app is None:
    app = QtWidgets.QApplication(sys.argv)

calib_duration = 10.0
start_time = time.time()
is_calibrated = False

R_cam_to_base = None
tvec_cam_to_base = None
best_rvec, best_tvec = None, None

obstacles = {}
ui_items = {} 
view = None

OBSTACLE_COLORS = [
    (0.0, 1.0, 1.0, 1.0), (1.0, 1.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0), 
    (0.0, 1.0, 0.0, 1.0), (1.0, 0.5, 0.0, 1.0), (0.5, 0.5, 1.0, 1.0)
]

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
        camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                                  [0, intrinsics.fy, intrinsics.ppy],
                                  [0, 0, 1]], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1))
        
        frame = np.asanyarray(c_frame.get_data())
        current_time = time.time()
        elapsed_time = current_time - start_time

        corners, ids, _ = detector.detectMarkers(frame)

        # ==========================================================
        # PHASE 1: CALIBRATION 
        # ==========================================================
        if not is_calibrated:
            countdown = max(0, calib_duration - elapsed_time)
            cv2.putText(frame, f"CALIBRATING BASE... {countdown:.1f}s", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 3)

            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                obj_pts, img_pts = board.matchImagePoints(corners, ids)
                
                # ADD THE 'is not None' CHECK HERE:
                if obj_pts is not None and len(obj_pts) >= 4:
                    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs)
                    if success:
                        best_rvec, best_tvec = rvec, tvec
                        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, best_rvec, best_tvec, 0.1, 4)
            
            if elapsed_time > calib_duration:
                if best_rvec is not None and best_tvec is not None:
                    is_calibrated = True
                    R_cam_to_base, _ = cv2.Rodrigues(best_rvec)
                    tvec_cam_to_base = best_tvec.flatten()
                    
                    view = gl.GLViewWidget()
                    view.setWindowTitle('PyQtGraph 3D Tracking')
                    view.resize(800, 600)
                    view.opts['distance'] = 150 
                    
                    grid = gl.GLGridItem()
                    grid.scale(10, 10, 1) 
                    view.addItem(grid)
                    
                    view.addItem(gl.GLLinePlotItem(pos=np.array([[0,0,0], [15,0,0]]), color=(1,0,0,1), width=3))
                    view.addItem(gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,15,0]]), color=(0,1,0,1), width=3))
                    view.addItem(gl.GLLinePlotItem(pos=np.array([[0,0,0], [0,0,15]]), color=(0,0,1,1), width=3))
                    
                    base_text = gl.GLTextItem(pos=[0, 0, -2], text="Robot Base", font=QtGui.QFont("Helvetica", 10))
                    view.addItem(base_text)
                    
                    view.show()
                else:
                    start_time = time.time() 

        # ==========================================================
        # PHASE 2: TRACKING & VISUALIZATION 
        # ==========================================================
        else:
            cv2.putText(frame, "TRACKING ACTIVE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, best_rvec, best_tvec, 0.1, 4)

            if ids is not None:
                for i in range(len(ids)):
                    marker_id = ids[i][0]
                    c = corners[i][0]
                    cx, cy = int(np.mean(c[:, 0])), int(np.mean(c[:, 1]))
                    
                    # Draw clean ID for ALL markers (Calibration + Obstacles) in 2D frame
                    cv2.putText(frame, f"ID: {marker_id}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    if marker_id >= 3:
                        depth_m = d_frame.get_distance(cx, cy)
                        
                        if depth_m > 0:
                            pt_cam = np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m))
                            pt_robot_m = R_cam_to_base.T @ (pt_cam - tvec_cam_to_base)
                            pt_robot_cm = pt_robot_m * 100 
                            
                            if marker_id not in obstacles:
                                obstacles[marker_id] = {'prev_pos': pt_robot_cm, 'time': current_time, 
                                                        'vel': 0.0, 'traj': deque(maxlen=40), 'last_seen': current_time}
                                
                                c_val = OBSTACLE_COLORS[marker_id % len(OBSTACLE_COLORS)]
                                scatter = gl.GLScatterPlotItem(color=c_val, size=15)
                                trail = gl.GLLinePlotItem(color=c_val, width=2)
                                q_color = QtGui.QColor(int(c_val[0]*255), int(c_val[1]*255), int(c_val[2]*255))
                                text_item = gl.GLTextItem(pos=list(pt_robot_cm), text="", color=q_color, font=QtGui.QFont("Helvetica", 10))
                                
                                view.addItem(scatter)
                                view.addItem(trail)
                                view.addItem(text_item)
                                ui_items[marker_id] = {'scatter': scatter, 'trail': trail, 'text': text_item}
                            
                            obs = obstacles[marker_id]
                            dt = current_time - obs['time']
                            
                            if dt > 0.05: 
                                dist_cm = np.linalg.norm(pt_robot_cm - obs['prev_pos'])
                                obs['vel'] = dist_cm / dt
                                obs['prev_pos'] = pt_robot_cm
                                obs['time'] = current_time
                            
                            obs['traj'].append(pt_robot_cm)
                            obs['last_seen'] = current_time # Reset the timeout clock
                            
                            ui_items[marker_id]['scatter'].setData(pos=np.array([pt_robot_cm]))
                            if len(obs['traj']) > 1:
                                ui_items[marker_id]['trail'].setData(pos=np.array(obs['traj']))
                            
                            info_str = f"ID: {marker_id}\nP: {pt_robot_cm.round(1)}\nV: {obs['vel']:.1f} cm/s"
                            ui_items[marker_id]['text'].setData(pos=list(pt_robot_cm + np.array([0, 0, 3])), text=info_str)

            # --- Delayed Cleanup Phase ---
            # Only delete if current time - last seen time is strictly greater than our timeout
            missing_ids = [m_id for m_id, obs in obstacles.items() if (current_time - obs['last_seen']) > TRACKING_TIMEOUT]
            
            for m_id in missing_ids:
                view.removeItem(ui_items[m_id]['scatter'])
                view.removeItem(ui_items[m_id]['trail'])
                view.removeItem(ui_items[m_id]['text'])
                del obstacles[m_id]
                del ui_items[m_id]

            if view is not None:
                app.processEvents()

        cv2.imshow('RealSense Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    if view is not None:
        view.close()