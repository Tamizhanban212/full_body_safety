# import jax
# import jax.numpy as jnp
# import sys
# import numpy as np
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtWidgets
# from danger_field_jax import DangerFields

# class DangerFieldExactPaperGUI(QtWidgets.QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.model = DangerFields()

#         # Workspace/Grid
#         self.N = 40  
#         self.xlim, self.ylim = (-5.0, 5.0), (-5.0, 5.0)
#         x_vals = np.linspace(*self.xlim, self.N)
#         y_vals = np.linspace(*self.ylim, self.N)
#         self.X, self.Y = np.meshgrid(x_vals, y_vals)
#         self.grid_points = np.stack([self.X.ravel(), self.Y.ravel(), np.zeros(self.N * self.N)], axis=1)
        
#         # Robot State & Initial Conditions
#         self.L = [1.5, 1.5, 1.5] 
#         # Start in a slightly bent configuration to avoid singularities
#         self.q_init = jnp.array([jnp.deg2rad(30), jnp.deg2rad(45), jnp.deg2rad(-30)])
#         self.q = self.q_init.copy()
        
#         self.target_point = None
#         self.obstacles = []
#         self.is_running = False
        
#         # Control Gains
#         self.k_task = 2.5
#         self.k_safe = 15.0  # Increased secondary gain so the elbow bends aggressively
#         self.dt = 0.02    

#         self._setup_gui()
        
#         self.timer = QtCore.QTimer()
#         self.timer.timeout.connect(self._update_loop)
#         self.timer.start(int(self.dt * 1000))

#     def _get_ee_pos(self, q):
#         th1, th12, th123 = q[0], q[0]+q[1], q[0]+q[1]+q[2]
#         x = self.L[0]*jnp.cos(th1) + self.L[1]*jnp.cos(th12) + self.L[2]*jnp.cos(th123)
#         y = self.L[0]*jnp.sin(th1) + self.L[1]*jnp.sin(th12) + self.L[2]*jnp.sin(th123)
#         return jnp.array([x, y, 0.0])

#     def _get_jax_kinematics(self, q):
#         th1, th12, th123 = q[0], q[0]+q[1], q[0]+q[1]+q[2]
#         p0 = jnp.array([0.,0.,0.])
#         p1 = p0 + jnp.array([self.L[0]*jnp.cos(th1), self.L[0]*jnp.sin(th1), 0.])
#         p2 = p1 + jnp.array([self.L[1]*jnp.cos(th12), self.L[1]*jnp.sin(th12), 0.])
#         p3 = p2 + jnp.array([self.L[2]*jnp.cos(th123), self.L[2]*jnp.sin(th123), 0.])
#         return jnp.vstack([p0, p1, p2]), jnp.vstack([p1, p2, p3])

#     def _compute_control(self):
#         if not self.target_point or not self.is_running:
#             return jnp.zeros(3)

#         target_p = jnp.array([self.target_point[0], self.target_point[1], 0.0])
        
#         def total_danger_func(q_in):
#             rs, re = self._get_jax_kinematics(q_in)
#             danger = 0.0
#             for obs in self.obstacles:
#                 obs_p = jnp.array([obs[0], obs[1], 0.0])
#                 danger += self.model._fast_total_scalar(obs_p, rs, re, jnp.zeros((3,3)), jnp.zeros((3,3)), 
#                                                        self.model.k1, self.model.k2, self.model.gamma)
#             return danger

#         # 1. Gradient of the Danger Field
#         _, grad_danger = jax.value_and_grad(total_danger_func)(self.q)

#         # 2. PRIMARY TASK: Reaching the target
#         ee_pos = self._get_ee_pos(self.q)
#         v_task_cartesian = self.k_task * (target_p - ee_pos)
        
#         # 2x3 Analytical Jacobian
#         J = jax.jacobian(self._get_ee_pos)(self.q)[:2, :] 
        
#         # Pseudo-inverse of Jacobian (J^#)
#         J_pinv = jnp.linalg.pinv(J) 
        
#         # Primary Task Velocity: J^# * v_task
#         dq_task = J_pinv @ v_task_cartesian[:2]

#         # 3. SECONDARY TASK: Safety (Danger Field Repulsion)
#         # q_dot_0 = -k_D * grad(DF)
#         dq_0 = -self.k_safe * grad_danger

#         # 4. NULL-SPACE PROJECTION
#         # N = (I - J^# * J)
#         N = jnp.eye(3) - J_pinv @ J
        
#         # Project safety task into the null-space of the primary task
#         dq_null = N @ dq_0

#         # EXACT PAPER IMPLEMENTATION (Section IV-B)
#         dq_final = dq_task + dq_null
        
#         return np.array(dq_final)

#     def _setup_gui(self):
#         self.setWindowTitle("Exact Paper Implementation: Section IV-B")
#         self.resize(800, 900)
        
#         main_widget = QtWidgets.QWidget()
#         self.setCentralWidget(main_widget)
#         layout = QtWidgets.QVBoxLayout(main_widget)

#         # Plotting Area
#         self.pw = pg.PlotWidget()
#         layout.addWidget(self.pw, stretch=1)
#         self.pw.setXRange(*self.xlim); self.pw.setYRange(*self.ylim)
#         self.pw.setAspectLocked(True)
#         self.pw.setTitle("1. Click Target | 2. Place Obstacles near elbow | 3. Start")
        
#         self.img = pg.ImageItem(); self.pw.addItem(self.img)
#         self.img.setRect(QtCore.QRectF(self.xlim[0], self.ylim[0], 10, 10))
#         self.img.setColorMap(pg.colormap.get('inferno'))
        
#         self.robot_plot = pg.PlotDataItem(pen=pg.mkPen('w', width=5), symbol='o')
#         self.pw.addItem(self.robot_plot)
        
#         self.target_marker = pg.ScatterPlotItem(size=15, brush='g')
#         self.pw.addItem(self.target_marker)
        
#         self.obs_markers = pg.ScatterPlotItem(size=12, brush='r', symbol='x')
#         self.pw.addItem(self.obs_markers)
        
#         self.pw.scene().sigMouseClicked.connect(self._on_click)

#         # Control Panel
#         ctrl_layout = QtWidgets.QHBoxLayout()
#         layout.addLayout(ctrl_layout)
        
#         self.btn_start = QtWidgets.QPushButton("Start")
#         self.btn_start.setMinimumHeight(40)
#         self.btn_start.clicked.connect(self._toggle_start)
#         ctrl_layout.addWidget(self.btn_start)
        
#         self.btn_reset = QtWidgets.QPushButton("Reset")
#         self.btn_reset.setMinimumHeight(40)
#         self.btn_reset.clicked.connect(self._reset_sim)
#         ctrl_layout.addWidget(self.btn_reset)

#     def _toggle_start(self):
#         self.is_running = not self.is_running
#         self.btn_start.setText("Pause" if self.is_running else "Start")

#     def _reset_sim(self):
#         self.is_running = False
#         self.btn_start.setText("Start")
#         self.q = self.q_init.copy()
#         self.target_point = None
#         self.obstacles = []
#         self.target_marker.setData([], [])
#         self.obs_markers.setData([], [])
#         self._update_loop() 

#     def _on_click(self, evt):
#         if not self.pw.sceneBoundingRect().contains(evt.scenePos()): return
#         pos = self.pw.plotItem.vb.mapSceneToView(evt.scenePos())
        
#         if self.target_point is None:
#             self.target_point = (pos.x(), pos.y())
#             self.target_marker.setData([pos.x()], [pos.y()])
#         else:
#             self.obstacles.append((pos.x(), pos.y()))
#             self.obs_markers.setData([o[0] for o in self.obstacles], [o[1] for o in self.obstacles])

#     def _update_loop(self):
#         self.q += self._compute_control() * self.dt
#         rs, re = self._get_jax_kinematics(self.q)
#         Z = self.model.compute_robot_danger_batch(self.grid_points, rs, re, jnp.zeros((3,3)), jnp.zeros((3,3)))
        
#         log_Z = jnp.log10(jnp.maximum(Z, 1e-12)).reshape(self.N, self.N).T
#         self.img.setImage(np.array(log_Z), autoLevels=False, levels=[-6.0, -1.0])
        
#         pts = jnp.vstack([rs, re[-1]])
#         self.robot_plot.setData(np.array(pts[:, 0]), np.array(pts[:, 1]))

# if __name__ == "__main__":
#     app = QtWidgets.QApplication(sys.argv)
#     gui = DangerFieldExactPaperGUI()
#     gui.show()
#     sys.exit(app.exec())


import jax
import jax.numpy as jnp
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from danger_field_jax import DangerFields

class DangerFieldHardcodedGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = DangerFields()

        self.N = 40  
        self.xlim, self.ylim = (-5.0, 5.0), (-5.0, 5.0)
        x_vals = np.linspace(*self.xlim, self.N)
        y_vals = np.linspace(*self.ylim, self.N)
        self.X, self.Y = np.meshgrid(x_vals, y_vals)
        self.grid_points = np.stack([self.X.ravel(), self.Y.ravel(), np.zeros(self.N * self.N)], axis=1)
        
        self.L = [1.5, 1.5, 1.5] 
        self.q_init = jnp.array([jnp.deg2rad(30), jnp.deg2rad(45), jnp.deg2rad(-30)])
        self.q = self.q_init.copy()
        
        self.target_point = None
        self.obstacles = []
        self.is_running = False
        
        # --- TUNABLE CONSTANTS ---
        self.k_task = 2.5
        self.k_safe = 15.0  
        self.dt = 0.02    
        
        # Safety Activation Parameters
        self.danger_threshold = 1e-6  # WHERE it reacts (Higher = gets closer to obstacles)
        self.delta = 0.001              # HOW smoothly it reacts (Lower = sharper transition)
        # -------------------------

        self._setup_gui()
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_loop)
        self.timer.start(int(self.dt * 1000))

    def _get_ee_pos(self, q):
        th1, th12, th123 = q[0], q[0]+q[1], q[0]+q[1]+q[2]
        x = self.L[0]*jnp.cos(th1) + self.L[1]*jnp.cos(th12) + self.L[2]*jnp.cos(th123)
        y = self.L[0]*jnp.sin(th1) + self.L[1]*jnp.sin(th12) + self.L[2]*jnp.sin(th123)
        return jnp.array([x, y, 0.0])

    def _get_jax_kinematics(self, q):
        th1, th12, th123 = q[0], q[0]+q[1], q[0]+q[1]+q[2]
        p0 = jnp.array([0.,0.,0.])
        p1 = p0 + jnp.array([self.L[0]*jnp.cos(th1), self.L[0]*jnp.sin(th1), 0.])
        p2 = p1 + jnp.array([self.L[1]*jnp.cos(th12), self.L[1]*jnp.sin(th12), 0.])
        p3 = p2 + jnp.array([self.L[2]*jnp.cos(th123), self.L[2]*jnp.sin(th123), 0.])
        return jnp.vstack([p0, p1, p2]), jnp.vstack([p1, p2, p3])

    def _compute_control(self):
        if not self.target_point or not self.is_running:
            return jnp.zeros(3)

        target_p = jnp.array([self.target_point[0], self.target_point[1], 0.0])
        
        def total_danger_func(q_in):
            rs, re = self._get_jax_kinematics(q_in)
            danger = 0.0
            for obs in self.obstacles:
                obs_p = jnp.array([obs[0], obs[1], 0.0])
                danger += self.model._fast_total_scalar(obs_p, rs, re, jnp.zeros((3,3)), jnp.zeros((3,3)), 
                                                       self.model.k1, self.model.k2, self.model.gamma)
            return danger

        # 1. Gradient of Danger Field
        d_val, grad_danger = jax.value_and_grad(total_danger_func)(self.q)

        # 2. Smooth Activation 'm' (Sigmoid/Tanh blend)
        normalized_diff = (d_val - self.danger_threshold) / (self.delta * self.danger_threshold + 1e-12)
        m = 0.5 * (1.0 + jnp.tanh(normalized_diff))

        # 3. Primary Task: Reaching Target
        ee_pos = self._get_ee_pos(self.q)
        v_task_cartesian = self.k_task * (target_p - ee_pos)
        J = jax.jacobian(self._get_ee_pos)(self.q)[:2, :] 
        J_pinv = jnp.linalg.pinv(J) 
        dq_task = J_pinv @ v_task_cartesian[:2]

        # 4. Secondary Task: Safety Repulsion
        dq_0 = -self.k_safe * grad_danger
        N = jnp.eye(3) - J_pinv @ J
        dq_null = N @ dq_0

        # 5. Composite Control with Smooth Switch 'm'
        dq_final = dq_task + (m * dq_null)
        
        return np.array(dq_final)

    def _setup_gui(self):
        self.setWindowTitle("Danger Field - Hardcoded Parameters")
        self.resize(800, 850)
        
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QVBoxLayout(main_widget)

        # Plotting Area
        self.pw = pg.PlotWidget()
        layout.addWidget(self.pw, stretch=1)
        self.pw.setXRange(*self.xlim); self.pw.setYRange(*self.ylim)
        self.pw.setAspectLocked(True)
        self.pw.setTitle("1. Target | 2. Obstacles | 3. Start")
        
        self.img = pg.ImageItem(); self.pw.addItem(self.img)
        self.img.setRect(QtCore.QRectF(self.xlim[0], self.ylim[0], 10, 10))
        self.img.setColorMap(pg.colormap.get('inferno'))
        
        self.robot_plot = pg.PlotDataItem(pen=pg.mkPen('w', width=5), symbol='o')
        self.pw.addItem(self.robot_plot)
        
        self.target_marker = pg.ScatterPlotItem(size=15, brush='g')
        self.pw.addItem(self.target_marker)
        
        self.obs_markers = pg.ScatterPlotItem(size=12, brush='r', symbol='x')
        self.pw.addItem(self.obs_markers)
        
        self.pw.scene().sigMouseClicked.connect(self._on_click)

        # Control Panel (Buttons Only)
        ctrl_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl_layout)
        
        self.btn_start = QtWidgets.QPushButton("Start")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self._toggle_start)
        ctrl_layout.addWidget(self.btn_start)
        
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.setMinimumHeight(40)
        self.btn_reset.clicked.connect(self._reset_sim)
        ctrl_layout.addWidget(self.btn_reset)

    def _toggle_start(self):
        self.is_running = not self.is_running
        self.btn_start.setText("Pause" if self.is_running else "Start")

    def _reset_sim(self):
        self.is_running = False
        self.btn_start.setText("Start")
        self.q = self.q_init.copy()
        self.target_point = None
        self.obstacles = []
        self.target_marker.setData([], [])
        self.obs_markers.setData([], [])
        self._update_loop() 

    def _on_click(self, evt):
        if not self.pw.sceneBoundingRect().contains(evt.scenePos()): return
        pos = self.pw.plotItem.vb.mapSceneToView(evt.scenePos())
        
        if self.target_point is None:
            self.target_point = (pos.x(), pos.y())
            self.target_marker.setData([pos.x()], [pos.y()])
        else:
            self.obstacles.append((pos.x(), pos.y()))
            self.obs_markers.setData([o[0] for o in self.obstacles], [o[1] for o in self.obstacles])

    def _update_loop(self):
        self.q += self._compute_control() * self.dt
        rs, re = self._get_jax_kinematics(self.q)
        Z = self.model.compute_robot_danger_batch(self.grid_points, rs, re, jnp.zeros((3,3)), jnp.zeros((3,3)))
        
        log_Z = jnp.log10(jnp.maximum(Z, 1e-12)).reshape(self.N, self.N).T
        self.img.setImage(np.array(log_Z), autoLevels=False, levels=[-6.0, -1.0])
        
        pts = jnp.vstack([rs, re[-1]])
        self.robot_plot.setData(np.array(pts[:, 0]), np.array(pts[:, 1]))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = DangerFieldHardcodedGUI()
    gui.show()
    sys.exit(app.exec())