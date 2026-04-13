import jax
import jax.numpy as jnp
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from collections import deque

from danger_field_jax import DangerFields

# Enable 64-bit precision for stable gradients
jax.config.update("jax_enable_x64", True)

class DangerFieldKeyboardGUI(QtWidgets.QMainWindow):
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
        self.last_dq = jnp.zeros(3)
        
        # --- Interactive & Keyboard State ---
        self.target_point = None
        self.obstacles = [] 
        self.selected_obs_idx = None
        self.is_running = False
        self.keys = {'up': False, 'down': False, 'left': False, 'right': False}
        self.danger_history = deque(maxlen=200)
        
        # --- TUNABLE CONSTANTS ---
        self.k_task = 2.0
        self.k_safe = 5.0  
        self.dt = 0.02    
        self.danger_threshold = 1e-5  
        self.delta = 0.1            
        self.keyboard_speed = 3.5  # Max m/s when pressing arrow keys
        self.friction = 0.15       # EMA weight for smoothing starts/stops
        # -------------------------

        # JAX Compilation: Vectorized Multi-Obstacle Gradient Calculator
        self.compiled_danger_grad = self._build_vectorized_grad_func()

        self._setup_gui()
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus) # Required to capture keystrokes
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_loop)
        self.timer.start(int(self.dt * 1000))

    def _build_vectorized_grad_func(self):
        """
        Creates a purely compiled JAX function that processes all obstacles in parallel 
        using vmap, eliminating all Python loop tracing overhead.
        """
        @jax.jit
        def calc_grad(q_in, last_dq_in, obs_p, obs_v, L_arr, k1, k2, gamma):
            # 1. Pure Kinematics Inside JIT
            th1, th12, th123 = q_in[0], q_in[0]+q_in[1], q_in[0]+q_in[1]+q_in[2]
            dth1, dth12, dth123 = last_dq_in[0], last_dq_in[0]+last_dq_in[1], last_dq_in[0]+last_dq_in[1]+last_dq_in[2]
            
            p0 = jnp.array([0.,0.,0.])
            p1 = p0 + jnp.array([L_arr[0]*jnp.cos(th1), L_arr[0]*jnp.sin(th1), 0.])
            p2 = p1 + jnp.array([L_arr[1]*jnp.cos(th12), L_arr[1]*jnp.sin(th12), 0.])
            p3 = p2 + jnp.array([L_arr[2]*jnp.cos(th123), L_arr[2]*jnp.sin(th123), 0.])
            rs = jnp.vstack([p0, p1, p2])
            re = jnp.vstack([p1, p2, p3])

            v0 = jnp.array([0.,0.,0.])
            v1 = v0 + jnp.array([-L_arr[0]*jnp.sin(th1)*dth1, L_arr[0]*jnp.cos(th1)*dth1, 0.])
            v2 = v1 + jnp.array([-L_arr[1]*jnp.sin(th12)*dth12, L_arr[1]*jnp.cos(th12)*dth12, 0.])
            v3 = v2 + jnp.array([-L_arr[2]*jnp.sin(th123)*dth123, L_arr[2]*jnp.cos(th123)*dth123, 0.])
            vs = jnp.vstack([v0, v1, v2])
            ve = jnp.vstack([v1, v2, v3])

            # 2. Vectorized Obstacle Evaluation
            def single_obs_danger(p, v):
                return self.model._fast_total_scalar(p, v, rs, re, vs, ve, k1, k2, gamma)
            
            # vmap maps the single_obs function over the arrays of shape (N, 3)
            return jnp.sum(jax.vmap(single_obs_danger)(obs_p, obs_v))

        return jax.value_and_grad(calc_grad)

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
            return jnp.zeros(3), 0.0

        target_p = jnp.array([self.target_point[0], self.target_point[1], 0.0])
        
        # 1. Fast Vectorized Danger/Gradient Computation
        if not self.obstacles:
            d_val, grad_danger = 0.0, jnp.zeros(3)
        else:
            obs_p = jnp.array([[o['x'], o['y'], 0.0] for o in self.obstacles])
            obs_v = jnp.array([[o['vx'], o['vy'], 0.0] for o in self.obstacles])
            
            d_val, grad_danger = self.compiled_danger_grad(
                self.q, self.last_dq, obs_p, obs_v, 
                jnp.array(self.L), self.model.k1, self.model.k2, self.model.gamma
            )

        # 2. Smooth Activation 'm'
        normalized_diff = (d_val - self.danger_threshold) / (self.delta * self.danger_threshold + 1e-12)
        m = 0.5 * (1.0 + jnp.tanh(normalized_diff))

        # 3. Primary Task
        ee_pos = self._get_ee_pos(self.q)
        v_task_cartesian = self.k_task * (target_p - ee_pos)
        J = jax.jacobian(self._get_ee_pos)(self.q)[:2, :] 
        J_pinv = jnp.linalg.pinv(J) 
        dq_task = J_pinv @ v_task_cartesian[:2]

        # 4. Secondary Task (Pure Null-Space)
        dq_0 = -self.k_safe * grad_danger
        N = jnp.eye(3) - J_pinv @ J
        dq_null = N @ dq_0
        
        # We completely remove the dq_avoid blend so the 
        # end-effector is never allowed to abandon the target.

        # 5. Composite Control & Numerical Cap (Strict Task Priority)
        dq_raw = dq_task + (m * dq_null)
        dq_norm = jnp.linalg.norm(dq_raw)
        max_dq = 7.0 
        dq_raw = jnp.where(dq_norm > max_dq, dq_raw * (max_dq / (dq_norm + 1e-12)), dq_raw)
        
        # 6. Smooth Application
        alpha = 0.2  
        dq_smoothed = alpha * dq_raw + (1.0 - alpha) * self.last_dq
        
        dq_smoothed = jnp.nan_to_num(dq_smoothed, nan=0.0)
        safe_d_val = float(jnp.nan_to_num(d_val, nan=0.0))
        
        self.last_dq = dq_smoothed 
        return np.array(dq_smoothed), safe_d_val

    # --- KEYBOARD EVENT HANDLERS ---
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Up: self.keys['up'] = True
        elif event.key() == QtCore.Qt.Key.Key_Down: self.keys['down'] = True
        elif event.key() == QtCore.Qt.Key.Key_Left: self.keys['left'] = True
        elif event.key() == QtCore.Qt.Key.Key_Right: self.keys['right'] = True

    def keyReleaseEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Up: self.keys['up'] = False
        elif event.key() == QtCore.Qt.Key.Key_Down: self.keys['down'] = False
        elif event.key() == QtCore.Qt.Key.Key_Left: self.keys['left'] = False
        elif event.key() == QtCore.Qt.Key.Key_Right: self.keys['right'] = False

    def _setup_gui(self):
        self.setWindowTitle("Keyboard Controlled Danger Field")
        self.resize(1000, 900)
        
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        layout = QtWidgets.QVBoxLayout(main_widget)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        layout.addWidget(splitter, stretch=1)

        self.pw = pg.PlotWidget(title="1. Click to set Target. 2. Click to spawn Obstacles. 3. Click Obstacles to Select (Yellow). 4. Move with Arrow Keys.")
        splitter.addWidget(self.pw)
        self.pw.setXRange(*self.xlim); self.pw.setYRange(*self.ylim)
        self.pw.setAspectLocked(True)
        
        self.img = pg.ImageItem(); self.pw.addItem(self.img)
        self.img.setRect(QtCore.QRectF(self.xlim[0], self.ylim[0], 10, 10))
        self.img.setColorMap(pg.colormap.get('inferno'))
        
        self.robot_plot = pg.PlotDataItem(pen=pg.mkPen('w', width=5), symbol='o')
        self.pw.addItem(self.robot_plot)
        
        self.target_marker = pg.ScatterPlotItem(size=15, brush='g')
        self.pw.addItem(self.target_marker)
        
        self.obs_markers = pg.ScatterPlotItem(size=15)
        self.pw.addItem(self.obs_markers)
        
        self.pw.scene().sigMouseClicked.connect(self._on_background_click)

        self.telemetry_pw = pg.PlotWidget(title="Danger Field Magnitude (Total D)")
        splitter.addWidget(self.telemetry_pw)
        self.telemetry_pw.setLabel('left', 'Magnitude')
        self.telemetry_pw.setLabel('bottom', 'Time steps')
        self.telemetry_pw.setLogMode(y=True) 
        self.danger_curve = self.telemetry_pw.plot(pen=pg.mkPen('r', width=2))
        
        splitter.setSizes([700, 200])

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
        self.setFocus() # Regain focus for arrow keys after clicking UI buttons

    def _reset_sim(self):
        self.is_running = False
        self.btn_start.setText("Start")
        self.q = self.q_init.copy()
        self.last_dq = jnp.zeros(3)
        self.danger_history.clear()
        self.danger_curve.setData([])
        
        self.target_point = None
        self.target_marker.setData([], [])
        
        for obs in self.obstacles:
            self.pw.removeItem(obs['vel_line'])
            self.pw.removeItem(obs['text'])
            
        self.obstacles = []
        self.selected_obs_idx = None
        self.obs_markers.setData([], [])
        
        self.setFocus()
        self._update_loop() 

    def _on_background_click(self, evt):
        if not self.pw.sceneBoundingRect().contains(evt.scenePos()): return
        pos = self.pw.plotItem.vb.mapSceneToView(evt.scenePos())
        px, py = pos.x(), pos.y()
        
        # 1. Check if we clicked an existing obstacle to select it
        for i, obs in enumerate(self.obstacles):
            dist = np.hypot(obs['x'] - px, obs['y'] - py)
            if dist < 0.4:  # Click tolerance radius
                self.selected_obs_idx = i
                return
        
        # 2. Handle Spawning Target or New Obstacles
        if self.target_point is None:
            self.target_point = (px, py)
            self.target_marker.setData([px], [py])
        else:
            vel_line = pg.PlotDataItem(pen=pg.mkPen('c', width=2))
            self.pw.addItem(vel_line)
            text = pg.TextItem(color='c', anchor=(0, 1))
            self.pw.addItem(text)
            
            self.obstacles.append({
                'x': px, 'y': py,
                'vx': 0.0, 'vy': 0.0,
                'vel_line': vel_line,
                'text': text
            })
            # Automatically select the newly spawned obstacle
            self.selected_obs_idx = len(self.obstacles) - 1

    def _update_loop(self):
        # 1. Update Obstacle Physics via Keyboard (Runs even if paused)
        for i, obs in enumerate(self.obstacles):
            target_vx, target_vy = 0.0, 0.0
            
            # Apply arrow keys ONLY to the selected obstacle
            if i == self.selected_obs_idx:
                if self.keys['up']: target_vy = self.keyboard_speed
                if self.keys['down']: target_vy = -self.keyboard_speed
                if self.keys['left']: target_vx = -self.keyboard_speed
                if self.keys['right']: target_vx = self.keyboard_speed
            
            # Smooth Acceleration / Deceleration via EMA
            obs['vx'] = self.friction * target_vx + (1.0 - self.friction) * obs['vx']
            obs['vy'] = self.friction * target_vy + (1.0 - self.friction) * obs['vy']
            
            # Update position
            obs['x'] += obs['vx'] * self.dt
            obs['y'] += obs['vy'] * self.dt
            
            # Update visual lines and text
            obs['vel_line'].setData([obs['x'], obs['x'] + obs['vx']], [obs['y'], obs['y'] + obs['vy']])
            v_mag = np.hypot(obs['vx'], obs['vy'])
            obs['text'].setText(f"|v|: {v_mag:.1f}")
            obs['text'].setPos(obs['x'], obs['y'])

        # Render all obstacles with highlighting for selection
        if self.obstacles:
            brushes = [pg.mkBrush('y') if i == self.selected_obs_idx else pg.mkBrush('r') for i in range(len(self.obstacles))]
            self.obs_markers.setData([o['x'] for o in self.obstacles], [o['y'] for o in self.obstacles], brush=brushes)
            
        # 2. Control & Telemetry (Only when running)
        if self.is_running:
            dq, d_val = self._compute_control()
            self.q += dq * self.dt
            
            self.danger_history.append(d_val)
            self.danger_curve.setData(list(self.danger_history))
        
        # 3. Heatmap Visualization
        rs, re = self._get_jax_kinematics(self.q)
        zeros_3x3 = jnp.zeros((3, 3))
        v_grid_zeros = jnp.zeros((self.N * self.N, 3))
        
        Z = self.model.compute_robot_danger_batch(
            self.grid_points, v_grid_zeros, rs, re, zeros_3x3, zeros_3x3
        )
        log_Z = jnp.log10(jnp.maximum(Z, 1e-12)).reshape(self.N, self.N).T
        self.img.setImage(np.array(log_Z), autoLevels=False, levels=[-6.0, -1.0])
        
        pts = jnp.vstack([rs, re[-1]])
        self.robot_plot.setData(np.array(pts[:, 0]), np.array(pts[:, 1]))

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = DangerFieldKeyboardGUI()
    gui.show()
    sys.exit(app.exec())