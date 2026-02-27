import jax
import jax.numpy as jnp
import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from danger_field_jax import DangerFields

class DangerFieldControlGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = DangerFields()

        # Workspace definition
        self.N = 40  
        self.xlim = (-5.0, 5.0)
        self.ylim = (-5.0, 5.0)

        # Precompute grid points for the Danger Field
        x_vals = np.linspace(*self.xlim, self.N)
        y_vals = np.linspace(*self.ylim, self.N)
        self.X, self.Y = np.meshgrid(x_vals, y_vals)
        self.grid_points = np.stack(
            [self.X.ravel(), self.Y.ravel(), np.zeros(self.N * self.N)], axis=1
        )
        
        # Robot Kinematic Parameters
        self.L = [1.5, 1.5, 1.5] 
        self.q = np.array([np.deg2rad(30), np.deg2rad(30), np.deg2rad(30)])
        self.dq = np.zeros(3) 
        
        # Control parameters
        self.obstacle_point = None
        self.k_rep = 5.0  # Increased gain for snappier response
        self.dt = 0.02    # 50 Hz control loop

        self._setup_gui()

        # High-speed timer for the control and animation loop
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_loop)
        self.timer.start(int(self.dt * 1000)) # Convert dt to milliseconds

    def _get_kinematics(self, q, dq):
        th1 = q[0]
        th2 = q[0] + q[1]
        th3 = q[0] + q[1] + q[2]
        
        p0 = np.array([0., 0., 0.])
        p1 = p0 + np.array([self.L[0]*np.cos(th1), self.L[0]*np.sin(th1), 0.])
        p2 = p1 + np.array([self.L[1]*np.cos(th2), self.L[1]*np.sin(th2), 0.])
        p3 = p2 + np.array([self.L[2]*np.cos(th3), self.L[2]*np.sin(th3), 0.])
        
        r_starts = np.vstack([p0, p1, p2])
        r_ends = np.vstack([p1, p2, p3])
        v_starts = np.zeros((3, 3))
        v_ends = np.zeros((3, 3))
        
        return r_starts, r_ends, v_starts, v_ends

    def _compute_danger_grid(self, r_starts, r_ends, v_starts, v_ends):
        danger_vals = self.model.compute_robot_danger_batch(
            self.grid_points, r_starts, r_ends, v_starts, v_ends
        )
        danger_vals = np.maximum(danger_vals, 1e-12)
        
        # Reshape and Transpose: PyQtGraph expects (X, Y) format, not Matplotlib's (Y, X)
        return np.log10(danger_vals).reshape(self.N, self.N).T

    def _compute_control_velocity(self):
        if self.obstacle_point is None:
            return np.zeros(3)

        obs_p = np.array([self.obstacle_point[0], self.obstacle_point[1], 0.0])
        
        # Use JAX to compute the gradient of danger with respect to q
        # Define a function to take q and return total danger
        def q_to_danger(q_test):
            # Inline flattened kinematics logic for JAX
            th1 = q_test[0]
            th2 = q_test[0] + q_test[1]
            th3 = q_test[0] + q_test[1] + q_test[2]
            
            p0 = jnp.array([0., 0., 0.])
            p1 = p0 + jnp.array([self.L[0]*jnp.cos(th1), self.L[0]*jnp.sin(th1), 0.])
            p2 = p1 + jnp.array([self.L[1]*jnp.cos(th2), self.L[1]*jnp.sin(th2), 0.])
            p3 = p2 + jnp.array([self.L[2]*jnp.cos(th3), self.L[2]*jnp.sin(th3), 0.])
            
            rs = jnp.vstack([p0, p1, p2])
            re = jnp.vstack([p1, p2, p3])
            # For control, we assume static evaluation or feed current dq if needed
            # Here we follow the paper's simpler static gradient for reactive control
            vs = jnp.zeros((3, 3))
            ve = jnp.zeros((3, 3))
            
            return self.model._fast_total_scalar(
                jnp.array(obs_p), rs, re, vs, ve, 
                self.model.k1, self.model.k2, self.model.gamma
            )

        # Compute both value and gradient efficiently
        d0, grad_q = jax.value_and_grad(q_to_danger)(self.q)
        
        if d0 < 1e-6: 
            return np.zeros(3)
            
        return -self.k_rep * np.array(grad_q)

    def _setup_gui(self):
        self.setWindowTitle("Real-Time Kinematic Danger Field Control")
        self.resize(800, 800)
        
        # Setup pyqtgraph widget
        self.pw = pg.PlotWidget()
        self.setCentralWidget(self.pw)
        self.pw.setXRange(*self.xlim)
        self.pw.setYRange(*self.ylim)
        self.pw.setAspectLocked(True)
        self.pw.setTitle("Click anywhere to place an obstacle")

        # 1. Heatmap (ImageItem)
        self.img = pg.ImageItem()
        self.pw.addItem(self.img)
        # Scale and position image to match our coordinate system (-5 to +5)
        rect = QtCore.QRectF(self.xlim[0], self.ylim[0], 
                             self.xlim[1]-self.xlim[0], self.ylim[1]-self.ylim[0])
        self.img.setRect(rect)
        self.img.setColorMap(pg.colormap.get('inferno'))
        self.img.setLevels([-6.0, -1.0]) # Lock color scaling

        # 2. Robot Manipulator Wireframe
        pen = pg.mkPen(color='w', width=5)
        self.robot_plot = pg.PlotDataItem(pen=pen, symbol='o', symbolSize=12, symbolBrush='w')
        self.pw.addItem(self.robot_plot)

        # 3. Obstacle Marker
        self.obs_scatter = pg.ScatterPlotItem(size=15, pen=pg.mkPen(None), brush=pg.mkBrush('r'), symbol='x')
        self.pw.addItem(self.obs_scatter)

        # Hook up mouse click event
        self.pw.scene().sigMouseClicked.connect(self._on_mouse_click)

        # Initial Draw
        r_starts, r_ends, v_starts, v_ends = self._get_kinematics(self.q, self.dq)
        Z = self._compute_danger_grid(r_starts, r_ends, v_starts, v_ends)
        self.img.setImage(Z, autoLevels=False)
        self._update_robot_plot(r_starts, r_ends)

    def _on_mouse_click(self, evt):
        # Convert scene coordinates to data coordinates
        if self.pw.sceneBoundingRect().contains(evt.scenePos()):
            mousePoint = self.pw.plotItem.vb.mapSceneToView(evt.scenePos())
            self.obstacle_point = (mousePoint.x(), mousePoint.y())
            self.obs_scatter.setData([mousePoint.x()], [mousePoint.y()])

    def _update_robot_plot(self, r_starts, r_ends):
        pts = np.vstack([r_starts, r_ends[-1]])
        self.robot_plot.setData(pts[:, 0], pts[:, 1])

    def _update_loop(self):
        # 1. Control step
        self.dq = self._compute_control_velocity()
        self.q = self.q + self.dq * self.dt
        
        # 2. Kinematics step
        r_starts, r_ends, v_starts, v_ends = self._get_kinematics(self.q, self.dq)
        
        # 3. Compute new Danger Field
        Z = self._compute_danger_grid(r_starts, r_ends, v_starts, v_ends)
        
        # 4. Render (Hardware Accelerated)
        self.img.setImage(Z, autoLevels=False)
        self._update_robot_plot(r_starts, r_ends)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    gui = DangerFieldControlGUI()
    gui.show()
    sys.exit(app.exec())