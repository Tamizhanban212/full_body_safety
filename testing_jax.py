import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from danger_field_jax import DangerFields  # Import the class we just created

# --- ROBOT SIMULATOR ---
class Simple3RManipulator:
    def __init__(self):
        self.lengths = [1.0, 0.8, 0.6]
        self.angles = [0.0, 0.0, 0.0]
        self.t = 0.0
        self.dt = 0.1
        self.update_step() # Initialize positions

    def update_step(self):
        self.t += self.dt
        # Arbitrary sinusoidal motion
        self.angles[0] = 45 + 30 * np.sin(self.t * 0.5)
        self.angles[1] = 30 * np.cos(self.t * 0.7)
        self.angles[2] = 30 * np.sin(self.t * 1.2)

        # Forward Kinematics
        x, y = 0.0, 0.0
        cum_angle = 0.0
        points = [[x, y, 0.0]]
        
        # Convert degrees to radians
        rads = np.radians(self.angles)
        
        for i, L in enumerate(self.lengths):
            cum_angle += rads[i]
            x += L * np.cos(cum_angle)
            y += L * np.sin(cum_angle)
            points.append([x, y, 0.0])
        
        curr_points = np.array(points)
        r_starts = curr_points[:-1]
        r_ends = curr_points[1:]

        # Finite Difference Velocity
        if not hasattr(self, 'prev_points'):
            self.prev_points = curr_points
            
        v_starts = (r_starts - self.prev_points[:-1]) / self.dt
        v_ends = (r_ends - self.prev_points[1:]) / self.dt
        
        self.prev_points = curr_points
        return r_starts, r_ends, v_starts, v_ends

# --- VISUALIZATION APP ---
class DangerApp:
    def __init__(self):
        self.robot = Simple3RManipulator()
        # Initialize JAX Solver
        self.df_solver = DangerFields(k1=2.0, k2=0.5, gamma=1.1)
        
        # Grid for Heatmap
        self.res = 40
        self.x_rng = np.linspace(-2.5, 3.5, self.res)
        self.y_rng = np.linspace(-2.5, 3.5, self.res)
        self.X, self.Y = np.meshgrid(self.x_rng, self.y_rng)
        self.Z = np.zeros_like(self.X)

        # Plot Setup
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)
        
        self.r_s, self.r_e, self.v_s, self.v_e = self.robot.update_step()
        self.quiver_artist = None
        self.click_marker = None
        self.clicked_point = None

        # Initial Computation
        self.compute_heatmap()
        self.draw_scene()

        # Interaction
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add "Step Simulation" Button
        ax_btn = plt.axes([0.7, 0.05, 0.2, 0.075])
        self.btn = Button(ax_btn, 'Step Simulation')
        self.btn.on_clicked(self.step_simulation)

        print("Simulation Ready. Click on plot to place obstacle.")
        plt.show()

    def step_simulation(self, event):
        # Update robot state
        self.r_s, self.r_e, self.v_s, self.v_e = self.robot.update_step()
        self.compute_heatmap()
        self.draw_scene()
        
        # If a point was previously clicked, update its vector too
        if self.clicked_point is not None:
            self.update_interaction_vectors(self.clicked_point)

    def compute_heatmap(self):
        # Compute danger for every point on the grid
        print("Updating Heatmap...")
        for i in range(self.res):
            for j in range(self.res):
                pt = [self.X[i, j], self.Y[i, j], 0.0]
                d, _, _ = self.df_solver.compute_robot_danger(
                    pt, self.r_s, self.r_e, self.v_s, self.v_e
                )
                self.Z[i, j] = d

    def draw_scene(self):
        self.ax.clear()
        self.ax.set_title("Danger Field Heatmap & Control Vectors")
        self.ax.set_xlim(-2.5, 3.5)
        self.ax.set_ylim(-2.5, 3.5)
        
        # 1. Heatmap
        self.ax.contourf(self.X, self.Y, self.Z, levels=30, cmap='viridis', alpha=0.8)
        
        # 2. Robot Arm
        robot_x = np.concatenate(([self.r_s[0,0]], self.r_e[:,0]))
        robot_y = np.concatenate(([self.r_s[0,1]], self.r_e[:,1]))
        self.ax.plot(robot_x, robot_y, 'w-o', linewidth=4, markersize=8, label='Robot')

        # 3. Restore Clicked Point if exists
        if self.clicked_point is not None:
            self.ax.plot(self.clicked_point[0], self.clicked_point[1], 'rx', markersize=12, markeredgewidth=3)
            # Re-draw arrow is handled by separate update usually, but we can redraw here
            # (Simplification: We re-trigger vector calc in step_simulation)

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax: return
        self.clicked_point = [event.xdata, event.ydata, 0.0]
        
        # Draw the marker immediately
        self.draw_scene() 
        self.update_interaction_vectors(self.clicked_point)

    def update_interaction_vectors(self, point):
        # Calculate Vectors specific to this obstacle point
        danger, grad, ctrl = self.df_solver.compute_robot_danger(
            point, self.r_s, self.r_e, self.v_s, self.v_e
        )
        
        print(f"Danger: {danger:.4f} | Control Vector: {ctrl}")

        # The control vector 'ctrl' is parallel to gradient (pointing uphill/towards danger).
        # To EVADE, the robot should move OPPOSITE to this.
        evasion = -ctrl 

        # Visualize Evasion Vector on the Robot's End Effector
        ee_pos = self.r_e[-1]
        
        # Remove old quiver if exists (clearing ax handles this in draw_scene, 
        # but if we just want to update overlay:)
        self.ax.quiver(ee_pos[0], ee_pos[1], evasion[0], evasion[1], 
                       color='cyan', scale=50, width=0.01, label='Evasion Force')
        
        self.fig.canvas.draw_idle()

if __name__ == "__main__":
    app = DangerApp()