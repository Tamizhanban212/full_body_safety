# test_manipulator.py
import tkinter as tk
from manipulator import ManipulatorApp   # assuming file is named manipulator.py
from danger_field_jax import DangerFields

root = tk.Tk()
app = ManipulatorApp(root)

df_solver = DangerFields(k1=2.0, k2=0.5, gamma=1.1)
df, grad, control_vec = df_solver.compute_robot_danger(
    r_target=[1.0, 1.0, 0.0],
    r_starts=[[0, 0, 0], [1, 0, 0], [1.5, 0.5, 0]],
    r_ends=[[1, 0, 0], [1.5, 0.5, 0], [2.0, 1.0, 0]],
    v_starts=[[0, 0, 0], [0.1, 0, 0], [0.1, 0.1, 0]],
    v_ends=[[0.1, 0, 0], [0.1, 0.1, 0], [0.1, 0.1, 0]]
)

# Set frame rate (example: 60 FPS)
app.manipulator.set_frame_rate(60)

# Test movements (uncomment one by one or run all)
app.start_angle(a1=30, a2=30, a3=30, duration=2)

root.mainloop()

# After running → check manipulator_points.csv