# test_manipulator.py
import tkinter as tk
from manipulator import ManipulatorApp   # assuming file is named manipulator.py
# from danger_field_jax import DangerFields

root = tk.Tk()
app = ManipulatorApp(root)

# Test movements (uncomment one by one or run all)
app.start_angle(a1=60, a2=120, a3=90, duration=2)

root.mainloop()

# After running → check manipulator_points.csv