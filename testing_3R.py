# test_manipulator.py
import tkinter as tk
from manipulator import ManipulatorApp   # assuming file is named manipulator.py
# from danger_field import DangerField

root = tk.Tk()
app = ManipulatorApp(root)

# Test movements (uncomment one by one or run all)
app.start_angle(a1=30, a2=30, a3=30, duration=2)

root.mainloop()

# After running → check manipulator_points.csv