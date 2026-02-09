# # test_manipulator.py
# import manipulator
# import tkinter as tk

# root = tk.Tk()
# app = manipulator.ManipulatorApp(root)

# # Test functions
# app.home()
# app.start_pos(3, 1)
# app.start_pos(2, 2)
# app.start_angle(0, 45, -45)
# app.home()

# root.mainloop()

# test_manipulator.py
import tkinter as tk
from manipulator import ManipulatorApp   # assuming file is named manipulator.py
from danger_field import DangerField

root = tk.Tk()
app = ManipulatorApp(root)

# Test movements (uncomment one by one or run all)
app.home(duration=0)
app.start_angle(a1=30, a2=30, a3=30, duration=10)

root.mainloop()

# After running → check manipulator_points.csv