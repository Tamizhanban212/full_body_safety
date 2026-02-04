# test_manipulator.py
import manipulator
import tkinter as tk

root = tk.Tk()
app = manipulator.ManipulatorApp(root)

# Test functions
app.home()
app.start_pos(3, 1)
app.start_pos(2, 2)
app.start_angle(0, 45, -45)
app.home()

root.mainloop()