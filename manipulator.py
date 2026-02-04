
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation
# import tkinter as tk
# from tkinter import ttk, messagebox
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# class Manipulator3R:
#     def __init__(self, link_lengths=(2, 1.5, 1)):
#         self.link_lengths = link_lengths
#         self.angles = [0, 0, 0]
#         self.fig, self.ax = plt.subplots()
#         # Colors for each link
#         self.link_colors = ['#1f77b4', '#2ca02c', '#d62728']  # blue, green, red
#         # Create a line for each link
#         self.link_lines = [self.ax.plot([], [], '-', lw=6, color=self.link_colors[i])[0] for i in range(3)]
#         # Black joints
#         self.joint_dots, = self.ax.plot([], [], 'o', color='black', markersize=10)
#         self.ax.set_xlim(-sum(link_lengths)-0.5, sum(link_lengths)+0.5)
#         self.ax.set_ylim(-sum(link_lengths)-0.5, sum(link_lengths)+0.5)
#         self.ax.set_aspect('equal')
#         self.target_angles = [0, 0, 0]
#         self.anim = None

#     def _forward_kinematics(self, angles):
#         x = [0]
#         y = [0]
#         theta = 0
#         for i, (l, a) in enumerate(zip(self.link_lengths, angles)):
#             theta += np.deg2rad(a)
#             x.append(x[-1] + l * np.cos(theta))
#             y.append(y[-1] + l * np.sin(theta))
#         return x, y

#     def angle(self, a1, a2, a3):
#         self.target_angles = [a1, a2, a3]
#         self._animate_to_angles()

#     def pos(self, x, y):
#         """
#         Move end-effector to (x, y) with phi=0 (default orientation)
#         """
#         l1, l2, l3 = self.link_lengths
#         phi = 0  # default orientation
#         wx = x - l3 * np.cos(np.deg2rad(phi))
#         wy = y - l3 * np.sin(np.deg2rad(phi))
#         D = (wx**2 + wy**2 - l1**2 - l2**2) / (2 * l1 * l2)
#         if np.abs(D) > 1:
#             raise ValueError('Position unreachable')
#         a2 = np.arctan2(np.sqrt(1 - D**2), D)
#         a1 = np.arctan2(wy, wx) - np.arctan2(l2 * np.sin(a2), l1 + l2 * np.cos(a2))
#         a3 = np.deg2rad(phi) - a1 - a2
#         self.target_angles = [np.rad2deg(a1), np.rad2deg(a2), np.rad2deg(a3)]
#         self._animate_to_angles()

#     def _animate_to_angles(self, block=True):
#         start = np.array(self.angles)
#         end = np.array(self.target_angles)
#         steps = 50
#         angles_seq = np.linspace(start, end, steps)
#         def update(frame):
#             self.angles = angles_seq[frame]
#             x, y = self._forward_kinematics(self.angles)
#             # Draw each link with its color
#             for i in range(3):
#                 self.link_lines[i].set_data([x[i], x[i+1]], [y[i], y[i+1]])
#             # Draw joints
#             self.joint_dots.set_data(x, y)
#             return (*self.link_lines, self.joint_dots)
#         if self.anim is not None and getattr(self.anim, 'event_source', None) is not None:
#             self.anim.event_source.stop()
#         self.anim = FuncAnimation(self.fig, update, frames=len(angles_seq), interval=30, blit=True, repeat=False)
#         if block:
#             self.fig.canvas.draw_idle()
#             self.fig.canvas.start_event_loop(steps * 0.03 + 0.5)


# # --- Tkinter GUI ---
# class ManipulatorApp:
#     def __init__(self, root):
#         self.root = root
#         self.root.title('3R Manipulator GUI')
#         self.manipulator = Manipulator3R()
#         # Link length colored labels
#         link_colors = self.manipulator.link_colors
#         link_lengths = self.manipulator.link_lengths
#         label_frame = tk.Frame(root)
#         label_frame.grid(row=0, column=0, columnspan=3, pady=(5,0))
#         for i, (color, length) in enumerate(zip(link_colors, link_lengths)):
#             tk.Label(label_frame, text=f'Link {i+1}: {length}', fg=color, font=('Arial', 12, 'bold')).pack(side='left', padx=10)
#         # Embed matplotlib figure in tkinter
#         self.canvas = FigureCanvasTkAgg(self.manipulator.fig, master=root)
#         self.canvas.get_tk_widget().grid(row=1, column=0, rowspan=10, columnspan=1, padx=10, pady=10)

#         # Angle inputs
#         ttk.Label(root, text='Angle 1 (deg):').grid(row=0, column=1, sticky='e')
#         self.angle1_entry = ttk.Entry(root, width=8)
#         self.angle1_entry.grid(row=0, column=2)
#         ttk.Label(root, text='Angle 2 (deg):').grid(row=1, column=1, sticky='e')
#         self.angle2_entry = ttk.Entry(root, width=8)
#         self.angle2_entry.grid(row=1, column=2)
#         ttk.Label(root, text='Angle 3 (deg):').grid(row=2, column=1, sticky='e')
#         self.angle3_entry = ttk.Entry(root, width=8)
#         self.angle3_entry.grid(row=2, column=2)

#         # XY inputs
#         ttk.Label(root, text='X position:').grid(row=3, column=1, sticky='e')
#         self.x_entry = ttk.Entry(root, width=8)
#         self.x_entry.grid(row=3, column=2)
#         ttk.Label(root, text='Y position:').grid(row=4, column=1, sticky='e')
#         self.y_entry = ttk.Entry(root, width=8)
#         self.y_entry.grid(row=4, column=2)

#         # Buttons
#         self.start_angle_btn = ttk.Button(root, text='Start (Angles)', command=self.start_angle)
#         self.start_angle_btn.grid(row=5, column=1, columnspan=2, pady=5)
#         self.start_pos_btn = ttk.Button(root, text='Start (X, Y)', command=self.start_pos)
#         self.start_pos_btn.grid(row=6, column=1, columnspan=2, pady=5)
#         self.home_btn = ttk.Button(root, text='Home', command=self.go_home)
#         self.home_btn.grid(row=7, column=1, columnspan=2, pady=5)

#     def start_angle(self):
#         try:
#             a1 = float(self.angle1_entry.get())
#             a2 = float(self.angle2_entry.get())
#             a3 = float(self.angle3_entry.get())
#             self.manipulator.target_angles = [a1, a2, a3]
#             self.manipulator._animate_to_angles(block=False)
#             self.canvas.draw()
#         except Exception as e:
#             messagebox.showerror('Error', f'Invalid angle input: {e}')

#     def start_pos(self):
#         try:
#             x = float(self.x_entry.get())
#             y = float(self.y_entry.get())
#             self.manipulator.pos(x, y)
#             self.canvas.draw()
#         except Exception as e:
#             messagebox.showerror('Error', f'Invalid position input: {e}')

#     def go_home(self):
#         self.manipulator.target_angles = [0, 0, 0]
#         self.manipulator._animate_to_angles(block=False)
#         self.canvas.draw()


# if __name__ == '__main__':
#     root = tk.Tk()
#     app = ManipulatorApp(root)
#     root.mainloop()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Manipulator3R:
    def __init__(self, link_lengths=(2, 1.5, 1)):
        self.link_lengths = link_lengths
        self.angles = [0, 0, 0]
        self.fig, self.ax = plt.subplots()
        self.link_colors = ['#1f77b4', '#2ca02c', '#d62728']
        self.link_lines = [self.ax.plot([], [], '-', lw=6, color=self.link_colors[i])[0] for i in range(3)]
        self.joint_dots, = self.ax.plot([], [], 'o', color='black', markersize=10)
        self.ax.set_xlim(-sum(link_lengths)-0.5, sum(link_lengths)+0.5)
        self.ax.set_ylim(-sum(link_lengths)-0.5, sum(link_lengths)+0.5)
        self.ax.set_aspect('equal')
        self.target_angles = [0, 0, 0]
        self.anim = None

    def _forward_kinematics(self, angles):
        x = [0]
        y = [0]
        theta = 0
        for i, (l, a) in enumerate(zip(self.link_lengths, angles)):
            theta += np.deg2rad(a)
            x.append(x[-1] + l * np.cos(theta))
            y.append(y[-1] + l * np.sin(theta))
        return x, y

    def angle(self, a1, a2, a3):
        self.target_angles = [a1, a2, a3]
        self._animate_to_angles()

    def pos(self, x, y):
        l1, l2, l3 = self.link_lengths
        phi = 0
        wx = x - l3 * np.cos(np.deg2rad(phi))
        wy = y - l3 * np.sin(np.deg2rad(phi))
        D = (wx**2 + wy**2 - l1**2 - l2**2) / (2 * l1 * l2)
        if np.abs(D) > 1:
            raise ValueError('Position unreachable')
        a2 = np.arctan2(np.sqrt(1 - D**2), D)
        a1 = np.arctan2(wy, wx) - np.arctan2(l2 * np.sin(a2), l1 + l2 * np.cos(a2))
        a3 = np.deg2rad(phi) - a1 - a2
        self.target_angles = [np.rad2deg(a1), np.rad2deg(a2), np.rad2deg(a3)]
        self._animate_to_angles()

    def _animate_to_angles(self, block=True):
        start = np.array(self.angles)
        end = np.array(self.target_angles)
        steps = 50
        angles_seq = np.linspace(start, end, steps)
        def update(frame):
            self.angles = angles_seq[frame]
            x, y = self._forward_kinematics(self.angles)
            for i in range(3):
                self.link_lines[i].set_data([x[i], x[i+1]], [y[i], y[i+1]])
            self.joint_dots.set_data(x, y)
            return (*self.link_lines, self.joint_dots)
        if self.anim is not None and getattr(self.anim, 'event_source', None) is not None:
            self.anim.event_source.stop()
        self.anim = FuncAnimation(self.fig, update, frames=len(angles_seq), interval=30, blit=True, repeat=False)
        if block:
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(steps * 0.03 + 0.5)

class ManipulatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title('3R Manipulator GUI')
        self.manipulator = Manipulator3R()
        link_colors = self.manipulator.link_colors
        link_lengths = self.manipulator.link_lengths
        label_frame = tk.Frame(root)
        label_frame.grid(row=0, column=0, columnspan=3, pady=(5,0))
        for i, (color, length) in enumerate(zip(link_colors, link_lengths)):
            tk.Label(label_frame, text=f'Link {i+1}: {length}', fg=color, font=('Arial', 12, 'bold')).pack(side='left', padx=10)
        self.canvas = FigureCanvasTkAgg(self.manipulator.fig, master=root)
        self.canvas.get_tk_widget().grid(row=1, column=0, rowspan=10, columnspan=1, padx=10, pady=10)
        ttk.Label(root, text='Angle 1 (deg):').grid(row=0, column=1, sticky='e')
        self.angle1_entry = ttk.Entry(root, width=8)
        self.angle1_entry.grid(row=0, column=2)
        ttk.Label(root, text='Angle 2 (deg):').grid(row=1, column=1, sticky='e')
        self.angle2_entry = ttk.Entry(root, width=8)
        self.angle2_entry.grid(row=1, column=2)
        ttk.Label(root, text='Angle 3 (deg):').grid(row=2, column=1, sticky='e')
        self.angle3_entry = ttk.Entry(root, width=8)
        self.angle3_entry.grid(row=2, column=2)
        ttk.Label(root, text='X position:').grid(row=3, column=1, sticky='e')
        self.x_entry = ttk.Entry(root, width=8)
        self.x_entry.grid(row=3, column=2)
        ttk.Label(root, text='Y position:').grid(row=4, column=1, sticky='e')
        self.y_entry = ttk.Entry(root, width=8)
        self.y_entry.grid(row=4, column=2)
        self.start_angle_btn = ttk.Button(root, text='Start (Angles)', command=self.start_angle)
        self.start_angle_btn.grid(row=5, column=1, columnspan=2, pady=5)
        self.start_pos_btn = ttk.Button(root, text='Start (X, Y)', command=self.start_pos)
        self.start_pos_btn.grid(row=6, column=1, columnspan=2, pady=5)
        self.home_btn = ttk.Button(root, text='Home', command=self.home)
        self.home_btn.grid(row=7, column=1, columnspan=2, pady=5)

    def start_angle(self, a1=None, a2=None, a3=None):
        try:
            if a1 is None:
                a1 = float(self.angle1_entry.get())
            if a2 is None:
                a2 = float(self.angle2_entry.get())
            if a3 is None:
                a3 = float(self.angle3_entry.get())
            self.manipulator.angle(a1, a2, a3)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror('Error', f'Invalid angle input: {e}')

    def start_pos(self, x=None, y=None):
        try:
            if x is None:
                x = float(self.x_entry.get())
            if y is None:
                y = float(self.y_entry.get())
            self.manipulator.pos(x, y)
            self.canvas.draw()
        except Exception as e:
            messagebox.showerror('Error', f'Invalid position input: {e}')

    def home(self):
        self.manipulator.angle(0, 0, 0)
        self.canvas.draw()

# Usage: Save as manipulator.py
# In another file: import manipulator; import tkinter as tk; root = tk.Tk(); app = manipulator.ManipulatorApp(root); app.start_pos(1, 2); root.mainloop()