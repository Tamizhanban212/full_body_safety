import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from danger_field_jax import DangerFields


class DangerFieldGUI:
    def __init__(self, csv_path=None):
        self.csv_path = csv_path or os.path.join(
            os.path.dirname(__file__), 'manipulator_points.csv'
        )
        self.model = DangerFields()

        # Grid definition
        self.N = 50
        self.xlim = (-4.5, 4.5)
        self.ylim = (-4.5, 4.5)

        self.X, self.Y = np.meshgrid(
            np.linspace(*self.xlim, self.N),
            np.linspace(*self.ylim, self.N)
        )
        self.grid_points = np.stack(
            [self.X.ravel(), self.Y.ravel(), np.zeros(self.N * self.N)], axis=1
        )

        self._load_data()
        self._setup_gui()

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        df['point_id'] = df['point_id'].astype(int)

        required_ids = {0, 1, 2, 3}
        self.frames = []

        for t, group in df.groupby('time'):
            group = group.set_index('point_id')
            if not required_ids.issubset(group.index):
                continue

            r_starts = np.vstack([group.loc[i, ['x', 'y', 'z']].values for i in (0, 1, 2)])
            r_ends   = np.vstack([group.loc[i, ['x', 'y', 'z']].values for i in (1, 2, 3)])
            v_starts = np.vstack([group.loc[i, ['vx', 'vy', 'vz']].values for i in (0, 1, 2)])
            v_ends   = np.vstack([group.loc[i, ['vx', 'vy', 'vz']].values for i in (1, 2, 3)])

            self.frames.append({
                "time": float(t),
                "r_starts": r_starts,
                "r_ends": r_ends,
                "v_starts": v_starts,
                "v_ends": v_ends
            })

        self.times = np.array([f["time"] for f in self.frames])

    def _compute_danger_grid(self, frame):
        danger_vals = np.zeros(self.grid_points.shape[0])

        for i, p in enumerate(self.grid_points):
            d, _, _ = self.model.compute_robot_danger(
                p, frame["r_starts"], frame["r_ends"],
                frame["v_starts"], frame["v_ends"]
            )
            danger_vals[i] = max(d, 1e-12)

        return np.log10(danger_vals).reshape(self.N, self.N)

    def _setup_gui(self):
        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        plt.subplots_adjust(bottom=0.18)

        # Initial frame
        self.current_idx = 0
        Z = self._compute_danger_grid(self.frames[0])

        self.hm = self.ax.imshow(
            Z,
            extent=[*self.xlim, *self.ylim],
            origin='lower',
            cmap='inferno',
            aspect='equal'
        )
        self.cbar = plt.colorbar(self.hm, ax=self.ax)
        self.cbar.set_label("log10(Danger)")

        # Manipulator plot
        self.link_plot, = self.ax.plot([], [], 'wo-', lw=2)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title(f"Time = {self.frames[0]['time']:.2f} s")

        # Slider
        ax_slider = plt.axes([0.15, 0.05, 0.7, 0.03])
        self.slider = Slider(
            ax_slider,
            "Time (s)",
            0.0,
            2.0,
            valinit=self.frames[0]["time"]
        )
        self.slider.on_changed(self._update)

        self._draw_manipulator(self.frames[0])
        plt.show()

    def _draw_manipulator(self, frame):
        pts = np.vstack([frame["r_starts"], frame["r_ends"][-1]])
        self.link_plot.set_data(pts[:, 0], pts[:, 1])

    def _update(self, val):
        idx = np.argmin(np.abs(self.times - val))
        frame = self.frames[idx]

        Z = self._compute_danger_grid(frame)
        self.hm.set_data(Z)
        self.ax.set_title(f"Time = {frame['time']:.2f} s")

        self._draw_manipulator(frame)
        self.fig.canvas.draw_idle()


if __name__ == "__main__":
    gui = DangerFieldGUI()
