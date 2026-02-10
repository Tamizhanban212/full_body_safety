import os
import numpy as np
import pandas as pd
from danger_field_jax import DangerFields

class DangerNLink:
    """
    Reads manipulator_points.csv, groups rows by time and computes danger for links:
      links: 0->1, 1->2, 2->3
    Edit self.r_target (x,y,z) below as needed.
    Output CSV has columns: time, danger
    """
    def __init__(self, csv_path=None):
        # Set r_target (x,y,z) here as requested
        self.r_target = (1, 1, 0.0)  # <-- edit this inside the class
        self.csv_path = csv_path or os.path.join(os.path.dirname(__file__), 'manipulator_points.csv')
        self.model = DangerFields()

    def plot_log_danger_vs_time(self):
        import matplotlib.pyplot as plt
        df = pd.read_csv(self.csv_path)
        times = []
        log_dangers = []
        required_ids = {0, 1, 2, 3}

        for t, group in df.groupby('time'):
            group = group.copy()
            group['point_id'] = group['point_id'].astype(int)
            group = group.set_index('point_id')
            if not required_ids.issubset(set(group.index)):
                continue

            r_starts = np.vstack([group.loc[i, ['x', 'y', 'z']].values.astype(float) for i in (0, 1, 2)])
            r_ends   = np.vstack([group.loc[i, ['x', 'y', 'z']].values.astype(float) for i in (1, 2, 3)])
            v_starts = np.vstack([group.loc[i, ['vx', 'vy', 'vz']].values.astype(float) for i in (0, 1, 2)])
            v_ends   = np.vstack([group.loc[i, ['vx', 'vy', 'vz']].values.astype(float) for i in (1, 2, 3)])

            danger, _, _ = self.model.compute_robot_danger(self.r_target, r_starts, r_ends, v_starts, v_ends)
            log_danger = np.log10(max(danger, 1e-12))
            times.append(float(t))
            log_dangers.append(log_danger)

        plt.figure(figsize=(8, 4))
        plt.plot(times, log_dangers, marker='o')
        plt.xlabel('Time (s)')
        plt.ylabel('log10(Danger)')
        plt.title('Log Danger vs Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    runner = DangerNLink()
    runner.plot_log_danger_vs_time()