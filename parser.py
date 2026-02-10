import csv
import numpy as np


class Parser:
    def __init__(self, filename, N=3):
        self.filename = filename
        self.N = N

    def parse(self):
        r_starts, r_ends, v_starts, v_ends = [], [], [], []
        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                r_start = list(map(float, row[0:self.N]))
                r_end = list(map(float, row[3:6]))
                v_start = list(map(float, row[6:9]))
                v_end = list(map(float, row[9:12]))
                r_starts.append(r_start)
                r_ends.append(r_end)
                v_starts.append(v_start)
                v_ends.append(v_end)
        return np.array(r_starts), np.array(r_ends), np.array(v_starts), np.array(v_ends)