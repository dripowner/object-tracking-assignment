from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import numpy as np
from trackers import IoU

def linerar_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

class KalmanBoxTracker():
    def __init__(self) -> None:
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        pass

