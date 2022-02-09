import numpy as np
from benchmarkFunctions.ackeley import ackley


def MOOackley(xx, c1=0.5, c2=-0.5, a=20, b=0.2, c=2*np.pi):
    return np.array([ackley(np.array([xx-c1])),ackley(np.array([xx-c2]))])