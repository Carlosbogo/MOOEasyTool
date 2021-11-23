"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
from math import sqrt
from scipy.stats import norm
import numpy as np

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.optimize import minimize

from models.GPProblem import GPProblem
from models.MESMOProblem import MESMOProblem
from models.GaussianProcess import GaussianProcess

def mesmo_acq(GP: GaussianProcess):

    ## Compute pareto front
    problem = GPProblem(GP)
    algorithm = NSGA2()
    termination = get_termination("n_gen", 40)
    res = minimize(problem,
                   algorithm,
                   termination)
    maximums = [min(res.F[:,i]) for i in range(GP.O)]

    ## Apply mesmoc acquisition function

    problem = MESMOProblem(GP, np.array(maximums))
    algorithm = NSGA2()
    termination = get_termination("n_gen", 40)
    res = minimize(problem, algorithm, termination)

    return res.X[0], res.F[0]

