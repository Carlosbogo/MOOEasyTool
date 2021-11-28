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
from models.MESProblem import MESProblem
from models.GaussianProcess import GaussianProcess

def mes_acq(GP: GaussianProcess, showplots: bool):

    ## Compute pareto front
    problem = GPProblem(GP)
    algorithm = NSGA2()
    termination = get_termination("n_gen", 40)
    res = minimize(problem,
                   algorithm,
                   termination)
    maximums = [min(res.F[:,i]) for i in range(GP.O)]

    ## Apply mesmoc acquisition function

    problem = MESProblem(GP, np.array(maximums))
    algorithm = NSGA2()
    termination = get_termination("n_gen", 40)
    res = minimize(problem, algorithm, termination)

    if showplots:        
        x_tries, acqs = problem.curve() 
        GP.plotMESMO(res.X[0], res.F[0], x_tries, acqs)

    return res.X[0], res.F[0]

