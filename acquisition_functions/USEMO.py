"""
@author: Miguel Taibo Martínez

Date: 24-Nov 2021
"""
import numpy as np
import tensorflow as tf

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.optimize import minimize
from models.EIProblem import EIProblem

from models.GPProblem import GPProblem
from models.GaussianProcess import GaussianProcess

def usemo_acq(GP: GaussianProcess, beta=0.01, showplots=False):

    ## Compute pareto front
    problem = GPProblem(GP)
    algorithm = NSGA2()
    termination = get_termination("n_gen", 40)
    res = minimize(problem,
                   algorithm,
                   termination)
    maximums = [min(res.F[:,i]) for i in range(GP.O)]
    
    ## Apply mesmoc acquisition function
    problem = EIProblem(GP, np.array(maximums))
    algorithm = NSGA2()
    termination = get_termination("n_gen", 40)
    res = minimize(problem, algorithm, termination)

    _, var = GP.GPR.predict_y(res.X)
    var_volume = tf.math.reduce_prod(var,1)
    idx = np.argmax(var_volume.numpy())

    if showplots:        
        x_tries, acqs = problem.curve() 
        GP.plotMESMO(res.X[idx],  var_volume.numpy()[idx], x_tries, acqs)

    return res.X[idx], var_volume.numpy()[idx]
