"""
@author: Miguel Taibo Martínez

Date: Nov 2021
"""
import tensorflow as tf

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_termination
from pymoo.optimize import minimize
import sobol_seq

from models.MESProblem import MESProblem
from models.GaussianProcess import GaussianProcess

def mes_acq(GP: GaussianProcess, N: int = 1_000, M: int = 5, showplots: bool = False):
    
    xx = sobol_seq.i4_sobol_generate(GP.d,1_000)
    samples = GP.GPR.predict_f_samples(xx,M)
    optimums = tf.math.reduce_min(samples, axis = 1)

    problem = MESProblem(GP, optimums)
    algorithm = NSGA2(pop_size=10)
    res = minimize(problem, algorithm)

    if showplots:        
        x_tries, acqs = problem.curve() 
        GP.plotMES(res.X[0], res.F[0], x_tries, acqs)

    return res.X[0], res.F[0]

